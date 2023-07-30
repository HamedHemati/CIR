from typing import Dict
import math
from operator import itemgetter

from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ReservoirSamplingBuffer
)

from avalanche.benchmarks.utils import (
    AvalancheSubset,
    AvalancheConcatDataset
)


class AdaptiveBuffer(ExemplarsBuffer):
    def __init__(
            self,
            max_size: int,
            total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """

        super().__init__(max_size)

        # Total number of classes
        self.total_num_groups = total_num_classes

        # A dictionary that stores one exemplar buffer per class
        self.buffer_groups: Dict[int, ExemplarsBuffer] = {}

        # Number of observations for each class c in the dataset
        # self.n_obs_c = {c: 0 for c in range(total_num_classes)}
        self.n_obs_c = {}

        # Seen classes
        self.seen_classes = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheSubset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # Update number of observations for the classes in the current exp
        for c in cl_datasets.keys():
            if c not in self.n_obs_c:
                self.n_obs_c[c] = 1
            else:
                self.n_obs_c[c] += 1

        # associate lengths to classes
        class_to_len, q_c = self.get_group_lengths()

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]

            if class_id in self.buffer_groups and \
                    len(self.buffer_groups[class_id].buffer) > 0:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(
                strategy, class_to_len[class_id]
            )

        # NEW STEP: Remove extra items
        total = sum([len(self.buffer_groups[c].buffer) for c in
                     self.buffer_groups.keys()])
        if total > self.max_size:
            c_sorted = sorted(q_c.items(), key=itemgetter(1),
                              reverse=False)
            c_sorted = [x[0] for x in c_sorted]
            extra = total - self.max_size
            for c in c_sorted:
                if c in self.buffer_groups:
                    remaining = total - self.max_size
                    n_to_remove = math.ceil(q_c[c] * extra)
                    n_to_remove = min(
                        remaining,
                        min(n_to_remove, len(self.buffer_groups[c].buffer)))

                    # Remove additional samples from class c
                    total -= n_to_remove
                    class_to_len[c] -= n_to_remove
                    self.buffer_groups[c].resize(strategy, class_to_len[c])

                    if total == self.max_size:
                        break

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffers."""
        self.max_size = new_size
        lens = self.get_group_lengths()
        for c, ll in lens.items():
            self.buffer_groups[c].resize(strategy, ll)

    @property
    def buffer_datasets(self):
        """Return group buffers as a list of `AvalancheDataset`s."""
        return [g.buffer for g in self.buffer_groups.values()]

    @property
    def buffer(self):
        return AvalancheConcatDataset(
            [g.buffer for g in self.buffer_groups.values()]
        )

    def get_group_lengths(self):
        # Calculate quota per class
        q_c = {}
        for c in self.seen_classes:
            q_c[c] = 1 / self.n_obs_c[c]

        # Normalize
        total = sum(list(q_c.values()))
        q_c = {k: v / total for k, v in q_c.items()}

        # Quota per class
        lengths = {k: math.ceil(v * self.max_size) for k, v in q_c.items()}

        return lengths, q_c
