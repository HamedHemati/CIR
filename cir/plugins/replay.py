from typing import Optional, TYPE_CHECKING
import torch
from itertools import cycle
import random

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer
)

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class ReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        buffer_size: int = 200,
        batch_size_buffer: int = None,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        replay_mode="normal"
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.batch_size_buffer = batch_size_buffer
        self.storage_policy = storage_policy
        assert storage_policy.max_size == self.buffer_size

        self.replay_mode = replay_mode
        self.rand_indices = None
        self.n_buff_samples = None

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training_epoch(
            self,
            strategy: "SupervisedTemplate",
            **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            return

        self.n_buff_samples = min(self.batch_size_buffer,
                                  len(self.storage_policy.buffer))
        if self.replay_mode == "normal":
            self.rand_indices = cycle(
                torch.randperm(len(self.storage_policy.buffer)))

        elif self.replay_mode == "class-balanced":
            # Set of available classes in the buffer
            cls = set(self.storage_policy.buffer.targets)

            # All indices in the buffer
            targets = torch.LongTensor(self.storage_policy.buffer.targets)

            # Class to index
            cls_to_idx = {}
            # Class to len
            cls_to_len = {}

            max_c_len = 0
            for c in cls:
                list_idx = list(torch.where(targets == c)[0].numpy())
                random.shuffle(list_idx)
                cls_to_idx[c] = list_idx
                cls_to_len[c] = len(cls_to_idx[c])
                if cls_to_len[c] > max_c_len:
                    max_c_len = cls_to_len[c]
            cls_to_idx = {c: cycle(v) for c, v in cls_to_idx.items()}

            # Number of samples from each to be present in each batch
            cls_to_num = {c: 0 for c in cls}
            total = 0
            for c in cycle(cls):
                if total == self.n_buff_samples:
                    break
                if cls_to_num[c] + 1 > cls_to_len[c]:
                    continue

                cls_to_num[c] += 1
                total += 1

            # Final rand indices
            self.rand_indices = []
            for _ in range(max_c_len):
                for c in cls:
                    self.rand_indices += [next(cls_to_idx[c]) for _ in
                                          range(cls_to_num[c])]

            self.rand_indices = cycle(torch.LongTensor(self.rand_indices))

        else:
            pass

    def before_training_iteration(
        self,
        strategy: "SupervisedTemplate",
        **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            return

        # Select random items from the buffer
        selected_indices = [next(self.rand_indices).item()
                            for _ in range(self.n_buff_samples)]
        selected_indices = torch.LongTensor(selected_indices)

        selected_items = self.storage_policy.buffer[selected_indices]

        # Buffer mask
        buffer_mask = [0] * strategy.mbatch[0].shape[0] + \
                      [1] * len(selected_indices)
        strategy.buffer_mask = torch.LongTensor(buffer_mask)

        # Combine buffer and main loader items
        for i in range(len(strategy.mbatch)):
            strategy.mbatch[i] = torch.cat(
                (strategy.mbatch[i], selected_items[i].to(strategy.device)),
                dim=0)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
