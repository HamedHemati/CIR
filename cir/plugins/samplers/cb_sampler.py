from itertools import cycle
import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    def __init__(
            self,
            dataset,
            classes_to_use: list,
            n_mbatches: int = 10,
            mbatch_size: int = 32,
    ):
        """

        :param dataset: the dataset to sample from.
        :param classes_to_use: list of classes to use in the sampler. Classes
            not available in the list will be ignored.
        :param n_mbatches: number of mini-batches.
        :param mbatch_size: size of each mini-batch.
        """
        self.dataset = dataset
        self.classes_to_use = classes_to_use
        self.n_mbatches = n_mbatches

        # Calculate number of samples per class to be used in each mini-batch
        n_samples_per_class = mbatch_size // len(classes_to_use)
        self.n_sc = {c: n_samples_per_class for c in classes_to_use}
        remaining = mbatch_size % len(classes_to_use)
        for i in range(remaining):
            c = self.classes_to_use[i]
            self.n_sc[c] += 1

        # Extract indices per class
        self.class_to_idx = {c: self.get_class_indices(c)[0] for c in
                             classes_to_use}
        self.class_to_ns = {c: self.get_class_indices(c)[1] for c in
                            classes_to_use}

    def get_class_indices(self, c):
        """ Returns indices for class c.
        """
        # Get list of indices for class c and shuffle them
        targets = torch.LongTensor(self.dataset.targets)
        idx_c = torch.where(targets == c)[0]
        idx_c = idx_c[torch.randperm(len(idx_c))]

        # Create a cyclic iterator for each class
        idx_c = list(idx_c.numpy())
        n_samples = len(idx_c)

        return cycle(idx_c), n_samples

    def __iter__(self):
        """ Returns iterator for the list of indices.
        """
        # For each mini-batch
        for i in range(self.n_mbatches):
            # For each class in the mini-batch
            for c in self.classes_to_use:
                # For each sample in the class
                for j in range(self.n_sc[c]):
                    idx_c = next(self.class_to_idx[c])
                    yield idx_c

    def __len__(self):
        return sum(self.class_to_ns.values())
