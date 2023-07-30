from typing import TYPE_CHECKING

import torch
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    concat_datasets,
)


if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ClassBalancedWithLogitsBuffer:
    """Buffer updated using class-balanced sampling."""

    def __init__(self, max_size: int, seed: int = 0):
        """
        :param max_size:
        """
        self.max_size = max_size
        self.seen_classes = set()
        self.buffer: AvalancheDataset = concat_datasets([])

        # Set random number generator
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        # Update seen classes
        self.seen_classes.update(set(new_data.targets))

        # Calculate number of samples per class
        n_samples_per_class = self.max_size // len(self.seen_classes)
        self.class_to_n_samples = {c: n_samples_per_class
                                   for c in self.seen_classes}

        # Add remaining free slots
        rem = self.max_size % n_samples_per_class
        for i, c in zip(range(rem), self.seen_classes):
            self.class_to_n_samples[c] += 1

        # Merge new data with buffer
        merged_data = concat_datasets([self.buffer, new_data])

        # Retrieve random subsets for each class in the merged dataset
        merged_targets = torch.LongTensor(merged_data.targets)

        def get_subset_for_class(c):
            indices = torch.where(merged_targets == c)[0]
            indices = indices[torch.randperm(len(indices), generator=self.rng)]
            indices = indices[:self.class_to_n_samples[c]]
            return indices

        subsets = [get_subset_for_class(c) for c in self.seen_classes]
        subset_indices = torch.cat(subsets)

        self.buffer = merged_data.subset(subset_indices)


__all__ = ["ClassBalancedWithLogitsBuffer"]
