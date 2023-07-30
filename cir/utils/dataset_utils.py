import torch

from avalanche.benchmarks.utils import AvalancheSubset


def get_dataset_subset_by_class_list(
        dataset,
        subset_classes,
        num_samples=None,
        map_classes=True,
) -> AvalancheSubset:
    """ Returns subset of a dataset from a list of classes
    """
    targets = torch.LongTensor(dataset.targets)

    def get_class_idx(c):
        idx_c = torch.where(targets == c)[0]
        if num_samples:
            idx_c = idx_c[torch.randperm(len(idx_c))]
            idx_c = idx_c[:num_samples]

        return idx_c

    all_idx = [get_class_idx(c) for c in subset_classes]
    all_idx = torch.cat(all_idx)

    # Map classes ?
    target_classes = list(range(len(subset_classes)))
    if map_classes:
        class_mapping = {k: v for (k, v) in zip(subset_classes, target_classes)}
    else:
        class_mapping = None

    # Create subset
    subset_ds = AvalancheSubset(dataset, all_idx, class_mapping=class_mapping)

    return subset_ds
