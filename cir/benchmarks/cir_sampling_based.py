import numpy as np
import math
import torch
from torchvision.transforms import transforms
import warnings
from itertools import cycle

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.datasets import CIFAR100, TinyImagenet
from avalanche.benchmarks.utils.classification_dataset import (
    classification_subset
)

from cir.benchmarks.cir_utils import get_pmf, get_per_entity_prob


def cir_generator(
        train_set,
        test_set,
        n_e,
        s_e,
        dist_first_occurrence=None,
        dist_recurrence=None,
        seed=0,
):
    """

    :param train_set: train set of the original dataset.
    :param test_set: test set of the original dataset.
    :param n_e: length of the stream (number of experiences).
    :param s_e: number of samples in each experience.
    :param dist_first_occurrence: probability distribution of the first-time
        occurrence for each class along the stream.
    :param dist_recurrence: per-class probability of repetition after its first
        occurrence.
    :param seed: random seed.

    :return: CIR stream.
    """
    # Initialize random generators for numpy as torch
    rng_np = np.random.RandomState(seed=seed)
    rng_torch = torch.Generator()
    rng_torch.manual_seed(seed)

    # List classes in the dataset and shuffle them
    classes = list(set(train_set.targets))
    rng_np.shuffle(classes)
    n_classes = len(classes)

    # Initialize stream table with zeros
    stream_table = torch.zeros(n_classes, n_e)

    # Get indices of samples by class
    train_targets = torch.LongTensor(train_set.targets)

    def get_class_indices(c):
        indices_c = torch.where(train_targets == c)[0]
        return indices_c

    class_to_indices = {c: get_class_indices(c) for c in range(n_classes)}

    # ----------> First occurence of each class
    # Get first occurrence probabilities via the PMF
    pmf_probs = get_pmf(n_e, **dist_first_occurrence)
    first_occurrences = rng_np.choice(list(range(n_e)), n_classes, p=pmf_probs,
                                      replace=True)

    for c in range(n_classes):
        stream_table[c, first_occurrences[c]] = 1

    # ----------> Recurrence of each class
    # Compute probability of repetition for each class
    p_r = get_per_entity_prob(n_classes, **dist_recurrence)

    # Randomly assign recurrence probability to each class
    rng_np.shuffle(p_r)

    # For recurrences for each class in the stream table
    for c in range(n_classes):
        # Sets recurrences for a given class
        remaining = n_e - (first_occurrences[c] + 1)
        rands = torch.rand(remaining, generator=rng_torch)
        mask = rands < p_r[c]
        rands[mask] = 1
        rands[~mask] = 0
        stream_table[c, first_occurrences[c] + 1:] = rands

    # ----------> Verify stream tables
    # Find empty experiences and remove them
    empty_exps = torch.where(torch.sum(stream_table, dim=0) == 0)[0]
    if len(empty_exps) != 0:
        non_empty = torch.where(torch.sum(stream_table, dim=0) != 0)[0]
        msg = f"\n\nRemoving {len(empty_exps)} empty experiences ...\n" + \
            f"Current number of experiences: {len(non_empty)}"
        warnings.warn(msg)
        stream_table = stream_table[:, non_empty]

    # Set number of experiences according to the new stream table
    n_e = stream_table.shape[1]

    # ----------> Calculate number of samples per class for each experience
    # Initialize number of samples table with zeros
    n_samples_table = torch.zeros(n_classes, n_e).long()

    for e_i in range(n_e):
        # Get classes in the current experience and shuffle them
        classes_e_i = torch.where(stream_table[:, e_i] == 1)[0].numpy()
        rng_np.shuffle(classes_e_i)

        # Calculate number of samples per class
        weights = [1 / len(classes_e_i) for _ in range(len(classes_e_i))]
        n_samples_per_class = [math.floor(w * s_e) for w in weights]

        # Add remaining samples
        remaining = s_e - sum(n_samples_per_class)
        if remaining > 0:
            for i in cycle(list(range(len(n_samples_per_class)))):
                n_samples_per_class[i] += 1
                remaining -= 1
                if remaining == 0:
                    break

        # Set number of samples per class in the table
        for c, n in zip(classes_e_i, n_samples_per_class):
            n_samples_table[c][e_i] = n

    # ----------> Sample indices for each class
    # Initialize dictionary of selected indices:
    # Dict[c][e] = indices of class c in experience e
    selected_indices = {}

    # For each class, sample indices for that class in each experience
    for c in range(n_classes):
        selected_indices[c] = {}
        indices_c = class_to_indices[c]
        exp_c = torch.where(n_samples_table[c] != 0)[0]
        for e, n_samp in zip(exp_c, n_samples_table[c][exp_c]):
            rnd_perm = torch.randperm(len(indices_c), generator=rng_torch)
            selected_c = indices_c[rnd_perm][:n_samp]
            selected_indices[c][e.item()] = selected_c

    # ----------> Create dataset per experience
    train_datasets = []
    samples_per_exp = []
    for exp_i in range(n_e):
        present_classes = torch.where(stream_table[:, exp_i] != 0)[0]
        all_indices_i = torch.cat([selected_indices[c.item()][exp_i]
                                   for c in present_classes])
        ds_train_i = classification_subset(train_set,
                                           indices=all_indices_i.tolist())

        train_datasets.append(ds_train_i)
        samples_per_exp.append(all_indices_i)

    # ----------> Create stream benchmark
    benchmark = dataset_benchmark(
        train_datasets=train_datasets,
        test_datasets=[test_set],
    )

    # ----------> Stream details
    n_samples_per_exp = [len(samples_per_exp[i])
                         for i in range(stream_table.shape[1])]
    benchmark.details = {
        "first_occurrences": first_occurrences,
        "stream_table": stream_table,
        "n_samples_per_exp": n_samples_per_exp
    }

    return benchmark


# ==========> Benchmark wrappers


def cir_cifar100(
        dataset_root,
        n_e: int = 10,
        s_e: int = 500,
        dist_first_occurrence: dict = None,
        dist_recurrence: dict = None,
        seed: int = 0,
):
    # Transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ]
    )

    # Train and test sets
    train_set = CIFAR100(root=dataset_root, train=True,
                         transform=train_transform, download=True)
    test_set = CIFAR100(root=dataset_root, train=False,
                        transform=eval_transform, download=True)

    # Benchmark
    benchmark = cir_generator(
        train_set,
        test_set,
        n_e=n_e,
        s_e=s_e,
        dist_first_occurrence=dist_first_occurrence,
        dist_recurrence=dist_recurrence,
        seed=seed
    )

    return benchmark


def cir_tinyimagenet(
        dataset_root,
        n_e: int = 10,
        s_e: int = 500,
        dist_first_occurrence: dict = None,
        dist_recurrence: dict = None,
        seed: int = 0,
):
    # Transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # Train and test sets
    train_set = TinyImagenet(root=dataset_root, train=True,
                             transform=train_transform)
    test_set = TinyImagenet(root=dataset_root, train=False,
                            transform=eval_transform)

    # Benchmark
    benchmark = cir_generator(
        train_set,
        test_set,
        n_e=n_e,
        s_e=s_e,
        dist_first_occurrence=dist_first_occurrence,
        dist_recurrence=dist_recurrence,
        seed=seed
    )

    return benchmark


# ==========> Test

if __name__ == "__main__":
    dataset_root = "./data/datasets"

    dist_first_occurrence = {'dist_type': 'geometric', 'p': 0.01}
    dist_recurrence = {'dist_type': 'fixed', 'p': 0.2}

    benchmark = cir_cifar100(
        dataset_root=dataset_root,
        n_e=200,
        s_e=500,
        dist_first_occurrence=dist_first_occurrence,
        dist_recurrence=dist_recurrence,
        seed=0,
    )
