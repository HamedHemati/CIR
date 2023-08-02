import torch
import numpy as np

import torchvision.transforms as transforms
from torchvision.transforms import (
    ToTensor,
    Compose,
    Normalize,
)
from torch.utils.data import random_split

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    AvalancheConcatDataset,
    AvalancheSubset
)
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.datasets import (
    CIFAR100,
    MNIST,
    TinyImagenet,
    MiniImageNetDataset
)


def cir_slot_based(
        train_set,
        test_set,
        N,
        K,
        seed=None
):
    """CIR Slot-based generator.

    Given a dataset, this function generates a CIR stream of length N.

    The generator splits the data for each class into equal subsets,
    obtaining in total N*K subsets. These are then divided equally and randomly
    among experiences. Inside each experience, the generator tries to assign
    slots of different classes to fill the experience, if possible. If this is
    not possible, two or more slots of the same class may appear in
    the same experience. In most settings, this will happen only for a
    few of the first experiences.

    It is suggested to use this generator only if the samples are distributed 
    roughly equally among classes.

    :param train_set: train set containing the stream train samples.
    :param test_set: test set containing the stream test samples.
    :param N: desired length of the stream.
    :param K: number of slots
    :param seed:
    :return: <avl_data_seq, slot_table>, where avl_data_seq is a list of
        AvalancheDatasets, and slot_table shows how the slot->experience 
        assignments.
    """
    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    known_classes = set(train_set.targets)
    C = len(known_classes)
    D = N*K // C  # min. number of splits for each class

    assert K <= C, \
        "The number of slots per experience K must be " \
        "less than or equal to the number of classes."

    # get sample idxs per class
    classes_idxs = {}
    for idx, cls in enumerate(train_set.targets):
        if cls not in classes_idxs:
            classes_idxs[cls] = []
        classes_idxs[cls].append(idx)

    # split classes into slots of data
    slots_per_class = {}
    num_slots_per_class = {}
    shuffled_classes = list(known_classes)
    np.random.shuffle(shuffled_classes)
    for ii, cls in enumerate(shuffled_classes):
        if ii < (N*K % C):
            # total slots are not divisible by classes. Some classes must
            # have one slot less than the others. We choose them randomly
            # since we shuffled the classes previously.
            num_slots_per_class[cls] = D + 1
        else:
            num_slots_per_class[cls] = D

        # split data by class
        perm_idxs = torch.randperm(len(classes_idxs[cls]))
        cls_idxs = torch.tensor(classes_idxs[cls])[perm_idxs]

        slots_per_class[cls] = []
        step = len(cls_idxs) // num_slots_per_class[cls]
        # split class-data into slots
        for i in range(num_slots_per_class[cls]):
            cls_subset = AvalancheSubset(
                train_set, cls_idxs[i*step:(i+1)*step])
            slots_per_class[cls].append(cls_subset)
    assert sum([len(el) for el in slots_per_class.values()]) == N*K

    # sample slot assignments
    slot_table = np.zeros((N, K))
    avl_data_seq = []
    for eid in reversed(range(N)):
        # choose the classes for the current experience
        if len(known_classes) >= K:  # we sample classes without replacement, if possible
            cls_chosen = np.random.choice(
                list(known_classes), replace=False, size=K)
            for c in cls_chosen:
                num_slots_per_class[c] -= 1
                # remove classes that don't have available slots
                if num_slots_per_class[c] == 0:
                    known_classes.remove(c)
        else:
            # we have to sample the same class multiple times,
            # but we must ensure that there are available slots
            cls_chosen = []
            for _ in range(K):
                cls = np.random.choice(list(known_classes), size=1)[0]
                cls_chosen.append(cls)
                num_slots_per_class[cls] -= 1
                if num_slots_per_class[cls] == 0:
                    known_classes.remove(cls)

        assert len(cls_chosen) == K
        for j, cidx in enumerate(cls_chosen):
            slot_table[eid, j] = cidx

        edatal = []
        for cls in cls_chosen:
            assert len(slots_per_class[cls]) > 0
            i = np.random.randint(0, len(slots_per_class[cls]))
            d = slots_per_class[cls].pop(i)
            edatal.append(d)
        avl_data_seq.insert(0, AvalancheConcatDataset(edatal))

    for ll in slots_per_class.values():  # we must use all the data
        assert len(ll) == 0

    # Create scenario table
    n_classes = len(set(train_set.targets))
    n_exp = slot_table.shape[0]
    scenario_table = torch.zeros(n_classes, n_exp)

    def set_classes_per_ex(i):
        for c in slot_table[i]:
            scenario_table[int(c), i] = 1

    _ = [set_classes_per_ex(i) for i in range(n_exp)]

    benchmark = dataset_benchmark(
        train_datasets=avl_data_seq,
        test_datasets=[test_set],
        complete_test_set_only=True,
    )

    benchmark.scenario_table = scenario_table
    benchmark.n_samples_per_exp = [len(ds) for ds in avl_data_seq]

    # Number of samples per class in the stream
    n_e = len(avl_data_seq)
    classes = set(train_set.targets)
    benchmark.samples_per_class = {c: [0] * n_e for c in classes}

    # Classes in each experience:
    benchmark.present_classes_in_each_exp = [list(set(exp.dataset.targets))
                                             for exp in benchmark.train_stream]
    benchmark.slot_table = slot_table

    # List of seen classes up to each experience
    classes_in_each_exp = []
    seen_classes = []
    seen_classes_up_to_exp = []
    for i, exp in enumerate(benchmark.train_stream):
        classes_in_each_exp.append(exp.classes_in_this_experience)
        seen_classes += exp.classes_in_this_experience
        seen_classes_up_to_exp.append(list(set(seen_classes)))

    # Benchmark details
    benchmark.details = {
        "stream_table": scenario_table,
        "n_samples_per_exp": benchmark.n_samples_per_exp,
        "classes_in_each_exp": classes_in_each_exp,
        "seen_classes_up_to_exp": seen_classes_up_to_exp,
    }

    def set_n_class_samples_per_exp(exp_id):
        targets = benchmark.train_stream[exp_id].dataset.targets
        targets = torch.LongTensor(targets)

        def _set_n_class_c(c):
            total_c = torch.sum(targets == c)
            benchmark.samples_per_class[c][exp_id] = total_c.item()

        _ = [_set_n_class_c(c.item()) for c in torch.unique(targets)]

    _ = [set_n_class_samples_per_exp(exp_id) for exp_id in range(n_e)]

    return benchmark


##############################
#      Helper Functions
##############################


def cir_slot_mnist(dataset_root, N, K, seed=None):
    train_transform = Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]
    )

    test_transform = Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))]
    )

    train_set = MNIST(root=dataset_root, train=True,
                      download=True, transform=train_transform)
    train_set.targets = list(train_set.targets.numpy())
    test_set = MNIST(root=dataset_root, train=False,
                     download=True, transform=test_transform)

    benchmark = cir_slot_based(train_set, test_set, N, K, seed=seed)
    return benchmark


def cir_slot_cifar100(dataset_root, N, K, seed=None):
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

    train_set = CIFAR100(root=dataset_root, train=True,
                         transform=train_transform, download=True)
    test_set = CIFAR100(root=dataset_root, train=False,
                        transform=eval_transform, download=True)
    benchmark = cir_slot_based(train_set, test_set, N, K, seed=seed)
    return benchmark


def cir_slot_tinyimagenet(dataset_root, N, K, seed=None):
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

    train_set = TinyImagenet(root=dataset_root, train=True,
                             transform=train_transform, download=True)
    test_set = TinyImagenet(root=dataset_root, train=False,
                            transform=eval_transform, download=True)

    benchmark = cir_slot_based(train_set, test_set, N, K, seed=seed)
    return benchmark


def cir_slot_miniimagenet(dataset_root, N, K, seed=None):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # from
            # https://github.com/Mattdl/RehearsalRevealed/blob/main/framework/cole/core.py
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # from
            # https://github.com/Mattdl/RehearsalRevealed/blob/main/framework/cole/core.py
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ]
    )

    train_set = MiniImageNetDataset(dataset_root, split='all')
    trl = len(train_set)
    perc_train = 0.7
    sizes = [int(trl*perc_train), trl-int(trl*perc_train)]
    if seed is not None:
        train_set, test_set = random_split(
            train_set, sizes, generator=torch.Generator().manual_seed(seed))
    else:
        train_set, test_set = random_split(train_set, sizes)
    train_set = AvalancheDataset(
        train_set, transform=transforms.Compose([train_transform]))
    test_set = AvalancheDataset(
        test_set, transform=transforms.Compose([eval_transform]))
    benchmark = cir_slot_based(train_set, test_set, N, K, seed=seed)
    return benchmark

##############################
#           Test
##############################


if __name__ == '__main__':
    dataset_root = "experiments/data/datasets"

    # class-incremental
    N, K = 5, 2
    print(f"Class-incremental (N={N}, K={K}):")
    benchmark = cir_slot_mnist(dataset_root, N, K, seed=123)
    print(benchmark.scenario_table)

    # CIR
    N, K = 10, 3
    print(f"CIR (N={N}, K={K}):")
    benchmark = cir_slot_mnist(dataset_root, N, K, seed=123)
    print(benchmark.scenario_table)

    # domain-incremental
    N, K = 10, 10
    print(f"Domain-incremental (N={N}, K={K}):")
    benchmark = cir_slot_mnist(dataset_root, N, K, seed=123)
    print(benchmark.scenario_table)

    # domain-incremental
    N, K = 9, 8
    print(f"Domain-incremental with uneven slots (N={N}, K={K}):")
    benchmark = cir_slot_mnist(dataset_root, N, K, seed=123)
    print(benchmark.scenario_table)
