import torchvision.transforms as transforms

from avalanche.benchmarks.datasets import TinyImagenet
from avalanche.benchmarks.datasets import Omniglot
from avalanche.benchmarks.datasets import CIFAR10, CIFAR100
from avalanche.benchmarks.utils import AvalancheDataset

from avalanche.benchmarks.classic.comniglot import \
    _default_omniglot_train_transform as train_transform_omniglot
from avalanche.benchmarks.classic.comniglot import \
    _default_omniglot_eval_transform as eval_transform_omniglot

from avalanche.benchmarks.classic.ccifar10 import \
    _default_cifar10_train_transform as train_transform_cifar10
from avalanche.benchmarks.classic.ccifar10 import \
    _default_cifar10_eval_transform as eval_transform_cifar10


def get_dataset(ds_name, dataset_root, train=True) -> AvalancheDataset:
    """ Returns an AVL dataset given the dataset name. """

    if ds_name == "tinyimagenet":
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

        if train:
            transform = train_transform
        else:
            transform = eval_transform

        dataset = TinyImagenet(
            root=dataset_root,
            download=False,
            train=train,
            transform=transform
        )

    elif ds_name == "omniglot":
        if train:
            transform = train_transform_omniglot
        else:
            transform = eval_transform_omniglot
        dataset = Omniglot(
            root=dataset_root,
            download=True,
            train=train,
            transform=transform
        )

    elif ds_name == "cifar-100":
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

        if train:
            transform = train_transform
        else:
            transform = eval_transform

        dataset = CIFAR100(
            root=dataset_root,
            download=True,
            train=train,
            transform=transform
        )

    elif ds_name == "cifar-10":
        if train:
            transform = train_transform_cifar10
        else:
            transform = eval_transform_cifar10

        dataset = CIFAR10(
            root=dataset_root,
            download=True,
            train=train,
            transform=transform
        )

    else:
        raise NotImplementedError()

    return dataset
