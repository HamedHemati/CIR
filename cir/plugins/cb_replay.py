from typing import Optional, TYPE_CHECKING
import math
from torch.utils.data import DataLoader
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)

from .samplers import ClassBalancedSampler

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class CBReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        mem_size: int = 200,
        batch_size: int = 32,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        sampling_type="class-balanced",
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.sampling_type = sampling_type

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        ds = strategy.experience.dataset
        classes_to_use = list(set(ds.targets))

        if len(self.storage_policy.buffer) > 0:
            buffer_ds = self.storage_policy.buffer
            classes_to_use = list(set(classes_to_use).union(
                set(buffer_ds.targets)))
            ds = AvalancheConcatDataset([ds, buffer_ds])

        n_batches = math.ceil(len(ds) / self.batch_size)

        if self.sampling_type == "class-balanced":
            sampler = ClassBalancedSampler(
                dataset=ds,
                classes_to_use=classes_to_use,
                n_mbatches=n_batches,
                mbatch_size=self.batch_size
            )
            shuffle = False
        elif self.sampling_type == "random":
            sampler = None
            shuffle = True
        else:
            raise NotImplementedError()

        strategy.dataloader = DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            shuffle=shuffle
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
