# Code adapted from https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/supervised/der.py


from typing import List, Optional, Optional, List, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from copy import deepcopy
from torch.nn import Module
import math

from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    concat_datasets,
)


@torch.no_grad()
def compute_dataset_logits(dataset, model, batch_size, device):
    model.eval()

    logits = []
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        logits.extend(list(out.cpu()))
    return logits


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


class FrequencyAwareBuffer:
    """Buffer updated using class-balanced sampling."""

    def __init__(self, max_size: int, total_num_classes: int = 0,
                 seed: int = 0):
        """
        :param max_size:
        """
        self.max_size = max_size
        self.seen_classes = set()
        self.buffer: AvalancheDataset = concat_datasets([])

        self.c_to_nobs = {c: 0 for c in range(total_num_classes)}
        self.c_to_nsamp = {c: 0 for c in range(total_num_classes)}

        # Set random number generator
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset

        # Add prediction logits as data attribute to the dataset
        logits = compute_dataset_logits(
            new_data.eval(),
            strategy.model,
            strategy.train_mb_size,
            strategy.device
        )
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(logits, name="logits", use_in_getitem=True)
            ],
        )

        self.update_from_dataset(new_data_with_logits)

    def get_quota_per_class(self, current_classes):
        # Calculate quota per class
        q_c = {}
        for c in self.seen_classes:
            q_c[c] = 1 / self.c_to_nobs[c]

        # Normalize
        total = sum(list(q_c.values()))
        quota_c = {k: v / total for k, v in q_c.items()}

        # Quota per class
        c_to_nsamp = {}
        for c, q in quota_c.items():
            v_ = math.floor(q * self.max_size)
            if c in current_classes:
                c_to_nsamp[c] = v_
            else:
                # Make sure the number fo assigned samples for missing classes
                # is not larger than existing number of samples in the buffer
                c_to_nsamp[c] = min(self.c_to_nsamp[c], v_)

        return c_to_nsamp

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        # Update seen classes
        current_classes = list(set(new_data.targets))
        self.seen_classes.update(current_classes)

        # Update number of observations for the classes in the current exp
        for c in set(new_data.targets):
            self.c_to_nobs[c] += 1

        # Calculate number of samples per class
        c_to_nsamp_new = self.get_quota_per_class(current_classes)

        # Merge new data with buffer
        merged_data = concat_datasets([self.buffer, new_data])
        merged_targets = torch.LongTensor(merged_data.targets)

        def get_indices_for_class(c, targets):
            """Returns indices of samples for class c in the merged ds."""
            indices_c = torch.where(targets == c)[0]
            return indices_c

        # Fill unused buffer slots
        if sum(c_to_nsamp_new.values()) < self.max_size:
            counter = 0
            while True:
                idx = counter % len(current_classes)
                c = current_classes[idx]
                # Add num samples for class c
                c_to_nsamp_new[c] += 1
                # Check if all unused slots are filled
                if sum(c_to_nsamp_new.values()) == self.max_size:
                    break

                counter += 1

        # Update self.c_to_nsamp
        self.c_to_nsamp.update(c_to_nsamp_new)

        # Retrieve random subsets for each class in the merged dataset
        def get_subset_indices_for_class(c, targets):
            indices_c = get_indices_for_class(c, targets)
            indices_c = indices_c[torch.randperm(len(indices_c),
                                                 generator=self.rng)]
            indices_c_to_use = indices_c[:self.c_to_nsamp[c]]
            self.c_to_nsamp[c] = len(indices_c_to_use)

            indices_c_rem = indices_c[self.c_to_nsamp[c]:]

            return indices_c_to_use, indices_c_rem

        subsets = [get_subset_indices_for_class(c, merged_targets)
                   for c in self.seen_classes]
        subset_indices_to_use = torch.cat([s[0] for s in subsets])
        subset_indices_rem = torch.cat([s[1] for s in subsets])

        # If number of selected indices is less than max_size, add more
        # from the remaining indices
        if len(subset_indices_to_use) < self.max_size:
            n_rem = self.max_size - len(subset_indices_to_use)
            rnd_rem = torch.randperm(len(subset_indices_rem))
            subset_indices_to_use = torch.cat(
                [subset_indices_to_use, subset_indices_rem[rnd_rem][:n_rem]])

        self.buffer = merged_data.subset(subset_indices_to_use.tolist())


class DER_Adaptive(SupervisedTemplate):
    """
    Implements the DER and the DER++ Strategy,
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        total_num_classes=None,
        batch_size_mem: int = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param alpha: float : Hyperparameter weighting the MSE loss
        :param beta: float : Hyperparameter weighting the CE loss,
                             when more than 0, DER++ is used instead of DER
        :param transforms: Callable: Transformations to use for
                                     both the dataset and the buffer data, on
                                     top of already existing
                                     test transformations.
                                     If any supplementary transformations
                                     are applied to the
                                     input data, it will be
                                     overwritten by this argument
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.storage_policy = FrequencyAwareBuffer(
            self.mem_size,
            total_num_classes=total_num_classes
        )
        self.replay_loader = None
        self.alpha = alpha
        self.beta = beta

    def _before_training_exp(self, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        super()._after_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(self.device),
            batch_y.to(self.device),
            batch_tid.to(self.device),
            batch_logits.to(self.device),
        )
        self.mbatch[0] = torch.cat((batch_x, self.mbatch[0]))
        self.mbatch[1] = torch.cat((batch_y, self.mbatch[1]))
        self.mbatch[2] = torch.cat((batch_tid, self.mbatch[2]))
        self.batch_logits = batch_logits

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            if self.replay_loader is not None:

                # DER Loss computation

                self.loss += F.cross_entropy(
                    self.mb_output[self.batch_size_mem:],
                    self.mb_y[self.batch_size_mem:],
                )

                self.loss += self.alpha * F.mse_loss(
                    self.mb_output[:self.batch_size_mem],
                    self.batch_logits,
                )
                self.loss += self.beta * F.cross_entropy(
                    self.mb_output[:self.batch_size_mem],
                    self.mb_y[:self.batch_size_mem],
                )

                # They are a few difference compared to the autors impl:
                # - Joint forward pass vs. 3 forward passes
                # - One replay batch vs two replay batches
                # - Logits are stored from the non-transformed sample
                #   after training on task vs instantly on transformed sample

            else:
                self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
