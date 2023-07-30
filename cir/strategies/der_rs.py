# Code adapted from https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/supervised/der.py


import copy
from typing import Callable, List, Optional, Dict, Optional, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from copy import deepcopy
from torch.nn import Module


from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import (BalancedExemplarsBuffer,
                                               ReservoirSamplingBuffer)
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
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


class ReservoirSamplingWithLogitsBuffer:
    """Buffer updated using class-balanced sampling."""

    def __init__(self, max_size: int, seed: int = 0):
        """
        :param max_size:
        """
        self.max_size = max_size
        self.seen_classes = set()

        self.storage_policy = ReservoirSamplingBuffer(max_size)

        # Set random number generator
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    @property
    def buffer(self):
        return self.storage_policy.buffer
    
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        new_data = strategy.experience.dataset
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

        self.storage_policy.update_from_dataset(new_data_with_logits)


class DER_RS(SupervisedTemplate):
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
        self.storage_policy = ReservoirSamplingWithLogitsBuffer(
            self.mem_size
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
