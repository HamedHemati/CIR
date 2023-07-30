from typing import Optional, TYPE_CHECKING

import copy
import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class EvalHeadAdapter(SupervisedPlugin):
    """
    This plugin trains the model's head with small subset of samples
    from each (previous) experience before evaluation on that experience.
    """

    def __init__(self,
                 n_samples_per_exp: int = 100,
                 n_training_steps: int = 1,
                 training_lr: float = 0.1
                 ):
        super().__init__()
        self.n_samples_per_exp = n_samples_per_exp
        self.n_training_steps = n_training_steps
        self.training_lr = training_lr

        self.samples_per_experience = {}
        self.original_model = None

    def before_eval(
            self, strategy: SupervisedTemplate, *args, **kwargs
    ):
        self.original_model = copy.deepcopy(strategy.model)

    def before_eval_exp(
            self, strategy: SupervisedTemplate, *args, **kwargs
    ):
        """
        For all
        """
        n_passed_exps = strategy.clock.train_exp_counter

        # Adapt only for previously seen experiences
        if n_passed_exps > 1:
            exp_id = strategy.experience.current_experience
            if exp_id < n_passed_exps - 1:
                self.train_model_head(strategy, exp_id)

    def train_model_head(self, strategy, exp_id):
        """
        Trains the model's head.
        """
        print(f"\n\nTraining head for experience {exp_id}")
        # Retrieve data for the current experience
        x = self.samples_per_experience[exp_id][:][0].to(strategy.device)
        y = self.samples_per_experience[exp_id][:][1].to(strategy.device)
        t = torch.zeros(len(x)).to(strategy.device)
        strategy.mbatch = (x, y, t)

        # Set model to training mode
        strategy.model.train()
        torch.set_grad_enabled(True)

        # Optimizer only for the classifier head
        optimizer = torch.optim.SGD(strategy.model.classifier.parameters(),
                                    lr=self.training_lr)

        # Train the model's head for `n_training_steps` iterations
        for i in range(self.n_training_steps):
            strategy.model.zero_grad()
            strategy.loss = 0.0
            strategy.mb_output = strategy.forward()
            strategy.loss += strategy.criterion()
            strategy.loss.backward()
            # print(strategy.loss)
            optimizer.step()

        # Set model's mode back to eval
        torch.set_grad_enabled(False)
        strategy.model.eval()

    def after_eval_exp(
            self, strategy: SupervisedTemplate, *args, **kwargs
    ):
        """
        Switch the model back to its original state.
        """
        strategy.model.load_state_dict(self.original_model.state_dict())

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        """
        Save random samples from the current experience to adapt the model's
        head later during evaluation.
        """
        exp_id = strategy.clock.train_exp_counter
        random_idx = torch.randperm(len(strategy.experience.dataset))[
            :self.n_samples_per_exp]
        self.samples_per_experience[exp_id] = copy.copy(
            strategy.experience.dataset[random_idx])
