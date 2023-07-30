import os
import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class CheckpointSaver(SupervisedPlugin):
    """
    This plugin saves model's checkpoint every `checkpoint_saving_steps` steps.
    """
    def __init__(self,
                 checkpoint_saving_steps: int = 50,
                 checkpoint_dir_path: str = None
    ):
        super().__init__()
        self.checkpoint_saving_steps = checkpoint_saving_steps
        self.checkpoint_dir_path = checkpoint_dir_path

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        exp_id = strategy.clock.train_exp_counter
        if exp_id % self.checkpoint_saving_steps == 0:
            ckpt_path = os.path.join(self.checkpoint_dir_path,
                                     f"ckpt_{exp_id}.pt")
            torch.save(strategy.model.state_dict(), ckpt_path)
