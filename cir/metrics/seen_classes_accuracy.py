import torch
from torch import Tensor
from typing import Dict
from typing import TYPE_CHECKING
import numpy as np
from torchmetrics import Accuracy

from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metric_definitions import get_metric_name
from avalanche.evaluation.metric_results import MetricValue
if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricResult
    from avalanche.training.templates.supervised import SupervisedTemplate


class PerClassAccuracy(Metric[float]):
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.acc_metric = None
        self.reset_mean_accuracy()

    def reset_mean_accuracy(self):
        self.acc_metric = Accuracy(average="none", task="multiclass", 
                                   num_classes=self.n_classes)
        self.acc_metric.to(self.device)

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor
    ) -> None:
        self.acc_metric.update(predicted_y, true_y)

    def result(self, task_label=None) -> Dict[int, float]:
        return {i: a.item() for (i, a) in enumerate(self.acc_metric.compute())}

    def reset(self, task_label=None) -> None:
        self.reset_mean_accuracy()


class SeenClassesAccuracyPluginMetric(GenericPluginMetric[float]):
    def __init__(self, reset_at, emit_at, mode, n_classes, device,
                 seen_classes_in_each_exp=None):
        self._accuracy = PerClassAccuracy(n_classes, device)
        super(SeenClassesAccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )
        self.n_classes = n_classes
        self.seen_classes_in_each_exp = seen_classes_in_each_exp

    def reset(self, strategy=None) -> None:
        self._metric.reset()

    def result(self, strategy=None) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._accuracy.update(strategy.mb_output, strategy.mb_y)

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result(strategy)
        plot_x_position = strategy.clock.train_iterations

        exp_id = strategy.clock.train_exp_counter - 1
        all_seen_values = [v for (k, v) in metric_value.items() if
                           k in self.seen_classes_in_each_exp[exp_id]]

        # Average
        metric_name = get_metric_name(
            self, strategy, add_experience=False, add_task=None
        )
        metric_name = metric_name.split("/")
        metric_name[0] += "-AVG"
        metric_name = "/".join(metric_name)
        seen_avg = np.mean(all_seen_values)

        metrics = [
            MetricValue(self, metric_name, seen_avg, plot_x_position)
        ]

        return metrics


##################################################################
#                           Wrappers                             #
##################################################################

class ExperienceSeenClassesAccuracy(SeenClassesAccuracyPluginMetric):
    def __init__(self, n_classes, device, seen_classes_in_each_exp=None):
        super(ExperienceSeenClassesAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval",
            n_classes=n_classes,
            device=device,
            seen_classes_in_each_exp=seen_classes_in_each_exp
        )

    def __str__(self):
        return "Top1_SeenClassesAcc_Experience"
