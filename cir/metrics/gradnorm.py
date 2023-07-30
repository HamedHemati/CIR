from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue


class GradientNormMetric(PluginMetric[float]):
    """
    This metric computes gradient norm of the model after every back-prop
    step.
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        self._grad_norm = 0.0

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._grad_norm = 0.0

    def result(self) -> float:
        """
        Emit the result
        """
        return self._grad_norm

    def after_backward(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        total_norm = 0.0
        for p in strategy.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self._grad_norm = total_norm

    def before_backward(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_iteration(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        return self._package_result(strategy)

    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self._grad_norm
        plot_x_position = strategy.clock.train_iterations
        metric_name = f"GradNorm/train_phase/train_stream/"

        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "GradNorm"
