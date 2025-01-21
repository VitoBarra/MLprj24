from enum import Enum

from Core.FeedForwardModel import ModelFeedForward
from Core.Callback.CallBack import CallBack


class Direction(Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"

class EarlyStopping(CallBack):
    """
    Implements early stopping to halt training when a monitored metric does not improve.

    :param watchedMetric: The name of the metric to monitor. Default is "val_loss".
    :param patience: The number of epochs with no improvement before stopping training. Default is 0.
    :param minimumImp: Minimum improvement in the metric to reset patience. Default is 0.
    :param direction: Direction of improvement - Direction.MINIMIZE or Direction.MAXIMIZE.
    """
    def __init__(self, watchedMetric: str = "val_loss", patience: int = 0, minimumImp: float = 0,
                 direction: Direction = Direction.MINIMIZE):
        self.patience = patience
        self.best_value = float('inf') if direction == Direction.MINIMIZE else float('-inf')
        self.no_improvement_count = 0
        self.WatchedMetric = watchedMetric
        self.minimumImp = minimumImp
        self.direction = direction

    def Reset(self):
        """
        Reset the no_improvement_count attribute and best_value.
        """
        self.no_improvement_count = 0
        self.best_value = float('inf') if self.direction == Direction.MINIMIZE else float('-inf')

    def Call(self, model: ModelFeedForward = None) -> None:
        """
        Monitors the metric and updates the early stopping condition.

        :param model: The model being trained, which provides metric results.
        """
        if self.WatchedMetric not in model.MetricResults:
            raise ValueError(f"Metric '{self.WatchedMetric}' not found in model.MetricResults.")

        watchedMetric = model.MetricResults[self.WatchedMetric][-1]

        if (self.direction == Direction.MINIMIZE and watchedMetric < self.best_value - self.minimumImp) or \
                (self.direction == Direction.MAXIMIZE and watchedMetric > self.best_value + self.minimumImp):
            self.best_value = watchedMetric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count > self.patience:
            print(f"Early stopping triggered. Best {self.WatchedMetric}: {self.best_value}")
            model.EarlyStop = True


