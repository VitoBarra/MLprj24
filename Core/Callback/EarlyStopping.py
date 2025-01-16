from Core.FeedForwardModel import ModelFeedForward
from Core.Callback.CallBack import CallBack


class EarlyStopping(CallBack):
    """
    Implements early stopping to halt training when a monitored metric does not improve.

    :param watchedMetric: The name of the metric to monitor. Default is "val_loss".
    :param patience: The number of epochs with no improvement before stopping training. Default is 0.
    """
    def __init__(self, watchedMetric: str = "val_loss", patience: int = 0, minimumImp: float = 0):
        self.patience = patience
        self.best_value = float('inf')
        self.no_improvement_count = 0
        self.WatchedMetric = watchedMetric
        self.minimumImp = minimumImp

    def Reset(self):
        """
        Reset the no_improvement_count attribute.
        """
        self.no_improvement_count = 0

    def Call(self, model: ModelFeedForward = None) -> None:
        """
        Monitors the metric and updates the early stopping condition.

        :param model: The model being trained, which provides metric results.
        """
        watchedMetric = model.MetricResults[self.WatchedMetric][-1]
        if watchedMetric < self.best_value - self.minimumImp:
            self.best_value = watchedMetric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count > self.patience:
            model.EarlyStop = True


