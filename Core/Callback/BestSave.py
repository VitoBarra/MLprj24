from Core.Callback.CallBack import CallBack
from Core.FeedForwardModel import ModelFeedForward


class BestSave(CallBack):
    """
    Saves the best model during training based on the monitored metric.

    :param savePath: The file path where the best model's metrics should be saved.
    :param watchedMetric: The name of the metric to monitor. Default is "val_loss".
    """
    def __init__(self, savePath: str, watchedMetric: str = "val_loss"):
        self.best_value = float('inf')
        self.WatchedMetric = watchedMetric
        self.SavePath = savePath

    def Reset(self):
        pass

    def Call(self, model: ModelFeedForward = None) -> None:
        """
        Monitors the metric and saves the model's metrics if the monitored metric improves.

        Checks the most recent value of the watched metric. If it improves, the model's
        metric results are saved to the specified file path.

        :param model: The model being trained, which provides metric results and a method to save them.
        """
        if model.MetricResults[self.WatchedMetric][-1] < self.best_value:
            self.best_value = model.MetricResults[self.WatchedMetric][-1]
            model.SaveMetricsResults(self.SavePath)
