from Core.FeedForwardModel import ModelFeedForward
from Core.callback.BestSave import BestSave


class EarlyStopping:
    patience: int
    model: ModelFeedForward
    validation: str

    def __init__(self, patience: int= 0, model : ModelFeedForward = None, validation: str = "val_loss"):
        """
            Initializes EarlyStopping to monitor a validation metric and stop training if no improvement is seen.

            :param patience: The number of epochs to wait for improvement before stopping. Default is 0.
            :param model: The model to monitor. It should have a MetricResults attribute and a SaveModel method.
            :param validation: The validation metric to monitor (default is "val_loss").
        """
        self.model = model
        self.patience = patience
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
        self.validation = validation

    def Call(self, bs:BestSave = None, e:int = 0) -> int:
        """
           Checks if the validation metric has improved. If so, updates the best model and resets the no-improvement counter.
           If no improvement is seen for a number of consecutive epochs equal to 'patience', it signals to stop training.

           :param bs: (Optional) An instance of the BestSave class to save the current best model.
                      If not provided, the model is only saved to a default path.
           :param e: The current epoch number, used to uniquely identify the saved model if 'bs' is provided.
           :return: 1 if training should stop due to no improvement for 'patience' epochs, 0 otherwise.
        """
        if self.model.MetricResults[self.validation][-1] < self.best_val_loss:
            print(
                f"Confronto: ultimo val_loss = {self.model.MetricResults[self.validation][-1]}, best_val_loss = {self.best_val_loss}")
            self.best_val_loss = self.model.MetricResults[self.validation][-1]
            self.no_improvement_count = 0
            if bs is not None:
                bs.Call(self.model, e)
            self.model.SaveModel("../MLprj24/Models/best_model.json")
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            return 1
        else:
            return 0

