from Core.FeedForwardModel import ModelFeedForward




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

    def Call(self) -> int:
        """
           Checks if the validation metric has improved. If so, updates the best model.
           If no improvement is seen for 'patience' epochs, stops the training.

           :return: 1 if training should stop, 0 otherwise.
        """
        if self.model.MetricResults[self.validation][-1] < self.best_val_loss:
            print(
                f"Confronto: ultimo val_loss = {self.model.MetricResults[self.validation][-1]}, best_val_loss = {self.best_val_loss}")
            self.best_val_loss = self.model.MetricResults[self.validation][-1]
            self.no_improvement_count = 0
            self.model.SaveModel("../MLprj24/Models/best_model.json")
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            return 1
        else:
            return 0

