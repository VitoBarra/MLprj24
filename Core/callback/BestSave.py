from Core.FeedForwardModel import ModelFeedForward


class BestSave:
    """
        A utility class to save the best models during training.

        Attributes:
            path (str): The directory or file path prefix where the models files will be saved.
    """
    path: str
    def __init__(self, path: str):
        """
        Initializes the BestSave object with the specified path.
        """
        self.path = path

    def Call(self, model: ModelFeedForward, epoch: int):
        """
        Saves the given model to a file with the epoch number appended to the file name.

        :param model: The feedforward model instance (ModelFeedForward) to be saved.
        :param epoch: The current epoch number, used to uniquely identify the saved model.
        """
        model.SaveModel(f"{self.path}{epoch}.json")
