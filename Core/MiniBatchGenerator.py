
import numpy as np

from .DataSet.DataSet import DataExamples


class MiniBatchGenerator:
    """
    A class that generates mini-batches of data for training models.

    This generator splits the dataset into smaller batches of size `BatchSize` for each training step.
    It is ideal for large datasets that do not fit entirely into memory.
    """

    Data: DataExamples
    IsBatchGenerationFinished: bool
    BatchSize: int

    LastPosition: int

    def __init__(self, data:DataExamples, batchSize:int):
        """
        Initializes the MiniBatchGenerator instance.

        :param data: The dataset containing the input data and target labels.
        :param batchSize: The size of each mini-batch to be generated.
        :raises ValueError: If the dataset is None, empty, or if BatchSize is not a positive integer.
        """
        if data is None:
            raise ValueError("DataSet cannot be None")
        if len(data) == 0:
            raise ValueError("DataSet is empty")

        if batchSize <= 0:
            raise ValueError("BatchSize must be a positive integer")

        self.Data = data
        self.BatchSize = batchSize
        self.LastPosition = 0
        self.IsBatchGenerationFinished = False

    def NextBatch(self) -> np.ndarray | None:
        """
         Generates the next mini-batch of data.

         :return: A tuple containing:
                  - batch_data: A numpy array containing up to `BatchSize` data points from the dataset.
                  - batch_target: A numpy array containing the corresponding target labels for the batch.
                  - If fewer than `BatchSize` data points remain, the batch will contain all remaining data.
                  - Returns `None, None` if all data has been processed and no further batches are available.
         """

        if self.LastPosition >= len(self.Data) or self.IsBatchGenerationFinished:
            return None , None

        batch_data = self.Data.Data[self.LastPosition : self.LastPosition + self.BatchSize]
        batch_target = self.Data.Label[self.LastPosition : self.LastPosition + self.BatchSize]

        self.LastPosition += self.BatchSize
        if self.LastPosition >= len(self.Data):
            self.IsBatchGenerationFinished = True
        return batch_data , batch_target

    def Reset(self) -> None:
        """
        Resets the mini-batch generation process. Typically called after each epoch.
        The next call to `NextBatch()` will start from the first data point.

        If `BatchSize` is 1 (online training), the data will be shuffled at the start of each epoch.
        """
        self.IsBatchGenerationFinished = False
        self.LastPosition = 0
        if self.BatchSize == 1:
            self.Data.Shuffle()

