
import numpy as np

from .DataSet.DataSet import DataExamples


class MiniBatchGenerator:

    Data: DataExamples
    IsBatchGenerationFinished: bool
    BatchSize: int

    LastPosition: int

    def __init__(self, data:DataExamples, batchSize:int):
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
       Generate the next mini-batch of data.

       :return: Return:
       - A list containing up to `BatchSize` data points, starting from the current position.
       - If fewer than `BatchSize` data points remain, the returned list will include all remaining data.
       - Returns `None` if all data has been processed and no further batches are available.
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
        reset the mini-batch generation. Usually called after each epoch.
        the next NextBatch() will start from the first data point.
        with BatchSize == 1 (online-training), the data will be shuffled at each epoch.
        """
        self.IsBatchGenerationFinished = False
        self.LastPosition = 0
        if self.BatchSize == 1:
            self.Data.Shuffle()

