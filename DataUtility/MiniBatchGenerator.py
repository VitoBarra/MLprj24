from DataUtility.DataSet import *
import numpy as np

from DataUtility.DataSet import DataExamples


class MiniBatchGenerator:

    Data: DataExamples
    IsBatchGenerationFinished: bool
    BatchSize: int

    LastPosition: int

    def __init__(self, Data:DataExamples, BatchSize:int):
        self.Data = Data
        self.BatchSize = BatchSize
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
            return None
        batch = self.Data[self.LastPosition : self.LastPosition + self.BatchSize]
        self.LastPosition += self.BatchSize
        if self.LastPosition >= len(self.Data):
            self.IsBatchGenerationFinished = True
        return batch

    def Reset(self) -> None:
        """
        reset the mini-batch generation.
        the next NextBatch() will start from the first data point.
        """
        self.IsBatchGenerationFinished = False
        self.LastPosition = 0
