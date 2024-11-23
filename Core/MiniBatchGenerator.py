from DataUtility.DataUtil import *

class MiniBatchGenerator:
    def __init__(self, Data:DataExamples, BatchSize:int):
        self.Data = Data
        self.BatchSize = BatchSize
        self.LastPosition = 0
        self.IsBatchGenerationFinished = False

    def NextBatch(self):
        if self.LastPosition >= len(self.Data):
            return None
        batch = self.Data[self.LastPosition : self.LastPosition + self.BatchSize]
        self.LastPosition += self.BatchSize
        if self.LastPosition >= len(self.Data):
            self.IsBatchGenerationFinished = True
        return batch

    def Reset(self):
        self.IsBatchGenerationFinished = False
        self.LastPosition = 0
