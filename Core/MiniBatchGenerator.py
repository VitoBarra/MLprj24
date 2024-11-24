class MiniBatchGenerator:
    def __init__(self, Data, BatchSize):
        self.Data = Data
        self.BatchSize = BatchSize
        self.LastPosition = 0

    def NextBatch(self):
        if self.LastPosition >= len(self.Data):
            return None
        batch = self.Data[self.LastPosition : self.LastPosition + self.BatchSize]
        self.LastPosition += self.BatchSize
        return batch

    def Reset(self):
        self.LastPosition = 0
