from Core.DataSet import DataSet
from Core.Tuner.HyperBag import HyperBag


class HyperModel:
    DataSetsVariant: dict[(str,str),DataSet]
    originalDataset: DataSet

    def __init__(self, originalDataset : DataSet):
        self.originalDataset = originalDataset
        self.DataSetsVariant = {}


    def GetDatasetVariant(self, hp):
        pass


    def GetModel(self, hp :HyperBag):
        pass

    def GetOptimizer(self, hp :HyperBag):
        pass

    def GetHyperParameters(self):
        pass



