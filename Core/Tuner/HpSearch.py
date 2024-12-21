import random
from itertools import product

from Core import FeedForwardModel
from Core.Callback.CallBack import CallBack
from Core.Metric import Metric
from Core.Tuner import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.DataSet import DataSet


class HyperParameterSearch:

    def __init__(self):
        pass

    def search(self, hp:HyperBag):
        pass

    def GetName(self):
        return "BaseClass"
    def TrialNumber(self):
        return 0

class GridSearch(HyperParameterSearch):


    def __init__(self):
        super().__init__()
        self.trials = 0


    def search(self, hp:HyperBag):
        keys = hp.Keys()
        values = hp.Values()
        prod = list(product(*values))
        self.trials = len(prod)
        for i,combination in enumerate(prod):
            hpsel=dict(zip(keys, combination))
            yield hpsel, i

    @staticmethod
    def GetName():
        return "GridSearch"
    def TrialNumber(self):
        return self.trials



class RandomSearch(HyperParameterSearch):

    trials: int
    def __init__(self, trials: int):
        super().__init__()
        self.trials = trials

    def search(self,hp) -> FeedForwardModel:
        keys = hp.Keys()
        values = hp.Values()

        for i in range(self.trials):
            combination = [random.choice(value_list) for value_list in values]
            hpsel = dict(zip(keys, combination))
            yield hpsel , i


    @staticmethod
    def GetName():
        return "RandomSearch"
    def TrialNumber(self):
        return self.trials



class GetBestSearch:
    def __init__(self, hp: HyperBag, search: HyperParameterSearch):
        self.hp = hp
        self.SearchObj = search

    def GetBestModel(self, hyperModel_fn, Data:DataSet, epoch:int, miniBatchSize: int | None=None, watchMetric ="val_loss", metric : list[Metric]= None, weightInizializer = GlorotInitializer(), callBack: list[CallBack] = None, ) -> (FeedForwardModel, dict[str, any]):
        best_model: FeedForwardModel = None
        best_watchMetric = float("inf")
        best_hpSel = None

        for  hpSel , i in self.SearchObj.search(self.hp):
            print(f"{self.SearchObj.GetName()}: Iteration {i} on {self.SearchObj.TrialNumber()}")
            hyperModel, optimizer= hyperModel_fn(hpSel)
            hyperModel.Build(weightInizializer)
            if metric is not None and len(metric) != 0:
                hyperModel.AddMetrics(metric)

            hyperModel.Fit(optimizer, Data.Training, epoch, miniBatchSize, Data.Validation, callBack)

            last_watchMetric=hyperModel.MetricResults[watchMetric][-1]
            if  last_watchMetric < best_watchMetric:
                best_watchMetric = last_watchMetric
                best_hpSel = hpSel
                best_model = hyperModel

        return best_model, best_hpSel



