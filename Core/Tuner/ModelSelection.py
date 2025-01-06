import numpy as np

from .. import FeedForwardModel
from ..Callback.CallBack import CallBack
from ..Metric import Metric
from ..Tuner.HpSearch import HyperParameterSearch
from ..Tuner.HyperBag import HyperBag
from ..WeightInitializer import GlorotInitializer
from ..DataSet.DataSet import DataSet


class ModelSelection:

    SearchObj: HyperParameterSearch
    hp: HyperBag
    SearchName: str
    TrialNumber: int

    def __init__(self, hp: HyperBag, search: HyperParameterSearch):
        self.hp = hp
        self.SearchObj = search
        self.SearchName = self.SearchObj.GetName()
        self.TrialNumber  = self.SearchObj.GetTrialNumber()
    def GetBestModel(self, hyperModel_fn, Data:DataSet, epoch:int, miniBatchSize: int | None=None,
                     watchMetric ="val_loss", metric : list[Metric]= None, weightInizializer = GlorotInitializer(),
                     callBack: list[CallBack] = None ) -> (FeedForwardModel, dict[str, any]):
        pass

class BestSearch(ModelSelection):


    def __init__(self, hp: HyperBag, search: HyperParameterSearch):
        super().__init__(hp, search)


    def GetBestModel(self, hyperModel_fn, Data:DataSet, epoch:int, miniBatchSize: int | None=None,
                     watchMetric ="val_loss", metric : list[Metric]= None, weightInizializer = GlorotInitializer(),
                     callBack: list[CallBack] = None ) -> (FeedForwardModel, dict[str, any]):
        best_model: FeedForwardModel = None
        best_watchMetric = float("inf")
        best_hpSel = None
        hyperParamWrapper = HyperBag()


        for hpSel  , i in self.SearchObj.Search(self.hp):
            hyperParamWrapper.Set(hpSel)
            print(f"{self.SearchName}: Iteration {i} / {self.TrialNumber} with param {hyperParamWrapper.GetHPString()}")
            hyperModel, optimizer= hyperModel_fn(hyperParamWrapper)
            hyperModel.Build(weightInizializer)
            if metric is not None and len(metric) != 0:
                hyperModel.AddMetrics(metric)

            hyperModel.Fit(optimizer, Data, epoch, miniBatchSize, callBack)

            last_watchMetric=hyperModel.MetricResults[watchMetric][-1]
            if  last_watchMetric < best_watchMetric:
                best_watchMetric = last_watchMetric
                best_hpSel = hpSel
                best_model = hyperModel

        hyperParamWrapper.Set(best_hpSel)

        return best_model, hyperParamWrapper

class BestSearchKFold(ModelSelection):


    def __init__(self, hp: HyperBag, search: HyperParameterSearch):
        super().__init__(hp, search)


    def GetBestModel(self, hyperModel_fn, allData:DataSet, epoch:int, miniBatchSize: int | None=None,
                     watchMetric ="val_loss", metrics : list[Metric]= None, weightInizializer = GlorotInitializer(),
                     callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):

        hyperParamWrapper = HyperBag()
        best_watchMetric = float("inf")
        best_hpSel = None

        for hpSel, i in self.SearchObj.Search(self.hp):
            fold_stat = []
            for j, (train, val) in enumerate(allData.Kfolds):
                DataTemp = DataSet.FromDataExampleTVT(train, val, allData.Test)
                hyperParamWrapper.Set(hpSel)
                print(f"{self.SearchName}: Fold {j+1}, Iteration {i+1} / {self.TrialNumber}  with param {hyperParamWrapper.GetHPString()}")
                hyperModel, optimizer= hyperModel_fn(hyperParamWrapper)
                hyperModel.Build(weightInizializer)
                if metrics is not None and len(metrics) != 0:
                    hyperModel.AddMetrics(metrics)

                hyperModel.Fit(optimizer, DataTemp, epoch, miniBatchSize, callBack)
                fold_stat.append(hyperModel.MetricResults[watchMetric][-1])


            last_watchMetric = np.array(fold_stat).mean()
            if  last_watchMetric < best_watchMetric or best_hpSel is None:
                best_watchMetric = last_watchMetric
                best_hpSel = hpSel


        hyperParamWrapper.Set(best_hpSel)

        best_model, Optimizer = hyperModel_fn(hyperParamWrapper)
        best_model.Build(weightInizializer)
        best_model.AddMetrics(metrics)

        d = DataSet.FromDataExampleTV(allData.Training,allData.Test)
        best_model.Fit(Optimizer,d, epoch, miniBatchSize)

        return best_model, hyperParamWrapper



