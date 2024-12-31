import numpy as np

from Core import FeedForwardModel
from Core.Callback.CallBack import CallBack
from Core.Metric import Metric
from Core.Tuner.HpSearch import HyperParameterSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from Utility.DataSet import DataSet


class ModelSelection:

    def __init__(self, hp: HyperBag, search: HyperParameterSearch):
        pass
    def GetBestModel(self, hyperModel_fn, Data:DataSet, epoch:int, miniBatchSize: int | None=None,
                     watchMetric ="val_loss", metric : list[Metric]= None, weightInizializer = GlorotInitializer(),
                     callBack: list[CallBack] = None ) -> (FeedForwardModel, dict[str, any]):
        pass

class BestSearch(ModelSelection):
    SearchObj: HyperParameterSearch
    hp: HyperBag

    def __init__(self, hp: HyperBag, search: HyperParameterSearch):
        self.hp = hp
        self.SearchObj = search
        self.SearchName = self.SearchObj.GetName()
        self.TrialNumber  = self.SearchObj.TrialNumber()

    def GetBestModel(self, hyperModel_fn, Data:DataSet, epoch:int, miniBatchSize: int | None=None,
                     watchMetric ="val_loss", metric : list[Metric]= None, weightInizializer = GlorotInitializer(),
                     callBack: list[CallBack] = None ) -> (FeedForwardModel, dict[str, any]):
        best_model: FeedForwardModel = None
        best_watchMetric = float("inf")
        best_hpSel = None
        hyperParamWrapper = HyperBag()


        for hpSel  , i in self.SearchObj.search(self.hp):
            hyperParamWrapper.Set(hpSel)
            print(f"{self.SearchName}: Iteration {i} / {self.TrialNumber}")
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

        return best_model, best_hpSel

class BestSearchKFold(ModelSelection):
    SearchObj: HyperParameterSearch
    hp: HyperBag
    folds: int

    def __init__(self, hp: HyperBag, search: HyperParameterSearch, folds:int):
        self.hp = hp
        self.SearchObj = search
        self.folds = folds
        self.SearchName = self.SearchObj.GetName()
        self.TrialNumber  = self.SearchObj.TrialNumber()

    def GetBestModel(self, hyperModel_fn, allData:DataSet, epoch:int, miniBatchSize: int | None=None,
                     watchMetric ="val_loss", metrics : list[Metric]= None, weightInizializer = GlorotInitializer(),
                     callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):

        test , forlds = allData.Kfold_TestHoldOut(self.folds, 0.15)
        hyperParamWrapper = HyperBag()
        best_watchMetric = float("inf")
        best_hpSel = None

        for hpSel, i in self.SearchObj.search(self.hp):
            fold_stat = []
            for j, (train, val) in enumerate(forlds):
                DataTemp = DataSet.FromDataExample(train,val,test)
                hyperParamWrapper.Set(hpSel)
                print(f"{self.SearchName}: Fold {j+1}, Iteration {i+1} / {self.TrialNumber}")
                hyperModel, optimizer= hyperModel_fn(hyperParamWrapper)
                hyperModel.Build(weightInizializer)
                if metrics is not None and len(metrics) != 0:
                    hyperModel.AddMetrics(metrics)

                hyperModel.Fit(optimizer, DataTemp, epoch, miniBatchSize, callBack)
                fold_stat.append(hyperModel.MetricResults[watchMetric][-1])


            last_watchMetric = np.array(fold_stat).mean()
            if  last_watchMetric < best_watchMetric:
                best_watchMetric = last_watchMetric
                best_hpSel = hpSel


        hyperParamWrapper.Set(best_hpSel)

        best_model, Optimizer = hyperModel_fn(hyperParamWrapper)
        best_model.Build(weightInizializer)
        best_model.AddMetrics(metrics)
        allData.Split(0.0,0.15)
        best_model.Fit(Optimizer,allData, epoch, miniBatchSize)

        return best_model, hyperParamWrapper



