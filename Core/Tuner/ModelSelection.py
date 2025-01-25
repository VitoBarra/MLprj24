import numpy as np

from .HyperModel import HyperModel
from .. import FeedForwardModel
from ..Callback.CallBack import CallBack
from ..Metric import Metric
from ..Tuner.HpSearch import HyperParameterSearch
from ..Tuner.HyperBag import HyperBag
from ..Initializer.WeightInitializer import GlorotInitializer
from ..DataSet.DataSet import DataSet


class ModelSelection:

    SearchObj: HyperParameterSearch
    SearchName: str
    TrialNumber: int

    def __init__(self, search: HyperParameterSearch):
        self.SearchObj = search
        self.SearchName = self.SearchObj.GetName()

    def GetBestModel(self, hyperModel_fn, hp:HyperBag,Data:DataSet, epoch:int,
                     watchMetric ="val_loss", metric : list[Metric]= None, weightInitializer = GlorotInitializer(),
                     callBack: list[CallBack] = None ) -> (FeedForwardModel, dict[str, any]):
        pass
    def GetBestModel_HyperModel(self, hyperModel:HyperModel, epoch:int,
                                watchMetric ="val_loss", metrics : list[Metric]= None, weightInitializer = GlorotInitializer(),
                                callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):
        pass

class BestSearch(ModelSelection):

    def __init__(self, search: HyperParameterSearch):
        super().__init__(search)

    def _train_and_evaluate(self, get_model_fn, get_optimizer_fn, get_data_fn,
                            hyper_param_source, epoch, watchMetric, metric,
                            weightInitializer, callBack):
        """
        Common logic for training and evaluating models during hyperparameter search.
        """
        best_model = None
        best_watchMetric = float("inf")
        best_hpSel = None
        hyperParamWrapper = HyperBag()

        for hpSel, i in self.SearchObj.Search(hyper_param_source):
            hyperParamWrapper.Set(hpSel)
            print(f"{self.SearchName}: Iteration {i} / {self.SearchObj.GetTrialNumber()} with param {hyperParamWrapper.GetHPString()}")

            # Fetch model, optimizer, and data
            data = get_data_fn(hyperParamWrapper)
            model = get_model_fn(hyperParamWrapper)
            optimizer = get_optimizer_fn(hyperParamWrapper)

            model.Build(weightInitializer)
            if metric is not None and len(metric) != 0:
                model.AddMetrics(metric)

            model.Fit(optimizer, data, epoch, callBack)

            last_watchMetric = model.MetricResults[watchMetric][-1]
            if last_watchMetric < best_watchMetric:
                best_watchMetric = last_watchMetric
                best_hpSel = hpSel
                best_model = model

        hyperParamWrapper.Set(best_hpSel)
        return best_model, hyperParamWrapper

    def GetBestModel(self, hyperModel_fn, hp: HyperBag, Data: DataSet, epoch: int,
                     watchMetric="val_loss", metric: list[Metric] = None,
                     weightInitializer=GlorotInitializer(), callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):
        return self._train_and_evaluate(
            get_model_fn=lambda hp_wrapper: hyperModel_fn(hp_wrapper)[0],
            get_optimizer_fn=lambda hp_wrapper: hyperModel_fn(hp_wrapper)[1],
            get_data_fn=lambda _: Data,
            hyper_param_source=hp,
            epoch=epoch,
            watchMetric=watchMetric,
            metric=metric,
            weightInitializer=weightInitializer,
            callBack=callBack
        )

    def GetBestModel_HyperModel(self, hyperModel: HyperModel, epoch: int,
                                watchMetric="val_loss", metric: list[Metric] = None,
                                weightInitializer=GlorotInitializer(), callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):
        return self._train_and_evaluate(
            get_model_fn=lambda hp_wrapper: hyperModel.GetModel(hp_wrapper),
            get_optimizer_fn=lambda hp_wrapper: hyperModel.GetOptimizer(hp_wrapper),
            get_data_fn=lambda hp_wrapper: hyperModel.GetDatasetVariant(hp_wrapper),
            hyper_param_source=hyperModel.GetHyperParameters(),
            epoch=epoch,
            watchMetric=watchMetric,
            metric=metric,
            weightInitializer=weightInitializer,
            callBack=callBack
        )

class BestSearchKFold(ModelSelection):

    def __init__(self, search: HyperParameterSearch):
        super().__init__(search)

    def _train_and_evaluate_kfold(self, get_model_fn, get_optimizer_fn, get_data_fn,
                                  hyper_param_source, epoch, watchMetric,
                                  metrics, weightInitializer, callBack):
        """
        Common logic for training and evaluating models with K-Fold cross-validation.
        """
        hyperParamWrapper = HyperBag()
        best_watchMetric = float("inf")
        best_hpSel = None

        for hpSel, i in self.SearchObj.Search(hyper_param_source):
            fold_stat = []
            allData = get_data_fn(hpSel)

            for j, (train, val,test) in enumerate(allData.Kfolds):
                DataTemp = DataSet.FromDataExampleTVT(train, val,test)
                hyperParamWrapper.Set(hpSel)
                print(f"{self.SearchName}: Fold {j+1}, Iteration {i+1} / {self.SearchObj.GetTrialNumber()} with param {hyperParamWrapper.GetHPString()}")

                # Fetch model and optimizer
                model = get_model_fn(hyperParamWrapper)
                optimizer = get_optimizer_fn(hyperParamWrapper)

                model.Build(weightInitializer)
                if metrics is not None and len(metrics) != 0:
                    model.AddMetrics(metrics)

                model.Fit(optimizer, DataTemp, epoch, callBack)
                fold_stat.append(model.MetricResults[watchMetric][-1])

            last_watchMetric = np.array(fold_stat).mean()
            if last_watchMetric < best_watchMetric or best_hpSel is None:
                best_watchMetric = last_watchMetric
                best_hpSel = hpSel

        hyperParamWrapper.Set(best_hpSel)
        return hyperParamWrapper

    def _final_training(self, best_model, optimizer, allData, epoch, metrics, weightInitializer):
        """
        Common logic for final training on the best hyperparameter configuration.
        """
        best_model.Build(weightInitializer)
        if metrics is not None and len(metrics) != 0:
            best_model.AddMetrics(metrics)

        dataset = DataSet.FromDataExampleTV(allData.Training, allData.Test)
        best_model.Fit(optimizer, dataset, epoch)

        return best_model

    def GetBestModel(self, hyperModel_fn, hp: HyperBag, allData: DataSet, epoch: int,
                     watchMetric="val_loss", metrics: list[Metric] = None,
                     weightInitializer=GlorotInitializer(), callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):
        hp_sel = self._train_and_evaluate_kfold(
            get_model_fn=lambda hp_wrapper: hyperModel_fn(hp_wrapper)[0],
            get_optimizer_fn=lambda hp_wrapper: hyperModel_fn(hp_wrapper)[1],
            get_data_fn=lambda _: allData,
            hyper_param_source=hp,
            epoch=epoch,
            watchMetric=watchMetric,
            metrics=metrics,
            weightInitializer=weightInitializer,
            callBack=callBack
        )

        model,optimizer = hyperModel_fn(hp_sel)
        best_model = self._final_training(
            best_model=model,
            optimizer=optimizer,
            allData=allData,
            epoch=epoch,
            metrics=metrics,
            weightInitializer=weightInitializer
        )

        return best_model, hp_sel

    def GetBestModel_HyperModel(self, hyperModel: HyperModel, epoch: int,
                                watchMetric="val_loss", metrics: list[Metric] = None,
                                weightInitializer=GlorotInitializer(), callBack: list[CallBack] = None) -> (FeedForwardModel, dict[str, any]):
        hp_sel = self._train_and_evaluate_kfold(
            get_model_fn=lambda hp_wrapper: hyperModel.GetModel(hp_wrapper),
            get_optimizer_fn=lambda hp_wrapper: hyperModel.GetOptimizer(hp_wrapper),
            get_data_fn=lambda hp_wrapper: hyperModel.GetDatasetVariant(hp_wrapper),
            hyper_param_source=hyperModel.GetHyperParameters(),
            epoch=epoch,
            watchMetric=watchMetric,
            metrics=metrics,
            weightInitializer=weightInitializer,
            callBack=callBack
        )

        best_model = self._final_training(
            best_model=hyperModel.GetModel(hp_sel) ,
            optimizer=hyperModel.GetOptimizer(hp_sel),
            allData= hyperModel.GetDatasetVariant(hp_sel),
            epoch=epoch,
            metrics=metrics,
            weightInitializer=weightInitializer
        )

        return best_model, hp_sel



