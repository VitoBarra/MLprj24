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
        """
        Initializes the ModelSelection object.

        :param search: An instance of HyperParameterSearch used for conducting the hyperparameter search.
        """
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

        :param get_model_fn: Function to generate the model given hyperparameters.
        :param get_optimizer_fn: Function to generate the optimizer given hyperparameters.
        :param get_data_fn: Function to generate the dataset for training/validation.
        :param hyper_param_source: Source of hyperparameter configurations.
        :param epoch: Number of training epochs.
        :param watchMetric: Metric used to evaluate model performance.
        :param metric: Additional metrics for evaluation.
        :param weightInitializer: A weight initialization strategy.
        :param callBack: List of callbacks to use during training.
        :returns: The best-performing model and its hyperparameter configuration.
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
        """
        Implements the model search using a hyperModel function for generating models and optimizers.

        :param hyperModel_fn: Function to generate the model and optimizer.
        :param hp: HyperBag containing the hyperparameter configurations.
        :param Data: DataSet for training and validation.
        :param epoch: Number of training epochs.
        :param watchMetric: The metric used to evaluate model performance.
        :param metric: Additional metrics for evaluation.
        :param weightInitializer: A weight initialization strategy.
        :param callBack: List of callbacks to use during training.
        :returns: The best-performing model and its hyperparameter configuration.
        """
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
        """
       Selects the best model using a HyperModel instance, by performing hyperparameter search and training.

       This method leverages the `_train_and_evaluate` function to run the hyperparameter search process,
       fetching the model, optimizer, and dataset from the HyperModel instance. The training process is
       performed and evaluated based on the specified metrics, with the best model and corresponding
       hyperparameters being returned.

       :param hyperModel: An instance of the HyperModel class, which contains model-specific methods.
       :param epoch: Number of training epochs.
       :param watchMetric: The metric to monitor for determining the best model (default is "val_loss").
       :param metric: A list of additional metrics to track during training.
       :param weightInitializer: An initializer for the model's weights (default is GlorotInitializer).
       :param callBack: A list of callbacks to be used during training.

       :return: A tuple consisting of the best FeedForwardModel and a dictionary of hyperparameters.
       """
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

        This method iterates through the hyperparameter configurations, performing K-Fold cross-validation
        and tracking the performance metric of interest to find the best configuration.

        :param get_model_fn: A function to get the model based on the current hyperparameters.
        :param get_optimizer_fn: A function to get the optimizer based on the current hyperparameters.
        :param get_data_fn: A function to get the dataset for training and validation.
        :param hyper_param_source: The source of hyperparameters for the search.
        :param epoch: Number of epochs to train the model.
        :param watchMetric: The metric to monitor for model performance (default is "val_loss").
        :param metrics: Additional metrics to track during training.
        :param weightInitializer: Weight initialization method to be used in the model.
        :param callBack: A list of callbacks for monitoring training.

        :return: The best hyperparameter configuration that minimizes the watchMetric.
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

        This method takes the best hyperparameter configuration and performs the final training on the
        full dataset to obtain the best model.

        :param best_model: The model to be trained with the best hyperparameters.
        :param optimizer: The optimizer used for training the model.
        :param allData: The dataset containing the training and test data.
        :param epoch: Number of epochs for the final training phase.
        :param metrics: List of metrics to be tracked during training.
        :param weightInitializer: The weight initializer to be used.

        :return: The trained best model.
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
        """
        Performs hyperparameter search and K-Fold cross-validation to find and train the best model.

        This method combines the processes of hyperparameter search, K-Fold cross-validation, and
        final model training to select the best-performing model.

        :param hyperModel_fn: A function that returns a tuple of model and optimizer based on hyperparameters.
        :param hp: The hyperparameter bag containing different hyperparameter configurations.
        :param allData: The full dataset to be used for K-Fold cross-validation.
        :param epoch: The number of epochs to train each model.
        :param watchMetric: The metric to monitor for determining the best model (default is "val_loss").
        :param metrics: A list of additional metrics to track during training.
        :param weightInitializer: Weight initialization method for model parameters (default is GlorotInitializer).
        :param callBack: A list of callbacks to be used during training.

        :return: A tuple containing the best model and the hyperparameters that resulted in it.
        """
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
        """
       Performs hyperparameter search and K-Fold cross-validation to find and train the best model
       using a HyperModel instance.

       This method combines the processes of hyperparameter search, K-Fold cross-validation, and
       final model training to select the best-performing model when using a HyperModel instance.

       :param hyperModel: The HyperModel instance containing the model, optimizer, and dataset methods.
       :param epoch: The number of epochs for training the model.
       :param watchMetric: The metric to monitor for determining the best model (default is "val_loss").
       :param metrics: A list of additional metrics to track during training.
       :param weightInitializer: Weight initialization method for model parameters (default is GlorotInitializer).
       :param callBack: A list of callbacks to be used during training.

       :return: A tuple containing the best model and the hyperparameters that resulted in it.
       """
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



