from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Layer import *
from Core.WeightInitializer import *
from Core.Metric import *
from Core.callback.EarlyStopping import EarlyStopping
from DataUtility.DataExamples import *
from Core.BackPropagation import *

from Core.BackPropagation import BackPropagation
from Core.LossFunction import MSELoss
from Core.Metric import *
from DataUtility.DataSet import DataSet
from Core.FeedForwardModel import ModelFeedForward
import os


def train_k_fold(data:DataSet, k:int, fn_buindModel, patience:int = None, validation: str = "val_loss") -> ModelFeedForward():
    """
       Performs k-fold cross-validation on a dataset and returns the best model based on the validation metric.

       :param data: Dataset object that implements the k_fold_cross_validation(k) method to split the data into k folds.
       :param k: The number of folds to use for cross-validation.
       :param fn_buindModel: Callable to create a new ModelFeedForward instance for each fold.
       :param patience: The patience for early stopping, determining how many epochs to wait without improvement before stopping.
       :param validation: The validation metric to monitor (default is "val_loss").
       :return: The best model based on the validation metric after performing k-fold cross-validation.
    """

    folds = data.k_fold_cross_validation_split(k)
    best_model = None
    best_metric_value = -np.inf
    best_fold_idx = -1
    for fold_idx, (train_set, test_set) in enumerate(folds):
        model = fn_buindModel()
        if patience is not None:
            es = EarlyStopping(patience, model, validation)
            model.Fit(BackPropagation(MSELoss()), train_set, 100, 12, test_set, [es])
        else:
            model.Fit(BackPropagation(MSELoss()), train_set, 100, 12, test_set)



        metrics = model.MetricResults
        final_metric = metrics["val_loss"][-1]
        if final_metric > best_metric_value:
            best_metric_value = final_metric
            best_model = model
            best_fold_idx = fold_idx
        #model.SaveModel(models_path)

    return best_model




