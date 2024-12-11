from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Layer import *
from Core.WeightInitializer import *
from Core.Metric import *
from DataUtility.DataExamples import *
from Core.BackPropagation import *

from Core.BackPropagation import BackPropagation
from Core.LossFunction import MSELoss
from Core.Metric import *
from DataUtility.DataSet import DataSet
from Core.FeedForwardModel import ModelFeedForward
import os


def train_k_fold(data:DataSet, k:int, fn_buindModel) -> ModelFeedForward():
    """
       Esegue la k-fold cross-validation su un dataset fornito.

       :param data: Un oggetto DataSet che implementa il metodo k_fold_cross_validation(k).
       :param k: Il numero di fold da utilizzare per la cross-validation.
       :param metric: La metrica da utilizzare per la valutazione (default: accuracy_score).
       :return: Una tupla (media_metriche, varianza_metriche).
    """

    folds = data.k_fold_cross_validation_split(k)
    best_model = None
    best_metric_value = -np.inf
    best_fold_idx = -1
    for fold_idx, (train_set, test_set) in enumerate(folds):
        model = fn_buindModel()

        model.Fit(BackPropagation(MSELoss()), train_set, 50, 12, test_set)

        metrics = model.MetricResults
        final_metric = metrics["val_loss"][-1]  # Seleziona l'ultimo valore della metrica MSE.
        if final_metric > best_metric_value:
            best_metric_value = final_metric
            best_model = model
            best_fold_idx = fold_idx
        #model.SaveModel(models_path)

    return best_model




