import numpy as np
from DataUtility.DataSet import DataSet
from Core.FeedForwardModel import ModelFeedForward
import os


def train_k_fold(data:DataSet, k:int, metric = 0) -> ModelFeedForward():
    """
       Esegue la k-fold cross-validation su un dataset fornito.

       :param data: Un oggetto DataSet che implementa il metodo k_fold_cross_validation(k).
       :param k: Il numero di fold da utilizzare per la cross-validation.
       :param metric: La metrica da utilizzare per la valutazione (default: accuracy_score).
       :return: Una tupla (media_metriche, varianza_metriche).
    """
    os.chdir("..")
    models_path = os.path.join(os.getcwd(), "Models")

    folds = data.k_fold_cross_validation(k)
    model= ModelFeedForward()
    best_model = None
    best_metric_value = -np.inf
    best_fold_idx = -1

    for fold_idx, (train_set, test_set) in enumerate(folds):
        model.Fit(train_set, 10, 0)
        metrics = model.MetricResults
        print(metrics)
        final_metric = metrics[metric][-1]
        if final_metric > best_metric_value:
            best_metric_value = final_metric
            best_model = model
            best_fold_idx = fold_idx
        #model.SaveModel(models_path)

    return best_model




