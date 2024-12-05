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
    """for i, (train_set, test_set) in enumerate(folds):
        print(f"Fold {i + 1}:")

        # Dati del training set
        print("Training Data:\n", train_set.Data)
        print("Training Labels:\n", train_set.Label)
        print("Training IDs:\n", train_set.Id)

        # Dati del test set
        print("Test Data:\n", test_set.Data)
        print("Test Labels:\n", test_set.Label)
        print("Test IDs:\n", test_set.Id)"""
    best_model = None
    best_metric_value = -np.inf
    best_fold_idx = -1
    for fold_idx, (train_set, test_set) in enumerate(folds):
        model = ModelFeedForward()
        model.AddLayer(Layer(1, Linear()))
        model.AddLayer(Layer(15, TanH()))
        model.AddLayer(Layer(15, TanH()))
        model.AddLayer(Layer(1, Linear()))
        model.Build(GlorotInitializer())
        model.AddMetrics([MSE(), RMSE(), MEE()])

        model.Fit(BackPropagation(MSELoss()), train_set, 15, 2, test_set)

        metrics = model.MetricResults
        print(metrics.items())
        final_metric = metrics["MSE"][-1]  # Seleziona l'ultimo valore della metrica MSE.
        if final_metric > best_metric_value:
            best_metric_value = final_metric
            best_model = model
            best_fold_idx = fold_idx
        #model.SaveModel(models_path)

    return best_model




