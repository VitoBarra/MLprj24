import random

from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Metric import *
from Core.Tuner.HpSearch import RandomSearch, GetBestSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
file_path_monk = "dataset/monk+s+problems/monks-1.train"


def HyperModel_Monk(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(),True, "input"))
    for i in range(hp["hlayer"]):
        model.AddLayer(Layer(hp["unit"], ReLU(),True, f"h{i}"))

    model.AddLayer(Layer(1, Sigmoid(), False,"output"))
    return model



def HyperModel_Monk_Full(hp):
    model = HyperModel_Monk(hp)
    #optimizer = BackPropagation(BinaryCrossEntropyLoss(), hp["eta"], hp["labda"], hp["alpha"])
    optimizer = BackPropagation(BinaryCrossEntropyLoss(), 0.05, None, None)
    return model, optimizer


if __name__ == '__main__':

    #MONK-1
    alldata = readMonk(file_path_monk)
    alldata.SplitDataset(0.15,0.5)
    training , val, test = alldata.Training , alldata.Validation , alldata.Test


    hp = HyperBag()

    hp.AddRange("labda", 0.04, 0.1, 0.02)
    hp.AddRange("alpha", 0.05, 0.5, 0.05)
    hp.AddRange("eta", 0.01, 0.4, 0.02)

    hp.AddChosen("hlayer", [1,2,3,4])
    hp.AddChosen("unit",[1,2,3])

    watched_metric = "val_loss"
    bestSearch = GetBestSearch(hp, RandomSearch(25))
    best_model,best_hpSel = bestSearch.GetBestModel(
        HyperModel_Monk_Full,
        alldata,
        500,
        None,
        watched_metric,
        None,
        GlorotInitializer(),
        [EarlyStopping(watched_metric, 5)])


    k = Binary()
    out_test= best_model.Predict(training.Data)
    out_test = k.Calculate(out_test)
    metric = Accuracy()

    print(f"accuracy on test: {metric.ComputeMetric(out_test, training.Label)}%")

    print(f"Best hp : {best_hpSel}")
    best_model.PlotModel()

    best_model.SaveMetricsResults("Data/Results/model1.mres")

    #print("\n\n")
    best_model.SaveModel("Data/Models/Test1.vjf")

    metrics = best_model.MetricResults
    val_metrics = {key: value for key, value in metrics.items() if key.startswith("")}

    # plot_losses_accuracy(
    #     metricDic=val_metrics,
    #     title="Metriche di Validazione",
    #     xlabel="Epoche",
    #     ylabel="Valore"
    # )