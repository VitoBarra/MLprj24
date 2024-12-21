import random

from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Metric import *
from Core.Tuner.HpSearch import RandomSearch, GetBestSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
file_path_monk1 = "dataset/monk+s+problems/monks-1.train"
file_path_monk2 = "dataset/monk+s+problems/monks-2.train"
file_path_monk3 = "dataset/monk+s+problems/monks-3.train"








def HyperModel_Monk(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(),True, "input"))
    for i in range(hp["hlayer"]):
        model.AddLayer(Layer(hp["unit"], ReLU(),True, f"h{i}"))

    model.AddLayer(Layer(1, Sigmoid(), False,"output"))

    optimizer = BackPropagation(BinaryCrossEntropyLoss(), hp["eta"], hp["labda"], hp["alpha"])
    return model, optimizer


def HyperBag_Monk():
    hp = HyperBag()

    hp.AddRange("labda", 0.04, 0.1, 0.02)
    hp.AddRange("alpha", 0.05, 0.5, 0.05)
    hp.AddRange("eta", 0.01, 0.4, 0.02)


    hp.AddRange("unit",3,5,1)
    hp.AddRange("hlayer",1,3,1)
    return hp

def HyperBag_Cap():
    hp = HyperBag()

    hp.AddRange("labda", 0.04, 0.1, 0.02)
    hp.AddRange("alpha", 0.05, 0.5, 0.05)
    hp.AddRange("eta", 0.01, 0.4, 0.02)


    hp.AddRange("unit",10,25,5)
    hp.AddRange("hlayer",5,10,1)
    return hp

def HyperModel_CAP(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(12, Linear(),True, "input"))
    for i in range(hp["hlayer"]):
        model.AddLayer(Layer(hp["unit"], TanH(),True, f"h{i}"))

    model.AddLayer(Layer(3, Linear(), False,"output"))

    optimizer = BackPropagation(MSELoss(), hp["eta"], hp["labda"], hp["alpha"])
    return model, optimizer

if __name__ == '__main__':

    alldata = readCUP(file_path_cup)

    for d, l, id in alldata:
        print(d,l, id)

    training , val, test = alldata.SplitDataset(0.15,0.5)


    watched_metric = "val_loss"
    bestSearch = GetBestSearch(HyperBag_Cap(), RandomSearch(50))
    best_model,best_hpSel = bestSearch.GetBestModel(
        HyperModel_CAP,
        alldata,
        500,
        None,
        watched_metric,
        [MEE()],
        GlorotInitializer(),
        [EarlyStopping(watched_metric, 10)])


    # k = Binary()
    # out_test= best_model.Predict(training.Data)
    # out_test = k.Calculate(out_test)
    # metric = Accuracy()
    #
    # print(f"accuracy on test: {metric.ComputeMetric(out_test, training.Label)}%")

    print(f"Best hp : {best_hpSel}")
    #best_model.PlotModel()

    best_model.SaveMetricsResults("Data/Results/model1.mres")

    #print("\n\n")
    best_model.SaveModel("Data/Models/Test1.vjf")

    metrics = best_model.MetricResults
    metric_to_plot = {key: value[2:] for key, value in metrics.items() if key.startswith("")}

    plot_metric(
        metricDic=metric_to_plot,
        title="Metriche di Validazione",
        xlabel="Epoche",
        ylabel="Valore"
    )