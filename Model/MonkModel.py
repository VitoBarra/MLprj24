from pickletools import optimize

from sklearn.linear_model import LogisticRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss
from Core.Metric import *
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Optimizer.BackPropagationMomentum import BackPropagationMomentum
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Core.Optimizer.Adam import Adam
from Core.Tuner.HpSearch import RandomSearch, GetBestSearch, GridSearch
from Core.Metric import Accuracy
from Core.Tuner.HpSearch import  RandomSearch, GridSearch
from Core.ModelSelection import *
from Core.Tuner.HyperBag import HyperBag
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

USE_CATEGORICAL = False

file_path_monk1 = "dataset/monk+s+problems/monks-1.train"
file_path_monk2 = "dataset/monk+s+problems/monks-2.train"
file_path_monk3 = "dataset/monk+s+problems/monks-3.train"


def HyperModel_Monk(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(),True, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"], Sigmoid(), hp["drop_out"], True, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], Sigmoid(), False, f"_h{i}"))


    if USE_CATEGORICAL:
        model.AddLayer(Layer(2, SoftARGMax(), False, "output_HotEncoding"))
        loss = CategoricalCrossEntropyLoss()
    else:
        model.AddLayer(Layer(1, Sigmoid(), False,"output"))
        loss = MSELoss()

    optimizer = BackPropagation(loss, hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    return model, optimizer


def HyperBag_Monk():
    hp = HyperBag()

    hp.AddRange("eta", 0.0005, 0.1, 0.005)
    hp.AddRange("labda", 0.001, 0.05, 0.005)
    hp.AddRange("alpha", 0.05, 0.9, 0.05)
    hp.AddRange("decay", 0.0001, 0.1, 0.0005)

    #hp.AddRange("drop_out", 0.2, 0.6, 0.1)

    hp.AddRange("unit", 1, 5, 1)
    hp.AddRange("hlayer", 0, 2, 1)
    return hp


if __name__ == '__main__':

    alldata = readMonk(file_path_monk1)
    alldata.Shuffle(195)
    if USE_CATEGORICAL:
        alldata.ToCategoricalLabel()
    alldata.PrintData()
    #alldata.Split(0.15, 0.5)

    BaselineMetric = Accuracy(Binary(0.5))
    watched_metric = "val_loss"
    bestSearchCrossValidation = BestSearchKFold(HyperBag_Monk(), RandomSearch(50), 4)
    callBacks = [EarlyStopping(watched_metric, 100)]
    best_model, best_hpSel = bestSearchCrossValidation.GetBestModel(
        HyperModel_Monk,
        alldata,
        100,
        64,
        watched_metric,
        [BaselineMetric],
        GlorotInitializer(),
        callBacks)


    print(f"Best hp : {best_hpSel}")


    best_model.PlotModel("MONK Model")

    best_model.SaveModel("Data/Models/Monk1.vjf")
    best_model.SaveMetricsResults("Data/Results/Monk1.mres")


    lin_model = LogisticRegression()
    if USE_CATEGORICAL:
        lin_model.fit(alldata.Training.Data, alldata.Training.Label[:,1])
        validationLable=alldata.Test.Label[:,1]
    else:
        lin_model.fit(alldata.Training.Data, alldata.Training.Label.reshape(-1))
        validationLable=alldata.Test.Label

    print(f"R2 on test: {lin_model.score(alldata.Test.Data, validationLable)}%")
    predictions = lin_model.predict(alldata.Test.Data)
    baseline= BaselineMetric(predictions.reshape(-1, 1), validationLable)

    #metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() if not key.startwith("")}
    metric_to_plot = {key: value[:] for key, value in best_model.MetricResults.items() }

    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        baselineName= f"baseline {BaselineMetric.Name}" ,
        limitYRange=None,
        title="MONK results",
        xlabel="Epoche",
        ylabel="")