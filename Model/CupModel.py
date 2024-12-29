from sklearn.linear_model import LinearRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss
from Core.Metric import *
from Core.ModelSelection import BestSearch
from Core.Optimizer.Adam import Adam
from Core.Tuner.HpSearch import RandomSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"

def HyperModel_CAP(hp :HyperBag ):
    model = ModelFeedForward()

    model.AddLayer(Layer(12, Linear(), True, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"], TanH(), hp["drop_out"], True, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], TanH(), True, f"_h{i}"))

    model.AddLayer(Layer(3, Linear(), False, "output"))

    optimizer = Adam(MSELoss(),hp["eta"], hp["labda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])
    return model, optimizer



def HyperBag_Cap():
    hp = HyperBag()

    hp.AddRange("eta", 0.001, 0.1, 0.01)
    hp.AddRange("labda", 0.001, 0.1, 0.001)
    hp.AddRange("alpha", 0.1, 0.9, 0.1)
    hp.AddRange("decay", 0.001, 0.1, 0.001)
    hp.AddRange("beta", 0.95, 0.99, 0.01)
    hp.AddRange("epsilon", 1e-13, 1e-8, 1e-1)

    #hp.AddRange("drop_out", 0.1, 0.5, 0.05)

    hp.AddRange("unit", 1, 15, 1)
    hp.AddRange("hlayer", 0, 3, 1)

    return hp



if __name__ == '__main__':
    alldata = readCUP(file_path_cup)
    alldata.PrintData()
    alldata.Split(0.15, 0.5)
    alldata.Standardize(True)

    watched_metric = "val_loss"
    bestSearch = BestSearch(HyperBag_Cap(), RandomSearch(250))
    best_model, best_hpSel = bestSearch.GetBestModel(
        HyperModel_CAP,
        alldata,
        500,
        128,
        watched_metric,
        [],
        GlorotInitializer(),
        [EarlyStopping(watched_metric, 10)])

    print(f"Best hp : {best_hpSel}")
    best_model.PlotModel("CUP Model")

    best_model.SaveModel("Data/Models/BestCup.vjf")
    best_model.SaveMetricsResults("Data/Results/BestCup.mres")

    lin_model = LinearRegression()
    lin_model.fit(alldata.Training.Data, alldata.Training.Label)
    print(f"R2 on test: {lin_model.score(alldata.Validation.Data, alldata.Validation.Label)}%")
    predictions = lin_model.predict(alldata.Validation.Data)
    m = MSE()
    baseline = m.ComputeMetric(predictions, alldata.Validation.Label)

    metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() if key.startswith("")}

    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        baselineName= f"Baseline ({m.Name})",
        limitYRange=None,
        title="CUP results",
        xlabel="Epoche",
        ylabel="")
