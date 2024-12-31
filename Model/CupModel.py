from sklearn.linear_model import LinearRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss
from Core.Metric import *
from Core.ModelSelection import BestSearch, BestSearchKFold
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readCUP


USE_KFOLD = False
USE_ADAM = True

def HyperModel_CAP(hp :HyperBag ):
    model = ModelFeedForward()

    act_fun = TanH()

    model.AddLayer(Layer(12, Linear(), True, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"],act_fun, hp["drop_out"], True, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], act_fun, True, f"_h{i}"))

    model.AddLayer(Layer(3, Linear(), False, "output"))

    loss = MSELoss()
    if USE_ADAM:
        optimizer = Adam(loss,hp["eta"], hp["labda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])
    else:
        optimizer = BackPropagation(loss, hp["eta"], hp["labda"], hp["alpha"],hp["decay"])

    return model, optimizer



def HyperBag_Cap():
    hp = HyperBag()

    hp.AddRange("eta", 0.001, 0.1, 0.01)
    hp.AddRange("labda", 0.001, 0.1, 0.001)
    hp.AddRange("alpha", 0.1, 0.9, 0.1)
    hp.AddRange("decay", 0.001, 0.1, 0.001)

    ## Only Adam
    if USE_ADAM:
        hp.AddRange("beta", 0.95, 0.99, 0.01)
        hp.AddRange("epsilon", 1e-13, 1e-8, 1e-1)

    #hp.AddRange("drop_out", 0.1, 0.5, 0.05)

    hp.AddRange("unit", 1, 15, 1)
    hp.AddRange("hlayer", 0, 3, 1)

    return hp

def ReadCUP(val_split:float = 0.15, test_split:float =0.5):
    file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
    all_data = readCUP(file_path_cup)
    all_data.PrintData()
    all_data.Split(val_split, test_split)
    all_data.Standardize(True)

    baseline_metric = MAE()
    return all_data,baseline_metric


if __name__ == '__main__':

    alldata, baselineMetric= ReadCUP(0.15, 0.05)


    if USE_KFOLD:
        ModelSelector = BestSearchKFold(HyperBag_Cap(), RandomSearch(50), 5)
    else:
        ModelSelector = BestSearch(HyperBag_Cap(), RandomSearch(50))


    watched_metric = "val_loss"
    best_model, best_hpSel = ModelSelector.GetBestModel(
        HyperModel_CAP,
        alldata,
        500,
        128,
        watched_metric,
        [baselineMetric],
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
    baseline = baselineMetric.ComputeMetric(predictions, alldata.Validation.Label)

    metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items()}

    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        baselineName= f"Baseline ({baselineMetric.Name})",
        limitYRange=None,
        title="CUP results",
        xlabel="Epoche",
        ylabel="")
