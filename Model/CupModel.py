from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Metric import *
from Core.Tuner.HpSearch import RandomSearch, GetBestSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *
from sklearn.linear_model import LogisticRegression, LinearRegression

file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"


def HyperBag_Cap():
    hp = HyperBag()

    hp.AddRange("eta", 0.01, 0.4, 0.02)
    hp.AddRange("labda", 0.005, 0.1, 0.005)
    hp.AddRange("alpha", 0.05, 0.5, 0.05)


    hp.AddRange("unit",1,10,1)
    hp.AddRange("hlayer",1,4,1)
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
    alldata.Normalize(True)
    alldata.PrintData()
    alldata.Split(0.15, 0.5)




    watched_metric = "val_loss"
    bestSearch = GetBestSearch(HyperBag_Cap(), RandomSearch(250))
    best_model,best_hpSel = bestSearch.GetBestModel(
        HyperModel_CAP,
        alldata,
        500,
        128,
        watched_metric,
        [MAE()],
        GlorotInitializer(),
        [EarlyStopping(watched_metric, 12)])


    print(f"Best hp : {best_hpSel}")
    best_model.PlotModel("CUP Model")

    best_model.SaveModel("Data/Models/BestCup.vjf")
    best_model.SaveMetricsResults("Data/Results/BestCup.mres")



    lin_model = LinearRegression()
    lin_model.fit(alldata.Training.Data, alldata.Training.Label)
    print(f"R2 on test: {lin_model.score(alldata.Validation.Data, alldata.Validation.Label)}%")
    predictions = lin_model.predict(alldata.Validation.Data)
    m = MSE()
    baseline= m.ComputeMetric(predictions, alldata.Validation.Label)

    metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() if key.startswith("")}

    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        limityRange=None,
        title="CUP",
        xlabel="Epoche",
        ylabel="")


