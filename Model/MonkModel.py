from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Metric import *
from Core.Tuner.HpSearch import RandomSearch, GetBestSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *
from sklearn.linear_model import LogisticRegression, LinearRegression

file_path_monk1 = "dataset/monk+s+problems/monks-1.train"
file_path_monk2 = "dataset/monk+s+problems/monks-2.train"
file_path_monk3 = "dataset/monk+s+problems/monks-3.train"



def HyperModel_Monk(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(),True, "input"))
    for i in range(hp["hlayer"]):
        model.AddLayer(Layer(hp["unit"], ReLU(),True, f"h{i}"))

    model.AddLayer(Layer(1, Sigmoid(), False,"output"))

    optimizer = BackPropagation(MSELoss(), hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    return model, optimizer


def HyperBag_Monk():
    hp = HyperBag()

    hp.AddRange("eta", 0.01, 0.4, 0.02)
    hp.AddRange("labda", 0.01, 0.1, 0.01)
    hp.AddRange("alpha", 0.05, 0.5, 0.05)
    hp.AddRange("decay", 0.001, 0.1, 0.005)


    hp.AddRange("unit",1,3,1)
    hp.AddRange("hlayer",0,2,1)
    return hp


if __name__ == '__main__':

    alldata = readMonk(file_path_monk1)
    alldata.PrintData()
    alldata.Split(0.15, 0.5)



    watched_metric = "val_loss"
    bestSearch = GetBestSearch(HyperBag_Monk(), RandomSearch(25))
    best_model,best_hpSel = bestSearch.GetBestModel(
        HyperModel_Monk,
        alldata,
        500,
        128,
        watched_metric,
        [Accuracy(Binary(0.5))],
        GlorotInitializer(),
        [EarlyStopping(watched_metric, 12)])


    print(f"Best hp : {best_hpSel}")
    best_model.PlotModel("MONK Model")

    best_model.SaveModel("Data/Models/Monk1.vjf")
    best_model.SaveMetricsResults("Data/Results/Monk1.mres")




    lin_model = LogisticRegression()
    lin_model.fit(alldata.Training.Data, alldata.Training.Label.reshape(-1))
    print(f"R2 on test: {lin_model.score(alldata.Validation.Data, alldata.Validation.Label)}%")
    predictions = lin_model.predict(alldata.Validation.Data)

    m = Accuracy(Binary())
    # m= MSE()
    baseline= m.ComputeMetric(predictions.reshape(-1,1), alldata.Validation.Label)

    metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() if not key.endswith("loss")}


    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        baselineName= f"Baseline ({m.Name})" ,
        limitYRange=None,
        title="MONK results",
        xlabel="Epoche",
        ylabel="")


