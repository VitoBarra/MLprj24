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
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_monk2 = "dataset/monk+s+problems/monks-2.train"

def HyperModel_Monk(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(),True, "input"))

    """
    #ADAM
    for i in range(2):
        model.AddLayer(Layer(5, Sigmoid(), True, f"_h{i}"))
    """

    for i in range(2):
        model.AddLayer(Layer(3, Sigmoid(), True, f"_h{i}"))

    model.AddLayer(Layer(1, Sigmoid(), False,"output"))

    #optimizer = BackPropagationMomentum(MSELoss(), 0.5, 0.015, 0.99, 0.02)
    #optimizer = BackPropagationNesterovMomentum(MSELoss(), 0.5, 0.9, 0.03, 0.02)
    optimizer = BackPropagation(MSELoss(), 0.7, 0.015, 0.99, 0.02)


    #optimizer = Adam(MSELoss(), 0.3, 0.007, 0.9, 0.99, 1e-8, 0.004) # monk2


    return model, optimizer


def HyperBag_Monk():
    hp = HyperBag()
    return hp


if __name__ == '__main__':

    alldata = readMonk(file_path_monk2)
    alldata.Shuffle(3)
    #alldata.Shuffle(11)
    alldata.PrintData()
    alldata.Split(0.15, 0.5)



    watched_metric = "val_loss"
    bestSearch = GetBestSearch(HyperBag_Monk(), RandomSearch(1))
    best_model,best_hpSel = bestSearch.GetBestModel(
        HyperModel_Monk,
        alldata,
        400,
        128,
        watched_metric,
        [Accuracy(Binary(0.5))],
        GlorotInitializer(2),
        [EarlyStopping(watched_metric, 400)])


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

    # metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() if not key.endswith("loss")}
    metric_to_plot = {key: value[:] for key, value in best_model.MetricResults.items() }

    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        baselineName= f"Baseline ({m.Name})" ,
        limitYRange=None,
        title="MONK results",
        xlabel="Epoche",
        ylabel="")


