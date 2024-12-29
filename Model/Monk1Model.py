from sklearn.linear_model import LogisticRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss
from Core.Metric import *
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GetBestSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_monk1 = "dataset/monk+s+problems/monks-1.train"

def HyperModel_Monk_manual(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(),True, "input"))

    """
    #ADAM
    for i in range(1):
        #model.AddLayer(Layer(5, TanH(), True, f"_h{i}"))
        model.AddLayer(Layer(10, Sigmoid(), True, f"_h{i}"))
    """

    #Back Prop
    for i in range(2):
        model.AddLayer(Layer(10, Sigmoid(), True, f"_h{i}"))


    model.AddLayer(Layer(1, Sigmoid(), False,"output"))

    #optimizer = BackPropagationMomentum(MSELoss(), 0.5, 0.015, 0.99, 0.02)
    #optimizer = BackPropagationNesterovMomentum(MSELoss(), 0.5, 0.9, 0.03, 0.02)
    #optimizer = BackPropagation(MSELoss(), 0.99, 0.003, 0.950, 0.0009)
    optimizer = Adam(MSELoss(), 0.1606, 0.002, 0.9, 0.99, 1e-8, 0.0059) # monk1

    return model, optimizer

def HyperModel_Monk(hp :HyperBag ):
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(), True, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"], TanH(), hp["drop_out"], True, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], Sigmoid(), True, f"_h{i}"))

    model.AddLayer(Layer(1, Sigmoid(), False, "output"))

    optimizer = BackPropagation(MSELoss(), 0.5, 0.015, 0.99, 0.02)
    #optimizer = Adam(MSELoss(), hp["eta"], hp["labda"], hp["alpha"],hp["beta"] ,hp["epsilon"],hp["decay"])
    return model, optimizer


def HyperBag_Monk():
    hp = HyperBag()

    hp.AddRange("eta", 0.001, 0.1, 0.01)
    hp.AddRange("labda", 0.001, 0.1, 0.001)
    hp.AddRange("alpha", 0.8, 0.99, 0.01)
    hp.AddRange("beta", 0.9, 0.99, 0.01)
    hp.AddRange("epsilon", 1e-13,1e-7, 1e-1)
    hp.AddRange("decay", 0.005,0.05, 0.001)

    #hp.AddRange("drop_out", 0.1, 0.5, 0.05)

    hp.AddRange("unit", 1, 15, 1)
    hp.AddRange("hlayer", 0, 3, 1)

    return hp


if __name__ == '__main__':

    alldata = readMonk(file_path_monk1)
    alldata.Shuffle(11)
    alldata.PrintData()
    alldata.Split(0.15, 0.5)



    watched_metric = "val_loss"
    bestSearch = GetBestSearch(HyperBag_Monk(), RandomSearch(100))
    best_model,best_hpSel = bestSearch.GetBestModel(
        HyperModel_Monk,
        alldata,
        150,
        128,
        watched_metric,
        [Accuracy(Binary(0.5))],
        GlorotInitializer(2),
        [EarlyStopping(watched_metric, 80)])


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


