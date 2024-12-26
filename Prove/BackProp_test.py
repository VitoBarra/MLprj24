from sklearn.linear_model import LogisticRegression

from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Metric import *
from Core.WeightInitializer import GlorotInitializer
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
file_path_monk1 = "dataset/monk+s+problems/monks-1.train"
file_path_monk2 = "dataset/monk+s+problems/monks-2.train"
file_path_monk3 = "dataset/monk+s+problems/monks-3.train"



if __name__ == '__main__':

    alldata = readMonk(file_path_monk1)
    alldata.PrintData()
    alldata.Split(0.15, 0.5)

    watched_metric = "val_loss"

    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear(), "Input"))
    model.AddLayer(Layer(1, ReLU(), "H1"))
    model.AddLayer(Layer(1, Linear(), "Output"))
    model.Build(GlorotInitializer())

    model.Fit(BackPropagation_momentum(MSELoss(), 0.15, 0.01, 0.6), alldata.Training , 1, 128, alldata.Validation , [EarlyStopping(watched_metric, 15)] )


    model.SaveMetricsResults("Data/Results/model1.mres")


    model.SaveModel("Data/Models/Test1.vjf")

    lin_model = LogisticRegression()
    lin_model.fit(alldata.Training.Data, alldata.Training.Label)
    print(f"R2 on test: {lin_model.score(alldata.Validation.Data, alldata.Validation.Label)}%")
    predictions = lin_model.predict(alldata.Validation.Data)
    predictions = predictions.reshape(-1, 1)

    m = MSE()
    baseline = m.ComputeMetric(predictions, alldata.Validation.Label)

    metric_to_plot = {key: value[2:] for key, value in model.MetricResults.items() if key.startswith("")}

    plot_metric(
        metricDic=metric_to_plot,
        baseline=baseline,
        limityRange=(0, 1),
        title="Metriche di Validazione",
        xlabel="Epoche",
        ylabel="Valore")




