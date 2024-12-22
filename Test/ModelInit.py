from Core.FeedForwardModel import *
from Core.FeedForwardModel import *
from Core.Metric import *
from Core.WeightInitializer import *
from DataUtility.PlotUtil import *
from DataUtility.ReadDatasetUtil import *

file_path_cup = "../dataset/CUP/ML-CUP24-TR.csv"
file_path_monk = "../dataset/monk+s+problems/monks-1.train"
if __name__ == '__main__':
    #MONK-1
    alldata = readMonk(file_path_monk)
    alldata.Split(0.15, 0.5)
    data , val = alldata.Training , alldata.Validation

    model1 = ModelFeedForward()

    model1.AddLayer(Layer(6, Linear(),"input"))
    model1.AddLayer(Layer(18, ReLU(),"h1"))
    model1.AddLayer(Layer(1, Sign(),"output"))
    model1.Build(GlorotInitializer())

    model1.AddMetrics([RMSE(), MAE()])
    model1.Fit(BackPropagation(MSELoss(),0.002, 0.001, 0.02), data, 120, 450, val)



    for k,v in model1.MetricResults.items() :
          #print(f"The {k} is {v}")
        pass
    model1.SaveMetricsResults("Data/Results/model1.mres")


    #print("\n\n")
    model1.SaveModel("Data/Models/Test1.vjf")

    metrics = model1.MetricResults
    print(f"metrics:{metrics}")
    # print(metrics["val_loss"])
    # Filtra le metriche che iniziano con "val_"
    val_metrics = {key: value for key, value in metrics.items() if key.startswith("loss")}
    print(f"val_metric : {val_metrics.values()}")
    # Creazione dinamica della matrice
    loss_matrix = np.array(list(val_metrics.values()))  # Converte i valori in una matrice
    labels = list(val_metrics.keys())  # Estrae i nomi delle metriche

    # Chiamata alla funzione
    plot_metric(
        metricDic=loss_matrix,
        labels=labels,
        title="Metriche di Validazione",
        xlabel="Epoche",
        ylabel="Valore"
    )



   #CUP
    """
    model_c = ModelFeedForward()

    model_c.AddLayer(Layer(12, Linear(),"input"))
    model_c.AddLayer(Layer(15, TanH(),"h1"))
    model_c.AddLayer(Layer(15, TanH(),"h2"))
    model_c.AddLayer(Layer(15, TanH(),"h3"))
    model_c.AddLayer(Layer(3, Linear(),"output"))
    model_c.Build(GlorotInitializer())

    #CUP
    alldata = readCUP(file_path_cup)
    alldata.SplitDataset(0.15,0.5)
    data , val = alldata.Training , alldata.Validation

    model_c.AddMetrics([MSE(), RMSE(), MEE()])
    model_c.Fit(BackPropagation(MSELoss(),0.2, 0.1), data, 50, 25, val)
    """
