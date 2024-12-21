import numpy as np

from Core.Validation_Alg import *
from DataUtility.PlotUtil import plot_metric
from Test.ModelInit import CreateFakeData


#Roba per testare

def f ():
    model = ModelFeedForward()
    model.AddLayer(Layer(6, Linear()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(1, Linear()))
    model.Build(GlorotInitializer())
    model.AddMetrics([MSE(), RMSE(), MAE()])
    return model

file_path = "dataset/CUP/ML-CUP24-TR.csv"
examples = CreateFakeData(12, 1, 1)

def fDrop():
    model = ModelFeedForward()

    model.AddLayer(Layer(6, Linear()))  # Primo layer
    model.AddLayer(Layer(15, TanH()))  # Secondo layer
    model.AddLayer(DropoutLayer(15, TanH(), 0.5))
    model.AddLayer(Layer(15, TanH()))  # Terzo layer
    model.AddLayer(DropoutLayer(15, TanH(), 0.5))
    model.AddLayer(Layer(1, Linear()))  # Layer di output

    model.Build(GlorotInitializer())

# Chiamata alla funzione
plot_metric(
    metricDic=loss_matrix,
    labels=labels,
    title="Metriche di Validazione",
    xlabel="Epoche",
    ylabel="Valore"
)

    return model

file_path_cup = "../dataset/CUP/ML-CUP24-TR.csv"
file_path_monk = "dataset/monk+s+problems/monks-1.train"

examples = readMonk(file_path_monk)
model2 = train_k_fold(examples, 10,fDrop)#farlo con 1 non ha senso
metrics = model2.MetricResults
#print(metrics["val_loss"])
# Filtra le metriche che iniziano con "val_"
val_metrics = {key: value for key, value in metrics.items() if key.startswith("loss")}
#print(val_metrics)
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







