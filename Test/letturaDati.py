import numpy as np

from Core.Validation_Alg import *
from DataUtility.PlotUtil import plot_metric
from DataUtility.ReadDatasetUtil import readMonk
from Test.ModelInit import CreateFakeData


#Roba per testare

def f ():
    model = ModelFeedForward()
    model.AddLayer(Layer(6, Linear()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(1, Linear()))
    model.Build(GlorotInitializer())
    return model

def fDrop():
    model = ModelFeedForward()

    model.AddLayer(DropoutLayer(6, Linear(),0, True, "input"))
    model.AddLayer(DropoutLayer(15, ReLU(),0, True, "h"))
    model.AddLayer(DropoutLayer(15, ReLU(), 0, True, "h"))
    model.AddLayer(DropoutLayer(1, Sigmoid(), 0, False, "output"))  # Layer di output

    model.Build(GlorotInitializer())

    return model




file_path_cup = "../dataset/CUP/ML-CUP24-TR.csv"
file_path_monk = "dataset/monk+s+problems/monks-1.train"

examples = readMonk(file_path_monk)
model2 = train_k_fold(examples, 10,fDrop)#farlo con 1 non ha senso
metrics = model2.MetricResults
#print(metrics["val_loss"])
# Filtra le metriche che iniziano con "val_"
val_metrics = {key: value for key, value in metrics.items() if key.startswith("")}
#print(val_metrics)
# Creazione dinamica della matrice
loss_matrix = np.array(list(val_metrics.values()))  # Converte i valori in una matrice
labels = list(val_metrics.keys())  # Estrae i nomi delle metriche



# Chiamata alla funzione
plot_metric(
    metricDic=val_metrics,
    baseline=None,
    title="Metriche di Validazione",
    xlabel="Epoche",
    ylabel="Valore"
)







