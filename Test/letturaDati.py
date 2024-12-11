from hypothesis import example

from DataUtility.ReadDatasetUtil import readCUP
from DataUtility.PlotUtil import plot_losses_accuracy
from Core.Validation_Alg import *
import numpy as np

from Test.ModelInit import CreateFakeData, CreateFakeData_dataset


#Roba per testare

def f ():
    model = ModelFeedForward()
    model.AddLayer(Layer(1, Linear()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(1, Linear()))
    model.Build(GlorotInitializer())
    model.AddMetrics([MSE(), RMSE(), MEE()])
    return model

file_path = "dataset/CUP/ML-CUP24-TR.csv"
examples = CreateFakeData_dataset(12,1,1)

model = train_k_fold(examples, 3,f)#farlo con 1 non ha senso
metrics = model.MetricResults
print(metrics["val_loss"])
# Filtra le metriche che iniziano con "val_"
val_metrics = {key: value for key, value in metrics.items() if key.startswith("val_")}
print(val_metrics)
# Creazione dinamica della matrice
loss_matrix = np.array(list(val_metrics.values()))  # Converte i valori in una matrice
labels = list(val_metrics.keys())  # Estrae i nomi delle metriche



# Chiamata alla funzione
plot_losses_accuracy(
    loss_matrix=loss_matrix,
    labels=labels,
    title="Metriche di Validazione",
    xlabel="Epoche",
    ylabel="Valore"
)






