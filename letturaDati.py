from hypothesis import example

from DataUtility.ReadDatasetUtil import readCUP
from DataUtility.PlotUtil import plot_losses_accuracy
from Core.Validation_Alg import *
import numpy as np

#Roba per testare

file_path = "dataset/CUP/ML-CUP24-TR.csv"
examples = readCUP(file_path)
"""
folds =examples.k_fold_cross_validation(5)
for i, (train_set, test_set) in enumerate(folds):
    print(f"Fold {i + 1}")
    print(f"Training Set: {len(train_set)} esempi")
    print(train_set.Data)
    print(train_set.Label)
    print(f"Test Set: {len(test_set)} esempi")
    print(test_set.Data)
    print(test_set.Label)

np.random.seed(42)  # Per riproducibilità
epochs = 50
loss1 = np.linspace(1, 0.1, epochs) + np.random.normal(0, 0.05, epochs)  # Perdita decrescente
loss2 = np.linspace(0.8, 0.2, epochs) + np.random.normal(0, 0.03, epochs)  # Decrescita più lenta
loss3 = np.abs(np.sin(np.linspace(0, 3 * np.pi, epochs))) * 0.5 + 0.1  # Oscillante con decrescita

# Creazione della matrice delle perdite
loss_matrix = np.vstack((loss1, loss2, loss3)).T

# Etichette delle perdite
labels = ["Model A Loss", "Model B Loss", "Model C Loss"]

# Richiamo della funzione
plot_losses_accuracy(
    loss_matrix=loss_matrix,
    labels=labels,
    title="Training Loss per Epoch",
    xlabel="Epochs",
    ylabel="Loss Value",
    path="output/loss_plot.png"
)"""
x = np.random.uniform(10, 5, (5,1))
y = np.random.uniform(0, 1, (5,1))
id = np.array(range(x.shape[0]))

# Stampa di y
print("y (etichetta):")
print(y)

print("Labels:")
print(examples.Data.Label)  # Stampa le etichette (Y)

data = DataSet(x,y, id)
model = train_k_fold(examples, 2)#farlo con 1 non ha senso







