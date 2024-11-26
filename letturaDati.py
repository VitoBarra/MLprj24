from hypothesis import example

from DataUtility.FileUtil import readCUP
from DataUtility.DataSet import DataSet


file_path = "dataset/CUP/ML-CUP24-TR.csv"

# Legge il file e crea un'istanza di DataExamples
examples = readCUP(file_path)
#dataset = DataSet(examples.Data,examples.Labels)

# Mostra i dati creati
#print("Dati:\n", examples.Data)
#print("Label:\n", examples.Label)
#print("ID:\n", examples.Id)

k = 5
folds =examples.k_fold_cross_validation(k)
for i, (train_set, test_set) in enumerate(folds):
    print(f"Fold {i + 1}")
    print(f"Training Set: {len(train_set)} esempi")
    print(train_set.Data)
    print(train_set.Label)
    print(f"Test Set: {len(test_set)} esempi")
    print(test_set.Data)
    print(test_set.Label)
