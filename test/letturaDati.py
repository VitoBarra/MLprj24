from DataUtility.FileUtil import readMonk, readCUP


if __name__ == '__main__':
    file_path = "dataset/CUP/ML-CUP24-TR.csv"

    # Legge il file e crea un'istanza di DataExamples
    examples = readCUP(file_path)

    # Mostra i dati creati
    print("Dati:\n", examples.Data)
    print("Label:\n", examples.Label)
    print("ID:\n", examples.Id)