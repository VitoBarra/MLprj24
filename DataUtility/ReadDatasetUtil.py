﻿import numpy as np

from DataUtility.DataSet import DataSet


def readMonk(file_path:str):
    data = []
    labels = []
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError("we want 8 elements in this file.")

            # I primi sei valori come array di dati
            data.append([int(x) for x in parts[:6]])
            # Il settimo valore come label
            labels.append(int(parts[6]))
            # L'ottavo valore come ID
            ids.append(parts[7])

    # Converti le liste in array NumPy
    data = np.array(data)
    labels = np.array(labels)
    #substitue all 1 with 0 and all 2 with 1
    f = np.vectorize(lambda x: x-1)
    labels = f(labels)
    labels = labels.reshape(labels.shape[0], 1)
    ids = np.array(ids)
    # for i,(d, l) in enumerate(zip(data,labels)):
    #     labels[i]=d[0]
    #     print(d,l)
    # Crea un'istanza di DataExamples
    examples:DataSet = DataSet(data, labels, ids)


    # Puoi assegnarla a un attributo globale o manipolarla ulteriormente
    return examples


def readCUP(file_path:str):
    data = []
    labels = []
    ids = []
    with open(file_path, 'r') as file:
        for _ in range(7):
            next(file)
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 16:
                raise ValueError("we want 16 elements in this file.")

            # ID (primo valore)
            ids.append(int(parts[0]))

            # Primo input (secondo valore)
            first_input = float(parts[1])

            # Label (successivi tre valori)
            current_labels = list(map(float, parts[2:5]))
            labels.append(current_labels)

            # Altri input (tutti i valori successivi dopo il quinto)
            other_inputs = list(map(float, parts[5:]))
            data.append([first_input] + other_inputs)

    # Converti le liste in array NumPy
    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)
    # Crea un'istanza di DataExamples
    examples:DataSet = DataSet(data, labels, ids)

    # Puoi assegnarla a un attributo globale o manipolarla ulteriormente
    return examples



def FakeDataset():
    pass