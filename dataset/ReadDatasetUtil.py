import numpy as np

from Utility.DataExamples import DataExamples
from Utility.DataSet import DataSet
from Utility.DataUtil import *


def readMonk(file_path:str) -> DataSet:
    data = []
    labels = []
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError("we want 8 elements in this file.")

            # I primi sei valori come array di dati
            data.append([int(x) for x in parts[1:7]])
            # Il settimo valore come label
            labels.append(int(parts[0]))
            # L'ottavo valore come ID
            ids.append(parts[7])


    # Converts list to array NumPy
    data = np.array(data)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], 1)
    ids = np.array(ids)

    return DataSet.FromData(data, labels, ids)




def readCUP(file_path:str) -> DataSet:
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

            # ID (first value)
            ids.append([int(parts[0])])

            # data (middle value)
            data.append(list(map(float, parts[1:13])))

            # Label (last 3 value)
            labels.append(list(map(float, parts[13:])))

    # Converti le liste in array NumPy
    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)
    return DataSet.FromData(data, labels, ids)




def CreateFakeData(nData:int, xdim :int=1, ydim:int=1) -> DataSet:
    x = np.random.uniform(0, 1, (nData,xdim))
    y = np.random.choice([0, 1], (nData, ydim))
    id = np.array(range(x.shape[0]))

    return DataSet.FromData(x,y, id)