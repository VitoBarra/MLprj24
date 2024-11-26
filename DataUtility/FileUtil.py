import os
import numpy as np
from DataUtility.DataExamples import DataExamples
from DataUtility.DataSet import DataSet


def GetDirectSubDir(path:str) -> list[os.DirEntry]:
    """
    :param path: the path of the directory
    :return: ara array with only subdirectories of the parameter path
    """
    return [f for f in os.scandir(path) if f.is_dir()]

def readMonk(file_path:str):
    data:int = []
    labels:int = []
    ids:str = []
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
    ids = np.array(ids)
    # Crea un'istanza di DataExamples
    examples:DataSet = DataSet(data, labels, ids)

    # Puoi assegnarla a un attributo globale o manipolarla ulteriormente
    return examples


def readCUP(file_path:str):
    data:float = []
    labels:float = []
    ids:int = []
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
