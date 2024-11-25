import os
import numpy as np
from DataUtility import DataExamples


def GetDirectSubDir(path:str) -> list[os.DirEntry]:
    """
    :param path: the path of the directory
    :return: ara array with only subdirectories of the parameter path
    """
    return [f for f in os.scandir(path) if f.is_dir()]

def readMonk(file_path):
    data = []
    labels = []
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError("Ogni riga deve contenere esattamente 8 elementi.")

            # I primi sei valori come array di dati
            data.append([int(x) for x in parts[:6]])
            # Il settimo valore come label
            labels.append(int(parts[6]))
            # L'ottavo valore come ID
            ids.append(parts[7])

    # Converti le liste in array NumPy
    #data = np.array(data)
    #labels = np.array(labels)
    #ids = np.array(ids)


    # Crea un'istanza di DataExamples
    examples = DataExamples(data, labels, id=ids)

    # Puoi assegnarla a un attributo globale o manipolarla ulteriormente
    return examples
