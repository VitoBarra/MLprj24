import os

import numpy as np

from Core.ActivationFunction import ActivationFunction


def GetDirectSubDir(path:str) -> list[os.DirEntry]:
    """
    :param path: the path of the directory
    :return: ara array with only subdirectories of the parameter path
    """
    return [f for f in os.scandir(path) if f.is_dir()]



def CreateDir(path,exist_ok =True):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=exist_ok)




def GetAllFileInDir(path: str) -> list[os.DirEntry]:
    """
    Returns a list of all files in a specified directory.

    :param path: The path of the directory to scan.
    :return: A list of os.DirEntry objects representing files in the directory.
    :raises Exception: If the path does not exist or is not a directory.
    """
    try:
        # Check if the path exists and is a directory
        if not os.path.isdir(path):
            raise NotADirectoryError(f"The specified path '{path}' is not a valid directory.")

        # Return a list of all files in the directory
        files = [entry for entry in os.scandir(path) if entry.is_file()]
        return files

    except Exception as e:
        print(f"Error accessing the directory '{path}': {e}")
        raise




# Convert NumPy arrays to lists
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, ActivationFunction):
        return obj.Name
    raise TypeError(f"Type {type(obj)} not serializable")

def SaveJson(direc, filename, data):

    if not os.path.exists(direc):
        os.makedirs(direc)

    with open(f"{direc}/{filename}", 'w') as f:
        json.dump(data, f, default=convert_to_serializable)

import json


def readJson(path):
    """
    Legge i dati da un file JSON e restituisce il contenuto come dizionario.

    :param path: Il percorso del file JSON da leggere.
    :return: I dati letti dal file JSON come dizionario.
    :raises Exception: Se il file non può essere letto o se il contenuto non è un JSON valido.
    """
    try:
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error while reading {path}: {e}")
        raise