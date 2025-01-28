import os

import numpy as np

from Core.ActivationFunction import ActivationFunction


def GetDirectSubDir(path:str) -> list[os.DirEntry]:
    """
   Returns a list of direct subdirectories in the given path.

   :param path: The path to check for subdirectories.
   :return: A list of os.DirEntry objects representing subdirectories.
   """
    return [f for f in os.scandir(path) if f.is_dir()]



def CreateDir(path,exist_ok =True):
    """
    Creates a directory at the specified path.

    :param path: The directory path to create.
    :param exist_ok: If True, does not raise an error if the directory already exists.
    """
    # Ensure the directory exists
    os.makedirs(path, exist_ok=exist_ok)




def GetAllFileInDir(path: str) -> list[os.DirEntry]:
    """
    Returns a list of all files in the specified directory.

    :param path: The path to the directory.
    :return: A list of os.DirEntry objects representing files in the directory.
    :raises NotADirectoryError: If the specified path is not a directory.
    :raises Exception: If an error occurs while accessing the directory.
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
   """
   Converts non-serializable objects (like NumPy arrays) into serializable formats for JSON.

   :param obj: The object to convert.
   :return: A serializable representation of the object.
   :raises TypeError: If the object type is not supported.
   """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, ActivationFunction):
        return obj.Name
    raise TypeError(f"Type {type(obj)} not serializable")

def SaveJson(direc, filename, data):
    """
    Saves data as a JSON file in the specified directory.

    :param direc: The directory where the JSON file will be saved.
    :param filename: The name of the JSON file.
    :param data: The data to save in JSON format.
    """
    if not os.path.exists(direc):
        os.makedirs(direc)

    with open(f"{direc}/{filename}", 'w') as f:
        json.dump(data, f, default=convert_to_serializable)

import json


def readJson(path):
    """
   Reads and returns data from a JSON file.

   :param path: The path to the JSON file.
   :return: The data read from the JSON file.
   :raises Exception: If an error occurs while reading the file.
   """
    try:
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error while reading {path}: {e}")
        raise