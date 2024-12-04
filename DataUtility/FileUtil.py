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



def CreateDir(path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Convert NumPy arrays to lists
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")