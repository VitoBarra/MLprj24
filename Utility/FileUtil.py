import os
import numpy as np
import json


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

def SaveJson(direc, filename, data):

    if not os.path.exists(direc):
        os.makedirs(direc)

    with open(f"{direc}/{filename}", 'w') as f:
        json.dump(data, f, default=convert_to_serializable)