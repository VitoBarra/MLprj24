import os


def GetDirectSubDir(path:str) -> list[os.DirEntry]:
    """
    :param path: the path of the directory
    :return: ara array with only subdirectories of the parameter path
    """
    return [f for f in os.scandir(path) if f.is_dir()]
