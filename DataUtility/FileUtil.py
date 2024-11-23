import os


def getDirectSubDir(path):
    return [f for f in os.scandir(path) if f.is_dir()]
