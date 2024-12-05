import numpy as np
from DataUtility.npUtil import *

class HyperBag:

    hpDic: dict[str,list[float]]

    def __init__(self):
        self.hpDic = {}

    def __getitem__(self, key:str):
        return self.hpDic[key]

    def __setitem__(self, key:str, value:float):
        self.hpDic[key] = value

    def __delitem__(self, key:str):
        del self.hpDic[key]

    def CheckHP(self, hpName:str) -> None:
        if hpName in self.hpDic:
            raise ValueError(f"Hyper parameter '{hpName}' has already bean registered")


    def AddRange(self, hpName:str , lower:float, upper:float, inc:float) -> None:
        if lower >=upper:
            raise ValueError("Lower bound must be smaller than upper bound")
        self.CheckHP(hpName)
        self.hpDic[hpName] = arrangeClosed(lower, upper, inc).tolist()

    def AddChosen(self, hpName:str, chosen:list[float]) -> None:
        if len(chosen) <1:
            raise ValueError("Chosen parameter must have at least length 1")
        self.CheckHP(hpName)
        self.hpDic[hpName] = chosen


    def Keys(self):
        return self.hpDic.keys()

    def Values(self):
        return self.hpDic.values()


