from Utility.npUtil import *

class HyperBag:

    hpDic: dict[str,list[float]]

    def __init__(self):
        self.hpDic = {}

    def __getitem__(self, key:str):
        if self.hpDic.keys().__contains__(key):
            return self.hpDic[key]
        else:
            return None

    def __setitem__(self, key:str, value:list[float]):
        self.hpDic[key] = value

    def __delitem__(self, key:str):
        del self.hpDic[key]

    def __str__(self):
        return self.GetHPString()

    def CheckHP(self, hpName:str) -> None:
        if hpName in self.hpDic:
            raise ValueError(f"Hyper parameter '{hpName}' has already bean registered")


    def AddRange(self, hpName:str , lower:float, upper:float, inc:float) -> None:
        if lower > upper:
            raise ValueError("Lower bound must be smaller than upper bound")
        self.CheckHP(hpName)
        self.hpDic[hpName] = arrangeClosed(lower, upper, inc).tolist()


    def AddChosen(self, hpName:str, chosen:list[any]) -> None:
        if len(chosen) <1:
            raise ValueError("Chosen parameter must have at least length 1")
        if len(chosen) != len(set(chosen)):
            raise ValueError("Chosen parameter cannot contain duplicates")

        self.CheckHP(hpName)
        self.hpDic[hpName] = chosen

    def Keys(self):
        return self.hpDic.keys()

    def Values(self):
        return self.hpDic.values()

    def Set(self, hpDic:dict[str,list[float]]):
        if hpDic is None:
            raise ValueError("hpDic cannot be None")
        self.hpDic = hpDic

    def GetHPString(self):
        return "| "+" | ".join(
            f"{key}:None" if value is None else
            f"{key}:{'True ' if value else 'False'}" if isinstance(value, bool) else
            f"{key}:{value:4}" if isinstance(value, int) else
            f"{key}:{value:.4f}" if isinstance(value, float) else
            f"{key}:{value}"
            for key, value in self.hpDic.items()
        )
