import random
from itertools import product

from Core.Tuner.HyperBag import HyperBag


class HyperParameterSearch:

    def __init__(self):
        pass

    def Search(self, hp:HyperBag) -> (dict[str, any], int):
        yield None,0

    def GetName(self):
        return "BaseClass"
    def GetTrialNumber(self):
        return 0

class GridSearch(HyperParameterSearch):


    def __init__(self):
        super().__init__()
        self.trials = 0


    def Search(self, hp:HyperBag) -> (dict[str, any], int):
        keys = hp.Keys()
        values = hp.Values()
        prod = list(product(*values))
        self.trials = len(prod)
        for i,combination in enumerate(prod):
            hpsel=dict(zip(keys, combination))
            yield hpsel, i


    def GetName(self):
        return "GridSearch"
    def GetTrialNumber(self):
        return self.trials



class RandomSearch(HyperParameterSearch):

    trials: int
    def __init__(self, trials: int):
        super().__init__()
        self.trials = trials

    def Search(self, hp) -> (dict[str, any], int):
        keys = hp.Keys()
        values = hp.Values()

        for i in range(self.trials):
            combination = [random.choice(value_list) for value_list in values]
            hpsel = dict(zip(keys, combination))
            yield hpsel , i



    def GetName(self):
        return "RandomSearch"
    def GetTrialNumber(self):
        return self.trials



