import random
from itertools import product

from Core import FeedForwardModel
from Core.Callback.CallBack import CallBack
from Core.Metric import Metric
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from Utility.DataSet import DataSet


class HyperParameterSearch:

    def __init__(self):
        pass

    def search(self, hp:HyperBag) -> (dict[str, any], int):
        yield None,0

    def GetName(self):
        return "BaseClass"
    def TrialNumber(self):
        return 0

class GridSearch(HyperParameterSearch):


    def __init__(self):
        super().__init__()
        self.trials = 0


    def search(self, hp:HyperBag) -> (dict[str, any], int):
        keys = hp.Keys()
        values = hp.Values()
        prod = list(product(*values))
        self.trials = len(prod)
        for i,combination in enumerate(prod):
            hpsel=dict(zip(keys, combination))
            yield hpsel, i


    def GetName(self):
        return "GridSearch"
    def TrialNumber(self):
        return self.trials



class RandomSearch(HyperParameterSearch):

    trials: int
    def __init__(self, trials: int):
        super().__init__()
        self.trials = trials

    def search(self,hp) -> (dict[str, any], int):
        keys = hp.Keys()
        values = hp.Values()

        for i in range(self.trials):
            combination = [random.choice(value_list) for value_list in values]
            hpsel = dict(zip(keys, combination))
            yield hpsel , i



    def GetName(self):
        return "RandomSearch"
    def TrialNumber(self):
        return self.trials



