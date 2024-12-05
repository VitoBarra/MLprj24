import random

import numpy as np

from Core import FeedForwardModel
from Core.Tuner import HyperBag
from itertools import product
from Core.FeedForwardModel import  ModelFeedForward

class HyperParameterSearch:
    hp: HyperBag

    def __init__(self, hp: HyperBag):
        self.hp = hp

    def search(self, hyperModel_fn):
        pass

class GridSearch(HyperParameterSearch):

    hp: HyperBag
    def __init__(self, hp: HyperBag):
        super().__init__(hp)

    def search(self, hyperModel_fn):
        keys = self.hp.Keys()
        values = self.hp.Values()

        for combination in product(*values):
            hpsel=dict(zip(keys, combination))
            yield hyperModel_fn(hpsel), hpsel


class RandomSearch(HyperParameterSearch):

    trials: int
    def __init__(self, hp: HyperBag, trials: int):
        super().__init__(hp)
        self.trials = trials

    def search(self, hyperModel_fn) -> FeedForwardModel:
        keys = self.hp.Keys()
        values = self.hp.Values()

        for _ in range(self.trials):
            combination = [random.choice(value_list) for value_list in values]
            yield hyperModel_fn(dict(zip(keys, combination)))

