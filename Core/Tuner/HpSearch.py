import random

import numpy as np
from Core.Tuner import HyperBag
from itertools import product

class HyperParameterSearch:
    hp: HyperBag

    def __init__(self, hp: HyperBag):
        self.hp = hp

    def search(self):
        pass

class GridSearch(HyperParameterSearch):

    hp: HyperBag
    def __init__(self, hp: HyperBag):
        super().__init__(hp)

    def search(self):
        keys = self.hp.Keys()
        values = self.hp.Values()

        for combination in product(*values):
            yield dict(zip(keys, combination))


class RandomSearch(HyperParameterSearch):

    trials: int
    def __init__(self, hp: HyperBag, trials: int):
        super().__init__(hp)
        self.trials = trials

    def search(self):
        keys = self.hp.Keys()
        values = self.hp.Values()

        for _ in range(self.trials):
            combination = [random.choice(value_list) for value_list in values]
            yield dict(zip(keys, combination))

