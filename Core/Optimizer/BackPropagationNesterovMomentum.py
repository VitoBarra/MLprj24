import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationNesterovMomentum(Optimizer):

    def __init__(self, loss_function: LossFunction, eta: float, alpha: float, lambda_: float | None = None, decay_rate: float = 0.0):
        super().__init__(loss_function, eta, lambda_, alpha, decay_rate)


    def PreProcessigWeights(self, layer: Layer):
        if self.momentum and layer.LastLayer.Gradient is not None:
            weights = layer.WeightToNextLayer + self.alpha * layer.Gradient
        else:
            weights = layer.WeightToNextLayer

        return weights

