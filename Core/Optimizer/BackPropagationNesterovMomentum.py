import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationNesterovMomentum(Optimizer):

    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float, lambda_: float | None = None, alpha: float| None = None, decay_rate: float = 0.0):
        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)


    def PreProcesWeights(self, layer: Layer):
        if self.momentum and layer.LastLayer.Gradient is not None:
            weights = layer.WeightToNextLayer + self.alpha * layer.LastLayer.Gradient
        else:
            weights = layer.WeightToNextLayer

        return weights

