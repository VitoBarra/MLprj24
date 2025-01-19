import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationNesterovMomentum(Optimizer):
    """
    Implements BackPropagation with Nesterov Accelerated Gradient (NAG) momentum.

    Nesterov momentum is a variation of standard momentum that computes the gradient
    at a future position, giving a "look-ahead" effect. This can lead to faster convergence
    and more stable optimization.

    Attributes:
        loss_function (LossFunction): The loss function used to compute the loss and its derivative.
        batchSize (int): The size of the mini-batch for gradient computation.
        eta (float): The learning rate for optimization.
        lambda_ (float | None): The regularization factor (L2 regularization). Default is None.
        alpha (float | None): The momentum factor for NAG. Default is None.
        decay_rate (float): The learning rate decay factor to reduce eta over time. Default is 0.0.
    """

    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float, lambda_: float | None = None, alpha: float| None = None, decay_rate: float = 0.0):
        """
        Initializes the BackPropagationNesterovMomentum optimizer with the given parameters.
        """
        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)


    def PreProcessWeights(self, layer: Layer):
        """
        Precomputes the adjusted weights for Nesterov momentum.

        This method calculates a "look-ahead" weight adjustment by applying momentum to
        the previous gradient. The adjustment allows the gradient to be computed at the
        anticipated future position, leading to accelerated convergence.

        :param layer: The current layer for which to preprocess weights.
        :return: The adjusted weights with Nesterov momentum applied.
        """
        if self.momentum and layer.LastLayer.Gradient is not None:
            weights = layer.WeightToNextLayer + self.alpha * layer.LastLayer.Gradient
        else:
            weights = layer.WeightToNextLayer

        return weights

