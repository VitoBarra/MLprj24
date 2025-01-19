import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class BackPropagation(Optimizer):
    """
      Implements the BackPropagation algorithm for optimizing neural network weights.

      Attributes:
          loss_function (LossFunction): The loss function used to compute the loss and its derivative.
          batchSize (int): The size of the mini-batch for gradient computation.
          eta (float): The learning rate for optimization.
          lambda_ (float | None): The regularization factor (L2 regularization). Default is None.
          alpha (float | None): The momentum factor. Default is None.
          decay_rate (float): The learning rate decay factor. Default is 0.0.
      """
    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float = 0.0):
        """
        Initializes the BackPropagation object with a specific loss function.
        """
        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)
