import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class BackPropagation(Optimizer):
    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float = 0.0):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function, eta, lambda_, alpha, decay_rate)


    def ApplyMomentum(self, layer: Layer, layer_grad: np.ndarray):
        if layer.LastLayer.Gradient is not None:
            if layer.LastLayer.Gradient.shape != layer_grad.shape:
                # Pad gradients to match shape, this is used only in the last mini-batch that usually have fewer data
                pad_size = layer.LastLayer.Gradient.shape[0] - layer_grad.shape[0]
                layer_grad = np.pad(layer_grad, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            mom = self.alpha * layer.LastLayer.Gradient
            layer_grad = layer_grad + mom
        layer.LastLayer.Gradient = layer_grad
        return layer_grad