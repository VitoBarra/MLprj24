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


    def optimize(self, layer: Layer, target: np.ndarray):
        """
        Computes the gradients of the weights for the given layer during backpropagation.

        :param layer: The layer for which to compute the gradients.
        :param target: The target values for the given inputs.
        """


        # Calculate delta
        if layer.LastLayer is None: # Input layer
            self.update_weights(layer)
            return
        else:
            self.calculate_delta(layer, target)

        # Calculate gradient
        layerGrad = self.calculate_gradient(layer)


        # Calculte and applay the momentum
        if self.momentum is True:
            layerGrad = self.compute_momentum(layer, layerGrad)


        # Optimize the weights
        self.compute_optimization(layer, layerGrad)


    def compute_momentum(self, layer: Layer, layerGrad: np.ndarray):
        if layer.LastLayer.Gradient is not None:
            if layer.LastLayer.Gradient.shape != layerGrad.shape:
                # Pad gradients to match shape, this is used only in the last mini-batch that usually have fewer data
                pad_size = layer.LastLayer.Gradient.shape[0] - layerGrad.shape[0]
                layerGrad = np.pad(layerGrad, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            mom = self.alpha * layer.LastLayer.Gradient
            layerGrad = layerGrad + mom
        layer.LastLayer.Gradient = layerGrad
        return layerGrad