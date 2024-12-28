import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationNesterovMomentum(Optimizer):

    def __init__(self, loss_function: LossFunction, eta: float, alpha: float, lambda_: float | None = None, decay_rate: float = 0.0):
        super().__init__(loss_function, eta, lambda_, alpha, decay_rate)


    def optimize(self, layer: Layer, target: np.ndarray):

        # Calculate delta
        if layer.LastLayer is None:  # Input layer
            self.update_weights(layer)
            return
        else:
            self.calculate_delta(layer, target, True)

        layer_grad = self.calculate_gradient(layer)
        layer.LastLayer.Gradient = layer_grad

        """ 
          if layer.LastLayer.Gradient is None:
               velocity = self.alpha * 0 + self.eta * layer_grad
           else:
               velocity = self.alpha * layer.LastLayer.Gradient + self.eta * layer_grad
           layer.LastLayer.Gradient = velocity
        """

        # Optimize the weights
        self.compute_optimization(layer, layer_grad)

