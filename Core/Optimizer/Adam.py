import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, beta: float | None = None, epsilon: float = 1e-8, decay_rate: float = 0.0):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function, eta, lambda_, alpha, decay_rate)

        self.beta = beta
        self.epsilon = epsilon
        self.timestep = 1

        if self.alpha is None:
            self.momentum = False
        else:
            self.momentum = True

        if self.lambda_ is None:
            self.regularization = False
        else:
            self.regularization = True

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
            self.deltas = self.CalculateDelta(layer, target)

        layer_grad = self.CalculateGradient(layer)

        # Calculate and apply the momentum
        if layer.LastLayer.Velocity is None:
            first_velocity = 0
            velocity = self.alpha * first_velocity + ((1 - self.alpha) * layer_grad)
        else:
            velocity = self.alpha * layer.LastLayer.Velocity + ((1 - self.alpha) * layer_grad)

        layer.LastLayer.Velocity = velocity

        # Compute RMS value
        if layer.LastLayer.Acceleration is None:
            first_acceleration = 0
            acceleration = self.beta * first_acceleration + ((1 - self.beta) * np.square(layer_grad))
        else:
            acceleration = self.beta * layer.LastLayer.Acceleration + ((1 - self.beta) * np.square(layer_grad))

        layer.LastLayer.Acceleration = acceleration

        # Bias corrections
        vel_hat  = layer.LastLayer.Velocity / (1 - self.alpha ** self.timestep)
        acc_hat = layer.LastLayer.Acceleration / (1 - self.beta ** self.timestep)

        layer_grad = self.eta * vel_hat / (np.sqrt(acc_hat) + self.epsilon)


        # Optimize the weights
        self.ComputeLayerUpdates(layer, layer_grad)
        self.timestep += 1

