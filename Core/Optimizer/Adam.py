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
        layer.LastLayer.Velocity = self.ApplyMomentum(layer, layer_grad)

        # Compute RMS value
        layer.LastLayer.Acceleration = self.ApplyRMS(layer, layer_grad)

        # Bias corrections
        vel_hat, acc_hat = self.BiasCorrection(layer)

        # Compute final gradient
        layer_grad = self.eta * vel_hat / (np.sqrt(acc_hat) + self.epsilon)
        # Apply regularization
        layer_update = self.ComputeRegularization(layer, layer_grad)

        # Optimize the weights
        self.updates.append(layer_update)
        self.timestep += 1


    def ApplyMomentum (self, layer: Layer, layer_grad: np.ndarray):
        # Calculate and apply the momentum
        if layer.LastLayer.Velocity is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        else:
            velocity = layer.LastLayer.Velocity * self.alpha + ((1 - self.alpha) * layer_grad)

        return velocity

    def ApplyRMS (self, layer: Layer, layer_grad: np.ndarray):
        if layer.LastLayer.Acceleration is None:
            first_acceleration = 0
            acceleration = self.beta * first_acceleration + ((1 - self.beta) * np.square(layer_grad))
        else:
            acceleration = self.beta * layer.LastLayer.Acceleration + ((1 - self.beta) * np.square(layer_grad))

        return acceleration

    def BiasCorrection(self, layer: Layer):
        vel_hat = layer.LastLayer.Velocity / (1 - self.alpha ** self.timestep)
        acc_hat = layer.LastLayer.Acceleration / (1 - self.beta ** self.timestep)
        return vel_hat, acc_hat