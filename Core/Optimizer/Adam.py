import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float =0.001, lambda_: float | None = None,
                 alpha: float  | None = 0.9, beta: float | None = 0.99, epsilon: float = 1e-8, decay_rate: float = 0.0):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)

        self.beta = beta
        self.epsilon = epsilon

    def Optimize(self, layer: Layer, target: np.ndarray):
        """
        Computes the gradients of the weights for the given layer during backpropagation.

        :param layer: The layer for which to compute the gradients.
        :param target: The target values for the given inputs.
        """


        # Calculate delta
        if layer.LastLayer is None: # Input layer
            self.UpdateWeights(layer)
            return
        else:
            self.deltas = self.CalculateDelta(layer, target)

        layer_grad = self.CalculateGradient(layer)

        # Calculate and apply the momentum
        layer.LastLayer.Gradient = self.ApplyMomentum(layer, layer_grad)

        # Compute RMS value
        layer.LastLayer.Acceleration = self.ApplyRMS(layer, layer_grad)

        # Bias corrections
        vel_hat, acc_hat = self.BiasCorrection(layer)

        # Compute final gradient
        layer_grad = self.eta * vel_hat / (np.sqrt(acc_hat) + self.epsilon)
        # Apply regularization
        layer_update = self.ApplyRegularization(layer, layer_grad)

        # Optimize the weights
        self.updates.append(layer_update)


    def ApplyMomentum (self, layer: Layer, layer_grad: np.ndarray):
        # Calculate and apply the momentum
        if layer.LastLayer.Gradient is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        else:
            velocity = layer.LastLayer.Gradient * self.alpha + ((1 - self.alpha) * layer_grad)

        return velocity

    def ApplyRMS (self, layer: Layer, layer_grad: np.ndarray):
        if layer.LastLayer.Acceleration is None:
            first_acceleration = 0
            acceleration = self.beta * first_acceleration + ((1 - self.beta) * np.square(layer_grad))
        else:
            acceleration = self.beta * layer.LastLayer.Acceleration + ((1 - self.beta) * np.square(layer_grad))

        return acceleration

    def BiasCorrection(self, layer: Layer):
        vel_hat = layer.LastLayer.Gradient / (1 - self.alpha ** self.iteration+1)
        acc_hat = layer.LastLayer.Acceleration / (1 - self.beta ** self.iteration+1)
        return vel_hat, acc_hat