import numpy as np

from Core import LossFunction, Layer
from Core.Optimizer.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float =0.001, lambda_: float | None = None,
                 alpha: float  | None = 0.9, beta: float | None = 0.99, epsilon: float = 1e-8, decay_rate: float = 0.0):
        """
        Initializes the Adam optimizer with the given parameters.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param batchSize: The size of the mini-batch for optimization.
        :param eta: The learning rate. Default is 0.001.
        :param lambda_: The regularization factor (L2 regularization). Default is None.
        :param alpha: The exponential decay rate for the first moment estimates (momentum). Default is 0.9.
        :param beta: The exponential decay rate for the second moment estimates (RMS). Default is 0.99.
        :param epsilon: A small constant for numerical stability during division. Default is 1e-8.
        :param decay_rate: The learning rate decay factor. Default is 0.0.
        """
        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)

        self.beta = beta
        self.epsilon = epsilon

    def Optimize(self, layer: Layer, target: np.ndarray):
        """
        Performs weight optimization for a given layer during backpropagation.

        :param layer: The layer for which to compute and apply weight updates.
        :param target: The target values for the current training input.
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
        """
        Applies momentum to the gradient.

        :param layer: The current layer.
        :param layer_grad: The gradient for the current layer.
        :return: The velocity-adjusted gradient.
        """
        # Calculate and apply the momentum
        if layer.LastLayer.Gradient is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        else:
            velocity = layer.LastLayer.Gradient * self.alpha + ((1 - self.alpha) * layer_grad)

        return velocity

    def ApplyRMS (self, layer: Layer, layer_grad: np.ndarray):
        """
        Applies RMS (Root Mean Square) scaling to the gradient.

        :param layer: The current layer.
        :param layer_grad: The gradient for the current layer.
        :return: The RMS-scaled gradient.
        """
        if layer.LastLayer.Acceleration is None:
            first_acceleration = 0
            acceleration = self.beta * first_acceleration + ((1 - self.beta) * np.square(layer_grad))
        else:
            acceleration = self.beta * layer.LastLayer.Acceleration + ((1 - self.beta) * np.square(layer_grad))

        return acceleration

    def BiasCorrection(self, layer: Layer):
        """
        Applies bias correction to the momentum and RMS values.

        :param layer: The current layer.
        :return: A tuple (vel_hat, acc_hat) of bias-corrected momentum and RMS.
        """
        vel_hat = layer.LastLayer.Gradient / (1 - self.alpha ** self.iteration+1)
        acc_hat = layer.LastLayer.Acceleration / (1 - self.beta ** self.iteration+1)
        return vel_hat, acc_hat