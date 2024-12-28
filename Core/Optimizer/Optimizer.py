from Core import FeedForwardModel
from Core.Layer import Layer
from Core.LossFunction import *


class Optimizer:
    deltas: list[float]
    updates: list[np.ndarray]
    momentum: bool
    regularization: bool

    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float = 0.0):
        """
        Base class for optimization algorithms.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        self.loss_function = loss_function
        self.eta = eta
        self.initial_eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.iteration = 0

        self.deltas = []
        self.updates = []

    def start_optimize(self, model: FeedForwardModel, target: np.ndarray):
        self.iteration += 1
        if self.decay_rate is not None:
            self.eta = self.initial_eta * np.exp(-self.decay_rate * self.iteration)

        for layer in reversed(model.Layers):
            self.optimize(layer, target)


    def optimize(self, layer: Layer, target: np.ndarray):
        """
        Abstract method to compute gradients for a given layer.
        """
        raise NotImplementedError("Must override Calculate method")

    def update_weights(self, layer: Layer):
        """
        Function to update the layer's weights.

        :param layer: The layer to update.
        """
        index = 1
        max_norm = 1.0  # Define the maximum norm for gradient clipping
        while layer.NextLayer is not None:
            # Clip the gradients before updating weights
            gradients = self.updates[-index]
            norm = np.linalg.norm(gradients)
            if norm > max_norm:
                gradients = gradients * (max_norm / norm)

            # Update the weights in the layer
            layer.WeightToNextLayer = layer.WeightToNextLayer - gradients
            layer = layer.NextLayer
            index += 1
        self.updates = []

