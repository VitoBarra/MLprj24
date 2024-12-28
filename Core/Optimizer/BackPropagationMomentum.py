from Core.Layer import Layer
from Core.LossFunction import *
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationMomentum(Optimizer):

    def __init__(self, loss_function: LossFunction, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float | None = 0.0):
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
        if layer.LastLayer is None:  # Input layer
            self.update_weights(layer)
            return

        else:  # Output and Hidden layer
           self.calculate_delta(layer, target)

        # Calculate Gradient
        layer_grad = self.calculate_gradient(layer)

        if self.momentum is True:
            layer_grad = self.compute_momentum(layer, layer_grad)

        self.compute_optimization(layer, layer_grad)



    def compute_momentum (self, layer: Layer, layer_grad: np.ndarray):
        # Calculate and apply the momentum
        if layer.LastLayer.Velocity is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        else:
            velocity = layer.LastLayer.Velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        layer.LastLayer.Velocity = velocity

        layer_grad = velocity + layer_grad

        return layer_grad

