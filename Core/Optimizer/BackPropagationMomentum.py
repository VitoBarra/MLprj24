from Core.Layer import Layer
from Core.LossFunction import *
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationMomentum(Optimizer):

    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float | None = 0.0):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)



    def ApplyMomentum (self, layer: Layer, layer_grad: np.ndarray):
        # Calculate and apply the momentum
        if layer.LastLayer.Gradient is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        else:
            velocity = layer.LastLayer.Gradient * self.alpha + ((1 - self.alpha) * layer_grad)

        return velocity