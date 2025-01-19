from Core.Layer import Layer
from Core.LossFunction import *
from Core.Optimizer.Optimizer import Optimizer


class BackPropagationMomentum(Optimizer):
    """
    Implements BackPropagation with momentum for optimizing neural network weights.

    Attributes:
        loss_function (LossFunction): The loss function used to compute the loss and its derivative.
        batchSize (int): The size of the mini-batch for gradient computation.
        eta (float): The learning rate for optimization.
        lambda_ (float | None): The regularization factor (L2 regularization). Default is None.
        alpha (float | None): The momentum factor to accelerate convergence. Default is None.
        decay_rate (float | None): The learning rate decay factor to reduce eta over time. Default is 0.0.
    """

    def __init__(self, loss_function: LossFunction, batchSize:int, eta: float, lambda_: float | None = None,
                 alpha: float | None = None, decay_rate: float | None = 0.0):
        """
                Initializes the BackPropagationMomentum optimizer with the given parameters.
        """

        super().__init__(loss_function,batchSize, eta, lambda_, alpha, decay_rate)



    def ApplyMomentum (self, layer: Layer, layer_grad: np.ndarray):
        """
        Applies momentum to the gradient for smoother and faster convergence.

        Momentum helps to accelerate convergence by combining the previous gradient
        (stored in `layer.LastLayer.Gradient`) with the current gradient. It simulates
        a velocity component that smoothens the optimization path.

        :param layer: The current layer for which to apply momentum.
        :param layer_grad: The computed gradient for the current layer.
        :return: The velocity-adjusted gradient with momentum applied.
        """

        # Calculate and apply the momentum
        if layer.LastLayer.Gradient is None:
            first_velocity = 0
            velocity = first_velocity * self.alpha + ((1 - self.alpha) * layer_grad)
        else:
            velocity = layer.LastLayer.Gradient * self.alpha + ((1 - self.alpha) * layer_grad)

        return velocity