from Core import LossFunction
from Optimizer import Optimizer

class Adam(Optimizer):

    def __init__(self, loss_function: LossFunction, eta: float | None, lambda_: float | None = None,
                 alpha: float | None = None):
        """
        Initializes the BackPropagation object with a specific loss function.

        :param loss_function: An instance of LossFunction used to compute the loss and its derivative.
        :param eta: The learning rate.
        """
        super().__init__(loss_function, eta, lambda_, alpha)
        if self.alpha is None:
            self.momentum = False
        else:
            self.momentum = True

        if self.lambda_ is None:
            self.regularization = False
        else:
            self.regularization = True