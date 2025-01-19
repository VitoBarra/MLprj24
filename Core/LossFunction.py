import numpy as np

from Core.Metric import Metric, MSE, RMSE


class LossFunction:
    """
    Base class for defining different loss functions.

    Attributes:
        None
    """

    def __init__(self):
        self.Name = "invalid_LossFunction"
        pass

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the loss between prediction and target.

        :param prediction: The predicted values from the model.
        :param target: The true values to compare against.
        :return: The computed loss value.
        """
        pass

    def CalculateDeriv(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the derivative of the loss function with respect to the predictions.

        :param prediction: The predicted values from the model.
        :param target: The true values to compare against.
        :return: The derivative of the loss function with respect to the predictions.
        """
        pass




class MSELoss(LossFunction):
    """
    Mean Squared Error (MSE) Loss Function.

    Attributes:
        MSEFun (MSE): The MSE metric instance used for computing the loss.
    """


    def __init__(self):
        """
        Initializes the MSELoss using the MSE metric.
        """
        LossFunction.__init__(self)
        self.MSEFun = MSE()
        self.Name = "MSELoss"


    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the MSE loss.

        :param prediction: The predicted values from the model.
        :param target: The true values to compare against.
        :return: The computed MSE loss.
        """
        return self.MSEFun.ComputeMetric(prediction, target)

    def CalculateDeriv(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the derivative of the MSE loss.

        :param prediction: The predicted values from the model.
        :param target: The true values to compare against.
        :return: The derivative of the MSE loss.
        """
        der = 2 * (prediction - target)
        return der


class RMSELoss(LossFunction):
    """
    Root Mean Squared Error (RMSE) Loss Function.

    Attributes:
        None
    """
    def __init__(self):
        """
        Initializes the RMSE loss function.
        """
        LossFunction.__init__(self)
        self.Name = "RMSELoss"

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the RMSE loss.

        :param prediction: The predicted values from the model.
        :param target: The true values to compare against.
        :return: The computed RMSE loss.
        """
        rmse=RMSE()
        return rmse.ComputeMetric(prediction, target)

    def CalculateDeriv(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the derivative of the RMSE loss.

        :param prediction: The predicted values from the model.
        :param target: The true values to compare against.
        :return: The derivative of the RMSE loss.
        """
        der = np.mean(2 * (prediction - target)) / np.sqrt(np.mean((prediction - target) ** 2))
        return der





class BinaryCrossEntropyLoss(LossFunction):
    """
    Binary Cross-Entropy Loss Function (used for binary classification).

    Attributes:
        None
    """
    def __init__(self):
        """
        Initializes the Binary Cross-Entropy loss function.
        """
        LossFunction.__init__(self)
        self.Name = "BinaryCrossEntropyLoss"

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        :param prediction: Model predictions (values between 0 and 1).
        :param target: Ground truth labels (0 or 1).
        :return: The computed binary cross-entropy loss.
        """
        epsilon = 1e-12  # To prevent log(0)
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
        return loss

    def CalculateDeriv(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of binary cross-entropy loss with respect to prediction.

        :param prediction: Model predictions (values between 0 and 1).
        :param target: Ground truth labels (0 or 1).
        :return: The derivative of the binary cross-entropy loss.
        """
        epsilon = 1e-12  # To prevent division by 0
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        deriv = -(target / prediction) + ((1 - target) / (1 - prediction))
        return deriv


class CategoricalCrossEntropyLoss(LossFunction):
    """
    Categorical Cross-Entropy Loss Function (used for multi-class classification).

    Attributes:
        None
    """
    def __init__(self):
        """
        Initializes the Categorical Cross-Entropy loss function.
        """
        LossFunction.__init__(self)
        self.Name = "CategoricalCrossEntropyLoss"

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss.

        :param prediction: Model predictions (probabilities for each class).
        :param target: Ground truth one-hot encoded labels.
        :return: The computed categorical cross-entropy loss.
        """
        epsilon = 1e-12  # To prevent log(0)
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.sum(target * np.log(prediction)) / target.shape[0]
        return loss

    def CalculateDeriv(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of categorical cross-entropy loss with respect to prediction.

        :param prediction: Model predictions (probabilities for each class).
        :param target: Ground truth one-hot encoded labels.
        :return: The derivative of the categorical cross-entropy loss.
        """
        epsilon = 1e-12  # To prevent division by 0
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        deriv = -target / prediction
        return deriv