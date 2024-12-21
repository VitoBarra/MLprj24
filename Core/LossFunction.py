import numpy as np

from Core.Metric import Metric, MSE, RMSE


class LossFunction(Metric):

    def __init__(self):
        Metric.__init__(self)

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pass

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pass




class MSELoss(LossFunction):

    def __init__(self):
        LossFunction.__init__(self)
        self.MSEFun = MSE()


    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.MSEFun.ComputeMetric(prediction, target)

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        der = 2 * (prediction - target)
        return der


class RMSELoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self)

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        rmse=RMSE()
        return rmse.ComputeMetric(prediction, target)

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        der = np.mean(2 * (prediction - target)) / np.sqrt(np.mean((prediction - target) ** 2))
        return der





class BinaryCrossEntropyLoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self)

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.
        prediction: Model predictions (values between 0 and 1).
        target: Ground truth labels (0 or 1).
        """
        epsilon = 1e-12  # To prevent log(0)
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
        return loss

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of binary cross-entropy loss.
        prediction: Model predictions (values between 0 and 1).
        target: Ground truth labels (0 or 1).
        """
        epsilon = 1e-12  # To prevent division by 0
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        deriv = -(target / prediction) + ((1 - target) / (1 - prediction))
        return deriv


class CategoricalCrossEntropyLoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self)

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss.
        prediction: Model predictions (probabilities for each class).
        target: Ground truth one-hot encoded labels.
        """
        epsilon = 1e-12  # To prevent log(0)
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        loss = -np.sum(target * np.log(prediction)) / target.shape[0]
        return loss

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of categorical cross-entropy loss.
        prediction: Model predictions (probabilities for each class).
        target: Ground truth one-hot encoded labels.
        """
        epsilon = 1e-12  # To prevent division by 0
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        deriv = -target / prediction
        return deriv