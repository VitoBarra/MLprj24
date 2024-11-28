import numpy as np

from Core.Metric import Metric, MSE, RMSE


class LossFunction(object):

    def __init__(self):
        Metric.__init__(self)

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pass

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pass




class MSELoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self)


    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        mse = MSE()
        return mse.ComputeMetric(prediction, target)

    def CalculateDerivLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        der = np.mean(2 * (prediction - target))
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