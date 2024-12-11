import numpy as np

from Core.Metric import Metric, MSE


class LossFunction(object):

    def __init__(self):
        Metric.__init__(self)

    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass




class MSELoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self)


    def CalculateLoss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        mse = MSE()
        return mse.ComputeMetric(prediction, target)