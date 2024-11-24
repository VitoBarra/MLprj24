import numpy as np


class ErrorFunction:
    def Error(self, val, target):
        raise NotImplementedError("Must override Error method")


class MSE(ErrorFunction):
    def Error(self, val, target):
        return np.mean((val - target) ** 2)


class RMSE(ErrorFunction):
    def Error(self, val, target):
        return np.sqrt(np.mean((val - target) ** 2))


class MEE(ErrorFunction):
    def Error(self, val, target):
        return np.mean(np.abs(val - target))