import numpy as np


class ErrorFunction:
    def Error(self, val: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Must override Error method")


class MSE(ErrorFunction):
    def Error(self, val: np.ndarray, target: np.ndarray) -> float:
        return np.mean((val - target) ** 2)


class RMSE(ErrorFunction):
    def Error(self, val: np.ndarray, target: np.ndarray) -> float:
        return np.sqrt(np.mean((val - target) ** 2))


class MEE(ErrorFunction):
    def Error(self, val: np.ndarray, target: np.ndarray) -> float:
        return np.mean(np.abs(val - target))