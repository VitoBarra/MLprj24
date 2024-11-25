import numpy as np


class ErrorFunction:
    """
    Base class for error functions used in machine learning.

    This class provides an interface for calculating the error between predicted
    and target values. Subclasses must override the `Error` method.
    """

    def Error(self, val: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Computes the error between predicted values and target values.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The computed error as a numpy array.
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Must override Error method")


class MSE(ErrorFunction):
    """
    Computes the Mean Squared Error (MSE) between predicted and target values.

    MSE is a commonly used error function that calculates the average of the
    squared differences between predicted and actual values.
    """

    def Error(self, val: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Mean Squared Error.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The Mean Squared Error as a float.
        """
        return np.mean((val - target) ** 2)


class RMSE(ErrorFunction):
    """
    Computes the Root Mean Squared Error (RMSE) between predicted and target values.

    RMSE is the square root of the Mean Squared Error, providing an error measure
    in the same units as the predicted values.
    """

    def Error(self, val: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Root Mean Squared Error.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The Root Mean Squared Error as a float.
        """
        return np.sqrt(np.mean((val - target) ** 2))


class MEE(ErrorFunction):
    """
    Computes the Mean Absolute Error (MAE) between predicted and target values.

    MAE measures the average magnitude of the errors without considering their direction.
    """

    def Error(self, val: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Mean Absolute Error.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The Mean Absolute Error as a float.
        """
        return np.mean(np.abs(val - target))
