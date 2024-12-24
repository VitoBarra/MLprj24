import numpy as np

from Core.ActivationFunction import ActivationFunction


class Metric:
    """
    Base class for error functions used in machine learning.

    This class provides an interface for calculating the error between predicted
    and target values. Subclasses must override the `Error` method.
    """

    Name: str

    def __init__(self):
        pass

    def ComputeMetric(self, val: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Computes the error between predicted values and target values.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The computed error as a numpy array.
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Must override Error method")


class MSE(Metric):
    """
    Computes the Mean Squared Error (MSE) between predicted and target values.

    MSE is a commonly used error function that calculates the average of the
    squared differences between predicted and actual values.
    """
    Name: str

    def __init__(self):
        super().__init__()
        self.Name = "MSE"

    def ComputeMetric(self, val: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Mean Squared Error.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The Mean Squared Error as a float.
        """
        if val.shape != target.shape:
            raise ValueError(f"The size of val and target must be the same but instead where {val.shape} and {target.shape}")
        return np.square(val - target).mean()


class RMSE(Metric):
    """
    Computes the Root Mean Squared Error (RMSE) between predicted and target values.

    RMSE is the square root of the Mean Squared Error, providing an error measure
    in the same units as the predicted values.
    """

    def __init__(self):
        super().__init__()
        self.Name = "RMSE"

    def ComputeMetric(self, val: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Root Mean Squared Error.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The Root Mean Squared Error as a float.
        """
        if val.shape != target.shape:
            raise ValueError(f"The size of val and target must be the same but instead where {val.shape} and {target.shape}")
        return np.sqrt(np.square(val - target).mean())


class MAE(Metric):
    """
    Computes the Mean Absolute Error (MAE) between predicted and target values.

    MAE measures the average magnitude of the errors without considering their direction.
    """

    def __init__(self):
        super().__init__()
        self.Name = "MAE"

    def ComputeMetric(self, val: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Mean Absolute Error.

        :param val: A numpy array of predicted values.
        :param target: A numpy array of target (ground truth) values.
        :return: The Mean Absolute Error as a float.
        """
        if val.shape != target.shape:
            raise ValueError(f"The size of val and target must be the same but instead where {val.shape} and {target.shape}")
        differences = val - target
        norms = np.linalg.norm(differences, axis=1)  # L2 norm for each sample
        mee = np.mean(norms)
        return mee  # Return the mean of the L2 norms


class Accuracy(Metric):
    """
        Computes the Accuracy between predicted and target values.

        Accuracy measures the percentage of correctly classified samples.
        """

    def __init__(self, inter:ActivationFunction = None):
        super().__init__()
        self.Name = "Accuracy"
        self.inter = inter


    def ComputeMetric(self, val: np.ndarray, target: np.ndarray) -> float:
        # If predictions are probabilities, take the class with the highest probability
        if len(val.shape) > 1 and val.shape[1] > 1:
            val = np.argmax(val, axis=1)

        # Compare predictions with targets and compute the mean of correct predictions
        if self.inter is None:
            out = val
        else:
            out = self.inter.Calculate(val)

        f = np.vectorize(lambda x,y: 1 if x == y else 0)

        correct_predictions = f(out,target)
        accuracy = np.mean(correct_predictions)
        return accuracy

