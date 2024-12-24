import numpy as np

class ActivationFunction:
    """
    Base class for activation functions used in neural networks.

    This class provides an interface for:
      - Calculating the activation of a given input.
      - Calculating the derivative of the activation function with respect to the input.

    Subclasses must implement the `Calculate` and `CalculateDerivative` methods.
    """

    def __init__(self):
        pass

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the activation value for a given input.

        :param z: The input value to the activation function.
        :return: The activation value as a float.
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Must override Calculate method")

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the derivative of the activation function for a given input.

        :param z: The input value to the activation function.
        :return: The derivative value as a float.
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Must override CalculateDerivative method")

    def GetName(self):
        return "NONE"

    @staticmethod
    def GetInstances(functionName):
        function = {
            "TanH": TanH(),
            "ReLU": ReLU(),
            "Sign": Sign(),
            "Linear":Linear(),
        }
        return function.get(functionName, "Invalid option")

class TanH(ActivationFunction):
    """
    Hyperbolic tangent activation function (TanH).

    This activation function squashes input values to the range [-1, 1].
    """

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the TanH activation for a given input.

        :param z: The input value.
        :return: The TanH activation value as a float.
        """
        return np.tanh(z)

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the derivative of the TanH activation function.

        :param z: The input value.
        :return: The derivative of TanH as a float.
        """
        return 1 - np.tanh(z) ** 2

    def GetName(self):
        return "TanH"

class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    This activation function outputs the input value if it is positive; otherwise, it outputs 0.
    """

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the ReLU activation for a given input.

        :param z: The input value.
        :return: The ReLU activation value as a float.
        """
        return np.maximum(0, z)

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the derivative of the ReLU activation function.

        :param z: The input value.
        :return: The derivative value, which is 1 for positive inputs and 0 otherwise.
        """
        return np.where(z > 0, 1, 0)

    def GetName(self):
        return "ReLU"


class Sign(ActivationFunction):
    """
    Sign activation function.

    This activation function outputs -1 for negative inputs, 0 for zero, and 1 for positive inputs.
    """

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the Sign activation for a given input.

        :param z: The input value.
        :return: The Sign activation value as a float (-1, 0, or 1).
        """
        f = np.vectorize(lambda x: 1 if x == 0 else x)
        return np.sign(z)

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        The derivative of the Sign function is not defined mathematically in most cases.

        :param z: The input value.
        :return: NotImplementedError as the derivative does not apply to the Sign function.
        :raises NotImplementedError: To indicate that the derivative is not defined.
        """
        raise NotImplementedError("Derivative does not make sense in this case")

    def GetName(self):
        return "Sign"


class Binary(ActivationFunction):
    """
    Binary activation function.

    This activation function outputs 0 or 1.
    """
    def __init__(self, trashold:float=0.0):
        self.Thrashold   = trashold

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):

        f = np.vectorize(lambda x: 1 if x>self.Thrashold else 0)
        return f(z)

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        The derivative of the Sign function is not defined mathematically in most cases.

        :param z: The input value.
        :return: NotImplementedError as the derivative does not apply to the Sign function.
        :raises NotImplementedError: To indicate that the derivative is not defined.
        """
        raise NotImplementedError("Derivative does not make sense in this case")

    def GetName(self):
        return "Binary"


class Linear(ActivationFunction):
    """
    Linear activation function.

    This activation function directly returns the input value (identity function).
    It is typically used in the output layer for regression tasks or as a building block for other operations.
    """

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the Linear activation (identity function) for a given input.

        :param z: The input value.
        :return: The same value as the input (identity function).
        """
        return np.array(z)

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        Computes the derivative of the Linear activation function.

        The derivative of the identity function is always 1.

        :param z: The input value (not used in the calculation as the derivative is constant).
        :return: The derivative value, which is 1.
        """
        return np.ones_like(z)

    def GetName(self):
        return "Linear"


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.

    This activation function outputs a value between 0 and 1.
    """

    def Calculate(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the Sigmoid activation function.

        :param z: The input value(s).
        :return: The output of the Sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def CalculateDerivative(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the derivative of the Sigmoid activation function.

        :param z: The input value(s).
        :return: The derivative of the Sigmoid function.
        """
        sigmoid = self.Calculate(z)
        return sigmoid * (1 - sigmoid)

    def GetName(self):
        """
        Return the name of the activation function.

        :return: Name of the activation function.
        """
        return "Sigmoid"