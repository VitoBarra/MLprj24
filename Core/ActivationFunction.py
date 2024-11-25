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

    def Calculate(self, z: float) -> float:
        """
        Computes the activation value for a given input.

        :param z: The input value to the activation function.
        :return: The activation value as a float.
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Must override Calculate method")

    def CalculateDerivative(self, z: float) -> float:
        """
        Computes the derivative of the activation function for a given input.

        :param z: The input value to the activation function.
        :return: The derivative value as a float.
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("Must override CalculateDerivative method")


class TanH(ActivationFunction):
    """
    Hyperbolic tangent activation function (TanH).

    This activation function squashes input values to the range [-1, 1].
    """

    def Calculate(self, z: float) -> float:
        """
        Computes the TanH activation for a given input.

        :param z: The input value.
        :return: The TanH activation value as a float.
        """
        return np.tanh(z)

    def CalculateDerivative(self, z: float) -> float:
        """
        Computes the derivative of the TanH activation function.

        :param z: The input value.
        :return: The derivative of TanH as a float.
        """
        return 1 - np.tanh(z) ** 2


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    This activation function outputs the input value if it is positive; otherwise, it outputs 0.
    """

    def Calculate(self, z: float) -> float:
        """
        Computes the ReLU activation for a given input.

        :param z: The input value.
        :return: The ReLU activation value as a float.
        """
        return np.maximum(0, z)

    def CalculateDerivative(self, z: float) -> float:
        """
        Computes the derivative of the ReLU activation function.

        :param z: The input value.
        :return: The derivative value, which is 1 for positive inputs and 0 otherwise.
        """
        return np.where(z > 0, 1, 0)


class Sign(ActivationFunction):
    """
    Sign activation function.

    This activation function outputs -1 for negative inputs, 0 for zero, and 1 for positive inputs.
    """

    def Calculate(self, z: float) -> float:
        """
        Computes the Sign activation for a given input.

        :param z: The input value.
        :return: The Sign activation value as a float (-1, 0, or 1).
        """
        return np.sign(z)

    def CalculateDerivative(self, z: float) -> float:
        """
        The derivative of the Sign function is not defined mathematically in most cases.

        :param z: The input value.
        :return: NotImplementedError as the derivative does not apply to the Sign function.
        :raises NotImplementedError: To indicate that the derivative is not defined.
        """
        raise NotImplementedError("Derivative does not make sense in this case")
