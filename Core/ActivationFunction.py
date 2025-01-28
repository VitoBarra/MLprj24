import numpy as np

class ActivationFunction:
    """
    Base class for activation functions used in neural networks.

    This class provides an interface for:
      - Calculating the activation of a given input.
      - Calculating the derivative of the activation function with respect to the input.

    Subclasses must implement the `Calculate` and `CalculateDerivative` methods.
    """
    Name: str

    def __init__(self):
        self.Name = "NONE"

    def __call__(self, z: float|np.array(float)) -> float|np.array(float):
        return self.Calculate(z)
    def __format__(self, format_spec):
        return self.Name


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

    @staticmethod
    def GetInstances(functionName):
        function = {
            "TanH": TanH(),
            "ReLU": ReLU(),
            "Sign": Sign(),
            "Linear":Linear(),
            "Sigmoid":Sigmoid(),
            "SoftARGMax":SoftARGMax(),
            "Binary":Binary(),
            "LeakyReLU": LeakyReLU(),

        }
        instances = function.get(functionName, "Invalid option")
        if instances == "Invalid option":
            raise Exception(functionName + " is not a valid activation function")
        return instances

class TanH(ActivationFunction):
    """
    Hyperbolic tangent activation function (TanH).

    This activation function squashes input values to the range [-1, 1].
    """
    def __init__(self):
        super().__init__()
        self.Name = "TanH"

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

class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    This activation function outputs the input value if it is positive; otherwise, it outputs 0.
    """
    def __init__(self):
        super().__init__()
        self.Name = "ReLU"

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

class LeakyReLU(ActivationFunction):
    """
    Leaky Rectified Linear Unit (leaky ReLU) activation function.

    This activation function allows a small, non-zero, gradient when the unit is not active.
    It is defined as:
      - f(z) = z if z > 0
      - f(z) = α * z if z <= 0
    where α (alpha) is a small positive constant.

    This resolves the "dying ReLU" problem by ensuring non-zero gradients for negative inputs.
    """

    def __init__(self, alpha: float = 0.01):
        """
        Initializes the LeakyReLU activation function with a given alpha value.

        :param alpha: The slope for the negative part of the function. Default is 0.01.
        """
        super().__init__()
        self.Name = "LeakyReLU"
        self.alpha = alpha

    def Calculate(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Computes the LeakyReLU activation for a given input.

        The function returns the input value if it is positive, and alpha times the input otherwise.

        :param z: The input value (float or numpy array).
        :return: The activation value as a float or numpy array.
        """
        return np.where(z > 0, z, self.alpha * z)

    def CalculateDerivative(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Computes the derivative of the LeakyReLU activation function.

        The derivative is:
          - 1 if z > 0
          - alpha if z <= 0

        :param z: The input value (float or numpy array).
        :return: The derivative value as a float or numpy array.
        """
        return np.where(z > 0, 1, self.alpha)



class Sign(ActivationFunction):
    """
    Sign activation function.

    This activation function outputs -1 for negative inputs, 0 for zero, and 1 for positive inputs.
    """
    def __init__(self):
        super().__init__()
        self.Name = "Sign"


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

class Binary(ActivationFunction):
    """
    Binary activation function.

    This activation function outputs 0 or 1.
    """
    def __init__(self, threshold: float = 0.0):

        super().__init__()
        self.Name = "Binary"
        self.Threshold   = threshold

    def Calculate(self, z: float|np.array(float)) -> float|np.array(float):

        f = np.vectorize(lambda x: 1 if x>self.Threshold else 0)
        return f(z)

    def CalculateDerivative(self, z: float|np.array(float)) -> float|np.array(float):
        """
        The derivative of the Sign function is not defined mathematically in most cases.

        :param z: The input value.
        :return: NotImplementedError as the derivative does not apply to the Sign function.
        :raises NotImplementedError: To indicate that the derivative is not defined.
        """
        raise NotImplementedError("Derivative does not make sense in this case")


class Linear(ActivationFunction):
    """
    Linear activation function.

    This activation function directly returns the input value (identity function).
    It is typically used in the output layer for regression tasks or as a building block for other operations.
    """

    def __init__(self):
        super().__init__()
        self.Name = "Linear"


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




class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.

    This activation function outputs a value between 0 and 1.
    """
    def __init__(self):
        super().__init__()
        self.Name = "Sigmoid"

    def Calculate(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the Sigmoid activation function.

        :param z: The input value(s).
        :return: The output of the Sigmoid function.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def CalculateDerivative(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the derivative of the Sigmoid activation function.

        :param z: The input value(s).
        :return: The derivative of the Sigmoid function.
        """
        sigmoid = self.Calculate(z)
        return sigmoid * (1 - sigmoid)




class SoftARGMax(ActivationFunction):
    """
    SoftMax activation function.

    This activation function outputs a probability distribution over a set of inputs.
    """
    def __init__(self):
        super().__init__()
        self.Name = "SoftARGMax"

    def Calculate(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the SoftMax activation function.

        :param z: The input array of values.
        :return: The output array of the SoftMax function, where each element is a probability.
        """
        exp_values = np.exp(z - np.max(z, axis=-1, keepdims=True))  # Shift for numerical stability
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


    def CalculateDerivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the SoftMax activation function.

        :param z: The input array of values.
        :return: The Jacobian matrix of the SoftMax function for each input sample.
        """

        # Compute the SoftMax output
        softmax_output = self.Calculate(z)  # 1D array of probabilities

        # Compute the outer product of the SoftMax output with itself
        softmax_outer = np.outer(softmax_output, softmax_output)

        # Create the Jacobian matrix (diagonal elements)
        jacobian = np.diag(softmax_output) - softmax_outer

        #TODO: This do not work, ih has to be reduced to a 1D array some how
        return jacobian


