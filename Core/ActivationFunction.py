import numpy as np

class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def Calculate(self, z):
        return self.function(z)

    def CalculateDerivative(self, z):
        return self.derivative(z)



class TanH(ActivationFunction):
    def Calculate(self, z):
        return np.tanh(z)

    def CalculateDerivative(self, z):
        return 1 - np.tanh(z)**2



class ReLU(ActivationFunction):
    def Calculate(self, z):
        return np.maximum(0, z)

    def CalculateDerivative(self, z):
        return np.where(z > 0, 1, 0)


class Sign(ActivationFunction):
    def Calculate(self, z):
        return np.sign(z)

    def CalculateDerivative(self, z):
        return np.zeros_like(z)
