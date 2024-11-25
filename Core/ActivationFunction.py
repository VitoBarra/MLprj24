import numpy as np

class ActivationFunction:
    def __init__(self):
        pass


    def Calculate(self, z:float)-> float:
        raise NotImplementedError("Must override Calculate method")

    def CalculateDerivative(self, z:float)-> float:
        raise NotImplementedError("Must override CalculateDerivative method")



class TanH(ActivationFunction):
    def Calculate(self, z:float)-> float:
        return np.tanh(z)

    def CalculateDerivative(self, z:float)-> float:
        return 1 - np.tanh(z)**2



class ReLU(ActivationFunction):
    def Calculate(self, z:float)-> float:
        return np.maximum(0, z)

    def CalculateDerivative(self, z):
        return np.where(z > 0, 1, 0)


class Sign(ActivationFunction):
    def Calculate(self, z:float)-> float:
        return np.sign(z)

    def CalculateDerivative(self, z:float)-> float:
        raise NotImplementedError("Derivative does not make sense in this case")

