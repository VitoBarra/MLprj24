import numpy as np

class WeightInitializer:
    def __init__(self):
        pass

    def GenerateWeight(self, Unit: int, Unit1: int) -> np.ndarray:
        raise NotImplementedError(" must be implemented in subclass")




class GlorotInitializer(WeightInitializer):
    def __init__(self):
        pass

    def GenerateWeight(self, input_unit: int, output_unit: int) -> np.ndarray:
        """
        Implements Glorot (Xavier) initialization for weights.

        Parameters:
        Unit (int): Number of units in the current layer.
        Unit1 (int): Number of units in the next layer.

        Returns:
        np.ndarray: Randomly initialized weight matrix of shape (Unit, Unit1).
        """
        # Compute the Glorot scaling factor
        limit = np.sqrt(6 / (input_unit + output_unit))
        # Generate the weights from a uniform distribution within [-limit, limit]
        return np.random.uniform(low=-limit, high=limit, size=(input_unit, output_unit))