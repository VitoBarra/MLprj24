import numpy as np

class WeightInitializer:
    """
    Abstract base class for weight initialization strategies.
    """
    def __init__(self, seed: int | None = None):
        """
        Initializes the weight initializer.

        :param seed: Optional seed for random number generation to ensure reproducibility.
        """
        pass

    def GenerateWeight(self, Unit: int, Unit1: int) -> np.ndarray:
        """
        Abstract method to generate a weight matrix.

        :param Unit: Number of units in the current layer.
        :param Unit1: Number of units in the next layer.
        :return: A weight matrix of shape (Unit, Unit1).
        """
        raise NotImplementedError(" must be implemented in subclass")




class GlorotInitializer(WeightInitializer):
    """
    Implements the Glorot (Xavier) initialization strategy for weights.
    """

    def __init__(self, seed: int | None = None):
        """
        Initializes the GlorotInitializer.

        :param seed: Optional seed for random number generation to ensure reproducibility.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)


    def GenerateWeight(self, input_unit: int, output_unit: int) -> np.ndarray:
        """
        Implements Glorot (Xavier) initialization for weights.

        This method generates a weight matrix with values sampled from a uniform distribution
        within the range [-limit, limit], where `limit` is calculated as:
            limit = sqrt(6 / (input_unit + output_unit))

        :param input_unit: Number of units in the input layer (current layer).
        :param output_unit: Number of units in the output layer (next layer).

        :return: A weight matrix of shape (input_unit, output_unit) initialized using Glorot strategy.
        """

        # Compute the Glorot scaling factor
        limit = np.sqrt(6 / (input_unit + output_unit))
        # Generate the weights from a uniform distribution within [-limit, limit]
        return np.random.uniform(low=-limit, high=limit, size=(input_unit, output_unit))
