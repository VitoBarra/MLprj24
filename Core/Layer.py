import numpy as np

from Core import WeightInitializer, BackPropagation
from Core.ActivationFunction import ActivationFunction


class Layer:
    """
    Represents a single layer in a feedforward neural network.

    The Layer class provides functionality for:
        - Initializing weights for connections to the next layer.
        - Computing the layer's output using an activation function.
        - Updating weights during backpropagation with a given optimizer.
        - Managing connections to previous and subsequent layers.

    Attributes:
        LayerOutput (np.ndarray | None): The computed output of the layer after applying the activation function.
        WeightToNextLayer (np.ndarray | None): The weight matrix connecting this layer to the next layer.
        NextLayer (Layer): A reference to the next layer in the network (if any).
        LastLayer (Layer): A reference to the previous layer in the network (if any).
        ActivationFunction (ActivationFunction): The activation function used to compute the output of the layer.
        Unit (int): The number of units (neurons) in the layer.
    """

    LayerOutput: np.ndarray | None
    LayerInputput: np.ndarray | None
    WeightToNextLayer: np.ndarray | None
    NextLayer: 'Layer'
    LastLayer: 'Layer'
    ActivationFunction: ActivationFunction
    Unit: int

    def __init__(self, unit: int, activationFunction: ActivationFunction):
        """
        Initializes the layer with the specified number of units and an activation function.

        :param unit: The number of neurons in the layer.
        :param activationFunction: The activation function used for this layer.
        """
        self.Unit = unit
        self.ActivationFunction = activationFunction
        self.NextLayer = None
        self.LastLayer = None
        self.WeightToNextLayer = None
        self.LayerOutput = None
        self.LayerInputput = None

    def Build(self, weightInitializer: WeightInitializer) -> bool:
        """
        Initializes the weights for this layer using the specified weight initializer.

        :param weightInitializer: The initialization method for the layer's weights.
        :return:
            - True: If the weight initialization occurred successfully.
            - False: If the weights were already initialized.
        """
        if self.WeightToNextLayer is not None:
            return False

        if self.NextLayer is not None:
            self.WeightToNextLayer = weightInitializer.GenerateWeight(self.Unit, self.NextLayer.Unit)
        else:
            self.WeightToNextLayer = np.random.uniform(-0.02, 0.02, (self.Unit, self.Unit))
        return True

    def Update(self, optimizer: BackPropagation) -> None:
        """
        Updates the layer's weights using the provided optimizer during backpropagation.

        :param optimizer: The optimizer used to compute weight updates.
        """
        gradient = optimizer.ComputeUpdate(self)
        self.WeightToNextLayer -= optimizer.Eta * gradient

    def Compute(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the output of the layer based on the given input and activation function.

        :param inputs: The input data to the layer.
        :return: The computed output of the layer.
        :raises ValueError: If the input data is None.
        """
        if inputs is None:
            raise ValueError("input cant be none")

        self.LayerInputput = inputs
        if self.LastLayer is not None:
            nets = self.LayerInputput @ self.LastLayer.WeightToNextLayer
            self.LayerOutput = np.array([self.ActivationFunction.Calculate(feature) for feature in [data for data in nets]])
        else:
            self.LayerOutput = self.LayerInputput

        return self.LayerOutput

    def get_weights(self) -> np.ndarray:
        """
        Retrieves the current weights of the layer.

        :return: The weight matrix of the layer.
        """
        return self.WeightToNextLayer

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Sets the weights of the layer.

        :param weights: The weight matrix to set for this layer.
        """
        self.WeightToNextLayer = weights
