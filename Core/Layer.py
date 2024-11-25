import numpy as np

from Core import WeightInitializer, BackPropagation


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
        NextLayer (Layer | None): A reference to the next layer in the network (if any).
        LastLayer (Layer | None): A reference to the previous layer in the network (if any).
        ActivationFunction (any): The activation function used to compute the output of the layer.
        Unit (int): The number of units (neurons) in the layer.
    """

    LayerOutput: np.ndarray | None
    WeightToNextLayer: np.ndarray | None
    NextLayer: 'Layer' | None
    LastLayer: 'Layer' | None
    ActivationFunction: any
    Unit: int

    def __init__(self, unit: int, activationFunction: any):
        """
        Initializes the layer with the specified number of units and an activation function.

        :param unit: The number of neurons in the layer.
        :param activationFunction: The activation function used for this layer.
        """
        self.Unit = unit
        self.ActivationFunction = activationFunction
        self.NextLayer = None
        self.LastLayer: Layer | None = None
        self.WeightToNextLayer = None
        self.LayerOutput = None

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
            self.WeightToNextLayer = np.random.uniform(-0.02, 0.02, (self.Unit, self.NextLayer.Unit))
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
        """
        z = inputs @ self.WeightToNextLayer
        self.LayerOutput = self.ActivationFunction.Calculate(z)
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
