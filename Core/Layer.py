import numpy as np


from Core import WeightInitializer
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
    LayerInput: np.ndarray | None
    LayerNets: np.ndarray | None
    bias : float | None
    WeightToNextLayer: np.ndarray | None
    Gradient: np.ndarray | None
    NextLayer: 'Layer'
    LastLayer: 'Layer'
    ActivationFunction: ActivationFunction
    Unit: int
    name: str

    def __init__(self, unit: int, activationFunction: ActivationFunction, useBias:bool=False,name:str = "layer" ):
        """
        Initializes the layer with the specified number of units and an activation function.

        :param unit: The number of neurons in the layer.
        :param activationFunction: The activation function used for this layer.
        """
        self.Unit = unit
        self.name=name
        self.ActivationFunction = activationFunction
        self.LayerNets = None
        self.NextLayer = None
        self.LastLayer = None
        self.WeightToNextLayer = None
        self.Gradient = None
        self.LayerOutput = None
        self.LayerInput = None
        self.LayerNets = None
        self.Name = name
        self.bias = 1
        self.UseBias = useBias

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
            self.WeightToNextLayer = weightInitializer.GenerateWeight(self.NextLayer.Unit, self.Unit+(1 if self.UseBias else 0) )
        return True


    def Compute(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the output of the layer based on the given input and activation function.

        :param inputs: The input data to the layer.
        :return: The computed output of the layer.
        :raises ValueError: If the input data is None.
        """
        if inputs is None:
            raise ValueError("input cant be none")

        self.LayerInput = inputs
        if self.LastLayer is not None:
            self.LayerNets=   self.LayerInput @ self.LastLayer.WeightToNextLayer.T
            self.LayerOutput = self.ActivationFunction.Calculate(self.LayerNets)
        else:
            self.LayerOutput = self.LayerInput

        if self.UseBias:
            biasCol = np.full((self.LayerOutput.shape[0], 1), self.bias)
            self.LayerOutput = np.hstack((self.LayerOutput,biasCol))

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


    def get_gradients(self) -> np.ndarray:
        """
        Retrieves the last gradients of the layer.
        :return: The matrix of last gradients of the layer.
        """
        return self.Gradient

    def set_gradients(self, gradients: np.ndarray) -> None:
        """
        Sets the gradients of the layer.
        :param gradients: New gradients of the layer.
        """
        self.Gradient = gradients


class DropoutLayer(Layer):
    """
    Represents a dropout layer in a feedforward neural network.

    The DropoutLayer class extends the functionality of a standard Layer by introducing
    dropout regularization. Dropout helps prevent overfitting by randomly deactivating
    neurons during training.

    Attributes:
        dropout_rate (float): The probability of dropping a neuron during training.
        mask (np.ndarray | None): The binary mask used to deactivate neurons during training.
        training (bool): Flag to indicate whether the layer is in training or inference mode.
    """

    def __init__(self, unit: int, activationFunction: ActivationFunction, dropout_rate: float,
                 name: str = "dropout_layer"):
        """
        Initializes the DropoutLayer with a specified number of units, activation function,
        and dropout rate.

        :param unit: The number of neurons in the layer.
        :param activationFunction: The activation function used for this layer.
        :param dropout_rate: The probability of deactivating a neuron during training (value between 0 and 1).
        :param name: Optional name for the layer (default: "dropout_layer").
        """
        super().__init__(unit, activationFunction, name=name)
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True  # Flag to distinguish between training and inference modes.

    def Build(self, weightInitializer: WeightInitializer) -> bool:
        """
        Initializes the weights for the layer and applies a connection mask during weight initialization.

        :param weightInitializer: The initialization method for the layer's weights.
        :return: True if the weights were successfully initialized, False otherwise.
        """
        success = super().Build(weightInitializer)
        if self.NextLayer is not None:
            # Generate a binary mask for the connections to the next layer
            connection_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.WeightToNextLayer.shape)
            self.WeightToNextLayer *= connection_mask  # Apply the mask to the weight matrix.

        return success

    def Compute(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the output of the layer, applying dropout during training.

        :param inputs: The input data to the layer.
        :return: The computed output of the layer after applying dropout (if training).
        """
        output = super().Compute(inputs)

        if self.training:
            # Apply dropout mask during training
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=output.shape)
            output *= self.mask
        else:
            # Scale output during inference to account for dropout rate
            output *= (1 - self.dropout_rate)

        return output

    def set_training(self, training: bool):
        """
        Sets the training mode for the layer.

        :param training: True to enable training mode, False for inference mode.
        """
        self.training = training
