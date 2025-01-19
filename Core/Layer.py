import numpy as np

from Core.Initializer import WeightInitializer
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
    WeightToNextLayer: np.ndarray | None
    Gradient: np.ndarray | None # Also interpreted as "velocity" of the function
    Acceleration: np.ndarray | None
    NextLayer: 'Layer'
    LastLayer: 'Layer'
    ActivationFunction: ActivationFunction
    Unit: int
    name: str

    def __init__(self, unit: int, activationFunction: ActivationFunction, useBias:bool=False,name:str = "layer", train:bool  = True):
        """
        Initializes the layer with the specified number of units and an activation function.

        :param unit: The number of neurons in the layer.
        :param activationFunction: The activation function used for this layer.
        :param useBias: Boolean indicating whether or not to use a bias term in the layer.
        :param name: The name of the layer.
        :param train: Boolean indicating whether the layer is in training mode or not.
        """
        self.Unit = unit
        self.name=name
        self.ActivationFunction = activationFunction
        self.LayerNets = None
        self.NextLayer = None
        self.LastLayer = None
        self.WeightToNextLayer = None
        self.Gradient = None
        self.Velocity = None
        self.Acceleration = None
        self.LayerOutput = None
        self.LayerInput = None
        self.LayerNets = None
        self.Name = name
        self.UseBias = useBias
        self.TrainMode = train

    def Build(self, weightInitializer: WeightInitializer) -> bool:
        """
        Initializes the weights for this layer using the specified weight initializer.

        :param weightInitializer: The initialization method for the layer's weights.
        :return:
            - True if the weight initialization occurred successfully.
            - False if the weights were already initialized.
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
            biasCol = np.full((self.LayerOutput.shape[0], 1), 1)
            self.LayerOutput = np.hstack((self.LayerOutput,biasCol))

        return self.LayerOutput

    def InferenceMode(self):
        """Sets the layer to inference mode (optional, can be customized)."""
        pass
    def TrainingMode(self):
        """Sets the layer to training mode (optional, can be customized)."""
        pass

    def SerializeLayer(self):
        """
        Serializes the layer to a dictionary format for saving.

        :return: Dictionary representing the layer.
        """
        return {
            "name":self.name,
            "unit":self.Unit,
            "activation":self.ActivationFunction.Name,
            "useBias":self.UseBias,
            "train":self.TrainMode,
            "weight": self.WeightToNextLayer
        }
    @classmethod
    def DeserializeLayer(self, layerDic:dict):
        """
        Deserializes a dictionary to create a Layer object.

        :param layerDic: Dictionary containing layer parameters.
        :return: A Layer instance.
        """
        layer  = Layer(
            layerDic["unit"],
            ActivationFunction.GetInstances(layerDic["activation"]),
            layerDic["useBias"],
            layerDic["name"],
            layerDic["train"])

        layer.WeightToNextLayer = np.array(layerDic["weight"])
        return layer


class DropoutLayer(Layer):
    """
    Represents a dropout layer in a feedforward neural network.

    The DropoutLayer class extends the functionality of a standard Layer by introducing
    dropout regularization. Dropout helps prevent overfitting by randomly deactivating
    neurons during training.

    Attributes:
        dropout_rate (float): The probability of dropping a neuron during training.
        mask (np.ndarray | None): The binary mask used to deactivate neurons during training.
        inferenceMode (bool): Flag to indicate whether the layer is in training or inference mode.
    """

    def __init__(self, unit: int, activationFunction: ActivationFunction, dropout_rate: float,
                useBias:bool=False,name:str = "layer", train:bool  = True):
        """
        Initializes the DropoutLayer with a specified number of units, activation function,
        and dropout rate.

        :param unit: The number of neurons in the layer.
        :param activationFunction: The activation function used for this layer.
        :param dropout_rate: The probability of deactivating a neuron during training (value between 0 and 1).
        :param name: Optional name for the layer (default: "dropout_layer").
        :param train: Boolean indicating whether the layer is in training mode.
        """
        super().__init__(unit, activationFunction,  useBias,name, train)
        self.inferenceMode = False
        self.dropout_rate = dropout_rate
        self.mask = None

    def Build(self, weightInitializer: WeightInitializer) -> bool:
        """
        Initializes the weights for the layer and applies a connection mask during weight initialization.

        :param weightInitializer: The initialization method for the layer's weights.
        :return: True if the weights were successfully initialized, False otherwise.
        """
        success = super().Build(weightInitializer)
        return success

    def Compute(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the output of the layer based on the given inputs. During training, applies dropout.

        :param inputs: The input data to the layer.
        :return: The computed output of the layer with dropout applied if in training mode.
        """

        if self.inferenceMode:
            self.WeightToNextLayer *= (1-self.dropout_rate)

        super().Compute(inputs)

        if not self.inferenceMode:  # Apply dropout only during training
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.LayerOutput.shape)
            self.LayerOutput *= self.mask


        return self.LayerOutput


    def InferenceMode(self):
        """
        Switches the layer to inference mode. This means no dropout will be applied.
        """
        self.inferenceMode = True
        self.mask = np.ones_like(self.LayerOutput)  # Set mask to all ones no neurons are dropped

    def TrainingMode(self):
        """
        Switches the layer to training mode. This means dropout will be applied.
        """
        self.inferenceMode = False  # In training


    def SerializeLayer(self):
        """
        Serializes the DropoutLayer to a dictionary format.

        :return: Dictionary representing the layer's configuration.
        """
        return {
            "name":self.name,
            "unit":self.Unit,
            "activation":self.ActivationFunction.Name,
            "dropout_rate":self.dropout_rate,
            "useBias":self.UseBias
        }

    @classmethod
    def DeserializeLayer(self, layerDic:dict):
        """
        Deserializes a DropoutLayer from a dictionary.

        :param layerDic: Dictionary containing the layer's parameters.
        :return: A DropoutLayer instance.
        """
        layer  = DropoutLayer(
            layerDic["unit"],
            ActivationFunction.GetInstances(layerDic["activation"]),
            layerDic["dropout_rate"],
            layerDic["useBias"],
            layerDic["name"],
            layerDic["train"])
        return layer