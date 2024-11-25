from typing import List, Any

import numpy as np
import DataUtility.MiniBatchGenerator as mb
import Layer
from Core import ErrorFunction
from Core.BackPropagation import BackPropagation


class ModelFeedForward:
    """
A feedforward neural network class for building, training, saving, and loading models.

This class provides methods to construct a neural network by adding layers,
train it using backpropagation with a defined optimizer, evaluate performance
using specified metrics, and save or load the model's parameters.

Attributes:
    Optimizer (BackPropagation): The optimizer used for updating weights during training.
    MetricResults (List[List[float]]): A list of computed metric results for each epoch, where each sublist
        contains the results of all defined metrics for that epoch.
    Metric (List[ErrorFunction]): A list of error functions (metrics) to evaluate model performance.
    Layers (List[Layer]): The sequence of layers in the model, from input to output.
    OutputLayer (Layer): The last layer in the network, which computes the final outputs.
    InputLayer (Layer): The first layer in the network, which processes the input data.
"""


    Optimizer: BackPropagation | None
    MetricResults: List[List[float]]
    Metric: List[ErrorFunction]
    Layers: List[Layer]
    OutputLayer: Layer
    InputLayer: Layer

    def __init__(self):
        """
        Initializes an empty feedforward model.
        """
        self.Layers = []
        self.InputLayer = None
        self.OutputLayer = None
        self.Optimizer = None
        self.Metric = []
        self.MetricResults = []

    def Fit(self, Input: Any, epoch: int, batchSize: int) -> None:
        """
        Trains the model using the provided input data.

        :param Input: The training data.
        :param epoch: The number of epochs to train.
        :param batchSize: The size of each mini-batch.
        :return: None
        """
        batch_generator = mb.MiniBatchGenerator(Input, batchSize)
        for e in range(epoch):
            batch_generator.Reset()
            while not batch_generator.IsBatchGenerationFinished:
                batch = batch_generator.NextBatch()
                inputs, targets = batch
                outputs = self._forward(inputs)
                self.Optimizer.BackPropagation(outputs, targets)
                self._update_weights()
            self._compute_metrics(e)

    def AddLayer(self, newLayer: Any) -> None:
        """
        Adds a new layer to the model.

        :param newLayer: The new layer to add.
        :return: None
        """
        if len(self.Layers) != 0:
            self.Layers[-1].NextLayer = newLayer
            newLayer.LastLayer = self.Layers[-1]
        self.Layers.append(newLayer)
        if not self.InputLayer:
            self.InputLayer = newLayer
        self.OutputLayer = newLayer

    def SaveModel(self, path: str) -> None:
        """
        Saves the model's weights to a file.

        :param path: The file path to save the model to.
        :return: None
        """
        model_data = {
            "layers": [layer.get_weights() for layer in self.Layers],
        }
        with open(path, 'wb') as f:
            np.save(f, model_data)

    def LoadModel(self, path: str) -> None:
        """
        Loads the model's weights from a file.

        :param path: The file path to load the model from.
        :return: None
        """
        with open(path, 'rb') as f:
            model_data = np.load(f, allow_pickle=True).item()
        for layer, weights in zip(self.Layers, model_data["layers"]):
            layer.set_weights(weights)

    def SetOptimizer(self, optimizer: Any) -> None:
        """
        Sets the optimizer for training the model.

        :param optimizer: The optimizer instance to use.
        :return: None
        """
        self.Optimizer = optimizer

    def AddMetric(self, metric: Any) -> None:
        """
        Adds a single metric for evaluation during training.

        :param metric: The metric instance to add.
        :return: None
        """
        self.Metric.append(metric)

    def AddMetrics(self, metrics: List[Any]) -> None:
        """
        Adds multiple metrics for evaluation during training.

        :param metrics: A list of metric instances to add.
        :return: None
        """
        self.Metric.extend(metrics)

    def _forward(self, inputs: Any) -> Any:
        """
        Performs a forward pass through the model.

        :param inputs: The input data.
        :return: The final output after the forward pass.
        """
        for layer in self.Layers:
            inputs = layer.Compute(inputs)
        return inputs

    def _update_weights(self) -> None:
        """
        Updates the weights of the layers using the optimizer.

        :return: None
        """
        for layer in reversed(self.Layers):
            layer.Update(self.Optimizer)

    def _compute_metrics(self, epoch: int) -> None:
        """
        Computes the metrics for the current epoch.

        :param epoch: The current epoch number.
        :return: None
        """
        results = [metric.Error(self.OutputLayer.LayerOutput, self.OutputLayer.targets) for metric in self.Metric]
        self.MetricResults.append(results)
