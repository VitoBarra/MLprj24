import json
from typing import List, Any

import numpy as np

from Core.ActivationFunction import ActivationFunction
from Core.LossFunction import LossFunction

import DataUtility.MiniBatchGenerator as mb
from Core.Layer import Layer
from Core import Metric
from Core.BackPropagation import BackPropagation
from Core.WeightInitializer import WeightInitializer
from DataUtility.DataExamples import DataExamples
from DataUtility.FileUtil import CreateDir, convert_to_serializable


class ModelFeedForward:
    """
A feedforward neural network class for building, training, saving, and loading models.

This class provides methods to construct a neural network by adding layers,
train it using backpropagation with a defined optimizer, evaluate performance
using specified metrics, and save or load the model's parameters.

Attributes:
    Optimizer (BackPropagation): The optimizer used for updating weights during training.
    MetricResults (dict[str, list[float]]): A list of computed metric results for each epoch, where each sublist
        contains the results of all defined metrics for that epoch.
    Metrics (List[Metric]): List of metrics value to evaluate model performance.
    Loss: (LossFunction): The loss function used for training.
    Layers (List[Layer]): The sequence of layers in the model, from input to output.
    OutputLayer (Layer): The last layer in the network, which computes the final outputs.
    InputLayer (Layer): The first layer in the network, which processes the input data.
"""

    MetricResults: dict[str, np.ndarray[float]]
    Metrics: List[Metric]
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
        self.Metrics = []
        self.MetricResults = {}

    def Fit(self, optimizer: BackPropagation, training: DataExamples, epoch: int, batchSize: int | None,
            validation: DataExamples) -> None:
        """
        Trains the model using the provided input data.

        :param validation: Validation Dataset
        :param optimizer: the Optimizer to use for training.
        :param training: The training data.
        :param epoch: The number of epochs to train.
        :param batchSize: The size of each mini-batch.
        :return: None
        """
        if batchSize is None:
            batchSize = training.Data.shape[0]

        metric = []
        val_metric = []
        batch_generator = mb.MiniBatchGenerator(training, batchSize)
        for e in range(epoch):
            batch_generator.Reset()
            batch_accumulator =[]
            while not batch_generator.IsBatchGenerationFinished:
                inputs_batch, targets_batch = batch_generator.NextBatch()

                outputs_batch = self.Forward(inputs_batch)
                batch_metrics = self._compute_metrics(outputs_batch, targets_batch, optimizer.LossFunction)
                batch_accumulator.append(batch_metrics)

                #TODO: bind for back prop
                self._update_weights(None)

            metric_epoch = np.mean(batch_accumulator, axis=0)

            val_outputs = self.Forward(validation.Data)
            val_metric_epoch = self._compute_metrics(val_outputs, validation.Label, optimizer.LossFunction)
            metric.append(metric_epoch)
            val_metric.append(val_metric_epoch)

        metric = np.array(metric).reshape(-1, epoch)
        val_metric = np.array(val_metric).reshape(-1, epoch)
        self.MetricResults["loss"] = metric[0]
        self.MetricResults["val_loss"] = val_metric[0]
        for i, m in enumerate(self.Metrics):
            self.MetricResults[f"{m.Name}"] = metric[i + 1]
            self.MetricResults[f"val_{m.Name}"] = val_metric[i + 1]

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
            "Units": [layer.Unit for layer in self.Layers],
            "Activations": [layer.ActivationFunction.GetName() for layer in self.Layers],
            "LayersWeights": [layer.get_weights() for layer in self.Layers],
            "Names": [layer.name for layer in self.Layers]
        }

        CreateDir(path)
        with open(path, "w") as file:
            json.dump(model_data, file, default= convert_to_serializable)


    def LoadModel(self, path: str) -> None:
        """
        Loads the model's weights from a file.

        :param path: The file path to load the model from.
        :return: None
        """
        self.Layers = []

        with open(path, "r") as file:
            model_data = json.load(file)

        for unit, act ,name in zip(model_data["Units"], model_data["Activations"],model_data["Names"]):
            self.AddLayer(Layer(unit, ActivationFunction.GetInstances(act),name))

        for layer, weights in zip(self.Layers, model_data["LayersWeights"]):
            layer.set_weights(np.array(weights))

    def AddMetric(self, metric: Metric) -> None:
        """
        Adds a single metric for evaluation during training.

        :param metric: The metric instance to add.
        :return: None
        """
        self.Metrics.append(metric)

    def AddMetrics(self, metrics: List[Metric]) -> None:
        """
        Adds multiple metrics for evaluation during training.

        :param metrics: A list of metric instances to add.
        :return: None
        """
        self.Metrics.extend(metrics)

    def SaveMetricsResults(self, path: str) -> None:
        """
        Saves the model's metric results to a file.

        :param path: The file path to save the metric results to.
        :return: None
        """

        CreateDir(path)
        with open(path, "w") as file:
            json.dump(self.MetricResults, file, default= convert_to_serializable)

    def Forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the model.

        :param input: The input data.
        :return: The final output after the forward pass.
        """

        # Compute Input Layer
        self.Layers[0].Compute(input)
        # Forward to other Layers
        out = 0.0
        for i in range(len(self.Layers) - 1):
            out = self.Layers[i + 1].Compute(self.Layers[i].LayerOutput)
        return out

    def Build(self, weightInizialization: WeightInitializer) -> None:
        """
        build each layer of the model.

        :param weightInitialization:  the weights' initializer.

        """
        for layer in self.Layers:
            layer.Build(weightInizialization)

    def _update_weights(self, optimizer) -> None:
        """
        Updates the weights of the layers using the optimizer.

        :type optimizer: optimizer to use
        :return: None
        """
        for layer in reversed(self.Layers):
            layer.set_weights(layer.get_weights() + np.random.uniform(-0.1, 0.1, layer.get_weights().shape))

    def _compute_metrics(self, output: np.ndarray, target: np.ndarray, lossFunction: LossFunction) -> np.ndarray:
        """
        Computes the metrics for the current epoch.

        :param output: the predicted outputs.
        :param target: the ground truth outputs.
        :return: None
        """

        result = [lossFunction.CalculateLoss(output, target)]
        for m in self.Metrics:
            result.append(m.ComputeMetric(output, target))

        return np.array(result)
