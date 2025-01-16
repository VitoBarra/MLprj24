import json
import os
from typing import List, Any

import numpy as np

from . import MiniBatchGenerator as mb
from . import Metric
from .Layer import Layer
from .LossFunction import LossFunction
from .Optimizer.Optimizer import Optimizer
from .Inizializer.WeightInitializer import WeightInitializer, GlorotInitializer
from .DataSet.DataSet import DataSet
from Utility.FileUtil import CreateDir, convert_to_serializable
from Utility.PlotUtil import plot_neural_network_with_transparency


class ModelFeedForward:
    """
A feedforward neural network class for building, training, saving, and loading models.

This class provides methods to construct a neural network by adding layers,
train it using backpropagation with a defined optimizer, evaluate performance
using specified metrics, and save or load the model's parameters.

Attributes:
    BackPropagation (Optimizer): The optimizer used for updating weights during training.
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
        self.EarlyStop = None
        self.Layers = []
        self.InputLayer = None
        self.OutputLayer = None
        self.Metrics = []
        self.MetricResults = {}

    def Fit(self, optimizer: Optimizer, data: DataSet, epoch: int, callbacks:List = []) -> None:
        """
        Trains the model using the provided input data.

        :param data: DataSet to use
        :param optimizer: the Optimizer to use for training.
        :param epoch: The number of epochs to train.
        :param batchSize: The size of each mini-batch.
        :param callbacks: List of functions
        :return: None
        """
        for layer in self.Layers:
            layer.TrainingMode()

        self.EarlyStop=False

        if optimizer.batchSize is None or optimizer.batchSize <= 0:
            batchSize = len(data.Training.Data)
        else:
            batchSize = optimizer.batchSize

        if callbacks is not None:
            for callback in callbacks:
                callback.Reset()

        metric = []
        val_metric = []
        test_metric= []

        batch_generator = mb.MiniBatchGenerator(data.Training, batchSize)
        for e in range(epoch):
            batch_generator.Reset()
            batch_accumulator =[]
            while not batch_generator.IsBatchGenerationFinished:
                inputs_batch, targets_batch = batch_generator.NextBatch()

                outputs_batch = self.Forward(inputs_batch)
                batch_metrics = self._ComputeMetrics(outputs_batch, targets_batch, optimizer.loss_function)
                batch_accumulator.append(batch_metrics)

                # Back Propagation
                optimizer.StartOptimize(self, targets_batch)


            metric_epoch = np.mean(batch_accumulator, axis=0)

            metric.append(metric_epoch)
            metric_array = np.array(metric).T

            # compute metric on validation
            if data.Validation is not None:
                val_outputs = self.Forward(data.Validation.Data)
                val_metric_epoch = self._ComputeMetrics(val_outputs, data.Validation.Label, optimizer.loss_function)
                val_metric.append(val_metric_epoch)
                val_metric_array = np.array(val_metric).T

            # compute metric on test
            if data.Test is not None:
                test_outputs = self.Forward(data.Test.Data)
                test_metric_epoch = self._ComputeMetrics(test_outputs, data.Test.Label, optimizer.loss_function)
                test_metric.append(test_metric_epoch)
                test_metric_array = np.array(test_metric).T

            #Save f metric at each epoch
            metricNames = ["loss"] + [m.Name for m in self.Metrics]
            for i, metricName in enumerate(metricNames):
                self.MetricResults[f"{metricName}"] = metric_array[i]
                if data.Validation is not None:
                    self.MetricResults[f"val_{metricName}"] = val_metric_array[i]
                if data.Test is not None:
                    self.MetricResults[f"test_{metricName}"] = test_metric_array[i]
            if callbacks is not None:
                for callback in callbacks:
                    callback(self)

            if self.EarlyStop is True:
                break


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




    def SaveModel(self, directory: str, filename: str = "model.json") -> None:
        """
        Saves the model's weights to a file.

        :param directory: The directory where the file will be saved.
        :param filename: The name of the file to save the model to. Defaults to 'model.json'.
        :return: None
        """
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Combine directory and filename to create the full file path
        file_path = os.path.join(directory, filename)

        # Serialize model data
        model_data = {"layers": [layer.SerializeLayer() for layer in self.Layers]}

        # Save the serialized data to the file
        with open(file_path, "w") as file:
            json.dump(model_data, file, default=convert_to_serializable)



    def LoadModel(self, path: str) -> None:
        """
        Loads the model's weights from a file.

        :param path: The file path to load the model from.
        :return: None
        """
        self.Layers = []

        with open(path, "r") as file:
            model_data = json.load(file)

        for layerDic in model_data["layers"]:
            layer = Layer.DeserializeLayer(layerDic)
            self.AddLayer(layer)



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


    def Predict(self, input: np.ndarray , post_processing = None) -> np.ndarray:
        for layer in self.Layers:
            layer.InferenceMode()
        out = self.Forward(input)
        return post_processing(out) if post_processing is not None else out


    def PlotModel(self, plotTitle:str = "Neural Network diagram"):
        w = [np.array(l.WeightToNextLayer) for l in self.Layers if l.WeightToNextLayer is not None]
        plot_neural_network_with_transparency(w,plotTitle)


    def Build(self, weightIni: WeightInitializer | None = None) -> None:
        """
        build each layer of the model.

        :param weightIni: the weights' initializer.

        """
        if weightIni is None:
            weightIni=  GlorotInitializer()

        for layer in self.Layers:
            layer.Build(weightIni)


    def _ComputeMetrics(self, output: np.ndarray, target: np.ndarray, lossFunction: LossFunction) -> np.ndarray:
        """
        Computes the metrics for the current epoch.

        :param output: the predicted outputs.
        :param target: the ground truth outputs.
        :return: None
        """

        result = [lossFunction.CalculateLoss(output, target)]


        for m in self.Metrics:
            result.append(m(output, target))

        return np.array(result)