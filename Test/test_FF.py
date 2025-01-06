import json
import unittest
import numpy as np
from unittest.mock import MagicMock

from Core.ActivationFunction import ActivationFunction, Linear
from Core.DataSet.DataSet import DataSet
from Core.FeedForwardModel import ModelFeedForward
from Core.Layer import Layer
from Core.LossFunction import LossFunction
from Core.Metric import Metric
from Core.Optimizer.Optimizer import Optimizer


class TestModelFeedForward(unittest.TestCase):

    def setUp(self):
        """Setup a basic model for testing."""
        self.model = ModelFeedForward()

        # Mock layers
        self.IN = Layer(1,Linear(),True,"in")
        self.layer1 = Layer(2,Linear(),True,"L1")
        self.layer2 = Layer(2,Linear(),False,"L2")
        self.OUT = Layer(2,Linear(),False,"out")
        self.IN.WeightToNextLayer = np.array([[1,1], [2,2]])
        self.layer1.WeightToNextLayer = np.array([[1, 2,2], [3, 4,4]])
        self.layer2.WeightToNextLayer = np.array([[4, 3], [2, 1]])

        # Add layers to the model
        self.model.AddLayer(self.IN)
        self.model.AddLayer(self.layer1)
        self.model.AddLayer(self.layer2)
        self.model.AddLayer(self.OUT)



    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        test_path = "Test/model/test_model.json"
        self.model.SaveModel(test_path)

        # Verify the saved file
        with open(test_path, "r") as f:
            saved_data = json.load(f)

        self.assertIn("layers", saved_data)


        # Load the model and verify
        new_model = ModelFeedForward()
        new_model.LoadModel(test_path)

        self.assertEqual(len(new_model.Layers), len(self.model.Layers))

        for i, (loaded_layer, original_layer) in enumerate(zip(new_model.Layers, self.model.Layers)):
            self.assertEqual(loaded_layer.Unit, original_layer.Unit)
            if loaded_layer.WeightToNextLayer is not None and original_layer.WeightToNextLayer is not None:
                np.testing.assert_array_almost_equal(
                    loaded_layer.WeightToNextLayer, original_layer.WeightToNextLayer
                )
            self.assertEqual(loaded_layer.name, original_layer.name)
            self.assertEqual(
                loaded_layer.ActivationFunction.Name, original_layer.ActivationFunction.Name
            )

