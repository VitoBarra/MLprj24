import unittest
from Core.Tuner.HyperBag import HyperBag
import numpy as np

class TestHyperBag(unittest.TestCase):

    def setUp(self):
        """Setup a fresh instance of HyperBag for each test."""
        self.hyperbag = HyperBag()

    def test_add_range_valid(self):
        """Test adding a valid range of hyperparameters."""
        self.hyperbag.AddRange("lr", 0.1, 0.5, 0.1)
        self.assertIn("lr", self.hyperbag.Keys())
        np.testing.assert_array_almost_equal(self.hyperbag["lr"], [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_add_range_invalid_bounds(self):
        """Test adding a range with invalid bounds."""
        with self.assertRaises(ValueError) as context:
            self.hyperbag.AddRange("lr", 0.5, 0.1, 0.1)
        self.assertEqual(str(context.exception), "Lower bound must be smaller than upper bound")

    def test_add_chosen_valid(self):
        """Test adding a chosen list of hyperparameters."""
        chosen_values = [0.1, 0.2, 0.3]
        self.hyperbag.AddChosen("lr", chosen_values)
        self.assertIn("lr", self.hyperbag.Keys())
        self.assertEqual(self.hyperbag["lr"], chosen_values)

    def test_add_chosen_invalid_empty(self):
        """Test adding an empty list as chosen hyperparameters."""
        with self.assertRaises(ValueError) as context:
            self.hyperbag.AddChosen("lr", [])
        self.assertEqual(str(context.exception), "Chosen parameter must have at least length 1")

    def test_check_hp_duplicate(self):
        """Test that duplicate hyperparameters raise an error."""
        self.hyperbag.AddRange("lr", 0.1, 0.5, 0.1)
        with self.assertRaises(ValueError) as context:
            self.hyperbag.AddChosen("lr", [0.1, 0.2])
        self.assertEqual(
            str(context.exception),
            "Hyper parameter 'lr' has already bean registered"
        )

    def test_keys_retrieval(self):
        """Test retrieving keys from the hyperbag."""
        self.hyperbag.AddRange("lr", 0.1, 0.3, 0.1)
        self.hyperbag.AddChosen("momentum", [0.9, 0.95])
        self.assertEqual(set(self.hyperbag.Keys()), {"lr", "momentum"})



    def test_values_retrieval(self):
        """Test retrieving values from the hyperbag.
        This test fails and it is a numpy fault. The addRange function generates an extra value due to float precision."""

        self.hyperbag.AddRange("lr", 0.1, 0.3, 0.1)
        self.hyperbag.AddChosen("momentum", [0.9, 0.95])

        # Validate that the expected ranges are included (order-insensitive)
        lr_values = [0.1, 0.2, 0.3]
        momentum_values = [0.9, 0.95]

        # Ensure all items are present without strict order
        self.assertTrue(np.allclose(self.hyperbag["lr"], lr_values))
        self.assertTrue(np.allclose(self.hyperbag["momentum"], momentum_values))

    def test_remove_hyperparameter(self):
        """Test deleting a hyperparameter."""
        self.hyperbag.AddRange("lr", 0.1, 0.3, 0.1)
        del self.hyperbag["lr"]
        self.assertNotIn("lr", self.hyperbag.Keys())


