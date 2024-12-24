from Core.Metric import *
import numpy as np
import unittest



# Test case class

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Test data
        self.val = np.array([1.0, 2.0, 3.0])
        self.target = np.array([1.0, 2.0, 2.0])


        self.high_dim_val = np.array([[0.1, 0.9], [0.8, 0.2]])
        self.high_dim_target = np.array([1, 0])

    def test_MSE(self):
        m = MSE()
        # Test case 1: Perfect prediction
        self.assertEqual(m.ComputeMetric(self.val, self.val), 0.0)
        # Test case 2: Prediction with errors
        self.assertAlmostEqual(m.ComputeMetric(self.val, self.target), (1.0 / self.val.shape[0]), places=5)

    def test_RMSE(self):
        m = RMSE()
        # Test case 1: Perfect prediction
        self.assertEqual(m.ComputeMetric(self.val, self.val), 0.0)
        # Test case 2: Prediction with errors
        self.assertAlmostEqual(m.ComputeMetric(self.val, self.target), np.sqrt((1.0 / self.val.shape[0])), places=6)

    def test_MAE(self):
        m = MAE()
        # Test case 1: Perfect prediction
        self.assertEqual(m.ComputeMetric(self.val, self.val), 0.0)
        # Test case 2: Prediction with errors
        self.assertAlmostEqual(m.ComputeMetric(self.val, self.target), (1.0 / self.val.shape[0]), places=6)

    def test_Accuracy(self):
        m = Accuracy()
        # Test case 1: Perfect prediction
        self.assertEqual(m.ComputeMetric(self.val, self.val), 1.0)
        # Test case 2: Incorrect predictions
        self.assertEqual(m.ComputeMetric(self.val, self.target), 2 / 3)

        # Test case 3: High-dimensional inputs
        self.assertEqual(m.ComputeMetric(self.high_dim_val, self.high_dim_target), 1.0)
if __name__ == '__main__':
    unittest.main()