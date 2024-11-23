import numpy as np
from WeightInitializer  import *


from Core.BackPropagation import BackPropagation


class Layer:

    LayerOutput:        np.ndarray| None
    WeightToNextLayer:  np.ndarray| None
    LastLayer:          Layer | None
    NextLayer:          Layer | None
    ActivationFunction: any
    Unit:               int

    def __init__(self, unit: int, activationFunction: any, weightInitializer: WeightInitializer) -> None:
        self.Unit = unit
        self.ActivationFunction = activationFunction
        self.NextLayer = None
        self.LastLayer:Layer|None = None
        self.WeightToNextLayer = None
        self.LayerOutput = None
        if self.WeightToNextLayer is None:
            self.WeightToNextLayer = weightInitializer.GenerateWeight(self.Unit,self.NextLayer.Unit)


    def Update(self, optimizer: BackPropagation) -> None:
        gradient = optimizer.ComputeUpdate(self)
        self.WeightToNextLayer -= optimizer.Eta * gradient

    def Compute(self, inputs):
        z = inputs @ self.WeightToNextLayer
        self.LayerOutput = self.ActivationFunction.Calculate(z)
        return self.LayerOutput

    def get_weights(self):
        return self.WeightToNextLayer

    def set_weights(self, weights):
        self.WeightToNextLayer = weights
