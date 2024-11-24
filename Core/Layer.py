class Layer:
    def __init__(self, unit, activationFunction):
        self.Unit = unit
        self.ActivationFunction = activationFunction
        self.NextLayer = None
        self.LastLayer = None
        self.WeightToNextLayer = None
        self.LastOutputs = None

    def Update(self, optimizer):
        gradient = optimizer.ComputeUpdate(self)
        self.WeightToNextLayer -= optimizer.Eta * gradient

    def Compute(self, inputs):
        if self.WeightToNextLayer is None:
            self.WeightToNextLayer = np.random.randn(inputs.shape[1], self.Unit) * 0.01
        z = inputs @ self.WeightToNextLayer
        self.LastOutputs = self.ActivationFunction.Calculate(z)
        return self.LastOutputs

    def get_weights(self):
        return self.WeightToNextLayer

    def set_weights(self, weights):
        self.WeightToNextLayer = weights
