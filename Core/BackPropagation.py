class BackPropagation:
    def __init__(self, Lambda=0, Momentum=0, Eta=0.01, LossFunction=None):
        self.Lambda = Lambda
        self.Momentum = Momentum
        self.Eta = Eta
        self.LossFunction = LossFunction
        self.Velocity = None  # For momentum

    def ComputeUpdate(self, layer):
        grad = self.LossFunction.ComputeMetric(layer.LayerOutput, layer.NextLayer.LayerOutput)
        if self.Velocity is None:
            self.Velocity = grad
        else:
            self.Velocity = self.Momentum * self.Velocity + (1 - self.Momentum) * grad
        return grad + self.Lambda * layer.WeightToNextLayer
