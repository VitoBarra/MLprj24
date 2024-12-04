from Core.LossFunction import LossFunction


class BackPropagation:
    LossFunction: LossFunction
    Velocity: None
    Eta: float
    Momentum: int
    Lambda: int

    def __init__(self, function: LossFunction ,Lambda=0, Momentum=0, Eta=0.01, ):
        self.Lambda = Lambda
        self.Momentum = Momentum
        self.Eta = Eta
        self.LossFunction = function
        self.Velocity = None  # For momentum

    def ComputeUpdate(self, layer):
        if (layer.NextLayer != None):
            grad = self.LossFunction.CalculateLoss(layer.LayerOutput, layer.NextLayer.LayerOutput)
            if self.Velocity is None:
                self.Velocity = grad
            else:
                self.Velocity = self.Momentum * self.Velocity + (1 - self.Momentum) * grad
        return 0 + self.Lambda * layer.WeightToNextLayer
