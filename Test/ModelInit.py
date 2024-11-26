import Core as Core
from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Layer import *
from Core.WeightInitializer import *
from Core.Metric import *

if __name__ == '__main__':

    model = ModelFeedForward()

    model.AddLayer(Layer(1,Linear()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(1, Linear()))
    model.Build(GlorotInitializer())

    x = np.random.uniform(10, 5, (5,1))
    y = np.random.uniform(0, 1, (5,1))

    res=model.Forward(x)

    print(f"prediction are \n{res} \n while target are\n {y}")
    for e in [MSE(),RMSE(),MEE()]:
          print(f"The {e.Name} is {e.ComputeMetric(res, y)}")





