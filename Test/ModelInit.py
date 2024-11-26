from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Layer import *
from Core.LossFunction import MSELoss
from Core.WeightInitializer import *
from Core.Metric import *
from DataUtility.DataExamples import *
from Core.BackPropagation import *

if __name__ == '__main__':

    model = ModelFeedForward()

    model.AddLayer(Layer(1,Linear()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(15, TanH()))
    model.AddLayer(Layer(1, Linear()))
    model.Build(GlorotInitializer())

    x = np.random.uniform(10, 5, (5,1))
    y = np.random.uniform(0, 1, (5,1))
    id = np.array(range(x.shape[0]))

    data = DataExamples(x,y, id)
    val = DataExamples(x, y, id)

    model.AddMetrics([MSE(),RMSE(),MEE()])


    model.Fit(BackPropagation(MSELoss()),data,15,2,val)

    for k,v in model.MetricResults.items() :
          print(f"The {k} is {v}")






