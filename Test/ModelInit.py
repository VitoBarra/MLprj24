from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Layer import *
from Core.LossFunction import MSELoss
from Core.WeightInitializer import *
from Core.Metric import *
from DataUtility.DataExamples import *
from Core.BackPropagation import *

if __name__ == '__main__':

    model1 = ModelFeedForward()

    model1.AddLayer(Layer(1, Linear()))
    model1.AddLayer(Layer(15, TanH()))
    model1.AddLayer(Layer(15, TanH()))
    model1.AddLayer(Layer(1, Linear()))
    model1.Build(GlorotInitializer())

    x = np.random.uniform(10, 5, (5,1))
    y = np.random.uniform(0, 1, (5,1))
    id = np.array(range(x.shape[0]))

    data = DataExamples(x,y, id)
    val = DataExamples(x, y, id)

    model1.AddMetrics([MSE(), RMSE(), MEE()])


    model1.Fit(BackPropagation(MSELoss()), data, 15, 2, val)

    for k,v in model1.MetricResults.items() :
          print(f"The {k} is {v}")


    print("\n\n")
    model1.SaveModel("models/Test1.vjf")

    model2 = ModelFeedForward()
    model2.LoadModel("models/Test1.vjf")
    model2.AddMetrics([MSE(), RMSE(), MEE()])

    model2.Fit(BackPropagation(MSELoss()), data, 15, 2, val)

    for k,v in model2.MetricResults.items() :
        print(f"The {k} is {v}")





