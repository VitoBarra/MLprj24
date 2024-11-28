import numpy as np

from Core.FeedForwardModel import *
from Core.WeightInitializer import *
from Core.Metric import *
from DataUtility.DataSet import DataSet
from DataUtility.DataExamples import *
from Core.BackPropagation import *



if __name__ == '__main__':


    model = ModelFeedForward()


    model.AddLayer(Layer(1, Linear(), "Input"))
    model.AddLayer(Layer(1, TanH(), "H1"))
    model.AddLayer(Layer(2, TanH(), "H2"))
    model.AddLayer(Layer(3, Linear(), "Output"))
    model.Build(GlorotInitializer())

    x = np.random.uniform(10, 5, (8,1))
    y = np.random.uniform(0, 1, (8,3))
    id_data = np.array(range(x.shape[0]))

    data = DataExamples(x,y, id_data)
    val = DataExamples(x, y, id_data)

    model.AddMetrics([MSE(),RMSE(),MEE()])

    model.Fit(BackPropagation(MSELoss(), 0.3, 0.1),data,1,4,val)

    for k,v in model.MetricResults.items() :
          print(f"The {k} is {v}")
    model.SaveMetricsResults("Data/Results/model1.mres")
