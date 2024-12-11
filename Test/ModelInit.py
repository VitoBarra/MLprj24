from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Layer import *
from Core.LossFunction import MSELoss
from Core.WeightInitializer import *
from Core.Metric import *
from DataUtility.DataExamples import *
from Core.BackPropagation import *
from DataUtility.DataSet import DataSet


def CreateFakeData(nData:int , xdim :int=1, ydim:int=1) ->(DataExamples,DataExamples):
    x = np.random.uniform(0, 1, (nData,xdim))
    y = np.random.choice([0, 1], (nData, ydim))
    id = np.array(range(x.shape[0]))

    data = DataExamples(x,y, id)
    val = DataExamples(x, y, id)
    return data, val

def CreateFakeData_dataset(nData:int , xdim :int=1, ydim:int=1) ->(DataExamples,DataExamples):
    x = np.random.uniform(0, 0.5, (nData,xdim))
    y = np.random.choice([0, 1], (nData, ydim))
    id = np.array(range(x.shape[0]))

    data = DataSet(x,y, id)
    return data


if __name__ == '__main__':

    model1 = ModelFeedForward()

    model1.AddLayer(Layer(2, Linear(),"input"))
    model1.AddLayer(Layer(15, TanH(),"h1"))
    model1.AddLayer(Layer(15, TanH(),"h2"))
    model1.AddLayer(Layer(2, Linear(),"output"))
    model1.Build(GlorotInitializer())

    data ,val = CreateFakeData(6, 2,2)
    model1.AddMetrics([MSE(), RMSE(), MEE()])


    model1.Fit(BackPropagation(MSELoss()), data, 15, 4, val)

    for k,v in model1.MetricResults.items() :
          print(f"The {k} is {v}")
    model1.SaveMetricsResults("Data/Results/model1.mres")


    print("\n\n")
    model1.SaveModel("Data/Models/Test1.vjf")

    model2 = ModelFeedForward()
    model2.LoadModel("Data/Models/Test1.vjf")
    model2.AddMetrics([MSE(), RMSE(), MEE()])

    model2.Fit(BackPropagation(MSELoss()), data, 15, 2, val)

    for k,v in model2.MetricResults.items() :
        print(f"The {k} is {v}")





