from Core.ActivationFunction import *
from Core.BackPropagation import BackPropagation
from Core.FeedForwardModel import ModelFeedForward
from Core.Layer import Layer
from Core.LossFunction import MSELoss
from Core.Metric import *
from Core.Tuner.HyperBag import HyperBag

from Core.Tuner.HpSearch import GridSearch, RandomSearch
from Core.WeightInitializer import GlorotInitializer
from Test.ModelInit import CreateFakeData


def HyperMode(hp):
    model = ModelFeedForward()

    model.AddLayer(Layer(1, Linear()))

    for i in range(hp["hlayer"]):
        model.AddLayer(Layer(15, TanH()))

    model.AddLayer(Layer(1, Linear()))
    return model


if __name__ == '__main__':
    hp = HyperBag()
    hp.AddRange("labda",0.1,0.5,0.05)
    hp.AddChosen("alpha",[0.3,0.4,0.5,0.6])
    hp.AddChosen("hlayer",[1,2,3,4,5,6])
    hp.AddRange("eta",0.05,0.3,0.05)

    for key in hp.Keys():
        print(f"{key}: {hp[key]}")


    try:
        hp.AddChosen("hlayer",[1,2,3,4,5,6])
    except Exception as exc:
        print(f"\n{exc}\n")



    data,val =CreateFakeData(10)
    gs= GridSearch(hp)
    hyperModel: ModelFeedForward
    for hyperModel, hpSel in gs.search(HyperMode):
        hyperModel.Build(GlorotInitializer())
        hyperModel.AddMetrics([MSE(), RMSE(), MEE()])
        hyperModel.Fit(BackPropagation(MSELoss(),hpSel["labda"],hpSel["alpha"],hpSel["eta"]),data,5,2,val)
        print(f"number of layer selected: {len(hyperModel.Layers)} with labda: {hpSel["labda"]}, alpha: {hpSel["alpha"]}, eta: {hpSel["eta"]}")

