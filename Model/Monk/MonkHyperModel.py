import numpy as np

from Core.ActivationFunction import Sign, Binary, ActivationFunction, Linear
from Core.DataSet.DataSet import DataSet
from Core.FeedForwardModel import ModelFeedForward
from Core.Layer import Layer
from Core.LossFunction import CategoricalCrossEntropyLoss, MSELoss
from Core.Metric import Accuracy
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Core.Tuner.HyperBag import HyperBag
from Core.Tuner.HyperModel import HyperModel
from . import MONK_NUM, OPTIMIZER, USE_KFOLD


class MONKHyperModel(HyperModel):


    def __init__(self, originalDataset : DataSet):
        super().__init__( originalDataset)
        self.k = None
        self.val_split = None
        self.InterpretationMetric = None

    def SetSlit(self,val_split, k):
        self.val_split = val_split
        self.k = k

    def GetHyperParameters(self) ->HyperBag:
        hp = HyperBag()

        # Optimizer
        hp.AddChosen("BatchSize",[-1,1,32,64,96,128])
        hp.AddRange("eta", 0.001, 0.2, 0.005)
        if MONK_NUM ==3:
            hp.AddRange("lambda", 0.000, 0.01, 0.005)
        hp.AddRange("alpha", 0.5, 0.9, 0.05)
        hp.AddRange("decay", 0.0003, 0.005, 0.0003)

        # only for adam
        if OPTIMIZER>2:
            hp.AddRange("beta", 0.97, 0.99, 0.01)
            hp.AddRange("epsilon", 1e-13, 1e-10, 1e-1)


        # Architecture
        hp.AddChosen("UseBiasIN",[True,False])
        hp.AddChosen("UseBias",[True,False])
        hp.AddRange("unit", 2, 8, 1)
        hp.AddRange("hlayer", 0, 3, 1)
        hp.AddChosen("ActFun",["TanH","Sigmoid","ReLU","LeakyReLU"])

        # Data format
        hp.AddChosen("oneHotInput",[True,False])
        hp.AddChosen("outFun",["TanH","Sigmoid","SoftARGMax"])



        #hp.AddRange("drop_out", 0.2, 0.6, 0.1)

        return hp

    def GetDatasetVariant(self, hp) ->DataSet:
        data_set = DataSet.Clone(self.originalDataset)
        if (hp["oneHotInput"],hp["outFun"]) not in self.DataSetsVariant:
            self.PreprocessInput(hp ,data_set)
            self.PreprocessOutput(hp,data_set)
            self.SplitAllDataset(data_set)
            self.DataSetsVariant[hp["oneHotInput"],hp["outFun"]] = data_set


        return self.DataSetsVariant[hp["oneHotInput"],hp["outFun"]]

    def SplitAllDataset(self,data_set):

        if USE_KFOLD:
            data_set.SetUp_Kfold_TestHoldOut(self.k)
        else:
            data_set.SplitTV(self.val_split)


    def PreprocessInput(self,hp, data_set):
        if hp["oneHotInput"]:
            data_set.ToOnHotOnData()

    def PreprocessOutput(self,hp, data_set : DataSet):
        if hp["outFun"] == "TanH": # TanH
            data_set.ApplyTransformationOnLabel(np.vectorize(lambda x: -1 if x == 0 else 1 ))
            Interpretation_metric = Accuracy(Sign())

        elif hp["outFun"] == "Sigmoid": #asigmoid
            Interpretation_metric = Accuracy(Binary(0.5))

        elif hp["outFun"] == "SoftARGMax": #  one Hot label
            data_set.ToOneHotLabel()
            Interpretation_metric = Accuracy()
        else:
            raise ValueError("value unknown")
        self.InterpretationMetric= Interpretation_metric


    def GetModel(self, hp :HyperBag):
        model = ModelFeedForward()

        output_act = ActivationFunction.GetInstances(hp["outFun"])
        hidden_act = ActivationFunction.GetInstances(hp["ActFun"])
        input_unit = 17 if hp["oneHotInput"] else 6
        output_unit = 2 if hp["outFun"] =="SoftARGMax" else  1


        model.AddLayer(Layer(input_unit, Linear(), hp["UseBiasIN"], "input"))
        for i in range(hp["hlayer"]):
            model.AddLayer(Layer(hp["unit"], hidden_act, hp["UseBias"], f"_h{i}"))


        model.AddLayer(Layer(output_unit ,output_act, False,f"output_{hp['outFun']}"))
        return model


    def GetOptimizer(self, hp :HyperBag):

        loss = CategoricalCrossEntropyLoss() if hp["outFun"] == "SoftARGMax" else MSELoss()


        if OPTIMIZER == 1:
            optimizer = BackPropagation(loss,hp["BatchSize"], hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
        elif OPTIMIZER == 2:
            optimizer = BackPropagationNesterovMomentum(loss,hp["BatchSize"], hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
        else:
            optimizer = Adam(loss,hp["BatchSize"], hp["eta"], hp["lambda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

        return  optimizer

