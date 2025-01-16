import random
from statistics import mean, variance

from sklearn.linear_model import LogisticRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss, CategoricalCrossEntropyLoss
from Core.Metric import Accuracy, MSE
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Core.Tuner.ModelSelection import *
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.DataSet.DataExamples import DataExamples
from Model import MONKRESUTLPATH, MONKPLOTPATH, MONKMODELPATH
from Model.ModelResults import PlotTableVarianceAndMean, PlotMultipleModels, PlotMetrics
from Model.TrainingFuction import ValidateSelectedModel
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readMonk

USE_CATEGORICAL_LABEL = False
USE_ONEHOT_VARIABLE_DATA = False
OPTIMIZER = None
OUT_ACT_FUN = None
USE_KFOLD = False
MONK_NUM= None


def HyperModel_Monk(hp :HyperBag ):
    model = ModelFeedForward()


    model.AddLayer(Layer(17 if USE_ONEHOT_VARIABLE_DATA else 6, Linear(), hp["UseBiasIN"], "input"))
    for i in range(hp["hlayer"]):
            model.AddLayer(Layer(hp["unit"], hp["ActFun"], hp["UseBias"], f"_h{i}"))


    if USE_CATEGORICAL_LABEL:
        model.AddLayer(Layer(2, SoftARGMax(), False, "output_HotEncoding"))
        loss = CategoricalCrossEntropyLoss()
    else:
        model.AddLayer(Layer(1, TanH() if OUT_ACT_FUN == 1 else Sigmoid(), False,"output"))
        loss = MSELoss()

    if OPTIMIZER == 1:
        optimizer = BackPropagation(loss,hp["BatchSize"], hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    elif OPTIMIZER == 2:
        optimizer = BackPropagationNesterovMomentum(loss,hp["BatchSize"], hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    else:
        optimizer = Adam(loss,hp["BatchSize"], hp["eta"], hp["labda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

    return model, optimizer




def ReadMonk(n: int, val_split: float = 0.15, seed: int = 0):
    if n <0 or n>3:
        raise Exception("n must be between 0 and 3")
    TR_file_path_monk = f"dataset/monk+s+problems/monks-{MONK_NUM}.train"
    TS_file_path_monk = f"dataset/monk+s+problems/monks-{MONK_NUM}.test"

    designSet = readMonk(TR_file_path_monk)
    testSet = readMonk(TS_file_path_monk)
    monkDataset = DataSet.FromDataExample(designSet)
    monkDataset.Test = testSet

    if OUT_ACT_FUN == 1:
        monkDataset.ApplayTranformationOnLabel(np.vectorize(lambda x: -1 if x == 0 else 1 ))
        baseline_metric = Accuracy(Sign())
    else:
        baseline_metric = Accuracy(Binary(0.5))

    monkDataset.Shuffle(seed)

    if USE_ONEHOT_VARIABLE_DATA:
        monkDataset.ToOnHotOnExamples()

    if USE_CATEGORICAL_LABEL:
    
        monkDataset.ToCategoricalLabel()
    monkDataset.PrintData()

    if USE_KFOLD:
        monkDataset.SetUp_Kfold_TestHoldOut(5)
    else:
        monkDataset.SplitTV(val_split)

    return monkDataset, baseline_metric




def HyperBag_Monk():
    hp = HyperBag()

    # Optimizer
    hp.AddChosen("BatchSize",[-1,1,32,64,96,128])
    hp.AddRange("eta", 0.001, 0.2, 0.005)
    if MONK_NUM ==3:
        hp.AddRange("labda", 0.000, 0.01, 0.005)
    hp.AddRange("alpha", 0.5, 0.9, 0.05)
    hp.AddRange("decay", 0.0003, 0.005, 0.0003)

    # only for adam
    if OPTIMIZER>2:
        hp.AddRange("beta", 0.9, 0.99, 0.01)
        hp.AddRange("epsilon", 1e-13, 1e-7, 1e-1)


    # Architecture
    hp.AddChosen("UseBiasIN",[True,False])
    hp.AddChosen("UseBias",[True,False])
    hp.AddChosen("ActFun",[TanH(),Sigmoid(),ReLU()])
    hp.AddRange("unit", 2, 8, 1)
    hp.AddRange("hlayer", 0, 3, 1)



    #hp.AddRange("drop_out", 0.2, 0.6, 0.1)

    return hp



def ModelSelection(monkDataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:
    if USE_KFOLD:
        ModelSelector:ModelSelection = BestSearchKFold(HyperBag_Monk(), RandomSearch(NumberOrTrial))
    else:
        ModelSelector:ModelSelection = BestSearch(HyperBag_Monk(), RandomSearch(NumberOrTrial))

    watched_metric = "val_loss"

    callBacks = [EarlyStopping(watched_metric, 100,0.0001)]
    best_model, hpSel = ModelSelector.GetBestModel(
        HyperModel_Monk,
        monkDataset,
        800,
        watched_metric,
        [BaselineMetric],
        GlorotInitializer(),
        callBacks)
    #best_model.PlotModel(f"MONK Model {MONK_NUM}")
    return best_model, hpSel







def GeneratePlot(AccuracyMetric, MetricResults, monkDataset,extraname:str=""):
    MSEmetric = MSE()

    lin_model = LogisticRegression()
    if USE_CATEGORICAL_LABEL:
        lin_model.fit(monkDataset.Training.Data, monkDataset.Training.Label[:, 1])
        test_Label = monkDataset.Test.Label[:, 1]
    else:
        lin_model.fit(monkDataset.Training.Data, monkDataset.Training.Label.reshape(-1))
        test_Label = monkDataset.Test.Label

    predictions = lin_model.predict(monkDataset.Test.Data)

    baseline_acc = AccuracyMetric(predictions.reshape(-1, 1), test_Label) *100
    baseline_mse = MSEmetric(predictions.reshape(-1, 1), test_Label)

    metric_to_plot_loss = {key: value[1:] for key, value in MetricResults.items() if key.endswith("loss")}
    metric_to_plot_Accuracy = {key: value[1:]*100 for key, value in MetricResults.items() if key.endswith("Accuracy")}

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    PlotMetrics(
        metricDic=metric_to_plot_loss,
        baseline=baseline_mse,
        baselineName=f"baseline {MSEmetric.Name}",
        limitYRange=None,
        title=f"MONK {MONK_NUM} loss",
        xlabel="Epochs",
        ylabel="",
        subplotAxes=axes[0])
    PlotMetrics(
        metricDic=metric_to_plot_Accuracy,
        baseline=baseline_acc,
        baselineName=f"baseline {AccuracyMetric.Name}",
        limitYRange=None,
        title=f"MONK {MONK_NUM} accuracy",
        xlabel="Epochs",
        ylabel="%",
        subplotAxes=axes[1])
    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{MONKPLOTPATH}{MONK_NUM}", f"Loss(MSE)-Accuracy{extraname}")
    plt.close(fig)

def GenerateTagName():
    tagName = ""
    if OPTIMIZER == 1:
        tagName += "_backprop"
    elif OPTIMIZER == 2:
        tagName += "_nasterov"
    else:
        tagName += "_adam"

    if USE_ONEHOT_VARIABLE_DATA:
        tagName += "_onehot"
    else:
        tagName += "_numeric"

    if OUT_ACT_FUN == 1:
        tagName += "_tanh"
    else:
        tagName += "_sigmoid"

    return tagName


def TrainMonkModel(NumberOrTrial_search:int, NumberOrTrial_mean:int) -> None:


    mode = HyperBag()

    # Data Processing
    mode.AddChosen("OneHot",[True,False])
    mode.AddChosen("OutActFun",[1,2])
    # Training
    mode.AddChosen("Optimizer",[1,2])



    global OUT_ACT_FUN
    global OPTIMIZER
    global USE_ONEHOT_VARIABLE_DATA
    global MONK_NUM

    gs = GridSearch()

    for monk in [1,2,3]:
        MONK_NUM = monk

        #Dataset Preparation
        monkDataset, BaselineMetric_Accuracy = ReadMonk(MONK_NUM, 0.15)
        mergedMonkDataset = DataSet.Clone(monkDataset)
        if not USE_KFOLD:
            mergedMonkDataset.MergeTrainingAndValidation()

        for modes, _ in gs.Search(mode):
            USE_ADAM=modes["Adam"]
            USE_ONEHOT_VARIABLE_DATA=modes["OneHot"]
            OUT_ACT_FUN=modes["OutActFun"]


            tagName = GenerateTagName()


            print(f"Training MONK {MONK_NUM}...")
            print(f"Run experiment with the following settings: {tagName}")


            best_model, best_hpSel = ModelSelection(monkDataset, BaselineMetric_Accuracy, NumberOrTrial_search)
            best_hpSel:HyperBag
            best_model.SaveModel( f"{MONKMODELPATH}{MONK_NUM}", f"MONK{MONK_NUM}{tagName}.vjf")

            #best_model.SaveMetricsResults(f"Data/Results/Monk{MONK_NUM}{tagName}.mres")

             #GeneratePlot(BaselineMetric_Accuracy, best_model.MetricResults, monkDataset,tagName)

            print(f"Best hp : {best_hpSel}")

            MetricToCheck = [key for key, _ in best_model.MetricResults.items() if not key.startswith("val_")]
            totalResult = ValidateSelectedModel(
                HyperModel_Monk,best_hpSel,
                NumberOrTrial_mean, MetricToCheck,
                BaselineMetric_Accuracy
                ,monkDataset.Test,mergedMonkDataset.Training,
                None,
                500,50,42 )
            SaveJson(f"{MONKRESUTLPATH}{MONK_NUM}",f"res{tagName}.json",totalResult)

            # PlotMultipleModels(totalResult["metrics"],["test_loss", "loss", "test_Accuracy", "Accuracy"],f"{MONKPLOTPATH}{MONK_NUM}",f"mean_MONK{tagName}.png" )








if __name__ == '__main__':
        TrainMonkModel(250,25)

