from sklearn.linear_model import LinearRegression
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Metric import *
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Model import CUPMODELPATH, CUPPLOTPATH, CUPRESULTSPATH
from Model.ModelResults import *
from Model.TrainingFuction import ValidateSelectedModel
from Utility.PlotUtil import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss
from Core.Tuner.ModelSelection import BestSearch, BestSearchKFold, ModelSelection
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.Inizializer.WeightInitializer import GlorotInitializer
from dataset.ReadDatasetUtil import readCUP
import random
import gc
from statistics import mean, variance


USE_KFOLD = False
OPTIMIZER = None


def HyperModel_CAP(hp: HyperBag):
    model = ModelFeedForward()


    model.AddLayer(Layer(12, Linear(), False, "input"))
    for i in range(hp["hlayer"]):
            model.AddLayer(Layer(hp["unit"], hp["actFun"], hp["UseBias"], f"_h{i}"))

    model.AddLayer(Layer(3, Linear(), False, "output"))

    loss = MSELoss()


    if OPTIMIZER == 1:
        optimizer = BackPropagation(loss,hp["batchSize"], hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    elif OPTIMIZER == 2:
        optimizer = BackPropagationNesterovMomentum(loss,hp["batchSize"], hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    else:
        optimizer = Adam(loss,hp["batchSize"],hp["eta"], hp["labda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

    return model, optimizer


def HyperBag_Cap():
    hp = HyperBag()
    
    # Optimizer
    hp.AddChosen("batchSize",[-1,1,64,128,160])
    hp.AddRange("eta", 0.001, 0.1, 0.01)
    hp.AddRange("labda", 0.001, 0.1, 0.001)
    hp.AddRange("alpha", 0.1, 0.9, 0.1)
    hp.AddRange("decay", 0.001, 0.1, 0.001)

    ## Only Adam
    if OPTIMIZER>2:
        hp.AddRange("beta", 0.95, 0.99, 0.01)
        hp.AddRange("epsilon", 1e-13, 1e-8, 1e-1)

    #architecture
    # hp.AddRange("drop_out", 0.1, 0.5, 0.05)
    hp.AddChosen("UseBiasIN",[True,False])
    hp.AddChosen("UseBias",[True,False])
    hp.AddChosen("actFun",[Sigmoid(),TanH(),ReLU(),LeakyReLU()])
    hp.AddRange("unit", 1, 25, 1)
    hp.AddRange("hlayer", 1, 5, 1)

    return hp


def ReadCUP(val_split: float = 0.15, test_split: float = 0.5,seed:int = 10):
    file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
    all_data = readCUP(file_path_cup)
    all_data.Shuffle(seed)
    all_data.PrintData()

    if not USE_KFOLD or MULTY:
        all_data.Split(val_split, test_split)
        #all_data.Standardize(True)

    return all_data, MEE()

def ModelSelection(dataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:

    if USE_KFOLD:
        ModelSelector = BestSearchKFold(HyperBag_Cap(), RandomSearch(NumberOrTrial))
    else:
        ModelSelector = BestSearch(HyperBag_Cap(), RandomSearch(NumberOrTrial))

    watched_metric = "val_loss"
    callback = [EarlyStopping(watched_metric, 10)]

    best_model, best_hpSel = ModelSelector.GetBestModel(
        HyperModel_CAP,
        dataset,
        500,
        watched_metric,
        [BaselineMetric],
        GlorotInitializer(),
        callback)
    #best_model.PlotModel("CUP Model")
    return best_model, best_hpSel




def GenerateTagName():
    tagName=""
    if OPTIMIZER == 1:
        tagName += "_backprop"
    elif OPTIMIZER == 2:
        tagName += "_nasterov"
    else:
        tagName += "_adam"

    return tagName


def  TrainCUPModel(NumberOrTrial:int, NumberOrTrial_mean:int):

    #DataSet Preparation
    SplitCUPDataset, BaselineMetric_MEE = ReadCUP(0.15, 0.20)
    mergedCUPDataset = DataSet.Clone(SplitCUPDataset)
    if not USE_KFOLD:
        mergedCUPDataset.MergeTrainingAndValidation()

    #Experiment parameter
    mode = HyperBag()
    mode.AddChosen("Optimizer",[1,2,3])

    global OPTIMIZER



    gs = GridSearch()
    for modes, _ in gs.Search(mode):
        OPTIMIZER = modes["Optimizer"]

        tagName = GenerateTagName()

        print(f"Run experiment with the following settings: {tagName}")

        best_model, best_hpSel = ModelSelection(SplitCUPDataset, BaselineMetric_MEE, NumberOrTrial)
        best_model:ModelFeedForward
        print(f"Best hp : {best_hpSel}")
        #best_model.PlotModel("CUP Model")

        #best_model.SaveMetricsResults(f"Data/Results/Cup{tagName}.mres")
        best_model.SaveModel(f"{CUPMODELPATH}",f"CUP{tagName}.vjf")

        #GeneratePlot(BaselineMetric_MEE, best_model.MetricResults, SplitCUPDataset, tagName)


        MetricToCheck = [key for key, _ in best_model.MetricResults.items() if not key.startswith("val_")]
        totalResult = ValidateSelectedModel(
            HyperModel_CAP,best_hpSel,
            NumberOrTrial_mean, MetricToCheck,
            BaselineMetric_MEE,
            SplitCUPDataset.Test,mergedCUPDataset.Training,
            None,
            500,50,42 )
        SaveJson(f"{CUPRESULTSPATH}", f"res_CUP{tagName}.json", totalResult)

        #PlotMultipleModels(totalResult["metrics"],"test_loss",f"{CUPPLOTPATH}",f"mean_CUP{tagName}.png" )



def GeneratePlot(BaselineMetric_MEE, MetricResults,CupDataset, extraname:str= ""):
    BaselineMetric_MSE = MSE()

    lin_model = LinearRegression()
    lin_model.fit(CupDataset.Training.Data, CupDataset.Training.Label)

    predictions = lin_model.predict(CupDataset.Test.Data)
    baseline_MSE = BaselineMetric_MSE.ComputeMetric(predictions, CupDataset.Test.Label)
    baseline_MEE = BaselineMetric_MEE.ComputeMetric(predictions, CupDataset.Test.Label)

    warm_up_epochs = 3
    metric_to_plot_loss = {key: value[warm_up_epochs:] for key, value in MetricResults.items() if key.endswith("loss")}
    metric_to_plot_MEE = {key: value[warm_up_epochs:] for key, value in MetricResults.items() if key.endswith(BaselineMetric_MEE.Name)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    PlotMetrics(
        metricDic=metric_to_plot_loss,
        baseline=baseline_MSE,
        baselineName=f"Baseline ({BaselineMetric_MSE.Name})",
        limitYRange=None,
        title=f"CUP results {BaselineMetric_MSE.Name}",
        xlabel="Epochs",
        ylabel="",
        subplotAxes=axes[0])

    PlotMetrics(
        metricDic=metric_to_plot_MEE,
        baseline=baseline_MEE,
        baselineName=f"Baseline ({BaselineMetric_MEE.Name})",
        limitYRange=None,
        title=f"CUP results {BaselineMetric_MEE.Name}",
        xlabel="Epochs",
        ylabel="",
        subplotAxes=axes[1])
    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{CUPPLOTPATH}", f"Loss(MSE)-MEE{extraname}")
    plt.close(fig)


if __name__ == '__main__':
        TrainCUPModel(250 ,250)
