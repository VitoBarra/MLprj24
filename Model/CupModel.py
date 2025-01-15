from sklearn.linear_model import LinearRegression
from Core.Callback.EarlyStopping import EarlyStopping
from Core.DataSet.DataExamples import DataExamples
from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Metric import *
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Model import CUPMODELPATH, CUPPLOTPATH, CUPRESULTSPATH
from Model.ModelResults import PlotMultipleModels, PlotTableVarianceAndMean
from Utility.PlotUtil import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss
from Core.Tuner.ModelSelection import BestSearch, BestSearchKFold, ModelSelection
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.WeightInitializer import GlorotInitializer
from dataset.ReadDatasetUtil import readCUP
import random
import gc
from statistics import mean, variance

USE_KFOLD = False
OPTIMIZER = None
BATCH_SIZE = None


def HyperModel_CAP(hp: HyperBag):
    model = ModelFeedForward()

    act_fun = TanH()

    model.AddLayer(Layer(12, Linear(), False, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"], act_fun, hp["drop_out"], True, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], act_fun, True, f"_h{i}"))

    model.AddLayer(Layer(3, Linear(), False, "output"))

    loss = MSELoss()


    if OPTIMIZER == 1:
        optimizer = BackPropagation(loss, hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    elif OPTIMIZER == 2:
        optimizer = BackPropagationNesterovMomentum(loss, hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    else:
        optimizer = Adam(loss,hp["eta"], hp["labda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

    return model, optimizer


def HyperBag_Cap():
    hp = HyperBag()

    hp.AddRange("eta", 0.001, 0.1, 0.01)
    hp.AddRange("labda", 0.001, 0.1, 0.001)
    hp.AddRange("alpha", 0.1, 0.9, 0.1)
    hp.AddRange("decay", 0.001, 0.1, 0.001)

    ## Only Adam
    if OPTIMIZER>2:
        hp.AddRange("beta", 0.95, 0.99, 0.01)
        hp.AddRange("epsilon", 1e-13, 1e-8, 1e-1)

    #hp.AddRange("drop_out", 0.1, 0.5, 0.05)

    hp.AddRange("unit", 1, 25, 1)
    hp.AddRange("hlayer", 1, 5, 1)

    return hp


def ReadCUP(val_split: float = 0.15, test_split: float = 0.5,seed:int = 10):
    file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
    all_data = readCUP(file_path_cup)
    all_data.Shuffle(seed)
    #all_data.PrintData()

    if not USE_KFOLD or MULTY:
        all_data.Split(val_split, test_split)
        #all_data.Standardize(True)

    return all_data, MEE()

def ModelSelection(dataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int, minibatchSize : int = 64) -> tuple[ModelFeedForward, HyperBag]:
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
        minibatchSize,
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

    if BATCH_SIZE is None:
        batchString = "batch"
    elif BATCH_SIZE == 1:
        batchString = "Online"
    else:
        batchString = f"b{BATCH_SIZE}"

    tagName += f"_{batchString}"
    return tagName


def  TrainCUPModel(NumberOrTrial:int, NumberOrTrial_mean:int):

    #DataSet Preparation
    SplittedCupDataset, BaselineMetric_MEE = ReadCUP(0.15, 0.20)
    mergedDataset = DataSet.Clone(SplittedCupDataset)
    if not USE_KFOLD:
        mergedDataset.MergeTrainingAndValidation()

    #Experiment parameter
    mode = HyperBag()
    mode.AddChosen("Optimizer",[1,2,3])
    mode.AddChosen("Adam", [True, False])
    mode.AddChosen("BatchSize",[1,64,128,160,None])

    global OPTIMIZER
    global BATCH_SIZE



    gs = GridSearch()
    for modes, _ in gs.Search(mode):
        OPTIMIZER = modes["Optimizer"]
        BATCH_SIZE = modes["BatchSize"]

        tagName = GenerateTagName()

        print(f"Run experiment with the following settings: {tagName}")

        best_model, best_hpSel = ModelSelection(SplittedCupDataset, BaselineMetric_MEE, NumberOrTrial, BATCH_SIZE)
        best_model:ModelFeedForward
        print(f"Best hp : {best_hpSel}")
        #best_model.PlotModel("CUP Model")

        #best_model.SaveMetricsResults(f"Data/Results/Cup{tagName}.mres")
        best_model.SaveModel(f"{CUPMODELPATH}",f"CUP{tagName}.vjf")

        GeneratePlot(BaselineMetric_MEE,best_model.MetricResults,SplittedCupDataset , tagName)

        totalResult = {"metrics": [], "HP": best_hpSel.hpDic}
        res = { key: [] for key, _ in best_model.MetricResults.items() if not key.startswith("val_") }

        tempDataset:DataSet = DataSet()
        tempDataset.Test = SplittedCupDataset.Test

        random.seed(42)
        seedList = [random.randint(0, 1000) for _ in range(NumberOrTrial_mean)]
        for i,seed in zip(range(NumberOrTrial_mean),seedList):
            training:DataExamples = DataExamples.Clone(mergedDataset.Training)
            training.Shuffle(seed)
            tempDataset.Training = training
            print(f"Training Model {i + 1}/{NumberOrTrial_mean}...")

            model, optimizer = HyperModel_CAP(best_hpSel)
            model.Build(GlorotInitializer())
            model.AddMetric(BaselineMetric_MEE)
            callbacks = [EarlyStopping("loss", 10, 0.0001)]
            model.Fit(optimizer, tempDataset, 500, BATCH_SIZE, callbacks)
            totalResult["metrics"].append(model.MetricResults)


            for key, value in model.MetricResults.items():
                res[key].append(value[-1])

            print(f"training model {i + 1} / {NumberOrTrial_mean} " + " | ".join(
                f"{key}:{value[-1]:.4f}" for key, value in res.items()))

        totalResult["MetricStat"] = {key: [mean(value),variance(value)] for key, value in res.items()}
        PlotMultipleModels(totalResult["metrics"],"test_loss",f"{CUPPLOTPATH}",f"mean_CUP{tagName}.png" )
        SaveJson(f"{CUPRESULTSPATH}", f"res_CUP{tagName}.json", totalResult)

        SaveJson(f"Data/FinalModel/CUP", f"res_CUP{tagName}.json", res)
        gc.collect()


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
    plot_metrics(
        metricDic=metric_to_plot_loss,
        baseline=baseline_MSE,
        baselineName=f"Baseline ({BaselineMetric_MSE.Name})",
        limitYRange=None,
        title=f"CUP results {BaselineMetric_MSE.Name}",
        xlabel="Epochs",
        ylabel="",
        subplotAxes=axes[0])

    plot_metrics(
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
        TrainCUPModel(150, 25)
