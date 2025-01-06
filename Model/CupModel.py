from sklearn.linear_model import LinearRegression
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Metric import *
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
from statistics import mean, variance

USE_KFOLD = False
USE_ADAM = False
MULTY = False


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
    if USE_ADAM:
        optimizer = Adam(loss, hp["eta"], hp["labda"], hp["alpha"], hp["beta"], hp["epsilon"], hp["decay"])
    else:
        optimizer = BackPropagation(loss, hp["eta"], hp["labda"], hp["alpha"], hp["decay"])

    return model, optimizer


def HyperBag_Cap():
    hp = HyperBag()

    hp.AddRange("eta", 0.001, 0.1, 0.01)
    hp.AddRange("labda", 0.001, 0.1, 0.001)
    hp.AddRange("alpha", 0.1, 0.9, 0.1)
    hp.AddRange("decay", 0.001, 0.1, 0.001)

    ## Only Adam
    if USE_ADAM:
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



def TrainMultipleModels(num_models: int = 5,NumberOrTrial:int=50) -> None:
    """
    Train multiple models and evaluate their performance.

    :param num_models: Number of models to train.
    """
    CupDataset, BaselineMetric = ReadCUP(0.15, 0.20)
    results = {}

    print("Performing initial Random Search for best hyperparameters...")


    best_model, best_hpSel = ModelSelection(CupDataset,BaselineMetric,NumberOrTrial,64)


    print(f"Best hyperparameters found: {best_hpSel}")

    SavedTraining = DataExamples.Clone(CupDataset.Training)
    if not USE_KFOLD:
        SavedTraining.Concatenate(CupDataset.Validation)
        CupDataset.Validation=None

    seedList = [random.randint(0, 1000) for _ in range(num_models)]
    for i,seed in zip(range(num_models),seedList):
        training:DataExamples = DataExamples.Clone(SavedTraining)
        training.Shuffle(seed)
        CupDataset.Training = training
        #monkDataset.Training.CutData(-55)
        print(f"Training Model {i + 1}/{num_models}...")

        model, optimizer = HyperModel_CAP(best_hpSel)
        model.Build(GlorotInitializer())
        model.AddMetric(BaselineMetric)
        model.Fit( optimizer,CupDataset, 250, 64,)

        model_name = f"CUP_Model_{i}"
        model.SaveModel(f"Data/Models/{model_name}.vjf")
        model.SaveMetricsResults(f"Data/Results/{model_name}.mres")

        # Save metrics
        results[model_name] = {
            "hyperparameters": best_hpSel,
            "metrics": model.MetricResults
        }
    PlotMultipleModels(results,"test_loss")
    PlotTableVarianceAndMean(results)


def GenerateTagName():
    tagName=""
    if USE_ADAM:
        tagName += "_Adam"
    else:
        tagName +="_Backprop"
    return tagName


def  TrainCUPModel(NumberOrTrial:int, NumberOrTrial_mean:int, minibatchSize:int = 160):

    mode = HyperBag()
    mode.AddChosen("Adam", [True, False])
    global USE_ADAM
    gs = GridSearch()
    for modes, _ in gs.Search(mode):
        USE_ADAM = modes["Adam"]

        tagName = GenerateTagName()

        CupDataset, BaselineMetric_MEE = ReadCUP(0.15, 0.20)
        best_model, best_hpSel = ModelSelection(CupDataset, BaselineMetric_MEE, NumberOrTrial, minibatchSize)
        best_model:ModelFeedForward
        print(f"Best hp : {best_hpSel}")
        best_model.PlotModel("CUP Model")

        best_model.SaveMetricsResults(f"Data/Results/Cup{tagName}.mres")
        best_model.SaveModel(f"Data/Models/Cup{tagName}.vjf")
        GeneratePlot(BaselineMetric_MEE,best_model.MetricResults,CupDataset , tagName)

        CupDataset.MergeTrainingAndValidation()
        res = {}
        for i in range(NumberOrTrial_mean):
            model, optimizer = HyperModel_CAP(best_hpSel)
            model.Build(GlorotInitializer())
            model.AddMetric(BaselineMetric_MEE)
            calbacks = [EarlyStopping("loss", 10, 0.0001)]
            model.Fit(optimizer, CupDataset, 500, minibatchSize, calbacks)
            for key, value in model.MetricResults.items():
                if key not in res:
                    res[key] = []
                res[key].append(value[-1])
            print(f"training model {i + 1} / {NumberOrTrial_mean} " + " | ".join(
                f"{key}:{value[-1]:.4f}" for key, value in res.items()))

        res = {key: [mean(value),variance(value)] for key, value in res.items()}
        res["HP"] = best_hpSel.hpDic

        SaveJson(f"Data/FinalModel/CUP", f"res_CUP{tagName}.json", res)



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
    ShowOrSavePlot(f"Data/Plots/CUP", f"Loss(MSE)-MEE{extraname}")


if __name__ == '__main__':

    if MULTY:
        TrainMultipleModels(40,50)
    else:
        TrainCUPModel(150, 50,160)
