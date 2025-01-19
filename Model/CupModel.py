from sklearn.linear_model import LinearRegression
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Metric import *
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Model import *
from Model.ModelPlots import *
from Model.TrainingFuction import ValidateSelectedModel
from Utility.PlotUtil import *
from Core.LossFunction import MSELoss
from Core.Tuner.ModelSelection import BestSearch, BestSearchKFold, ModelSelection
from Core.Tuner.HyperModel import HyperModel
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.Initializer.WeightInitializer import GlorotInitializer
from dataset.ReadDatasetUtil import readCUP
import random
import gc
from statistics import mean, variance


USE_KFOLD = True
OPTIMIZER = 1


def HyperModel_CAP(hp: HyperBag):
    """
    Builds a feedforward neural network model based on the provided hyperparameters.

    :param hp: A HyperBag instance containing the hyperparameters.
    :return: The constructed model and the selected optimizer.
    """
    model = ModelFeedForward()


    model.AddLayer(Layer(12, Linear(), False, "input"))
    for i in range(hp["hlayer"]):
            model.AddLayer(Layer(hp["unit"], hp["actFun"], hp["UseBias"], f"_h{i}"))

    model.AddLayer(Layer(3, Linear(), False, "output"))

    loss = MSELoss()


    if OPTIMIZER == 1:
        optimizer = BackPropagation(loss,hp["batchSize"], hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
    elif OPTIMIZER == 2:
        optimizer = BackPropagationNesterovMomentum(loss,hp["batchSize"], hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
    else:
        optimizer = Adam(loss,hp["batchSize"],hp["eta"], hp["lambda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

    return model, optimizer


def HyperBag_Cap():
    """
    Defines the hyperparameter search space for the model.

    :return: A HyperBag instance containing the hyperparameter search space.
    """
    hp = HyperBag()
    
    # Optimizer
    hp.AddChosen("batchSize",[-1,1,64,128,160])
    hp.AddRange("eta", 0.001, 0.1, 0.01)
    #hp.AddRange("lambda", 0.001, 0.1, 0.001)
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
    hp.AddRange("unit", 1, 25, 1)
    hp.AddRange("hlayer", 1, 5, 1)
    hp.AddChosen("actFun",[Sigmoid(),TanH(),ReLU(),LeakyReLU()])

    return hp


def ReadCUP(val_split: float = 0.15, test_split: float = 0.5,seed:int = 10):
    """
    Reads the CUP dataset, shuffles it, and splits it into training, validation, and test sets.

    :param val_split: Proportion of data to use for validation.
    :param test_split: Proportion of data to use for testing.
    :param seed: Random seed for shuffling.
    :return: The dataset split and a metric for evaluation.
    """
    file_path_cup = "dataset/CUP/ML-CUP24-TR.csv"
    all_data = readCUP(file_path_cup)
    all_data.Shuffle(seed)
    all_data.PrintData()

    if not USE_KFOLD or MULTY:
        all_data.Split(val_split, test_split)
        #all_data.Standardize(True)
    else:
        all_data.SetUp_Kfold_TestHoldOut(5,test_split)

    return all_data, MEE()

def ModelSelection(dataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:
    """
    Selects the best model using either K-fold cross-validation or a standard search.

    :param dataset: The dataset to train the model on.
    :param BaselineMetric: The metric used for model evaluation.
    :param NumberOrTrial: The number of trials to run.
    :return: The best model and the selected hyperparameters.
    """

    if USE_KFOLD:
        ModelSelector = BestSearchKFold(RandomSearch(NumberOrTrial))
    else:
        ModelSelector = BestSearch(RandomSearch(NumberOrTrial))

    watched_metric = "val_loss"
    callback = [EarlyStopping(watched_metric, 10)]

    best_model, best_hpSel = ModelSelector.GetBestModel(
        HyperModel_CAP, HyperBag_Cap(),
        dataset,
        500,
        watched_metric,
        [BaselineMetric],
        GlorotInitializer(),
        callback)
    #best_model.PlotModel("CUP Model")
    return best_model, best_hpSel



def GenerateTagNameFromSettings(settings):
    """
    Generates a tag name based on the selected optimizer.

    :return: A string tag that represents the chosen optimizer.
    """
    tagName=""

    if settings["optimizer"] == 1:
        tagName += "_BackPropagation"
    elif settings["optimizer"] == 2:
        tagName += "_Nesterov"
    else:
        tagName += "_adam"

    return tagName


def  TrainCUPModel(NumberOrTrial:int, NumberOrTrial_mean:int):
    """
    Trains the CUP model with the specified number of trials.

    :param NumberOrTrial: Number of trials to run.
    :param NumberOrTrial_mean: The mean number of trials for evaluation.
    """

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

        settingDict = {"optimizer": OPTIMIZER}
        tagName = GenerateTagNameFromSettings(settingDict)

        print(f"Run experiment with the following settings: {tagName}")

        best_model, best_hpSel = ModelSelection(SplitCUPDataset, BaselineMetric_MEE, NumberOrTrial)
        best_model:ModelFeedForward
        print(f"Best hp : {best_hpSel}")
        #best_model.PlotModel("CUP Model")

        #best_model.SaveMetricsResults(f"Data/Results/Cup{tagName}.mres")
        best_model.SaveModel(f"{CUP_MODEL_PATH}",f"CUP{tagName}.vjf")

        #GeneratePlot(BaselineMetric_MEE, best_model.MetricResults, SplitCUPDataset, tagName)


        MetricToCheck = [key for key, _ in best_model.MetricResults.items() if not key.startswith("val_")]
        totalResult = ValidateSelectedModel(
            HyperModel_CAP,best_hpSel,
            NumberOrTrial_mean, MetricToCheck,
            BaselineMetric_MEE,
            SplitCUPDataset.Test,mergedCUPDataset.Training,
            500,50,42 )
        totalResult["settings"] = settingDict
        SaveJson(f"{CUP_RESULTS_PATH}", f"res_CUP{tagName}.json", totalResult)

        #PlotMultipleModels(totalResult["metrics"],"test_loss",f"{CUP_PLOT_PATH}",f"mean_CUP{tagName}.png" )



def GeneratePlot_ForCUP(BaselineMetric_MEE, MetricResults, CupDataset, extraname:str= ""):
    """
    Generates plots to visualize the model's performance (loss and MEE).

    :param BaselineMetric_MEE: The baseline metric for evaluation.
    :param MetricResults: The model's metric results.
    :param CupDataset: The dataset used for testing and evaluation.
    :param extraname: Optional extra name to append to the plot file.
    """
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
    ShowOrSavePlot(f"{CUP_PLOT_PATH}", f"Loss(MSE)-MEE{extraname}")
    plt.close(fig)

def GeneratePlotAverage_ForCUP(Results: list[dict], Metrics: list[str], path=f"{CUP_PLOT_PATH}", name: str = f"CUP_MEAN", tag: str = ""):
    """
    Generate plot for CUP using the plot for the mean of individual trials.

    :param Results: List of dictionaries. Each dictionary represents an individual trial and contains metrics with respective values.
    :param Metrics: List of metrics to plot.
    :param path: The path where the plot will be saved.
    :param name: The name of the file for the plot.
    :param tag: Extra information for the file name.
    """
    # Organize data by metric name
    loss_metrics = {metric: [] for metric in Metrics if metric.endswith("loss")}
    mee_metrics = {metric: [] for metric in Metrics if metric.endswith("MEE")}

    warm_up_epochs = 5
    for trial in Results:
        for metric in Metrics:
            if metric.endswith("loss") and metric in trial:
                loss_metrics[metric].append(trial[metric][warm_up_epochs:])
            elif metric.endswith("MEE") and metric in trial:
                mee_metrics[metric].append(trial[metric][warm_up_epochs:])

    # Plot the metrics
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    PlotAverage(
        metricDict=loss_metrics,
        limitYRange=None,
        WarmUpEpochs=warm_up_epochs,
        title=f"CUP Loss",
        xlabel="Epochs",
        ylabel="Loss",
        subplotAxes=axes[0],
    )

    PlotAverage(
        metricDict=mee_metrics,
        limitYRange=None,
        WarmUpEpochs=warm_up_epochs,
        title=f"CUP MEE",
        xlabel="Epochs",
        ylabel="Accuracy",
        subplotAxes=axes[1]
    )

    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{path}", f"{name}{tag}")
    plt.close(fig)

if __name__ == '__main__':
        #TrainCUPModel(300,50)

        
        jsonFiles = GetAllFileInDir(f"{CUP_RESULTS_PATH}")
        for jsonFile in jsonFiles:
            data = readJson(jsonFile)
            GeneratePlotAverage_ForCUP(
                Results=data['metrics'],
                Metrics=["test_loss", "loss", "test_MEE", "MEE"],
                path=f"{CUP_PLOT_PATH}",
                tag=GenerateTagNameFromSettings(data['settings'])
            )