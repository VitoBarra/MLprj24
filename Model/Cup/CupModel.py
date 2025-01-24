from sklearn.linear_model import LinearRegression
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.ActivationFunction import *
from Core.Metric import *
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Model.ModelPlots import *
from Model.TrainingFuction import AsesSelectedModel
from Utility.PlotUtil import *
from Core.LossFunction import MSELoss
from Core.Tuner.ModelSelection import BestSearch, BestSearchKFold, ModelSelection
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.Initializer.WeightInitializer import GlorotInitializer
from dataset.ReadDatasetUtil import readCUP, readCUPTest
from . import *



def HyperModel_CUP(hp: HyperBag):
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
        optimizer = BackPropagation(loss,BATCH_SIZE, hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
    elif OPTIMIZER == 2:
        optimizer = BackPropagationNesterovMomentum(loss,BATCH_SIZE, hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
    else:
        optimizer = Adam(loss,BATCH_SIZE,hp["eta"], hp["lambda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

    return model, optimizer


def HyperBag_CUP():
    """
    Defines the hyperparameter search space for the model.

    :return: A HyperBag instance containing the hyperparameter search space.
    """
    hp = HyperBag()
    
    # Optimizer
    if BATCH_SIZE == 1:
        hp.AddRange("eta", 0.0001, 0.005, 0.0005)
        hp.AddRange("lambda", 1e-8, 1e-7, 1e-6)
        hp.AddRange("alpha", 0.1, 0.9, 0.1)
        hp.AddRange("decay", 0.001, 0.01, 0.01)
    elif BATCH_SIZE == -1:
        hp.AddRange("eta", 0.001, 0.1, 0.001)
        hp.AddRange("lambda", 0.0001, 0.001, 0.001)
        hp.AddRange("alpha", 0.1, 0.9, 0.1)
        hp.AddRange("decay", 0.001, 0.01, 0.001)
    elif BATCH_SIZE == 64:
        hp.AddRange("eta", 0.001, 0.1, 0.001)
        hp.AddRange("lambda", 0.0001, 0.001, 0.001)
        hp.AddRange("alpha", 0.1, 0.9, 0.1)
        hp.AddRange("decay", 0.001, 0.01, 0.001)
    else: #BATCH_SIZE == 128
        if OPTIMIZER == 3:
            hp.AddRange("eta", 0.01, 0.05, 0.01)
            hp.AddRange("lambda", 0.0001, 0.001, 0.01)
            hp.AddRange("alpha", 0.1, 0.9, 0.1)
            hp.AddRange("decay", 0.01, 0.1, 0.01)
        else:
            hp.AddRange("eta", 0.001, 0.1, 0.001)
            hp.AddRange("lambda", 0.0001, 0.001, 0.001)
            hp.AddRange("alpha", 0.1, 0.9, 0.1)
            hp.AddRange("decay", 0.001, 0.01, 0.001)


    # Only Adam
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

    data = readCUP(DATASET_PATH_CUP_TR)
    data.Shuffle(seed)
    data.PrintData()
    dataUnsplit = DataSet.Clone(data)
    if not USE_KFOLD:
        data.Split(val_split, test_split)
        #all_data.Standardize(True)
    else:
        data.SetUp_Kfold_TestHoldOut(5,test_split)

    dataUnsplit.Training = dataUnsplit._Data

    return data,dataUnsplit, MEE()

def ModelSelection_Cup(dataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:
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
    callback = [EarlyStopping(watched_metric, 40)]

    best_model, best_hpSel = ModelSelector.GetBestModel(
        HyperModel_CUP, HyperBag_CUP(),
        dataset,
        500,
        watched_metric,
        [BaselineMetric],
        GlorotInitializer(),
        callback)
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
        tagName += "_ADAM"

    if settings["batch_size"] == 64:
        tagName += "_BatchSize64"
    elif settings["batch_size"] == 128:
        tagName += "_BatchSize128"
    elif settings["batch_size"] == 1:
        tagName += "_Online"
    else:
        tagName += "_BatchSize-1"


    return tagName


def  TrainCUPModel(NumberOrTrial:int, NumberOrTrial_mean:int):
    """
    Trains the CUP model with the specified number of trials.

    :param NumberOrTrial: Number of trials to run.
    :param NumberOrTrial_mean: The mean number of trials for evaluation.
    """

    #DataSet Preparation
    SplitCUPDataset,CUPDataset, BaselineMetric_MEE = ReadCUP(0.15, 0.20)
    mergedCUPDataset = DataSet.Clone(SplitCUPDataset)


    if not USE_KFOLD:
        mergedCUPDataset.MergeTrainingAndValidation()


    #Experiment parameter
    mode = HyperBag()
    mode.AddChosen("Optimizer",[1,2,3])
    mode.AddChosen("Batch_size", [-1,1,64,128])

    global OPTIMIZER
    global BATCH_SIZE


    gs = GridSearch()
    for modes, _ in gs.Search(mode):
        OPTIMIZER = modes["Optimizer"]
        BATCH_SIZE = modes["Batch_size"]

        settingDict = {"optimizer": OPTIMIZER, "batch_size": BATCH_SIZE}
        tagName = GenerateTagNameFromSettings(settingDict)

        print(f"Run experiment with the following settings: {tagName}")

        best_model, best_hpSel = ModelSelection_Cup(SplitCUPDataset, BaselineMetric_MEE, NumberOrTrial)
        best_model:ModelFeedForward
        print(f"Best hp : {best_hpSel}")
        #best_model.PlotModel("CUP Model")


        totalResult = AsesSelectedModel(
            HyperModel_CUP,best_hpSel,
            NumberOrTrial_mean,
            BaselineMetric_MEE,
            SplitCUPDataset.Test,mergedCUPDataset.Training,
            500,50,42 )
        totalResult["settings"] = settingDict
        SaveJson(f"{CUP_RESULTS_PATH}", f"res_CUP{tagName}.json", totalResult)


        final_model ,optimizer= HyperModel_CUP(best_hpSel)
        final_model.Build(GlorotInitializer(42))
        final_model.AddMetric(BaselineMetric_MEE)
        final_model.Fit( optimizer,CUPDataset, 500)
        final_model.SaveModel(f"{CUP_MODEL_PATH}",tagName)





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

def GeneratePlotAverage_ForCUP(Results: list[dict],
                               BaselineMetric_MEE, CupDataset, extra_name: str = ""):
    """
    Generate plot for CUP using the plot for the mean of individual trials.

    :param Results: List of dictionaries. Each dictionary represents an individual trial and contains metrics with respective values.
    :param Metrics: List of metrics to plot.
    :param path: The path where the plot will be saved.
    :param name: The name of the file for the plot.
    :param extra_name: Extra information for the file name.
    """
    BaselineMetric_MSE = MSE()

    lin_model = LinearRegression()
    lin_model.fit(CupDataset.Training.Data, CupDataset.Training.Label)

    predictions = lin_model.predict(CupDataset.Test.Data)
    baseline_MSE = BaselineMetric_MSE(predictions, CupDataset.Test.Label)
    baseline_MEE = BaselineMetric_MEE(predictions, CupDataset.Test.Label)


    # Organize data by metric name
    warm_up_epochs = 5
    metric_to_plot_loss =[]
    metric_to_plot_MEE = []
    # Organize data by metric name
    for trialsRes in Results:
        metric_to_plot_loss.append({key: value for key, value in  trialsRes.items() if key.endswith("loss")})
        metric_to_plot_MEE.append({key: value for key, value in trialsRes.items() if key.endswith(BaselineMetric_MEE.Name)})

    # Plot the metrics
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    PlotAverage(metricList=metric_to_plot_loss, title=f"CUP Loss (MSE)", xlabel="Epochs", ylabel="MSE",
                limitYRange=None, WarmUpEpochs=warm_up_epochs, baseline=baseline_MSE,
                baselineName=f"Baseline ({BaselineMetric_MSE.Name})", subplotAxes=axes[0])

    PlotAverage(metricList=metric_to_plot_MEE, title=f"CUP MEE", xlabel="Epochs", ylabel="MEE", limitYRange=None,
                WarmUpEpochs=warm_up_epochs, baseline=baseline_MEE,
                baselineName=f"Baseline ({BaselineMetric_MSE.Name})", subplotAxes=axes[1])

    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(CUP_PLOT_PATH, f"CUP_MEAN{extra_name}")



def GenerateAllPlot_CUP():
    CreateDir(CUP_RESULTS_PATH)


    SplitCupDataset, all_data,  BaselineMetric_MEE = ReadCUP(0.15, 0.20)

    jsonFiles = GetAllFileInDir(CUP_RESULTS_PATH)
    for jsonFile in jsonFiles:
        data = readJson(jsonFile)
        GeneratePlotAverage_ForCUP(
            CupDataset = SplitCupDataset,
            Results=data['metrics'],
            extra_name=GenerateTagNameFromSettings(data['settings']),
            BaselineMetric_MEE=BaselineMetric_MEE,

        )


    best_model: ModelFeedForward



    best_model = ModelFeedForward()
    best_model.LoadModel(CUP_MODEL_PATH,"_adam")


    labels= all_data._Data.Label
    test = readCUPTest(DATASET_PATH_CUP_TS)
    result = best_model.Predict(test._Data.Data)
    plot_CAP_3d(
        labels,  # First dataset: actual labels
        result,  # Second dataset: predictions
        labels=["Training Labels", "Predictions"],  # Legend labels
        colors=["blue", "red"],  # Colors for each dataset
        markers=["o", "^"]  # Markers for each dataset
    )