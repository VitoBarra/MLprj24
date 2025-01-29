import csv

from sklearn.linear_model import LinearRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.DataSet.DataExamples import DataExamples
from Core.FeedForwardModel import *
from Core.Initializer.WeightInitializer import GlorotInitializer
from Core.LossFunction import MSELoss
from Core.Metric import *
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Core.Tuner.ModelSelection import BestSearch, BestSearchKFold
from Model.ModelPlots import *
from Model.TrainingFuction import AssessmentSelectedModel
from Utility.PlotUtil import *
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


    if OPTIMIZER_CUP == 1:
        optimizer = BackPropagation(loss,BATCH_SIZE, hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
    elif OPTIMIZER_CUP == 2:
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
    if OPTIMIZER_CUP == 3:
        # Optimizer
        hp.AddRange("beta", 0.99 ,0.99, 0.01) # fixed to ADAM default 0.99
        hp.AddRange("epsilon", 1e-8, 1e-8, 1e-8) # fixed to ADAM default
        hp.AddRange("alpha", 0.9, 0.9, 0.1) # fixed to ADAM default
        if BATCH_SIZE == 1:
            hp.AddRange("eta", 0.0001, 0.005, 0.0005)
            hp.AddRange("lambda", 0, 1e-7, 1e-8)
            hp.AddRange("decay", 0.001, 0.01, 0.001)
        elif BATCH_SIZE == -1:
            hp.AddRange("eta", 0.02, 0.05, 0.005)
            hp.AddRange("lambda", 0.0004, 0.0009, 0.0001)
            hp.AddRange("decay", 0.001, 0.008, 0.001)
            """hp.AddRange("eta", 0.03, 0.03, 0.01)
            hp.AddRange("lambda", 0.0008, 0.0008, 0.0001)
            hp.AddRange("decay", 0.005, 0.005, 0.001)"""

        elif BATCH_SIZE == 64:
            #hp.AddRange("eta", 0.03, 0.05, 0.005)
            #hp.AddRange("lambda", 0.0001, 0.0004, 0.0001)
            #hp.AddRange("decay", 0.005, 0.011, 0.002)
            hp.AddRange("eta", 0.005, 0.03, 0.001)
            hp.AddRange("lambda", 0.0003, 0.0008, 0.0001)
            hp.AddRange("decay", 0.0001, 0.0011, 0.0002)

        else: #BATCH_SIZE == 128
            # Optimizer
            hp.AddRange("eta", 0.03, 0.5, 0.001)
            hp.AddRange("lambda", 0.0003, 0.0008, 0.0001)
            hp.AddRange("decay", 0.0001, 0.0011, 0.0002)
            """hp.AddRange("eta", 0.032, 0.032, 0.01)
            hp.AddRange("lambda", 0.0003, 0.0003, 0.0001)
            hp.AddRange("decay", 0.0007, 0.0007, 0.0001)"""

    #architecture
    # hp.AddRange("drop_out", 0.1, 0.5, 0.05)
    hp.AddChosen("UseBiasIN",[True,False])
    hp.AddChosen("UseBias",[True,False])
    hp.AddRange("unit", 15, 30, 1)
    hp.AddRange("hlayer", 1, 2, 1)
    hp.AddChosen("actFun",[ReLU(), LeakyReLU()])
    return hp


def ReadCUP(val_split: float = VAL_SPLIT_CUP, test_split: float = TEST_SPLIT_CUP,seed:int = DATA_SHUFFLE_SEED_CUP):
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

    if not USE_KFOLD_CUP:
        data.Split(val_split, test_split)
    else:
        data.SetUp_Kfold_TestHoldOut(KFOLD_NUM_CUP,test_split)

    if STANDARDIZE:
        data.Standardize(False)

    dataUnsplit.Training = dataUnsplit.Data

    return data,dataUnsplit, MEE()

def ModelSelection_Cup(dataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:
    """
    Selects the best model using either K-fold cross-validation or a standard search.

    :param dataset: The dataset to train the model on.
    :param BaselineMetric: The metric used for model evaluation.
    :param NumberOrTrial: The number of trials to run.
    :return: The best model and the selected hyperparameters.
    """

    #e = GridSearch()
    e = RandomSearch(NumberOrTrial)
    if USE_KFOLD_CUP:
        ModelSelector = BestSearchKFold(e)
    else:
        ModelSelector = BestSearch(e)

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

    if settings["Optimizer"] == 1:
        tagName += "_BackPropagation"
    elif settings["Optimizer"] == 2:
        tagName += "_Nesterov"
    else:
        tagName += "_ADAM"

    if settings["Batch_size"] == 64:
        tagName += "_BatchSize64"
    elif settings["Batch_size"] == 128:
        tagName += "_BatchSize128"
    elif settings["Batch_size"] == 1:
        tagName += "_Online"
    else:
        tagName += "_BatchSize-1"


    return tagName


def  TrainCUPModel(NumberOrTrial:int, NumberOrTrial_mean:int, optimizer = None, minibatch_size= None):
    """
    Trains the CUP model with the specified number of trials.

    :param minibatch_size: a list of Mini-batch sizes to run.
    :param optimizer: a list of optimizers to run.
    :param NumberOrTrial: Number of trials to run.
    :param NumberOrTrial_mean: The mean number of trials for evaluation.
    """

    #DataSet Preparation
    SplitCUPDataset,CUPDataset, BaselineMetric_MEE = ReadCUP(VAL_SPLIT_CUP, TEST_SPLIT_CUP,DATA_SHUFFLE_SEED_CUP)

    if minibatch_size is None:
        minibatch_size = [-1, 1, 32, 64, 128]
    if  optimizer is None:
        optimizer = [1,2,3]

    #Experiment parameter
    SettingsList = HyperBag()
    SettingsList.AddChosen("Optimizer",optimizer)
    SettingsList.AddChosen("Batch_size", minibatch_size)

    global OPTIMIZER_CUP
    global BATCH_SIZE
    global STANDARDIZE


    gs = GridSearch()
    for settings, _ in gs.Search(SettingsList):
        OPTIMIZER_CUP = settings["Optimizer"]
        BATCH_SIZE = settings["Batch_size"]

        settingDict = {"Optimizer": OPTIMIZER_CUP, "Batch_size": BATCH_SIZE, "kFold" : USE_KFOLD_CUP}
        tagName = GenerateTagNameFromSettings(settingDict)

        print(f"\nRun Cup experiments with the following settings: {tagName}\n")

        best_model, best_hpSel = ModelSelection_Cup(SplitCUPDataset, BaselineMetric_MEE, NumberOrTrial)
        best_model:ModelFeedForward
        print(f"Best hp : {best_hpSel}")
        cupData = DataExamples.Clone(SplitCUPDataset.Training)
        cupData.Concatenate(SplitCUPDataset.Validation)
        cupData.Concatenate(SplitCUPDataset.Test)
        totalResult = AssessmentSelectedModel(
            HyperModel_CUP,best_hpSel,
            NumberOrTrial_mean,
            BaselineMetric_MEE,
            cupData,VAL_SPLIT_CUP,TEST_SPLIT_CUP,None,
            500,150,42 )
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
        limitYRange=(0,1),
        title=f"CUP results {BaselineMetric_MSE.Name}",
        xlabel="Epochs",
        ylabel=BaselineMetric_MSE.Name,
        subplotAxes=axes[0])

    PlotMetrics(
        metricDic=metric_to_plot_MEE,
        baseline=baseline_MEE,
        baselineName=f"Baseline ({BaselineMetric_MEE.Name})",
        limitYRange=(0,1),
        title=f"CUP results {BaselineMetric_MEE.Name}",
        xlabel="Epochs",
        ylabel=BaselineMetric_MEE.Name,
        subplotAxes=axes[1])
    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{CUP_PLOT_PATH}", f"Loss(MSE)-MEE{extraname}")
    plt.close(fig)

def GeneratePlotAverage_ForCUP(Results: list[dict],
                               BaselineMetric_MEE, CupDataset, extra_name: str = ""):
    """
    Generate and save a plot for the CUP dataset based on the mean of individual trials. This function creates
    two subplots: one for the mean squared error (MSE) and another for the mean Euclidean error (MEE).

    :param Results: List of dictionaries. Each dictionary represents an individual trial and contains metrics
                    (e.g., loss, MEE, etc.) with their respective values over multiple epochs.
    :param BaselineMetric_MEE: A callable object representing the baseline metric calculation for MEE.
    :param CupDataset: A dataset object that contains training and testing data and labels.
    :param extra_name: Optional string that will be appended to the filename when saving the plot.
                       Default is an empty string.
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
                limitYRange=(0,1), WarmUpEpochs=warm_up_epochs, baseline=baseline_MSE,
                baselineName=f"Baseline ({BaselineMetric_MSE.Name})", subplotAxes=axes[0])

    PlotAverage(metricList=metric_to_plot_MEE, title=f"CUP MEE", xlabel="Epochs", ylabel="MEE", limitYRange=(0.4, 1.5),
                WarmUpEpochs=warm_up_epochs, baseline=baseline_MEE,
                baselineName=f"Baseline ({BaselineMetric_MSE.Name})", subplotAxes=axes[1])

    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(CUP_PLOT_PATH, f"CUP_MEAN{extra_name}")



def GenerateAllPlot_CUP():
    CreateDir(CUP_RESULTS_PATH)



    SplitCupDataset, all_data,  BaselineMetric_MEE = ReadCUP(VAL_SPLIT_CUP, TEST_SPLIT_CUP,DATA_SHUFFLE_SEED_CUP)

    jsonFiles = GetAllFileInDir(CUP_RESULTS_PATH)
    for jsonFile in jsonFiles:
        data = readJson(jsonFile)
        GeneratePlotAverage_ForCUP(
            CupDataset = SplitCupDataset,
            Results=data['metrics'],
            extra_name=GenerateTagNameFromSettings(data['settings']),
            BaselineMetric_MEE=BaselineMetric_MEE,

        )


    models = GetAllFileInDir(CUP_MODEL_PATH)
    for model in models:
        model_name = os.path.splitext(model.name)[0]
        best_model = ModelFeedForward()
        best_model.LoadModel(CUP_MODEL_PATH,model_name)


        labels = all_data.Data.Label
        test = readCUPTest(DATASET_PATH_CUP_TS)
        if STANDARDIZE:
            test.Data.Standardize(False,SplitCupDataset.StandardizationStat[0])
        result = best_model.Predict(test.Data.Data)

        WriteToCSV(result, CUP_CSV_RESULTS_PATH,f"{model_name}.csv")

        plot_CUP_3d(
            labels,  # First dataset: actual labels
            result,  # Second dataset: predictions
            path=f"{CUP_PLOT_PATH}/3D",
            title=f"{model_name}",
            labels=["Training Labels", "Predictions"],  # Legend labels
            colors=["blue", "red"],  # Colors for each dataset
            markers=["o", "^"]  # Markers for each dataset
        )




def WriteToCSV(data, path,fileName):
    CreateDir(path)

    # Open a CSV file to write
    with open(f"{path}/{fileName}", mode='w', newline='') as file:
        writer = csv.writer(file)


        writer.writerow(['Vito Barra ','Jessica Ferrari ','Francesco Romeo'])
        writer.writerow(['LMSTeam'])
        writer.writerow(['ML-CUP24 v1'])
        writer.writerow(['29/01/2025'])

        # Write each row with an auto-incrementing ID
        for idx, row in enumerate(data, start=1):
            writer.writerow([idx] + list(row))