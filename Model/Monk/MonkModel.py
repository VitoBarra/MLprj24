from sklearn.linear_model import LogisticRegression
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.LossFunction import  CategoricalCrossEntropyLoss
from Core.Metric import MSE
from Core.Tuner.ModelSelection import *
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Model.ModelPlots import  *
from . import *
from Model.Monk.MonkHyperModel import MONKHyperModel
from Model.TrainingFuction import AsesSelectedModel
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readMonk






def ReadMonk(n: int, seed: int = 0) -> DataSet:
    if n <0 or n>3:
        raise Exception("n must be between 0 and 3")

    designSet = readMonk(DATASET_PATH_MONK_TR)
    testSet = readMonk(DATASET_PATH_MONK_TS)
    monkDataset = DataSet.FromDataExample(designSet)
    monkDataset.Test = testSet
    monkDataset.Shuffle(seed)
    return monkDataset








def ModelSelection_MONK( hyperModel:MONKHyperModel ,NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:


    if USE_KFOLD:
        ModelSelector:ModelSelection = BestSearchKFold( RandomSearch(NumberOrTrial))
    else:
        ModelSelector:ModelSelection = BestSearch( RandomSearch(NumberOrTrial))

    watched_metric = "val_loss"
    callBacks = [EarlyStopping(watched_metric, 100,0.0001)]

    best_model, hpSel = ModelSelector.GetBestModel_HyperModel(
        hyperModel,
        800,
        watched_metric,
        None,
        GlorotInitializer(),
        callBacks)
    #best_model.PlotModel(f"MONK Model {MONK_NUM}")
    return best_model, hpSel





def GeneratePlotAverage_ForMonk(Results: list[dict],
                                BaselineMetric_ACC, monkDataset, hpUsed, monk_number):
    """
    Generate plot for MONK using the plot for the mean of individual trials.

    :param Results: List of dictionaries. Each dictionary represents an individual trial and contains metrics with respective values.
    :param path: The path where the plot will be saved.
    :param name: The name of the file for the plot.
    :param tag: Extra information for the file name.
    """
    BaselineMetric_Loss = CategoricalCrossEntropyLoss() if hpUsed["outFun"] == "SoftARGMax" else MSE()

    lin_model = LogisticRegression()
    if  hpUsed["oneHotInput"]:
        lin_model.fit(monkDataset.Training.Data, monkDataset.Training.Label.reshape(-1))
        test_Label = monkDataset.Test.Label
    else:
        lin_model.fit(monkDataset.Training.Data, monkDataset.Training.Label[:, 1])
        test_Label = monkDataset.Test.Label[:, 1]

    predictions = lin_model.predict(monkDataset.Test.Data)

    baseline_loss = BaselineMetric_Loss(predictions.reshape(-1, 1), test_Label)
    baseline_acc = BaselineMetric_ACC(predictions.reshape(-1, 1), test_Label) *100



    warm_up_epochs = 3
    metric_to_plot_loss =[]
    metric_to_plot_Accuracy =[]
    # Organize data by metric name
    for trialsRes in Results:
        metric_to_plot_loss.append({key: value for key, value in  trialsRes.items() if key.endswith("loss")})
        metric_to_plot_Accuracy.append({key: [x *100 for x in value] for key, value in trialsRes.items() if key.endswith(BaselineMetric_ACC.Name)})


    # Plot the metrics
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    PlotAverage(metricList=metric_to_plot_loss, title=f"MONK {monk_number} Loss ({BaselineMetric_Loss.Name})", xlabel="Epochs", ylabel=f"{BaselineMetric_Loss.Name}",
                limitYRange=None, WarmUpEpochs=warm_up_epochs, baseline=baseline_loss, baselineName=f"baseline ({BaselineMetric_Loss.Name})", subplotAxes=axes[0])

    PlotAverage(metricList=metric_to_plot_Accuracy, title=f"MONK {monk_number} Accuracy", xlabel="Epochs", ylabel="Accuracy",
                limitYRange=None, WarmUpEpochs=warm_up_epochs, baseline=baseline_acc, baselineName=f"baseline ({BaselineMetric_ACC.Name})", subplotAxes=axes[1])

    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{MONK_PLOT_PATH}{monk_number}", f"MONK{monk_number}_MEAN{GenerateTagNameFromSettings(hpUsed)}")
    plt.close(fig)


def GenerateTagNameFromSettings(settings:dict):
    """
    Generate a tag name based on the selected optimizer, data preprocessing, and activation function.

    :return: A string tag name describing the model configuration.
    """
    tagName = ""

    if settings["optimizer"] == 1:
        tagName += "_BackPropagation"
    elif settings["optimizer"] == 2:
        tagName += "_NesterovMomentum"
    else:
        tagName += "_ADAM"

    return tagName


def TrainMonkModel(NumberOrTrial_search:int, NumberOrTrial_mean:int, monk_To_Test=None) -> None:
    """
    Train the MONK model using grid search and random search for hyperparameter optimization.

    :param NumberOrTrial_search: Number of trials for hyperparameter search.
    :param NumberOrTrial_mean: Number of trials to evaluate the model's performance.
    :param monk_To_Test : It's which monk dataset will be tested.
    """
    if monk_To_Test is None:
        monk_To_Test = [1, 2, 3]

    mode = HyperBag()

    # Training
    mode.AddChosen("Optimizer",[1,2,3])

    global OPTIMIZER
    global MONK_NUM

    gs = GridSearch()

    for monk_num in monk_To_Test:
        MONK_NUM = monk_num

        for modes, _ in gs.Search(mode):
            OPTIMIZER=modes["Optimizer"]


            settingDict = { "optimizer":OPTIMIZER}
            tagName = GenerateTagNameFromSettings(settingDict)

            hyperModel = MONKHyperModel(ReadMonk(MONK_NUM,DATA_SHUFFLE_SEED))
            hyperModel.SetSlit(VAL_SPLIT ,KFOLD_NUM)

            print(f"Training MONK {MONK_NUM}...")
            print(f"Run experiment with the following settings: {tagName}")


            best_model, best_hpSel = ModelSelection_MONK(hyperModel, NumberOrTrial_search)
            best_hpSel:HyperBag
            best_model.SaveModel( f"{MONK_MODEL_PATH}{MONK_NUM}", f"MONK{MONK_NUM}{tagName}.vjf")

            print(f"Best hp : {best_hpSel}")


            monk_data = hyperModel.GetDatasetVariant(best_hpSel)
            hyperModel_fn = lambda hp: (hyperModel.GetModel(hp), hyperModel.GetOptimizer(hp))
            totalResult = AsesSelectedModel(
                hyperModel_fn,best_hpSel,
                NumberOrTrial_mean,
                hyperModel.InterpretationMetric
                ,monk_data.Test,monk_data.Training,
                500,50,42 )

            totalResult["settings"] = settingDict
            SaveJson(f"{MONK_RESUTL_PATH}{MONK_NUM}", f"res{tagName}.json", totalResult)

            dataset_variant = hyperModel.GetDatasetVariant(best_hpSel)
            dataset_variant.MergeAll()
            optimizer = hyperModel.GetOptimizer(best_hpSel)
            final_model= hyperModel.GetModel(best_hpSel)

            final_model.Build(GlorotInitializer(42))
            final_model.AddMetric(hyperModel.InterpretationMetric)
            final_model.Fit( optimizer,dataset_variant, 500)
            final_model.SaveModel(f"{MONK_MODEL_PATH}",tagName)





def GenerateAllPlot_MONK(monkNumList=None):
    if monkNumList is None:
        monkNumList = [1, 2, 3]

    CreateDir(MONK_RESUTL_PATH)
    global MONK_NUM
    for monk_num in monkNumList:
        hm = MONKHyperModel(ReadMonk(monk_num,DATA_SHUFFLE_SEED))
        hm.SetSlit(VAL_SPLIT ,KFOLD_NUM)
        jsonFiles = GetAllFileInDir(f"{MONK_RESUTL_PATH}{monk_num}")
        for jsonFile in jsonFiles:
            data = readJson(jsonFile)
            hp = HyperBag()
            hp.Set(data["HP"])

            GeneratePlotAverage_ForMonk(
                Results=data["metrics"],
                hpUsed=hp,
                monkDataset=hm.GetDatasetVariant(hp),
                BaselineMetric_ACC=hm.InterpretationMetric,
                monk_number=monk_num
            )

