from sklearn.linear_model import LogisticRegression

from Core.Callback.EarlyStopping import EarlyStopping
from Core.DataSet.DataExamples import DataExamples
from Core.FeedForwardModel import *
from Core.LossFunction import CategoricalCrossEntropyLoss
from Core.Metric import MSE
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.ModelSelection import *
from Model.ModelPlots import *
from Model.Monk.MonkHyperModel import MONKHyperModel
from Model.TrainingFuction import AssessmentSelectedModel
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readMonk
from . import *


def ReadMonk(n: int, seed: int = 0) -> DataSet:
    if n <0 or n>4:
        raise Exception("n must be between 0 and 3")

    designSet = readMonk(DATASET_PATH_MONK_TR)
    testSet = readMonk(DATASET_PATH_MONK_TS)
    monkDataset = DataSet.FromDataExample(designSet)
    monkDataset.Test = testSet
    monkDataset.Shuffle(seed)
    return monkDataset








def ModelSelection_MONK( hyperModel:MONKHyperModel ,NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:


    if USE_KFOLD_MONK:
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
    return best_model, hpSel





def GeneratePlotAverage_ForMonk(Results: list[dict],
                                BaselineMetric_ACC, monkDataset, hpUsed, monk_number ,settings):

    BaselineMetric_Loss = CategoricalCrossEntropyLoss() if hpUsed["outFun"] == "SoftARGMax" else MSE()

    lin_model = LogisticRegression()
    if  hpUsed["outFun"] == "SoftARGMax":
        lin_model.fit(monkDataset.Training.Data, monkDataset.Training.Label[:, 1])
        test_Label = monkDataset.Test.Label[:, 1]
    else:
        lin_model.fit(monkDataset.Training.Data, monkDataset.Training.Label.reshape(-1))
        test_Label = monkDataset.Test.Label

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
    ShowOrSavePlot(f"{MONK_PLOT_PATH}{monk_number}", f"MONK{monk_number}_MEAN{GenerateTagNameFromSettings(settings)}")
    plt.close(fig)


def GenerateTagNameFromSettings(settings:dict):

    tagName = ""

    if settings["Optimizer"] == 1:
        tagName += "_BackPropagation"
    elif settings["Optimizer"] == 2:
        tagName += "_NesterovMomentum"
    else:
        tagName += "_ADAM"

    if settings["Batch_size"] == 32:
        tagName += "_BatchSize32"
    elif settings["Batch_size"] == 64:
        tagName += "_BatchSize64"
    elif settings["Batch_size"] == 1:
        tagName += "_Online"
    else:
        tagName += "_BatchSize-1"

    return tagName


def TrainMonkModel(NumberOrTrial_search:int, NumberOrTrial_mean:int, monk_To_Test=None, optimizer = None, minibatch_size =None) -> None:
    if monk_To_Test is None:
        monk_To_Test = [1, 2, 3,4]
    if optimizer is None:
        optimizer = [1,2,3]
    if minibatch_size is None:
        minibatch_size = [-1, 1, 32, 64]

    settingsList = HyperBag()

    # Training
    settingsList.AddChosen("Optimizer",optimizer)
    settingsList.AddChosen("Batch_size", minibatch_size)

    global OPTIMIZER_MONK
    global MONK_NUM
    global MINI_BATCH_SIZE

    gs = GridSearch()

    for monk_num in monk_To_Test:
        MONK_NUM = monk_num

        for settings, _ in gs.Search(settingsList):
            OPTIMIZER_MONK=settings["Optimizer"]
            MINI_BATCH_SIZE = settings["Batch_size"]


            settingDict = { "Optimizer":OPTIMIZER_MONK, "Batch_size": MINI_BATCH_SIZE ,"MONK":MONK_NUM, "kFold": USE_KFOLD_MONK}
            tagName = GenerateTagNameFromSettings(settingDict)

            hyperModel = MONKHyperModel(ReadMonk(MONK_NUM,DATA_SHUFFLE_SEED),settingDict)
            hyperModel.SetSlit(VAL_SPLIT_MONK ,KFOLD_NUM)

            print(f"Training MONK {MONK_NUM}...")
            print(f"Run MONK experiment with the following settings: {tagName}")


            best_model, best_hpSel = ModelSelection_MONK(hyperModel, NumberOrTrial_search)
            best_hpSel:HyperBag

            print(f"Best hp : {best_hpSel}")


            monk_data = hyperModel.GetDatasetVariant(best_hpSel)
            dataset_variant = DataExamples.Clone(monk_data.Training)
            dataset_variant.Concatenate(monk_data.Validation)
            dataset_variant.Concatenate(monk_data.Test)
            hyperModel_fn = lambda hp: (hyperModel.GetModel(hp), hyperModel.GetOptimizer(hp))
            totalResult = AssessmentSelectedModel(
                hyperModel_fn,best_hpSel,
                NumberOrTrial_mean,
                hyperModel.GetInterpretationMetric(best_hpSel)
                ,dataset_variant, VAL_SPLIT_MONK ,None, 400,
                500,150,42 )

            totalResult["settings"] = settingDict
            SaveJson(f"{MONK_RESUTL_PATH}{MONK_NUM}", f"res{tagName}.json", totalResult)

            dataset_variant = hyperModel.GetDatasetVariant(best_hpSel)
            dataset_variant.MergeTrainingAndValidation()
            optimizer = hyperModel.GetOptimizer(best_hpSel)
            final_model= hyperModel.GetModel(best_hpSel)

            final_model.Build(GlorotInitializer(42))
            final_model.AddMetric(hyperModel.GetInterpretationMetric(best_hpSel))
            final_model.Fit( optimizer,dataset_variant, 500)
            final_model.SaveModel( f"{MONK_MODEL_PATH}{MONK_NUM}", f"MONK{MONK_NUM}{tagName}")





def GenerateAllPlot_MONK(monkNumList=None):
    if monkNumList is None:
        monkNumList = [1, 2, 3, 4]

    for monk_num in monkNumList:
        jsonFiles = GetAllFileInDir(f"{MONK_RESUTL_PATH}{monk_num}")
        hm = MONKHyperModel(ReadMonk(monk_num,DATA_SHUFFLE_SEED) )
        hm.SetSlit(VAL_SPLIT_MONK ,KFOLD_NUM)
        for jsonFile in jsonFiles:
            data = readJson(jsonFile)
            hm.UpdateSettings(data["settings"])
            hp = HyperBag()
            hp.Set(data["HP"])

            GeneratePlotAverage_ForMonk(
                Results=data["metrics"],
                hpUsed=hp,
                monkDataset=hm.GetDatasetVariant(hp),
                BaselineMetric_ACC=hm.GetInterpretationMetric(hp),
                monk_number=monk_num,
                settings =  data["settings"]
            )

