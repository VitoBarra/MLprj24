from sklearn.linear_model import LogisticRegression
from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.LossFunction import MSELoss, CategoricalCrossEntropyLoss
from Core.Metric import Accuracy, MSE
from Core.Optimizer.BackPropagationNesterovMomentum import BackPropagationNesterovMomentum
from Core.Tuner.ModelSelection import *
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Model import MONK_RESUTL_PATH, MONK_PLOT_PATH, MONK_MODEL_PATH
from Model.ModelPlots import  *
from Model.TrainingFuction import ValidateSelectedModel
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readMonk

OPTIMIZER = 1

USE_KFOLD = True
KFOLD_NUM = 5
VAL_SPLIT = 0.15

MONK_NUM= 1


class HyperModel_MONK(HyperModel):


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
        hp.AddChosen("ActFun",[TanH(),Sigmoid(),ReLU()])

        # Data format
        hp.AddChosen("oneHotInput",[True,False])
        hp.AddChosen("outFun",[TanH(),Sigmoid(),SoftARGMax()])



        #hp.AddRange("drop_out", 0.2, 0.6, 0.1)

        return hp

    def GetDatasetVariant(self, hp):
        data_set = DataSet.Clone(self.originalDataset)
        if (hp["oneHotInput"],hp["outFun"]) not in self.DataSetsVariant:
            self.PreprocessInput(hp ,data_set)
            self.PreprocessOutput(hp,data_set)
            self.SplitAllDataset(data_set)
            self.DataSetsVariant[hp["oneHotInput"],hp["outFun"].Name] = data_set


        return self.DataSetsVariant[hp["oneHotInput"],hp["outFun"].Name]

    def SplitAllDataset(self,data_set):

        if USE_KFOLD:
            data_set.SetUp_Kfold_TestHoldOut(self.k)
        else:
            data_set.SplitTV(self.val_split)


    def PreprocessInput(self,hp, data_set):
        if hp["oneHotInput"]:
            data_set.ToOnHotOnData()

    def PreprocessOutput(self,hp, data_set : DataSet):
        if hp["outFun"].Name == "TanH": # TanH
            data_set.ApplyTransformationOnLabel(np.vectorize(lambda x: -1 if x == 0 else 1 ))
            Interpretation_metric = Accuracy(Sign())

        elif hp["outFun"].Name == "Sigmoid": #asigmoid
            Interpretation_metric = Accuracy(Binary(0.5))

        elif hp["outFun"].Name == "SoftARGMax": #  one Hot label
            data_set.ToOneHotLabel()
            Interpretation_metric = Accuracy()
        else:
            raise ValueError("value unknown")
        self.InterpretationMetric= Interpretation_metric


    def GetModel(self, hp :HyperBag):
        model = ModelFeedForward()


        model.AddLayer(Layer(17 if hp["oneHotInput"] else 6, Linear(), hp["UseBiasIN"], "input"))
        for i in range(hp["hlayer"]):
                model.AddLayer(Layer(hp["unit"], hp["ActFun"], hp["UseBias"], f"_h{i}"))

        output_act = hp["outFun"]

        model.AddLayer(Layer(1 if output_act.Name !="SoftARGMax" else 2 ,
                             output_act, False,f"output_{output_act.Name}"))
        return model


    def GetOptimizer(self, hp :HyperBag):

        loss = CategoricalCrossEntropyLoss() if hp["outFun"].Name == "SoftARGMax" else MSELoss()


        if OPTIMIZER == 1:
            optimizer = BackPropagation(loss,hp["BatchSize"], hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
        elif OPTIMIZER == 2:
            optimizer = BackPropagationNesterovMomentum(loss,hp["BatchSize"], hp["eta"], hp["lambda"], hp["alpha"],hp["decay"])
        else:
            optimizer = Adam(loss,hp["BatchSize"], hp["eta"], hp["lambda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])

        return  optimizer





def ReadMonk(n: int, seed: int = 0) -> DataSet:
    if n <0 or n>3:
        raise Exception("n must be between 0 and 3")
    TR_file_path_monk = f"dataset/monk+s+problems/monks-{MONK_NUM}.train"
    TS_file_path_monk = f"dataset/monk+s+problems/monks-{MONK_NUM}.test"

    designSet = readMonk(TR_file_path_monk)
    testSet = readMonk(TS_file_path_monk)
    monkDataset = DataSet.FromDataExample(designSet)
    monkDataset.Test = testSet
    monkDataset.Shuffle(seed)
    return monkDataset








def ModelSelection( hyperModel:HyperModel_MONK ,NumberOrTrial: int) -> tuple[ModelFeedForward, HyperBag]:


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






def GeneratePlot_ForMonk(AccuracyMetric, MetricResults, monkDataset, extra_name:str= ""):
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

    warm_up_epochs = 3
    metric_to_plot_loss = {key: value[warm_up_epochs:] for key, value in MetricResults.items() if key.endswith("loss")}
    metric_to_plot_Accuracy = {key: value[warm_up_epochs:]*100 for key, value in MetricResults.items() if key.endswith(baseline_acc.name)}

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
        baselineName=f"baseline {AccuracyMetric.name}",
        limitYRange=None,
        title=f"MONK {MONK_NUM} accuracy",
        xlabel="Epochs",
        ylabel="%",
        subplotAxes=axes[1])
    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{MONK_PLOT_PATH}{MONK_NUM}", f"Loss(MSE)-Accuracy{extra_name}")
    plt.close(fig)

def GeneratePlotAverage_ForMonk(Results: list[dict], Metrics: list[str], path=f"{MONK_PLOT_PATH}{MONK_NUM}", name: str = f"MONK{MONK_NUM}_MEAN", tag: str = ""):
    """
    Generate plot for MONK using the plot for the mean of individual trials.

    :param Results: List of dictionaries. Each dictionary represents an individual trial and contains metrics with respective values.
    :param Metrics: List of metrics to plot.
    :param path: The path where the plot will be saved.
    :param name: The name of the file for the plot.
    :param tag: Extra information for the file name.
    """
    # Organize data by metric name
    loss_metrics = {metric: [] for metric in Metrics if metric.endswith("loss")}
    accuracy_metrics = {metric: [] for metric in Metrics if metric.endswith("Accuracy")}

    warm_up_epochs = 5
    for trial in Results:
        for metric in Metrics:
            if metric.endswith("loss") and metric in trial:
                loss_metrics[metric].append(trial[metric][warm_up_epochs:])
            elif metric.endswith("Accuracy") and metric in trial:
                accuracy_metrics[metric].append(trial[metric][warm_up_epochs:])

    # Plot the metrics
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    PlotAverage(
        metricDict=loss_metrics,
        limitYRange=None,
        WarmUpEpochs=warm_up_epochs,
        title=f"MONK {MONK_NUM} Loss",
        xlabel="Epochs",
        ylabel="Loss",
        subplotAxes=axes[0],
    )

    PlotAverage(
        metricDict=accuracy_metrics,
        limitYRange=None,
        WarmUpEpochs=warm_up_epochs,
        title=f"MONK {MONK_NUM} Accuracy",
        xlabel="Epochs",
        ylabel="Accuracy",
        subplotAxes=axes[1]
    )

    # Adjust layout and save the entire figure
    fig.tight_layout()
    ShowOrSavePlot(f"{path}", f"{name}{tag}")
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

    for monk in monk_To_Test:
        MONK_NUM = monk

        #Dataset Preparation
        monkDataset, BaselineMetric_Accuracy = ReadMonk(MONK_NUM, 0.15)
        mergedMonkDataset = DataSet.Clone(monkDataset)
        if not USE_KFOLD:
            mergedMonkDataset.MergeTrainingAndValidation()

        for modes, _ in gs.Search(mode):
            OPTIMIZER=modes["Optimizer"]

            #Dataset Preparation
            monkDataset = ReadMonk(MONK_NUM)

            settingDict = { "optimizer":OPTIMIZER}
            tagName = GenerateTagNameFromSettings(settingDict)

            hyperModel = HyperModel_MONK(monkDataset)
            hyperModel.SetSlit(VAL_SPLIT ,KFOLD_NUM)
            monkDataset.SplitTV()

            print(f"Training MONK {MONK_NUM}...")
            print(f"Run experiment with the following settings: {tagName}")


            best_model, best_hpSel = ModelSelection(hyperModel, NumberOrTrial_search)
            best_hpSel:HyperBag
            best_model.SaveModel( f"{MONK_MODEL_PATH}{MONK_NUM}", f"MONK{MONK_NUM}{tagName}.vjf")

            print(f"Best hp : {best_hpSel}")

            MetricToCheck = [key for key, _ in best_model.MetricResults.items() if not key.startswith("val_")]

            monk_data = hyperModel.GetDatasetVariant(best_hpSel)
            hyperModel_fn = lambda hp: (hyperModel.GetModel(hp), hyperModel.GetOptimizer(hp))
            totalResult = ValidateSelectedModel(
                hyperModel_fn,best_hpSel,
                NumberOrTrial_mean, MetricToCheck,
                hyperModel.InterpretationMetric
                ,monk_data.Test,monk_data.Training,
                500,50,42 )

            totalResult["settings"] = settingDict
            SaveJson(f"{MONK_RESUTL_PATH}{MONK_NUM}", f"res{tagName}.json", totalResult)







if __name__ == '__main__':
        monkNumList = [3]
        TrainMonkModel(100,50, monkNumList)

        for monk_num in monkNumList:
            jsonFiles = GetAllFileInDir(f"{MONK_RESUTL_PATH}{monk_num}")
            for jsonFile in jsonFiles:
                data = readJson(jsonFile)

                GeneratePlotAverage_ForMonk(
                    Results=data["metrics"],
                    Metrics=["test_loss", "loss", "test_Accuracy", "Accuracy"],
                    path=f"{MONK_PLOT_PATH}{monk_num}",
                    tag=GenerateTagNameFromSettings(data['settings'])
                )


            #GeneratePlot_ForMonk(BaselineMetric_Accuracy, best_model.MetricResults, monkDataset,tagName)

