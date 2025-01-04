import random
from statistics import mean, variance

from sklearn.linear_model import LogisticRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss, CategoricalCrossEntropyLoss
from Core.Metric import Accuracy, MSE
from Core.ModelSelection import *
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch, GridSearch
from Core.Tuner.HyperBag import HyperBag
from Model.CupModel import PlotMultipleModels
from Model.ModelResults import PlotTableVarianceAndMean
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readMonk

USE_CATEGORICAL_LABLE = False
MULTY = False

USE_ONEHOT_VARIABLE_DATA = False
USE_ADAM = False
USE_TANH = True
USE_KFOLD = False
MONK_NUM=1

def HyperModel_Monk(hp :HyperBag ):
    model = ModelFeedForward()
    if USE_TANH:
        act_fn = TanH()
    else:
        act_fn = Sigmoid()

    model.AddLayer(Layer(17 if USE_ONEHOT_VARIABLE_DATA else 6, Linear(), False, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"], act_fn, hp["drop_out"], False, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], act_fn, False, f"_h{i}"))


    if USE_CATEGORICAL_LABLE:
        model.AddLayer(Layer(2, SoftARGMax(), False, "output_HotEncoding"))
        loss = CategoricalCrossEntropyLoss()
    else:
        model.AddLayer(Layer(1, act_fn, False,"output"))
        loss = MSELoss()

    if USE_ADAM:
        optimizer = Adam(loss,hp["eta"], hp["labda"], hp["alpha"], hp["beta"] , hp["epsilon"] ,hp["decay"])
    else:
        optimizer = BackPropagation(loss, hp["eta"], hp["labda"], hp["alpha"],hp["decay"])
    return model, optimizer




def ReadMonk(n: int, val_split: float = 0.15, test_split: float = 0.05, seed: int = 0):
    if n <0 or n>3:
        raise Exception("n must be between 0 and 3")
    TR_file_path_monk = f"dataset/monk+s+problems/monks-{MONK_NUM}.train"
    TS_file_path_monk = f"dataset/monk+s+problems/monks-{MONK_NUM}.test"

    designSet = readMonk(TR_file_path_monk)
    testSet = readMonk(TS_file_path_monk)
    monkDataset = DataSet.FromDataExample(designSet)
    monkDataset.Test = testSet

    if USE_TANH:
        monkDataset.ApplayTranformationOnLable(np.vectorize(lambda x: -1 if x == 0 else 1 ))
        baseline_metric = Accuracy(Sign())
    else:
        baseline_metric = Accuracy(Binary(0.5))


    if USE_ONEHOT_VARIABLE_DATA:
        monkDataset.ToOnHotOnExamples()

    monkDataset.Shuffle(seed)

    if USE_CATEGORICAL_LABLE:
        monkDataset.ToCategoricalLabel()
    monkDataset.PrintData()

    if USE_KFOLD:
        monkDataset.SetUp_Kfold_TestHoldOut(5)
    else:
        monkDataset.SplitTV(val_split)

    return monkDataset, baseline_metric




def HyperBag_Monk():
    hp = HyperBag()

    hp.AddRange("eta", 0.05, 0.2, 0.005)
    if MONK_NUM ==3:
        hp.AddRange("labda", 0.000, 0.01, 0.005)
    hp.AddRange("alpha", 0.5, 0.9, 0.05)
    # hp.AddRange("decay", 0.0003, 0.005, 0.0003)

    # only adam
    if USE_ADAM:
        hp.AddRange("beta", 0.9, 0.99, 0.01)
        hp.AddRange("epsilon", 1e-13, 1e-7, 1e-1)

    #hp.AddRange("drop_out", 0.2, 0.6, 0.1)

    hp.AddRange("unit", 1, 8, 1)
    hp.AddRange("hlayer", 0, 2, 1)
    return hp



def ModelSelection(monkDataset:DataSet, BaselineMetric:Metric, NumberOrTrial: int, minibatchSize : int = 64) -> tuple[ModelFeedForward, HyperBag]:
    if USE_KFOLD:
        ModelSelector:ModelSelection = BestSearchKFold(HyperBag_Monk(), RandomSearch(NumberOrTrial))
    else:
        ModelSelector:ModelSelection = BestSearch(HyperBag_Monk(), RandomSearch(NumberOrTrial))

    watched_metric = "val_loss"

    callBacks = [EarlyStopping(watched_metric, 50,0.0001)]
    best_model, hpSel = ModelSelector.GetBestModel(
        HyperModel_Monk,
        monkDataset,
        500,
        minibatchSize,
        watched_metric,
        [BaselineMetric],
        GlorotInitializer(),
        callBacks)
    best_model.PlotModel(f"MONK Model {MONK_NUM}")
    return best_model, hpSel



def TrainMultipleModels(num_models: int = 5, NumberOrTrial:int = 50 ) -> None:
    """
    Train multiple models and evaluate their performance.

    :param num_models: Number of models to train.
    """

    monkDataset, BaselineMetric = ReadMonk(MONK_NUM, 0.15, seed=159)
    results = {}
    print("Performing initial Random Search for best hyperparameters...")

    best_model, best_hpSel = ModelSelection(monkDataset, BaselineMetric, NumberOrTrial)
    best_hpSel:HyperBag
    print(f"Best hyperparameters found: {best_hpSel.GetHPString()}")

    SavedTraining = DataExamples.Clone(monkDataset.Training)
    if not USE_KFOLD:
        SavedTraining.Concatenate(monkDataset.Validation)
        monkDataset.Validation=None

    seedList = [random.randint(0, 1000) for _ in range(num_models)]
    for i,seed in zip(range(num_models),seedList):
        training:DataExamples = DataExamples.Clone(SavedTraining)
        training.Shuffle(seed)
        monkDataset.Training = training
        #monkDataset.Training.CutData(-55)

        print(f"Training Model {i + 1}/{num_models}...")



        model, optimizer = HyperModel_Monk(best_hpSel)
        model.Build(GlorotInitializer())
        model.AddMetric(BaselineMetric)
        model.Fit( optimizer,monkDataset, 500, 64)


        model_name = f"Monk{MONK_NUM}_Model_{i}"
        model.SaveModel(f"Data/Models/{model_name}.vjf")
        model.SaveMetricsResults(f"Data/Results/{model_name}.mres")
        # Save metrics
        results[model_name] = {
            "hyperparameters": model,
            "metrics": model.MetricResults
        }
    PlotMultipleModels(results)
    PlotTableVarianceAndMean(results)






def GeneratePlot(AccuracyMetric, MetricResults, monkDataset,extraname:str=""):
    MSEmetric = MSE()

    lin_model = LogisticRegression()
    if USE_CATEGORICAL_LABLE:
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
    plot_metrics(
        metricDic=metric_to_plot_loss,
        baseline=baseline_mse,
        baselineName=f"baseline {MSEmetric.Name}",
        limitYRange=None,
        title=f"MONK {MONK_NUM} loss",
        xlabel="Epochs",
        ylabel="",
        subplotAxes=axes[0])
    plot_metrics(
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
    ShowOrSavePlot(f"Data/Plots/MONK {MONK_NUM}", f"Loss(MSE)-Accuracy{extraname}")

def GenerateTagName():
    tagName = ""
    if USE_ADAM:
        tagName += "_adam"
    else:
        tagName += "_backprop"
    if USE_ONEHOT_VARIABLE_DATA:
        tagName += "_onehot"
    else:
        tagName += "_numeric"
    if USE_TANH:
        tagName += "_tanh"
    else:
        tagName += "_sigmoid"
    return tagName


def TrainMonkModel(NumberOrTrial_search:int, NumberOrTrial_mean:int) -> None:
    mode = HyperBag()
    mode.AddChosen("Adam",[True,False])
    mode.AddChosen("OneHot",[True,False])
    mode.AddChosen("Tanh",[True,False])
    global USE_TANH
    global USE_ADAM
    global USE_ONEHOT_VARIABLE_DATA

    gs = GridSearch()
    for modes, _ in gs.search(mode):
        USE_ADAM=modes["Adam"]
        USE_ONEHOT_VARIABLE_DATA=modes["OneHot"]
        USE_TANH=modes["Tanh"]


        tagName = GenerateTagName()

        global MONK_NUM
        for monk in [1,2,3]:
            MONK_NUM = monk
            print(f"Training MONK {MONK_NUM}...")
            monkDataset, BaselineMetric_Accuracy = ReadMonk(MONK_NUM, 0.15, 0.05)

            best_model, best_hpSel = ModelSelection(monkDataset, BaselineMetric_Accuracy, NumberOrTrial_search, 64)
            best_hpSel:HyperBag
            best_model.SaveModel(f"Data/Models/Monk{MONK_NUM}{tagName}.vjf")
            best_model.SaveMetricsResults(f"Data/Results/Monk{MONK_NUM}{tagName}.mres")

            GeneratePlot(BaselineMetric_Accuracy, best_model.MetricResults, monkDataset,tagName)


            print(f"Best hp : {best_hpSel}")

            res = {}

            for i in range(NumberOrTrial_mean):
                model, optimizer = HyperModel_Monk(best_hpSel)
                model.Build(GlorotInitializer())
                model.AddMetric(BaselineMetric_Accuracy)
                model.Fit( optimizer,monkDataset, 300, 64)
                for key, value in model.MetricResults.items():
                    if key not in res:
                        res[key] = []
                    res[key].append(value[-1])
                print(f"training model {i + 1} / {NumberOrTrial_mean} " +  " | ".join(f"{key}:{value[-1]:.4f}" for key, value in res.items()))

            res = {key: [mean(value),variance(value)] for key, value in res.items()}
            res["HP"] = best_hpSel.hpDic

            SaveJson(f"Data/FinalModel/MONK {MONK_NUM}",f"res{tagName}.json",res)
            print(res)





if __name__ == '__main__':

    if MULTY:
        TrainMultipleModels(50,250)
    else:
        TrainMonkModel(150,50)
