from sklearn.linear_model import LogisticRegression

from Core.ActivationFunction import *
from Core.Callback.EarlyStopping import EarlyStopping
from Core.FeedForwardModel import *
from Core.Layer import DropoutLayer
from Core.LossFunction import MSELoss, CategoricalCrossEntropyLoss
from Core.Metric import Accuracy
from Core.ModelSelection import *
from Core.Optimizer.Adam import Adam
from Core.Optimizer.BackPropagation import BackPropagation
from Core.Tuner.HpSearch import RandomSearch
from Core.Tuner.HyperBag import HyperBag
from Utility.PlotUtil import *
from dataset.ReadDatasetUtil import readMonk
import os
import json
from statistics import mean, variance
import matplotlib.pyplot as plt

USE_CATEGORICAL_LABLE = False
USE_ONEHOT_VARIABLE_DATA = False
USE_ADAM = True
USE_TANH = True
USE_KFOLD = False
MULTI = True
MONK=2

def HyperModel_Monk(hp :HyperBag ):
    model = ModelFeedForward()
    if USE_TANH:
        act_fn = TanH()
    else:
        act_fn = Sigmoid()

    model.AddLayer(Layer(17 if USE_ONEHOT_VARIABLE_DATA else 6, Linear(), True, "input"))
    for i in range(hp["hlayer"]):
        if hp["drop_out"] is not None:
            model.AddLayer(DropoutLayer(hp["unit"], act_fn, hp["drop_out"], True, f"drop_out_h{i}"))
        else:
            model.AddLayer(Layer(hp["unit"], act_fn, True, f"_h{i}"))


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




def ReadMonk(n: int, val_split: float = 0.15, test_split: float= 0.05):
    if n <0 or n>3:
        raise Exception("n must be between 0 and 3")
    file_path_monk = f"dataset/monk+s+problems/monks-{n}.train"
    all_data = readMonk(file_path_monk)

    if USE_TANH:
        all_data.ApplayTranformationOnLable(np.vectorize(lambda x: -1 if x == 0 else 1 ))
        baseline_metric = Accuracy(Sign())
    else:
        baseline_metric = Accuracy(Binary(0.5))


    if USE_ONEHOT_VARIABLE_DATA:
        all_data.ToOnHotOnExamples()

    all_data.Shuffle(195)

    if USE_CATEGORICAL_LABLE:
        all_data.ToCategoricalLabel()
    all_data.PrintData()
    if not USE_KFOLD:
        if MULTI:
            file_path_monk = f"dataset/monk+s+problems/monks-{MONK}.test"
            all_data.Split2(val_split, file_path_monk)
        else:
            all_data.Split(val_split, test_split)
    return all_data , baseline_metric




def HyperBag_Monk():
    hp = HyperBag()

    hp.AddRange("eta", 0.05, 0.2, 0.005)
    # hp.AddRange("labda", 0.000, 0.01, 0.005)
    hp.AddRange("alpha", 0.5, 0.9, 0.05)
    # hp.AddRange("decay", 0.0003, 0.005, 0.0003)

    # only adam
    if USE_ADAM:
        hp.AddRange("beta", 0.9, 0.99, 0.01)
        hp.AddRange("epsilon", 1e-13, 1e-7, 1e-1)

    #hp.AddRange("drop_out", 0.2, 0.6, 0.1)

    hp.AddRange("unit", 1, 5, 1)
    hp.AddRange("hlayer", 0, 1, 1)
    return hp




def SaveResults(results, path: str = "Data/Results/model_results.json") -> None:
    """
    Save the model results to a JSON file.

    :param results: Dictionary of model results (metrics and hyperparameters).
    :param path: Path to save the results JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

def TrainMultipleModels(num_models: int = 5) -> None:
    """
    Train multiple models and evaluate their performance.

    :param num_models: Number of models to train.
    """
    results = {}
    for i in range(num_models):
        print(f"Training Model {i + 1}/{num_models}...")

        if USE_KFOLD:
            ModelSelector = BestSearchKFold(HyperBag_Monk(), RandomSearch(50), 5)
        else:
            ModelSelector = BestSearch(HyperBag_Monk(), RandomSearch(50))

        watched_metric = "val_loss"
        callBacks = [EarlyStopping(watched_metric, 50)]

        best_model, best_hpSel = ModelSelector.GetBestModel(
            HyperModel_Monk,
            alldata,
            500,
            64,
            watched_metric,
            [BaselineMetric],
            GlorotInitializer(),
            callBacks
        )

        model_name = f"Monk{MONK}_Model_{i}"
        best_model.SaveModel(f"Data/Models/{model_name}.vjf")
        best_model.SaveMetricsResults(f"Data/Results/{model_name}.mres")
        best_model.PlotModel(f"MONK {MONK} Model {i}")

        # Save metrics
        results[model_name] = {
            "hyperparameters": best_hpSel,
            "metrics": best_model.MetricResults
        }
    PlotMultipleModels(results)
    CalculateVarianceAndMean(results)

def CalculateVarianceAndMean(results: dict) -> None:
    """
    Calculate and display mean and variance for each model's metrics.

    :param results: Dictionary containing model metrics.
    """
    metric_summary = {}
    for model_name, model_data in results.items():
        for metric, values in model_data["metrics"].items():
            if metric not in metric_summary:
                metric_summary[metric] = []
            metric_summary[metric].append(values[-1])

    final_summary = {}
    metrics = []
    means = []
    variances = []

    for metric, values in metric_summary.items():
        mean_val = mean(values)
        variance_val = variance(values)
        final_summary[metric] = {
            "mean": mean_val,
            "variance": variance_val
        }
        metrics.append(metric)
        means.append(mean_val)
        variances.append(variance_val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = list(zip(metrics, means, variances))
    table = plt.table(cellText=table_data,
                      colLabels=["Metric", "Mean", "Variance"],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title("Table of Means and Variances")
    plt.show()

    SaveResults(final_summary)

def PlotMultipleModels(results: dict) -> None:
    """
    Plot the test loss for multiple models, including individual model losses and the mean curve.

    :param results: Dictionary containing model metrics.
    """
    plt.figure(figsize=(12, 8))

    all_losses = []
    min_length = min(len(model_data["metrics"]["test_loss"]) for model_data in results.values())

    for model_name, model_data in results.items():
        loss_values = model_data["metrics"]["test_loss"][:min_length]
        all_losses.append(loss_values)
        plt.plot(loss_values, color='dodgerblue', alpha=0.2)

    all_losses = np.array(all_losses)
    mean_loss = np.mean(all_losses, axis=0)
    plt.plot(mean_loss, color='blue', linewidth=2, label='Mean')

    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Comparison for Multiple Models")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    alldata, BaselineMetric = ReadMonk(MONK, 0.15, 0.05)


    if MULTI:
        TrainMultipleModels(5)
    else:



        if USE_KFOLD:
            ModelSelector = BestSearchKFold(HyperBag_Monk(), RandomSearch(50), 5)
        else:
            ModelSelector = BestSearch(HyperBag_Monk(), RandomSearch(50))

        watched_metric = "val_loss"
        callBacks = [EarlyStopping(watched_metric, 100)]
        best_model, best_hpSel = ModelSelector.GetBestModel(
            HyperModel_Monk,
            alldata,
            500,
            64,
            watched_metric,
            [BaselineMetric],
            GlorotInitializer(),
            callBacks)


        print(f"Best hp : {best_hpSel}")


        best_model.PlotModel(f"MONK {MONK} Model")

        best_model.SaveModel(f"Data/Models/Monk{MONK}.vjf")
        best_model.SaveMetricsResults(f"Data/Results/Monk{MONK}.mres")


        lin_model = LogisticRegression()
        if USE_CATEGORICAL_LABLE:
            lin_model.fit(alldata.Training.Data, alldata.Training.Label[:,1])
            validationLable=alldata.Test.Label[:,1]
        else:
            lin_model.fit(alldata.Training.Data, alldata.Training.Label.reshape(-1))
            validationLable=alldata.Test.Label

        print(f"R2 on test: {lin_model.score(alldata.Test.Data, validationLable)}%")
        predictions = lin_model.predict(alldata.Test.Data)
        baseline= BaselineMetric(predictions.reshape(-1, 1), validationLable)

        #metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() if not key.startwith("")}
        metric_to_plot = {key: value[2:] for key, value in best_model.MetricResults.items() }

        plot_metric(
            metricDic=metric_to_plot,
            baseline=baseline,
            baselineName= f"baseline {BaselineMetric.Name}" ,
            limitYRange=None,
            title=f"MONK {MONK} results",
            xlabel="Epoche",
            ylabel="")