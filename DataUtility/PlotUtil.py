import pickle
from matplotlib import pyplot as plt
from DataUtility.FileUtil import *
import time as t




# Funzione per calcolare TP, FP, TN, FN a ogni soglia
"""
Uso:
fpr, tpr, thresholds = calculate_roc(y_scores, y_true)
auc = calculate_auc(fpr, tpr)
"""
def calculate_roc(y_scores: list[float], y_true:list[int]) -> tuple[list[float], list[float], list[float]]:
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for score, true_label in zip(y_scores, y_true):
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and true_label == 1:
                tp += 1
            elif predicted == 1 and true_label == 0:
                fp += 1
            elif predicted == 0 and true_label == 0:
                tn += 1
            elif predicted == 0 and true_label == 1:
                fn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilità
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - Specificità
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


# Funzione per calcolare l'AUC usando la formula del trapezio
def calculate_auc(fpr: list[float], tpr: list[float]) -> float:
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc

def printAUC(fpr: list[float], tpr: list[float], auc: float) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', linestyle='-', color='b', label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Classificatore casuale')  # Linea random
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (calcolata manualmente)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

#si può usare anche oer l'accuracu, basta passargli il valore oer training e validation con i corretti label
def plot_losses_accuracy(loss_matrix: list[list[float]] | np.ndarray, labels: list[str] = None,
                         title: str = "Loss per Epoch", xlabel: str = "Epochs", ylabel: str = "Loss Value",
                         path: str = None) -> None:
    """
    Plots loss curves over epochs using a given loss matrix, using different line styles for better distinction in black and white.

    Args:
        loss_matrix (list of lists or numpy.ndarray): A matrix where each row represents an epoch
                                                      and each column corresponds to a different loss function.
        labels (list of str, optional): Labels for each loss function. Should match the number of columns in the matrix.
                                        Default is None, in which case generic labels will be used.
        title (str, optional): Title of the plot. Default is "Loss per Epoch".
        xlabel (str, optional): Label for the x-axis. Default is "Epochs".
        ylabel (str, optional): Label for the y-axis. Default is "Loss Value".
        path (str, optional): Path to save the plot. Default is None (plot is shown but not saved).

    Returns:
        None
    """
    # Determine the number of loss functions based on the matrix shape
    num_losses = loss_matrix.shape[0]

    # If no labels are provided, create generic ones
    if labels is None:
        labels = [f"Loss {i + 1}" for i in range(num_losses)]

    # Define different line styles and markers for black-and-white readability
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^']

    plt.figure(figsize=(10, 6))

    # Plot each loss function with different styles
    for i in range(num_losses):
        plt.plot(
            loss_matrix[i,:],
            label=labels[i],
            linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
            marker=markers[i % len(markers)],  # Cycle through markers
            markersize=5,
            linewidth=1.5
        )

    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # Show grid and save the plot if a path is provided
    plt.grid(True)
    if path is not None:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path, format='png', dpi=300)
        print(f"Plot saved to {path}")

    plt.show()




def ShowOrSavePlot(path=None, filename=None):
    if path is None or path == '':
        plt.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None or filename == '':
            filename = 'model'
        plt.savefig(f"{path}/{filename}.png")
        plt.clf()


def SaveTrainingDataByName(data_path, problem_name, test_name, history, result):
    dir = f"{data_path}/{problem_name}/{test_name}"
    SaveTrainingData(dir, history.history, result)


def SaveTrainingData(path, history, result):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/history.bin", "wb") as outfile:
        pickle.dump(history, outfile)
    with open(f"{path}/result.bin", "wb") as outfile:
        pickle.dump(result, outfile)


def ReadTrainingDataByName(data_path, problem_name, test_name):
    dir = f"{data_path}/{problem_name}/{test_name}"
    # Writing to sample.json
    history, result = ReadTrainingData(dir)
    return history, result


def ReadTrainingData(path):
    with open(f"{path}/history.bin", "rb") as inputfile:
        history = pickle.load(inputfile)
    with open(f"{path}/result.bin", "rb") as inputfile:
        result = pickle.load(inputfile)
    return history, result


def PrityPlot(loss, mse=None, accuracy=None, baseline=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Define font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # Plot the baseline
    if baseline is not None:
        ax.plot(
            [loss[0], max(loss)],
            [baseline, baseline],
            label="Baseline",
            color="lightgray",
            linestyle="--",
            linewidth=1,
        )
        # Plot the baseline text
        ax.text(
            loss[-1] * 1.01,
            baseline,
            "Baseline",
            color="lightgray",
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

    # Define a nice color palette:
    colors = ["#2B2F42", "#8D99AE", "#EF233C"]
    labels = ["mse"]

    # Plot each of the main lines
    for i, label in enumerate(labels):
        # Line
        ax.plot(loss, label=label, color=colors[i], linewidth=2)

        # Text
        ax.text(
            loss["epoch"][-1] * 1.01,
            loss["loss"][i][-1],
            label,
            color=colors[i],
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(min(loss["epoch"]), max(loss["epoch"]))

    ax.set_xlabel(r"epoch")
    ax.set_ylabel("loss")
    plt.savefig("great.png", dpi=300)



def ReadAndPlot(data_path, plot_path, problem_name, test_name, classification):
    data_dir = f"{data_path}/{problem_name}"
    plot_dir = f"{plot_path}/{problem_name}"
    try:
        history, result = ReadTrainingDataByName(data_path, problem_name, test_name)
    except FileNotFoundError as e:
        print(f"some file in  {data_dir}/{test_name} not found {e}")
        return

    if classification:
        PlotModelAccuracy(history, test_name, plot_dir, test_name)
    else:
        PlotModelLossMSE(history, test_name, plot_dir, test_name)


def ReadAndPlotAll(data_path, plot_path, problemName, classification):
    dataPath = f"{data_path}/{problemName}"
    for dir in GetDirectSubDir(dataPath):
        ReadAndPlot(data_path, plot_path, problemName, dir.name, classification)


def PrintAllDataByName(data_path, problem_name, classification):
    print(f"------------------------{problem_name}------------------------")
    PrintAllData(f"{data_path}/{problem_name}", classification)


def PrintAllData(path, classification):
    if classification:
        metric_in_history = "val_accuracy"
        metric_to_print = "acc"
    else:
        metric_in_history = "val_mae"
        metric_to_print = "mae"

    print(f"|---------model_name----------|-Val_{metric_to_print}-|-t_{metric_to_print}-|total_time-|")
    PrintTestInFolder(path, metric_in_history)


def PrintTestInFolder(path, metric_in_history):
    for test_name in GetDirectSubDir(f"{path}"):
        history, result = ReadTrainingData(f"{path}/{test_name.name}")
        try:
            print(
                f"{test_name.name:30}  & {history[metric_in_history][-1]:5.5f} & {result[1]:5.5f} & {t.strftime('%H:%M:%S', t.gmtime(sum(history['time'])))}  \\\\")
        except Exception as e:
            # if len(history[metric_in_history]) == 0:
            #     history[metric_in_history] = [history[metric_in_history]]
            print(
                f"{test_name.name:30}  & {history[metric_in_history]:5.5f} & {result[1]:5.5f} & {t.strftime('%H:%M:%S', t.gmtime(sum(history['time'])))} \\\\")


def PrintAllDataAllSubProblem(data_path, problem_name, classification):
    print(f"---------------------{problem_name}---------------------")
    for dir in GetDirectSubDir(f"{data_path}/{problem_name}"):
        print(f"---------------------{dir.name}---------------------")
        CleanData(f"{data_path}/{problem_name}/{dir.name}")
        PrintAllData(f"{data_path}/{problem_name}/{dir.name}", classification)


def RenameDicKey(mydict, oldName, new_name):
    mydict[new_name] = mydict.pop(oldName)


def CleanData(path):
    for test_name in GetDirectSubDir(f"{path}"):
        history, result = ReadTrainingData(f"{path}/{test_name.name}")
        modified = False
        if hasattr(history, "history"):
            history = history.history
            modified = True

        if 'mean_squared_error' in history:
            RenameDicKey(history, 'mean_squared_error', 'loss')
            modified = True
        if 'val_mean_squared_error' in history:
            RenameDicKey(history, 'val_mean_squared_error', 'val_loss')
            modified = True
        if 'mean_absolute_error' in history:
            RenameDicKey(history, 'mean_absolute_error', 'mae')
            modified = True
        if 'val_mean_absolute_error' in history:
            RenameDicKey(history, 'val_mean_absolute_error', 'val_mae')
            modified = True

        if modified:
            SaveTrainingData(f"{path}/{test_name.name}", history, result)
