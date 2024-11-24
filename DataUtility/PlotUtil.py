import pickle
from matplotlib import pyplot as plt
from DataUtility.FileUtil import *
import time as t


def PlotModelLossMSE(history, plotTitle='Problems', path=None, filename=None):
    plt.title(plotTitle)

    plt.plot(history['loss'])
    if hasattr(history, "val_loss"):
        plt.plot(history['val_loss'])
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.ylabel('loss(mse)')
    plt.xlabel('epoch')
    ShowOrSavePlot(path, filename)

def PlotModelAccuracy(history, plotTitle='Problems', path=None, filename=None):
    """
    :param history: an object containing metrics values
    :param plotTitle: the title of the plot
    :param path: the path of where to save the plot
    :param filename: the name of where to save the plot
    """
    plt.title(plotTitle)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.plot(history['loss'])
    if hasattr(history, "val_loss"):
        ax1.plot(history['val_loss'])
        ax1.legend(['train', 'validation'], loc='upper left')
    else:
        ax1.legend(['train'], loc='upper left')

    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    if hasattr(history, "val_accuracy"):
        ax2.plot(history['val_accuracy'])
        ax2.legend(['train', 'validation'], loc='upper left')
    else:
        ax1.legend(['train'], loc='upper left')
    ax2.plot(history['accuracy'])
    ShowOrSavePlot(path, filename)


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
    for dir in getDirectSubDir(dataPath):
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
    for test_name in getDirectSubDir(f"{path}"):
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
    for dir in getDirectSubDir(f"{data_path}/{problem_name}"):
        print(f"---------------------{dir.name}---------------------")
        CleanData(f"{data_path}/{problem_name}/{dir.name}")
        PrintAllData(f"{data_path}/{problem_name}/{dir.name}", classification)


def RenameDicKey(mydict, oldName, new_name):
    mydict[new_name] = mydict.pop(oldName)


def CleanData(path):
    for test_name in getDirectSubDir(f"{path}"):
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
