import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from Core.DataSet.DataSet import DataSet
from Core.Metric import Metric
from Utility.PlotUtil import ShowOrSavePlot




def PlotAverage(metricList: list[dict[str, list[float]]], title="Title", xlabel="X-axis", ylabel="Y-axis",
                limitYRange=None,limitXRange=None, WarmUpEpochs: int = 0, baseline=None, baselineName=None, subplotAxes=None, NumbersOfMarker:int = 45):
    """
    Plot the average of multiple metrics along with individual trials for each metric.

    :param baselineName:
    :param baseline:
    :param limitXRange:
    :param metricList: Dictionary where keys are metric names and values are lists of lists,
                       where each inner list represents the metric values for a trial.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param limitYRange: (Optional) Tuple specifying the y-axis limits (min, max).
    :param WarmUpEpochs: Initial limit for the x-axis.
    :param subplotAxes: (Optional) A matplotlib Axes object to use for a subplot. If None, a new figure is created.
    :param title: Title of the plot.
    """
    if subplotAxes is None:
        plt.figure(figsize=(18, 6))
        ax = plt.gca()
    else:
        ax = subplotAxes

    # Define colors, different line styles, and markers for black-and-white readability
    colors = ["blue", "black", "red" , "orange", "green", "purple", "brown"]
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 'D', 's', '^', 'v', '*']

    # Track the global min and max of the averages for dynamic y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    min_len = 0

    metric_names = [key for key,_ in metricList[0].items()]
    dataObj = {}

    # Find the minimum length across all trials for this metric
    for metric_name in metric_names:
        metric_data= {}
        if min_len == 0:
            min_len=min([len(metricObj[metric_name]) for metricObj in metricList]) - WarmUpEpochs
        metric_data["data_sync"] = [metricObj[metric_name][WarmUpEpochs:min_len+WarmUpEpochs] for metricObj in metricList]
        metric_data["average_curve"] = np.mean(metric_data["data_sync"], axis=0)

        dataObj[metric_name] = metric_data



    for  idx,metric_name in enumerate(metric_names):

        metric_data =dataObj[metric_name]
        # Plot individual trials for the current metric
        for trial in metric_data["data_sync"]:
            ax.plot(
                    trial,
                    alpha=0.1,
                    color=colors[idx % len(colors)],
                    linestyle='-',
                    markersize=5,
                    linewidth=1)

        # Plot average for the current metric
        global_min = min(global_min, metric_data["average_curve"].min())  # Update global min
        global_max = max(global_max, metric_data["average_curve"].max())  # Update global max


        # Define marker spacing dynamically based on x-axis range
        marker_spacing = max(1, min_len // NumbersOfMarker) if min_len > 50 else 1
        marker_indices = list(range(0, min_len, marker_spacing))


        ax.plot(metric_data["average_curve"],
                label=f"{metric_name} (mean)",
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)],
                markevery=marker_indices,
                markersize=5,
                linewidth=2)


    # Plot the baseline
    if baseline is not None:
        ax.plot(
            [baseline for _ in range(min_len)],
            label=baselineName,
            color="green",
            linestyle="--",
            linewidth=3)

    # Adjust y-axis limits dynamically if not explicitly provided
    if limitYRange is None:
        padding = (global_max - global_min) * 0.1  # Add 10% padding
        limitYRange = (global_min - padding, global_max + padding)


    if limitXRange is None:
        limitXRange = (WarmUpEpochs, min_len)

    # Add title, labels, and legend
    PlotStyle(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, limitYRange=limitYRange, limitXRange =limitXRange)



def PlotMetrics(metricDic, baseline=None, baselineName="Baseline", title="Title", xlabel="X-axis", ylabel="Y-axis", limitYRange=None, path=None, subplotAxes=None):
    """
     Plot metrics with optional baseline as a standalone plot or as part of a subplot.

     :param metricDic: A dictionary where keys are labels and values are lists of metric values to plot.
     :param baseline: (Optional) A constant baseline value to plot. Default is None.
     :param baselineName: (Optional) The label for the baseline line. Default is "Baseline".
     :param title: (Optional) The title of the plot or subplot. Default is "Title".
     :param xlabel: (Optional) The label for the x-axis. Default is "X-axis".
     :param ylabel: (Optional) The label for the y-axis. Default is "Y-axis".
     :param limitYRange: (Optional) A tuple specifying the y-axis limits (min, max). Default is None.
     :param path: (Optional) The path to save the plot if provided. Default is None.
     :param subplotAxes: (Optional) A matplotlib Axes object to use for a subplot. If None, a new figure is created.
     """
    # If no subplot axes are provided, create a new figure
    if subplotAxes is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()  # Get the current axes
    else:
        ax = subplotAxes

    # Define different line styles and markers for black-and-white readability
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^']

    # Plot each loss function with different styles
    labels = metricDic.keys()
    for i, label in enumerate(labels):
        ax.plot(
            metricDic[label],
            label=label,
            linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
            marker=markers[i % len(markers)],  # Cycle through markers
            markersize=5,
            linewidth=1.5
        )

    # Plot the baseline
    if baseline is not None:
        ax.plot(
            [baseline for _ in range(len(list(metricDic.values())[0]))],
            label=baselineName,
            color="magenta",
            linestyle="--",
            linewidth=1,
        )

    # Add title, labels, and legend
    PlotStyle(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, limitYRange=limitYRange)

    # Save the plot if path is provided, only for standalone plots
    if subplotAxes is None:
        ShowOrSavePlot(path, title)


def PlotStyle (ax, title, xlabel, ylabel, limitYRange=None,limitXRange = None):
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if limitYRange is not None:
        ax.set_ylim(limitYRange)
    if limitXRange is not None:
        ax.set_xlim(limitXRange)
    ax.legend(fontsize=14)
    ax.grid(True)


def plot_CUP_3d(*arrays, title="3D graph", labels=None, colors=None, markers=None, alpha=0.7, path=None, values=None, cmap='viridis'):
    """
    Visualize 3D points from one or more 3D arrays.

    :param title: The title of the plot.
    :param arrays: A list of NumPy arrays with shape (n, 3), each representing a set of 3D points.
    :param labels: A list of strings, one for each array, used for the legend (optional).
    :param colors: A list of colors (e.g., 'blue', 'red', etc.) for each array (optional).
    :param markers: A list of markers (e.g., 'o', '^', etc.) for each array (optional).
    :param alpha: The transparency of the points (default: 0.7).
    :param path: The path to save the plot (optional).
    :param values: A list of arrays with values for colorizing the points (optional).
    :param cmap: The colormap to use for colorizing the points (default: 'viridis').
    :raises ValueError: If any of the arrays do not have exactly 3 columns.
    """
    # Check that all arrays have a valid shape
    for i, array in enumerate(arrays):
        if array.shape[1] != 3:
            raise ValueError(f"Array {i + 1} does not have 3 columns. Each array must have exactly 3 dimensions.")

    # Set default colors and markers if not provided
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan'] * len(arrays)
    if markers is None:
        markers = ['o', '^', 's', 'D', 'P', '*'] * len(arrays)
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(arrays))]
    # Create the 3D figure
    if path is None and os.name == "posix":
        matplotlib.use("TkAgg") #This could now work depending on os
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    # Add each array to the plot
    for i, array in enumerate(arrays):
        x = array[:, 0]
        y = array[:, 1]
        z = array[:, 2]
        if values is not None:
            sc = ax.scatter(x, y, z, c=values, cmap=cmap, marker=markers[i], alpha=alpha, edgecolor='k', label=labels[i])
            plt.colorbar(sc, ax=ax, label='Value')
        else:
            ax.scatter(x, y, z, c=colors[i], marker=markers[i], alpha=alpha, edgecolor='k', label=labels[i])


    # Configure the axes
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend(loc='upper right', fontsize=10)

    # Show the plot
    ShowOrSavePlot(path,title)


def CupOutWithDistanceGradient(cup_all:DataSet, mee:Metric):
    mean_distance = []
    for data_point_label in cup_all.Data.Label:

        # Specify the shape and the number to fill
        shape = cup_all.Data.Label.shape
        # Create the array
        allbleClone = np.full(shape, data_point_label)
        e =cup_all.Data.Label
        destance = mee(allbleClone, e)
        print(destance)
        mean_distance.append(destance)
    plot_CUP_3d(cup_all.Data.Label, values=mean_distance)