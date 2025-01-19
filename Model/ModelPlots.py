import json
import os
from statistics import mean, variance

import numpy as np
from matplotlib import pyplot as plt
from Utility.PlotUtil import ShowOrSavePlot

from scipy.constants import metric_ton


def PlotAverage(metricDict: dict[str, list[list]], xlabel="X-axis", ylabel="Y-axis", limitYRange=None, WarmUpEpochs:int = 0, subplotAxes=None, title="Title"):
    """
    Plot the average of multiple metrics along with individual trials for each metric.

    :param metricDict: Dictionary where keys are metric names and values are lists of lists,
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
    colors = ["blue", "red", "orange", "green", "purple", "brown"]
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 'D', 's', '^', 'v', '*']

    # Track the global min and max of the averages for dynamic y-axis limits
    global_min = float('inf')
    global_max = float('-inf')

    for idx, (metric_name, metricList) in enumerate(metricDict.items()):
        # Find the minimum length across all trials for this metric
        min_length = min(len(trial) for trial in metricList)

        # Synchronize all trials to the same length
        synced_trials = [trial[:min_length] for trial in metricList]


        # Plot individual trials for the current metric
        for trial in synced_trials:
            x_values = range(WarmUpEpochs, WarmUpEpochs + len(trial))
            ax.plot(x_values,
                    trial,
                    alpha=0.1,
                    color=colors[idx % len(colors)],
                    linestyle='-',
                    markersize=5,
                    linewidth=1)

        # Plot average for the current metric
        if synced_trials:
            mean_metric = np.mean(synced_trials, axis=0)
            global_min = min(global_min, np.min(mean_metric))  # Update global min
            global_max = max(global_max, np.max(mean_metric))  # Update global max

            x_values = range(WarmUpEpochs, WarmUpEpochs + len(mean_metric))

            # Define marker spacing dynamically based on x-axis range
            marker_spacing = max(1, len(mean_metric) // 45) if len(mean_metric) > 50 else 1
            marker_indices = [i for i in range(len(mean_metric)) if i % marker_spacing == 0]

            ax.plot(x_values,
                    mean_metric,
                    label=f"{metric_name} (mean)",
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    marker=markers[idx % len(markers)],
                    markevery=marker_indices,
                    markersize=5,
                    linewidth=2)

    # Adjust y-axis limits dynamically if not explicitly provided
    if limitYRange is None:
        padding = (global_max - global_min) * 0.1  # Add 10% padding
        ax.set_ylim(global_min - padding, global_max + padding)
    else:
        ax.set_ylim(limitYRange)


    # Add title, labels, and legend
    PlotStyle(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, limitYRange=limitYRange, WarmUpEpoch=WarmUpEpochs)



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


def PlotStyle (ax, title, xlabel, ylabel, limitYRange=None, WarmUpEpoch = None):
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if limitYRange is not None:
        ax.set_ylim(*limitYRange)
    if WarmUpEpoch is not None:
        ax.set_xlim(WarmUpEpoch, None)
    ax.legend(fontsize=14)
    ax.grid(True)
