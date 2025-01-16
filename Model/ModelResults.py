import json
import os
from statistics import mean, variance

import numpy as np
from matplotlib import pyplot as plt
from Utility.PlotUtil import ShowOrSavePlot

from scipy.constants import metric_ton




def PlotTableVarianceAndMean(results: dict) -> None:
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


def PlotMultipleModels(results: list[dict], metrics: list[str], path: str = None, filename: str = None) -> None:
    """
    Plot multiple metrics for multiple models, including individual model curves and the mean curve for each metric.

    Metrics containing similar keywords (e.g., "loss" or "accuracy") will be grouped into the same subplot.

    :param results: List of dictionaries containing model metrics.
    :param metrics: List of metrics to plot.
    :param path: (Optional) Path to save the plot.
    :param filename: (Optional) Filename to save the plot.
    """
    # Group metrics by keyword (e.g., "loss", "accuracy")
    grouped_metrics = {}
    for metric in metrics:
        if "loss" in metric:
            key = "loss"
        elif "Accuracy" in metric:
            key = "Accuracy"
        else:
            key = metric

        if key not in grouped_metrics:
            grouped_metrics[key] = []
        grouped_metrics[key].append(metric)

    num_groups = len(grouped_metrics)

    # Create subplots
    fig, axes = plt.subplots(1, num_groups, figsize=(12 * num_groups, 6), squeeze=False)

    colors = ["blue", "green", "orange", "purple", "red"]  # Predefined color palette

    for idx, (group, group_metrics) in enumerate(grouped_metrics.items()):
        ax = axes[0, idx]

        for i, metric in enumerate(group_metrics):
            # Extract data for the current metric
            all_metric = [metric_data[metric] for metric_data in results]
            min_length = min(len(model_data) for model_data in all_metric)
            all_metric = [model_data[:min_length] for model_data in all_metric]

            color = colors[i % len(colors)]  # Assign color for each metric

            # Plot individual model curves
            for model_data in all_metric:
                ax.plot(model_data, alpha=0.1, color=color)  # Individual curves, no label, very light

            # Compute and plot the mean curve
            all_metric = np.array(all_metric)
            mean_metric = np.mean(all_metric, axis=0)
            ax.plot(mean_metric, linewidth=2, label=f'{metric} (mean)', color=color)

        # Set axis labels and title
        ax.set_xlabel("Epochs", fontsize=14)
        ax.set_ylabel(f"{group}", fontsize=14)
        ax.set_title(f"Metrics: {group}", fontsize=16)
        ax.legend(fontsize=10)

    plt.tight_layout()

    # Save or show the plot
    if path and filename:
        plt.savefig(f"{path}/{filename}")
    else:
        plt.show()




def SaveResults(results, path: str = "DataSet/Results/model_results.json") -> None:
    """
    Save the model results to a JSON file.

    :param results: Dictionary of model results (metrics and hyperparameters).
    :param path: Path to save the results JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def PlotMetrics(metricDic, baseline=None, baselineName="Baseline", title="Title",
                xlabel="X-axis", ylabel="Y-axis", limitYRange=None, path=None, subplotAxes=None):
    """
    brief : Plots metrics with optional baseline, as either a standalone plot or a subplot.

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
    ax.set_title(title,fontsize=22)
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    if limitYRange is not None:
        ax.set_ylim(*limitYRange)
    ax.legend()
    ax.legend(fontsize=18)
    ax.grid(True)

    # Save the plot if path is provided, only for standalone plots
    if subplotAxes is None:
        ShowOrSavePlot(path, title)