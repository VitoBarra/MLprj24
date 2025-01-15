import json
import os
from statistics import mean, variance

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import metric_ton
from Utility.PlotUtil import ShowOrSavePlot




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

import matplotlib.pyplot as plt
import numpy as np

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
