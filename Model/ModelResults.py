import json
import os
from statistics import mean, variance

import numpy as np
from matplotlib import pyplot as plt
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

def PlotMultipleModels(results: dict,metric: str = "test_loss") -> None:
    """
    Plot the test loss for multiple models, including individual model losses and the mean curve.

    :param results: Dictionary containing model metrics.
    """
    plt.figure(figsize=(12, 8))

    all_losses = []
    min_length = min(len(model_data["metrics"][metric]) for model_data in results.values())

    for model_name, model_data in results.items():
        metric_values = model_data["metrics"][metric][2:min_length]
        all_losses.append(metric_values)
        plt.plot(metric_values, color='dodgerblue', alpha=0.3)

    all_losses = np.array(all_losses)
    mean_loss = np.mean(all_losses, axis=0)
    plt.plot(mean_loss, color='blue', linewidth=2, label=f'{metric} Mean')

    plt.xlabel("Epochs")
    plt.ylabel(f"{metric}")
    plt.title(f"{metric} Comparison for Multiple Models")
    plt.legend()
    plt.show()



def SaveResults(results, path: str = "Data/Results/model_results.json") -> None:
    """
    Save the model results to a JSON file.

    :param results: Dictionary of model results (metrics and hyperparameters).
    :param path: Path to save the results JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
