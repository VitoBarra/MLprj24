import pickle

import matplotlib.pyplot as plt
import networkx as nx

from Utility.FileUtil import *



def plot_metrics(metricDic, baseline=None, baselineName="Baseline", title="Title",
                 xlabel="X-axis", ylabel="Y-axis", limitYRange=None, path=None, subplotAxes=None):
    """
    :brief Plots metrics with optional baseline, as either a standalone plot or a subplot.

    :param metricDic A dictionary where keys are labels and values are lists of metric values to plot.
    :param baseline (Optional) A constant baseline value to plot. Default is None.
    :param baselineName (Optional) The label for the baseline line. Default is "Baseline".
    :param title (Optional) The title of the plot or subplot. Default is "Title".
    :param xlabel (Optional) The label for the x-axis. Default is "X-axis".
    :param ylabel (Optional) The label for the y-axis. Default is "Y-axis".
    :param limitYRange (Optional) A tuple specifying the y-axis limits (min, max). Default is None.
    :param path (Optional) The path to save the plot if provided. Default is None.
    :param subplotAxes (Optional) A matplotlib Axes object to use for a subplot. If None, a new figure is created.
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






def plot_neural_network_with_transparency(weights , plotName = "Neural Network Diagram"):
    """
    Plots a neural network visualization using transparency for edge weights.

    Args:
        weights: A list of 2D numpy arrays representing weight matrices between layers.
    """
    # Initialize graph
    G = nx.DiGraph()

    # First layer size + sizes from weight matrices
    edges = []
    edge_colors = []
    edge_alphas = []

    shapes = [w.shape for w in weights]
    midlePoint= np.array(shapes).max()/2

    #reigster the intuput and internal Node
    for i,weight_matrix in enumerate(weights):
        src_dim=weight_matrix.shape[1]
        rowStartPoint =midlePoint-(src_dim/2)
        for src_idx in range(src_dim):
            G.add_node(f"L{i}_N{src_idx}", pos=(rowStartPoint+ src_idx, i))

    # register the output node
    weight_matrix = weights[-1]
    tgt_dim=weight_matrix.shape[0]
    rowStartPoint =midlePoint-(tgt_dim/2)
    for tgt_idx in range(tgt_dim):
        G.add_node(f"L{len(weights)}_N{tgt_idx}", pos=(rowStartPoint+tgt_idx, len(weights)))

    # Add edges with weights
    for i,weight_matrix in enumerate(weights):
        tgt_dim=weight_matrix.shape[0]
        src_dim=weight_matrix.shape[1]
        max_weight_module = np.max(np.abs(weight_matrix))
        for tgt_idx in range(tgt_dim):
            for src_idx in range(src_dim):
                src_node = f"L{i}_N{src_idx}"
                tgt_node = f"L{i+1}_N{tgt_idx}"
                weight = weight_matrix[tgt_idx,src_idx ]
                G.add_edge(src_node, tgt_node, weight=weight)
                edges.append((src_idx, tgt_idx))
                # Calculate transparency and color based on weight magnitude
                intensity = abs(weight) / max_weight_module
                # Transparency (smaller weight = more transparent)
                if intensity <= 0:
                    alpha = 0.05
                    print(f"{intensity=}")
                elif intensity > 1:
                    alpha = 1
                    print(f"{intensity=}")
                else:
                    alpha = intensity
                color = "red" if weight < 0 else "blue"
                edge_colors.append(color)
                edge_alphas.append(alpha)

    # Draw the graph
    plt.figure(figsize=(10, 8))

    # Node positions
    pos = {node: (data["pos"][0], data["pos"][1]) for node, data in G.nodes(data=True)}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightgreen")

    # Draw edges with transparency
    for edge, color, alpha in zip(G.edges(), edge_colors, edge_alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge],
            edge_color=color,
            alpha=alpha,
            width=2
        )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels={node: f"{node}" for node in G.nodes()}, font_size=8)

    # Draw edge labels (weights)
    edge_labels = {(src, tgt): f"{G.edges[src, tgt]['weight']:.2f}" for src, tgt in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(plotName, fontsize=16)
    plt.axis("off")
    plt.show()






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