import matplotlib.pyplot as plt
import networkx as nx
from .FileUtil import *
import matplotlib.pyplot as plt
import networkx as nx

from .FileUtil import *


def ShowOrSavePlot(path=None, filename=None):
    """
    Displays a plot or saves it to a specified path.

    :param path: Directory path to save the plot. If None, the plot is displayed.
    :param filename: Name of the file (without extension) to save the plot. Defaults to 'img' if not specified.
    """
    if path is None or path == '':
        plt.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None or filename == '':
            filename = 'img'
        plt.savefig(f"{path}/{filename}.png")
        plt.close()



def plot_neural_network_with_transparency(weights , plotName = "Neural Network Diagram"):
    """
    Plots a neural network diagram with edge transparency based on weight magnitudes.

    :param weights: List of 2D numpy arrays representing weight matrices between layers.
    :param plotName: Title of the plot.
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
def calculate_roc(y_scores: list[float], y_true: list[int]) -> tuple[list[float], list[float], list[float]]:
    """
    Calculates the ROC curve.

    :param y_scores: Predicted scores or probabilities.
    :param y_true: True binary labels (0 or 1).
    :return: A tuple (fpr_list, tpr_list, thresholds).
    """
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list, fpr_list = [], []
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
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list, thresholds


# Funzione per calcolare l'AUC usando la formula del trapezio
def calculate_auc(fpr: list[float], tpr: list[float]) -> float:
    """
    Calculates the Area Under the Curve (AUC) using the trapezoidal rule.

    :param fpr: List of false positive rates.
    :param tpr: List of true positive rates.
    :return: The calculated AUC.
    """
    return sum((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2 for i in range(1, len(fpr)))



def printAUC(fpr: list[float], tpr: list[float], auc: float) -> None:
    """
    Plots the ROC curve and displays the AUC.

    :param fpr: List of false positive rates.
    :param tpr: List of true positive rates.
    :param auc: Calculated AUC value.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', linestyle='-', color='b', label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


