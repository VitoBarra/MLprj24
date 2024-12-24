
import matplotlib.pyplot as plt
import networkx as nx


from DataUtility.FileUtil import *



#si può usare anche oer l'accuracu, basta passargli il valore oer training e validation con i corretti label
def plot_metric(metricDic: dict[list[float]] | np.ndarray, baseline: float = None,
                title: str = "Loss per Epoch", xlabel: str = "Epochs", ylabel: str = "Loss Value", limityRange = None,
                path: str = None) -> None:
    """
    Plots loss curves over epochs using a given loss matrix, using different line styles for better distinction in black and white.

    Args:
        metricDic (list of lists or numpy.ndarray): A matrix where each row represents an epoch
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

    labels = metricDic.keys()

    # Define different line styles and markers for black-and-white readability
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^']

    plt.figure(figsize=(10, 6))


    # Plot each loss function with different styles
    for i,label in enumerate(labels):
        plt.plot(
            metricDic[label],
            label=label,
            linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
            marker=markers[i % len(markers)],  # Cycle through markers
            markersize=5,
            linewidth=1.5
        )

        # Plot the baseline
    if baseline is not None:
        plt.plot(
            [baseline for _ in range(len(list(metricDic.values())[0]))],
            label="Baseline",
            color="magenta",
            linestyle="--",
            linewidth=1,
        )

    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if limityRange is not None:
        plt.ylim(*limityRange)
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
                alpha = intensity  # Transparency (smaller weight = more transparent)
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

