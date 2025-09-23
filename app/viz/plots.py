from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: str) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], out_path: str, title: str = "Heatmap") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(max(6, len(col_labels) * 0.4), max(4, len(row_labels) * 0.3)))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels, cmap="magma", cbar=True)
    plt.title(title)
    plt.xlabel("Merchants")
    plt.ylabel("Cards")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ring_graph(G_sub: nx.Graph, out_path: str, title: str = "Ring Subgraph") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G_sub, seed=42, k=0.4)
    colors = {"card": "tab:blue", "merchant": "tab:orange", "device": "tab:green", "ip": "tab:red"}
    for ntype, color in colors.items():
        nodes = [n for n in G_sub.nodes if n[0] == ntype]
        nx.draw_networkx_nodes(G_sub, pos, nodelist=nodes, node_color=color, node_size=200, label=ntype)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    nx.draw_networkx_labels(G_sub, pos, labels={n: f"{n[0][0]}{n[1]}" for n in G_sub.nodes}, font_size=6)
    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

