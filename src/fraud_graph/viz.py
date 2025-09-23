from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def fraud_heatmap(
    card_ids: List[str],
    merchant_ids: List[str],
    pairs: np.ndarray,
    probs: np.ndarray,
    cluster_node_to_type_id: Dict[int, Tuple[str, str]] | None = None,
) -> bytes:
    num_cards = len(card_ids)
    num_merchants = len(merchant_ids)
    mat = np.zeros((num_cards, num_merchants), dtype=float)
    for (ci, mi), p in zip(pairs, probs):
        if 0 <= ci < num_cards and 0 <= mi < num_merchants:
            mat[int(ci), int(mi)] = max(mat[int(ci), int(mi)], float(p))

    plt.figure(figsize=(max(6, num_merchants * 0.3), max(4, num_cards * 0.3)))
    sns.heatmap(mat, cmap="Reds", cbar=True)
    plt.xlabel("Merchants")
    plt.ylabel("Cards")
    plt.title("Fraud Risk Heatmap (Card x Merchant)")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf.read()


def export_cluster_graph_json(
    nodes: List[int],
    edges: List[Tuple[int, int, float]],
    index_to_type_id: Dict[int, Tuple[str, str]],
) -> Dict:
    node_list = []
    node_index_map = {n: i for i, n in enumerate(nodes)}
    for n in nodes:
        t, nid = index_to_type_id[n]
        node_list.append({"id": int(node_index_map[n]), "global_id": int(n), "type": t, "name": nid})

    edge_list = []
    for u, v, w in edges:
        if u in node_index_map and v in node_index_map:
            edge_list.append({
                "source": int(node_index_map[u]),
                "target": int(node_index_map[v]),
                "weight": float(w),
            })

    return {"nodes": node_list, "edges": edge_list}

