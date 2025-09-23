from typing import Dict, Any, List, Tuple
import os
import math
import json
import networkx as nx
import numpy as np
import torch
from community import community_louvain


def hetero_to_nx(data) -> nx.Graph:
    G = nx.Graph()
    # Add nodes with type
    for node_type in ["card", "merchant", "device", "ip"]:
        num = int(data[node_type].x.size(0))
        for i in range(num):
            G.add_node((node_type, int(i)), ntype=node_type)

    # Add edges (undirected)
    for edge_type in [
        ("card", "transacts", "merchant"),
        ("card", "uses", "device"),
        ("device", "routes", "ip"),
        ("card", "from_ip", "ip"),
    ]:
        if edge_type in data.edge_types:
            ei = data[edge_type].edge_index
            src_type, _, dst_type = edge_type
            for s, t in zip(ei[0].tolist(), ei[1].tolist()):
                s_key = (src_type, int(s))
                t_key = (dst_type, int(t))
                if G.has_edge(s_key, t_key):
                    G[s_key][t_key]["weight"] += 1
                else:
                    G.add_edge(s_key, t_key, weight=1)
    return G


def louvain_partition(G: nx.Graph) -> Dict[Any, int]:
    # Convert to a simple graph with weights
    return community_louvain.best_partition(G, weight="weight")


def summarize_communities(
    G: nx.Graph,
    partition: Dict[Any, int],
    card_labels: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    # Group nodes by community id
    comm_to_nodes: Dict[int, List[Any]] = {}
    for node, cid in partition.items():
        comm_to_nodes.setdefault(cid, []).append(node)

    summaries: List[Dict[str, Any]] = []
    for cid, nodes in comm_to_nodes.items():
        type_counts = {"card": 0, "merchant": 0, "device": 0, "ip": 0}
        card_indices: List[int] = []
        for (ntype, idx) in nodes:
            type_counts[ntype] += 1
            if ntype == "card":
                card_indices.append(idx)

        if len(card_indices) == 0:
            continue
        labels = card_labels[card_indices]
        fraud_rate = float(labels.mean()) if len(labels) > 0 else 0.0
        majority = max(labels.sum(), len(labels) - labels.sum())
        purity = float(majority) / float(len(labels))
        ring_score = fraud_rate * math.log(1 + len(card_indices))
        summaries.append(
            {
                "community_id": int(cid),
                "num_cards": int(type_counts["card"]),
                "num_merchants": int(type_counts["merchant"]),
                "num_devices": int(type_counts["device"]),
                "num_ips": int(type_counts["ip"]),
                "fraud_rate": round(fraud_rate, 4),
                "purity": round(purity, 4),
                "ring_score": round(ring_score, 4),
            }
        )

    summaries.sort(key=lambda x: (x["ring_score"], x["fraud_rate"], x["num_cards"]), reverse=True)
    return summaries[:top_k]


def extract_community_subgraph(G: nx.Graph, community_id: int, partition: Dict[Any, int]) -> nx.Graph:
    nodes = [n for n, cid in partition.items() if cid == community_id]
    return G.subgraph(nodes).copy()


def build_card_merchant_heatmap_data(subgraph: nx.Graph) -> Tuple[np.ndarray, List[str], List[str]]:
    cards = [n for n in subgraph.nodes if n[0] == "card"]
    merchants = [n for n in subgraph.nodes if n[0] == "merchant"]
    card_to_idx = {n: i for i, n in enumerate(cards)}
    merch_to_idx = {n: i for i, n in enumerate(merchants)}
    mat = np.zeros((len(cards), len(merchants)), dtype=np.float32)
    for u, v, d in subgraph.edges(data=True):
        if (u[0] == "card" and v[0] == "merchant") or (u[0] == "merchant" and v[0] == "card"):
            if u[0] == "card":
                cu, mv = u, v
            else:
                cu, mv = v, u
            mat[card_to_idx[cu], merch_to_idx[mv]] += float(d.get("weight", 1))
    row_labels = [f"card_{n[1]}" for n in cards]
    col_labels = [f"merch_{n[1]}" for n in merchants]
    return mat, row_labels, col_labels

