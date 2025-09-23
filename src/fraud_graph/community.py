from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from community import community_louvain


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    avg_pred_prob: float
    fraud_rate: float
    node_ids: List[int]


def build_weighted_graph(num_nodes: int, edges: List[Tuple[int, int, float]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for u, v, w in edges:
        if u == v:
            continue
        g.add_edge(u, v, weight=float(w))
    return g


def louvain_clusters(g: nx.Graph) -> Dict[int, List[int]]:
    part = community_louvain.best_partition(g, weight="weight")
    clusters: Dict[int, List[int]] = {}
    for node, cid in part.items():
        clusters.setdefault(cid, []).append(node)
    return clusters


def compute_cluster_stats(
    clusters: Dict[int, List[int]],
    edge_list: List[Tuple[int, int, float, int]],
) -> List[ClusterInfo]:
    node_to_cluster: Dict[int, int] = {}
    for cid, nodes in clusters.items():
        for n in nodes:
            node_to_cluster[n] = cid

    per_cluster_edges: Dict[int, List[Tuple[float, int]]] = {}
    for u, v, p, label in edge_list:
        cu = node_to_cluster.get(u)
        cv = node_to_cluster.get(v)
        if cu is None or cv is None or cu != cv:
            continue
        per_cluster_edges.setdefault(cu, []).append((p, label))

    out: List[ClusterInfo] = []
    for cid, nodes in clusters.items():
        e = per_cluster_edges.get(cid, [])
        if len(e) == 0:
            avg_prob = 0.0
            fraud_rate = 0.0
        else:
            probs = [p for p, _ in e]
            labels = [l for _, l in e]
            avg_prob = float(np.mean(probs))
            fraud_rate = float(np.mean(labels))
        out.append(ClusterInfo(
            cluster_id=cid,
            size=len(nodes),
            avg_pred_prob=avg_prob,
            fraud_rate=fraud_rate,
            node_ids=sorted(nodes),
        ))
    out.sort(key=lambda c: (c.avg_pred_prob, c.fraud_rate, c.size), reverse=True)
    return out

