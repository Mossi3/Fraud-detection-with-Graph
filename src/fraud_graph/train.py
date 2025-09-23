from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

from .data import load_transactions, train_test_split_time
from .graph import NodeIndexer, build_indexer_from_df, build_adjacency, transactions_to_pairs
from .model import TrainConfig, train_link_predictor, predict_scores
from .metrics import pr_auc
from .community import build_weighted_graph, louvain_clusters, compute_cluster_stats
from .viz import fraud_heatmap, export_cluster_graph_json


def _get_offsets(by_type: Dict[str, List[str]]) -> Dict[str, int]:
    offset = 0
    offsets = {}
    for t in ["card", "merchant", "device", "ip"]:
        offsets[t] = offset
        offset += len(by_type.get(t, []))
    return offsets


def run_training(data_path: str, epochs: int, threshold: float, out_dir: str) -> Dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = load_transactions(data_path)
    splits = train_test_split_time(df, test_frac=0.2)

    indexer = build_indexer_from_df(splits.train)
    by_type = indexer.export()
    offsets = _get_offsets(by_type)
    num_cards = len(by_type["card"])
    num_merchants = len(by_type["merchant"])

    adj, deg = build_adjacency(splits.train, indexer)
    pairs_train_global, labels_train = transactions_to_pairs(splits.train, indexer)
    pairs_test_global, labels_test = transactions_to_pairs(splits.test, indexer)

    # Merchant global indices for negative sampling
    merchant_global_indices = np.arange(offsets["merchant"], offsets["merchant"] + num_merchants)

    model, train_info = train_link_predictor(
        adj=adj, deg=deg,
        train_pairs_global=pairs_train_global, train_labels=labels_train,
        merchant_global_indices=merchant_global_indices,
        cfg=TrainConfig(epochs=epochs)
    )

    probs_test = predict_scores(model, adj, deg, pairs_test_global)
    ap = pr_auc(labels_test, probs_test)

    # Build weighted graph for clustering using predicted probabilities on test pairs
    edges_w: List[Tuple[int, int, float]] = []
    edges_l: List[Tuple[int, int, float, int]] = []
    for (u, v), p, y in zip(pairs_test_global, probs_test, labels_test):
        if p >= threshold:
            edges_w.append((int(u), int(v), float(p)))
            edges_l.append((int(u), int(v), float(p), int(y)))

    g = build_weighted_graph(indexer.size(), edges_w)
    clusters = louvain_clusters(g)
    stats = compute_cluster_stats(clusters, edges_l)

    # Persist artifacts
    artifacts_dir = Path(out_dir)
    with open(artifacts_dir / "node_types.json", "w") as f:
        json.dump(by_type, f)
    torch.save(model.state_dict(), artifacts_dir / "model.pt")
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump({"pr_auc": ap, **train_info}, f)

    # Prepare a heatmap for the top cluster (if present)
    if len(stats) > 0:
        top = stats[0]
        # Rebuild the submatrix indices for cards/merchants only
        cards = by_type["card"]
        merchants = by_type["merchant"]
        # Map global pairs back to local card/merchant indices
        sub_pairs: List[Tuple[int, int]] = []
        sub_probs: List[float] = []
        for (u, v), p in zip(pairs_test_global, probs_test):
            if u in top.node_ids and v in top.node_ids:
                # Map back to local indices
                ci = int(u - offsets["card"]) if u >= offsets["card"] and u < offsets["merchant"] else -1
                mi = int(v - offsets["merchant"]) if v >= offsets["merchant"] and v < offsets["device"] else -1
                if ci >= 0 and mi >= 0:
                    sub_pairs.append((ci, mi))
                    sub_probs.append(float(p))
        if len(sub_pairs) > 0:
            hm_png = fraud_heatmap(cards, merchants, np.array(sub_pairs), np.array(sub_probs))
            with open(artifacts_dir / "top_cluster_heatmap.png", "wb") as f:
                f.write(hm_png)

        # Export the graph JSON for visualization
        # Build global index -> (type, id)
        index_to_type_id: Dict[int, Tuple[str, str]] = {}
        base = 0
        for t in ["card", "merchant", "device", "ip"]:
            for nid in by_type[t]:
                index_to_type_id[base] = (t, nid)
                base += 1
        cluster_json = export_cluster_graph_json(top.node_ids, edges_w, index_to_type_id)
        with open(artifacts_dir / "top_cluster_graph.json", "w") as f:
            json.dump(cluster_json, f)

    return {
        "pr_auc": float(ap),
        "num_clusters": len(stats),
        "clusters": [c.__dict__ for c in stats[:10]],
        "artifacts_dir": str(artifacts_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/mock_transactions.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--out", type=str, default="artifacts")
    args = parser.parse_args()

    info = run_training(args.data, args.epochs, args.threshold, args.out)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

