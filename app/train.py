import os
import json
import argparse
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from sklearn.metrics import average_precision_score

from app.data.generate_mock_data import generate_mock_transactions, save_mock_csv
from app.data.loaders import load_relational_csv
from app.graph.build_graph import build_hetero_graph
from app.models.gnn import HeteroSAGE
from app.graph.community import hetero_to_nx, louvain_partition, summarize_communities, extract_community_subgraph, build_card_merchant_heatmap_data
from app.viz.plots import plot_pr_curve, plot_heatmap, plot_ring_graph


ARTIFACTS_DIR = os.path.join("artifacts")
OUTPUTS_DIR = os.path.join("outputs")
DATA_DIR = os.path.join("data")


def train_model(
    use_mock: bool = True,
    csv_path: str = "",
    num_transactions: int = 20000,
    seed: int = 42,
) -> Dict[str, Any]:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if use_mock:
        df = generate_mock_transactions(num_transactions=num_transactions, seed=seed)
        csv_path = os.path.join(DATA_DIR, "mock_relational.csv")
        save_mock_csv(df, csv_path)
    else:
        if not csv_path:
            raise ValueError("csv_path must be provided when use_mock=False")
        df = load_relational_csv(csv_path)

    data, id_maps = build_hetero_graph(df)
    in_dims = {ntype: int(data[ntype].x.size(1)) for ntype in ["card", "merchant", "device", "ip"]}
    model = HeteroSAGE(in_dims=in_dims, hidden_channels=64, dropout=0.2, num_layers=2)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train/val split on cards
    y = data["card"].y.numpy()
    num_cards = y.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(num_cards)
    rng.shuffle(indices)
    split = int(0.8 * num_cards)
    train_idx = torch.tensor(indices[:split], dtype=torch.long)
    val_idx = torch.tensor(indices[split:], dtype=torch.long)

    def forward_pass():
        logits_card, _ = model(data.x_dict, data.edge_index_dict)
        return logits_card

    best_ap = 0.0
    best_state = None
    for epoch in range(1, 16):  # small number for school demo
        model.train()
        optimizer.zero_grad()
        logits = forward_pass()
        loss = criterion(logits[train_idx], data["card"].y[train_idx])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            logits = forward_pass()
            probs = torch.softmax(logits[val_idx], dim=1)[:, 1].cpu().numpy()
            ap = float(average_precision_score(data["card"].y[val_idx].cpu().numpy(), probs))
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final scores on all cards
    model.eval()
    with torch.no_grad():
        logits, hidden = model(data.x_dict, data.edge_index_dict)
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # Metrics and plots
    ap_all = float(average_precision_score(y, scores))
    metrics = {"pr_auc": round(ap_all, 6), "num_cards": int(len(y))}
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_pr_curve(y_true=y, y_score=scores, out_path=os.path.join(OUTPUTS_DIR, "pr_curve.png"))

    # Community detection
    G = hetero_to_nx(data)
    part = louvain_partition(G)
    ring_summaries = summarize_communities(G, part, card_labels=y, top_k=10)
    with open(os.path.join(ARTIFACTS_DIR, "rings.json"), "w") as f:
        json.dump(ring_summaries, f, indent=2)

    if len(ring_summaries) > 0:
        top_cid = ring_summaries[0]["community_id"]
        sub = extract_community_subgraph(G, top_cid, part)
        plot_ring_graph(sub, out_path=os.path.join(OUTPUTS_DIR, "ring_graph.png"), title="Top Ring Subgraph")
        mat, rows, cols = build_card_merchant_heatmap_data(sub)
        if mat.size > 0:
            plot_heatmap(mat, rows, cols, out_path=os.path.join(OUTPUTS_DIR, "card_merchant_heatmap.png"), title="Card-Merchant Co-occurrence")

    # Save artifacts
    torch.save({
        "state_dict": model.state_dict(),
        "in_dims": in_dims,
    }, os.path.join(ARTIFACTS_DIR, "model.pt"))
    with open(os.path.join(ARTIFACTS_DIR, "id_maps.json"), "w") as f:
        json.dump(id_maps, f)
    np.save(os.path.join(ARTIFACTS_DIR, "card_scores.npy"), scores)

    return {"metrics": metrics, "rings": ring_summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--num-transactions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_model(use_mock=args.use_mock, csv_path=args.csv_path, num_transactions=args.num_transactions, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

