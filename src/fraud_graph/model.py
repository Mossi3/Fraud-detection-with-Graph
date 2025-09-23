from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        neigh_sum = torch.sparse.mm(adj, h)
        neigh_mean = neigh_sum / deg.unsqueeze(1)
        out = self.lin_self(h) + self.lin_neigh(neigh_mean) + self.bias
        return F.relu(out)


class GraphSAGEModel(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_nodes, embed_dim)
        self.gnn1 = GraphSAGELayer(embed_dim, hidden_dim)
        self.gnn2 = GraphSAGELayer(hidden_dim, hidden_dim)

    def forward(self, adj: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        h0 = self.emb.weight
        h1 = self.gnn1(h0, adj, deg)
        h2 = self.gnn2(h1, adj, deg)
        return h2


@dataclass
class TrainConfig:
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    negative_ratio: int = 1


def sample_negative_pairs(
    pairs: np.ndarray,
    merchant_global_indices: np.ndarray,
    negative_ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_pos = pairs.shape[0]
    cards = pairs[:, 0]
    neg_cards = np.repeat(cards, negative_ratio)
    neg_merchants = rng.choice(merchant_global_indices, size=num_pos * negative_ratio, replace=True)
    neg_pairs = np.stack([neg_cards, neg_merchants], axis=1)
    return neg_pairs


def train_link_predictor(
    adj: torch.Tensor,
    deg: torch.Tensor,
    train_pairs_global: np.ndarray,
    train_labels: np.ndarray,
    merchant_global_indices: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[GraphSAGEModel, Dict[str, float]]:
    device = torch.device("cpu")
    num_nodes = adj.shape[0]

    model = GraphSAGEModel(num_nodes=num_nodes, embed_dim=64, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(42)

    labels_t = torch.tensor(train_labels, dtype=torch.float32, device=device)

    for epoch in range(cfg.epochs):
        model.train()

        neg_pairs = sample_negative_pairs(train_pairs_global.copy(), merchant_global_indices, cfg.negative_ratio, rng)
        neg_labels = np.zeros(len(neg_pairs), dtype=np.int64)

        all_pairs = np.concatenate([train_pairs_global, neg_pairs], axis=0)
        all_labels = np.concatenate([train_labels, neg_labels], axis=0)

        perm = rng.permutation(len(all_pairs))
        all_pairs = all_pairs[perm]
        all_labels = all_labels[perm]

        all_pairs_t = torch.tensor(all_pairs, dtype=torch.long, device=device)
        all_labels_t = torch.tensor(all_labels, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        z = model(adj, deg)
        card_idx = all_pairs_t[:, 0]
        merch_idx = all_pairs_t[:, 1]
        scores = (z[card_idx] * z[merch_idx]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(scores, all_labels_t)
        loss.backward()
        optimizer.step()

    metrics = {"final_loss": float(loss.item())}
    return model, metrics


@torch.no_grad()
def predict_scores(
    model: GraphSAGEModel,
    adj: torch.Tensor,
    deg: torch.Tensor,
    pairs_global: np.ndarray,
) -> np.ndarray:
    device = torch.device("cpu")
    z = model(adj, deg)
    pairs_t = torch.tensor(pairs_global, dtype=torch.long, device=device)
    card_idx = pairs_t[:, 0]
    merch_idx = pairs_t[:, 1]
    scores = (z[card_idx] * z[merch_idx]).sum(dim=1)
    probs = torch.sigmoid(scores).cpu().numpy()
    return probs

