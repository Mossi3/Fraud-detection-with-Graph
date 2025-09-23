from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


NODE_TYPES = ["card", "merchant", "device", "ip"]


@dataclass
class NodeMapping:
    node_type: str
    node_id: str
    global_index: int


class NodeIndexer:
    def __init__(self) -> None:
        self._id_to_index: Dict[Tuple[str, str], int] = {}
        self._index_to_type_id: List[Tuple[str, str]] = []
        self._oov_by_type: Dict[str, int] = {}
        for t in NODE_TYPES:
            self._add_node((t, "__OOV__"))

    def _add_node(self, type_id: Tuple[str, str]) -> int:
        if type_id in self._id_to_index:
            return self._id_to_index[type_id]
        idx = len(self._index_to_type_id)
        self._id_to_index[type_id] = idx
        self._index_to_type_id.append(type_id)
        return idx

    def add_type_id(self, node_type: str, node_id: str) -> int:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node_type {node_type}")
        if node_id is None or node_id == "":
            return self._get_oov(node_type)
        return self._add_node((node_type, str(node_id)))

    def _get_oov(self, node_type: str) -> int:
        for idx, (t, nid) in enumerate(self._index_to_type_id):
            if t == node_type and nid == "__OOV__":
                return idx
        raise RuntimeError("OOV not initialized")

    def get_index(self, node_type: str, node_id: str) -> int:
        return self._id_to_index.get((node_type, str(node_id)), self._get_oov(node_type))

    def size(self) -> int:
        return len(self._index_to_type_id)

    def export(self) -> Dict[str, List[str]]:
        by_type: Dict[str, List[str]] = {t: [] for t in NODE_TYPES}
        for t, nid in self._index_to_type_id:
            by_type[t].append(nid)
        return by_type

    @staticmethod
    def from_export(by_type: Dict[str, List[str]]) -> "NodeIndexer":
        # This method cannot reconstruct the original global order reliably.
        # Prefer using from_global(). Keeping for backward compatibility when order doesn't matter.
        ni = NodeIndexer()
        ni._id_to_index.clear()
        ni._index_to_type_id.clear()
        for t in NODE_TYPES:
            for nid in by_type.get(t, []):
                ni._add_node((t, nid))
        return ni

    def export_global(self) -> List[Tuple[str, str]]:
        return list(self._index_to_type_id)

    @staticmethod
    def from_global(global_list: List[Tuple[str, str]]) -> "NodeIndexer":
        ni = NodeIndexer()
        ni._id_to_index.clear()
        ni._index_to_type_id.clear()
        for t, nid in global_list:
            ni._add_node((t, nid))
        return ni

    def type_global_indices(self, node_type: str) -> List[int]:
        return [i for i, (t, _) in enumerate(self._index_to_type_id) if t == node_type]


def build_indexer_from_df(df: pd.DataFrame) -> NodeIndexer:
    indexer = NodeIndexer()
    for node_type, col in [
        ("card", "card_id"),
        ("merchant", "merchant_id"),
        ("device", "device_id"),
        ("ip", "ip_address"),
    ]:
        for nid in df[col].astype(str).unique().tolist():
            indexer.add_type_id(node_type, nid)
    return indexer


def build_adjacency(df: pd.DataFrame, indexer: NodeIndexer) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[int] = []
    cols: List[int] = []
    for _, row in df.iterrows():
        card = indexer.get_index("card", row["card_id"])
        merchant = indexer.get_index("merchant", row["merchant_id"])
        device = indexer.get_index("device", row["device_id"])
        ip = indexer.get_index("ip", row["ip_address"])
        edges = [
            (card, merchant),
            (card, device),
            (card, ip),
            (merchant, device),
            (merchant, ip),
            (device, ip),
        ]
        for a, b in edges:
            rows.append(a)
            cols.append(b)
            rows.append(b)
            cols.append(a)

    num_nodes = indexer.size()
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
    return adj.coalesce(), deg


def transactions_to_pairs(df: pd.DataFrame, indexer: NodeIndexer) -> Tuple[np.ndarray, np.ndarray]:
    # Returns GLOBAL indices for (card, merchant)
    card_idx = df["card_id"].astype(str).map(lambda x: indexer.get_index("card", x)).to_numpy()
    merchant_idx = df["merchant_id"].astype(str).map(lambda x: indexer.get_index("merchant", x)).to_numpy()
    pairs = np.stack([card_idx, merchant_idx], axis=1)
    labels = df["is_fraud"].astype(int).to_numpy()
    return pairs, labels

