from typing import Dict, Tuple, List
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData


def _index_mapping(values: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(sorted(set(values)))}


def build_hetero_graph(df: pd.DataFrame) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
    """Construct a heterogeneous graph from transactional DataFrame.

    Node types: card, merchant, device, ip
    Edge types: (card, transacts, merchant), (card, uses, device), (device, routes, ip), (card, from_ip, ip)
    """
    card_map = _index_mapping(df["card_id"].tolist())
    merchant_map = _index_mapping(df["merchant_id"].tolist())
    device_map = _index_mapping(df["device_id"].tolist())
    ip_map = _index_mapping(df["ip"].tolist())

    # Basic aggregations for features
    card_feats = (
        df.groupby("card_id")
        .agg(
            tx_count=("transaction_id", "count"),
            amount_mean=("amount", "mean"),
            amount_std=("amount", "std"),
            fraud_rate=("label", "mean"),
        )
        .fillna(0.0)
        .reset_index()
    )
    merchant_feats = (
        df.groupby("merchant_id")
        .agg(
            tx_count=("transaction_id", "count"),
            amount_mean=("amount", "mean"),
            amount_std=("amount", "std"),
            fraud_rate=("label", "mean"),
        )
        .fillna(0.0)
        .reset_index()
    )
    device_feats = (
        df.groupby("device_id")
        .agg(tx_count=("transaction_id", "count"), fraud_rate=("label", "mean"))
        .fillna(0.0)
        .reset_index()
    )
    ip_feats = (
        df.groupby("ip")
        .agg(tx_count=("transaction_id", "count"), fraud_rate=("label", "mean"))
        .fillna(0.0)
        .reset_index()
    )

    def norm_cols(frame: pd.DataFrame, cols: List[str]) -> np.ndarray:
        arr = frame[cols].to_numpy(dtype=np.float32)
        if arr.size == 0:
            return arr
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-6
        return (arr - mean) / std

    card_x = np.zeros((len(card_map), 4), dtype=np.float32)
    if len(card_map) > 0:
        card_x[_reindex(card_feats["card_id"].map(card_map).to_numpy()), :] = norm_cols(
            card_feats, ["tx_count", "amount_mean", "amount_std", "fraud_rate"]
        )

    merchant_x = np.zeros((len(merchant_map), 4), dtype=np.float32)
    if len(merchant_map) > 0:
        merchant_x[_reindex(merchant_feats["merchant_id"].map(merchant_map).to_numpy()), :] = norm_cols(
            merchant_feats, ["tx_count", "amount_mean", "amount_std", "fraud_rate"]
        )

    device_x = np.zeros((len(device_map), 2), dtype=np.float32)
    if len(device_map) > 0:
        device_x[_reindex(device_feats["device_id"].map(device_map).to_numpy()), :] = norm_cols(
            device_feats, ["tx_count", "fraud_rate"]
        )

    ip_x = np.zeros((len(ip_map), 2), dtype=np.float32)
    if len(ip_map) > 0:
        ip_x[_reindex(ip_feats["ip"].map(ip_map).to_numpy()), :] = norm_cols(
            ip_feats, ["tx_count", "fraud_rate"]
        )

    # Edges: build index arrays
    e_card_merchant = (
        df[["card_id", "merchant_id"]]
        .assign(
            s=lambda x: x["card_id"].map(card_map),
            t=lambda x: x["merchant_id"].map(merchant_map),
        )
        [["s", "t"]]
        .to_numpy()
        .T
    )
    e_card_device = (
        df[["card_id", "device_id"]]
        .assign(s=lambda x: x["card_id"].map(card_map), t=lambda x: x["device_id"].map(device_map))
        [["s", "t"]]
        .to_numpy()
        .T
    )
    e_device_ip = (
        df[["device_id", "ip"]]
        .assign(s=lambda x: x["device_id"].map(device_map), t=lambda x: x["ip"].map(ip_map))
        [["s", "t"]]
        .to_numpy()
        .T
    )
    e_card_ip = (
        df[["card_id", "ip"]]
        .assign(s=lambda x: x["card_id"].map(card_map), t=lambda x: x["ip"].map(ip_map))
        [["s", "t"]]
        .to_numpy()
        .T
    )

    data = HeteroData()
    data["card"].x = torch.tensor(card_x, dtype=torch.float32)
    data["merchant"].x = torch.tensor(merchant_x, dtype=torch.float32)
    data["device"].x = torch.tensor(device_x, dtype=torch.float32)
    data["ip"].x = torch.tensor(ip_x, dtype=torch.float32)

    data["card", "transacts", "merchant"].edge_index = torch.tensor(e_card_merchant, dtype=torch.long)
    data["card", "uses", "device"].edge_index = torch.tensor(e_card_device, dtype=torch.long)
    data["device", "routes", "ip"].edge_index = torch.tensor(e_device_ip, dtype=torch.long)
    data["card", "from_ip", "ip"].edge_index = torch.tensor(e_card_ip, dtype=torch.long)

    # Labels: card-level label if card has any fraudulent tx
    card_label_series = df.groupby("card_id")["label"].max()
    y = np.zeros(len(card_map), dtype=np.int64)
    for cid, lbl in card_label_series.items():
        y[card_map[cid]] = int(lbl)
    data["card"].y = torch.tensor(y, dtype=torch.long)

    # Save id maps for API use
    id_maps = {
        "card": card_map,
        "merchant": merchant_map,
        "device": device_map,
        "ip": ip_map,
    }
    return data, id_maps


def _reindex(idx: np.ndarray) -> np.ndarray:
    # idx already numeric mapping; ensure int and no NaNs
    idx = np.asarray(idx)
    return idx.astype(np.int64)

