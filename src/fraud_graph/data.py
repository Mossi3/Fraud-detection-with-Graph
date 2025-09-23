from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_address",
    "amount",
    "timestamp",
    "is_fraud",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame


def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["card_id"] = df["card_id"].astype(str)
    df["merchant_id"] = df["merchant_id"].astype(str)
    df["device_id"] = df["device_id"].astype(str)
    df["ip_address"] = df["ip_address"].astype(str)
    df["amount"] = df["amount"].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def train_test_split_time(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> DatasetSplits:
    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    split_idx = int((1.0 - test_frac) * n)
    train = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test = df_sorted.iloc[split_idx:].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        # Fallback to random split if timestamps are not valid
        rng = np.random.default_rng(seed)
        mask = rng.random(n) < (1.0 - test_frac)
        train = df_sorted[mask].reset_index(drop=True)
        test = df_sorted[~mask].reset_index(drop=True)
    return DatasetSplits(train=train, test=test)

