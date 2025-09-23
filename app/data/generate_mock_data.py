import os
import math
import random
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


def _random_ip(rng: random.Random) -> str:
    return f"{rng.randint(1, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def generate_mock_transactions(
    num_transactions: int = 20000,
    num_cards: int = 3000,
    num_merchants: int = 400,
    num_devices: int = 1200,
    num_ips: int = 800,
    num_rings: int = 12,
    ring_size_cards: Tuple[int, int] = (8, 20),
    ring_size_merchants: Tuple[int, int] = (3, 8),
    ring_size_devices: Tuple[int, int] = (5, 15),
    ring_size_ips: Tuple[int, int] = (3, 10),
    fraud_rate_background: float = 0.01,
    fraud_rate_ring: float = 0.45,
    start_datetime: str = "2024-01-01T00:00:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate mock relational transactions with fraud rings.

    Columns: transaction_id,card_id,merchant_id,device_id,ip,amount,timestamp,label
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_card_ids = [f"c_{i}" for i in range(num_cards)]
    all_merchant_ids = [f"m_{i}" for i in range(num_merchants)]
    all_device_ids = [f"d_{i}" for i in range(num_devices)]
    all_ip_ids = [f"ip_{i}" for i in range(num_ips)]

    # Build fraud rings: groups of cards, merchants, devices, IPs with higher fraud rate
    ring_card_sets: List[List[str]] = []
    ring_merchant_sets: List[List[str]] = []
    ring_device_sets: List[List[str]] = []
    ring_ip_sets: List[List[str]] = []

    available_cards = set(all_card_ids)
    available_merchants = set(all_merchant_ids)
    available_devices = set(all_device_ids)
    available_ips = set(all_ip_ids)

    for _ in range(num_rings):
        rc = rng.randint(ring_size_cards[0], ring_size_cards[1])
        rm = rng.randint(ring_size_merchants[0], ring_size_merchants[1])
        rd = rng.randint(ring_size_devices[0], ring_size_devices[1])
        ri = rng.randint(ring_size_ips[0], ring_size_ips[1])

        cards = rng.sample(list(available_cards), min(rc, len(available_cards)))
        ring_card_sets.append(cards)
        for c in cards:
            available_cards.discard(c)

        merchants = rng.sample(list(available_merchants), min(rm, len(available_merchants)))
        ring_merchant_sets.append(merchants)
        for m in merchants:
            available_merchants.discard(m)

        devices = rng.sample(list(available_devices), min(rd, len(available_devices)))
        ring_device_sets.append(devices)
        for d in devices:
            available_devices.discard(d)

        ips = rng.sample(list(available_ips), min(ri, len(available_ips)))
        ring_ip_sets.append(ips)
        for ip in ips:
            available_ips.discard(ip)

    # Time range
    start_dt = datetime.fromisoformat(start_datetime)

    records = []

    # Define a simple background distribution for amounts
    def background_amount(size: int) -> np.ndarray:
        # mix of log-normal small amounts and occasional larger values
        small = np.exp(np_rng.normal(3.0, 0.7, size))  # ~20-80
        spikes = np_rng.exponential(100.0, size)
        mix = 0.85 * small + 0.15 * spikes
        return mix.clip(1, 5000)

    # Ring amounts slightly higher on average
    def ring_amount(size: int) -> np.ndarray:
        base = np.exp(np_rng.normal(3.4, 0.6, size)) + np_rng.exponential(120.0, size)
        return base.clip(5, 7000)

    # Decide ring vs background for each transaction
    ring_prob = min(0.35, max(0.1, num_rings * 0.02))
    for t_idx in range(num_transactions):
        is_ring = rng.random() < ring_prob and len(ring_card_sets) > 0
        if is_ring:
            r = rng.randrange(len(ring_card_sets))
            card_id = rng.choice(ring_card_sets[r])
            merchant_id = rng.choice(ring_merchant_sets[r])
            device_id = rng.choice(ring_device_sets[r])
            ip = rng.choice(ring_ip_sets[r])
            label = 1 if rng.random() < fraud_rate_ring else 0
            amount = float(ring_amount(1)[0])
        else:
            card_id = rng.choice(all_card_ids)
            merchant_id = rng.choice(all_merchant_ids)
            device_id = rng.choice(all_device_ids)
            ip = rng.choice(all_ip_ids)
            label = 1 if rng.random() < fraud_rate_background else 0
            amount = float(background_amount(1)[0])

        # Timestamp jitters within ~120 days
        dt = start_dt + timedelta(minutes=rng.randint(0, 60 * 24 * 120))
        records.append(
            {
                "transaction_id": f"t_{t_idx}",
                "card_id": card_id,
                "merchant_id": merchant_id,
                "device_id": device_id,
                "ip": _random_ip(rng) if rng.random() < 0.2 else ip,
                "amount": round(amount, 2),
                "timestamp": dt.isoformat(),
                "label": int(label),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def save_mock_csv(df: pd.DataFrame, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

