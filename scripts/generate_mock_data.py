import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_mock_transactions(n_tx: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    num_cards = 80
    num_merchants = 50
    num_devices = 60
    num_ips = 100

    cards = [f"card_{i}" for i in range(num_cards)]
    merchants = [f"merchant_{i}" for i in range(num_merchants)]
    devices = [f"device_{i}" for i in range(num_devices)]
    ips = [f"10.0.{i//256}.{i%256}" for i in range(num_ips)]

    start_time = datetime(2025, 1, 1)

    num_rings = 4
    ring_specs = []
    for r in range(num_rings):
        ring_cards = rng.choice(cards, size=rng.integers(5, 10), replace=False).tolist()
        ring_merchants = rng.choice(merchants, size=rng.integers(4, 8), replace=False).tolist()
        ring_devices = rng.choice(devices, size=rng.integers(3, 6), replace=False).tolist()
        ring_ips = rng.choice(ips, size=rng.integers(3, 6), replace=False).tolist()
        ring_specs.append((ring_cards, ring_merchants, ring_devices, ring_ips))

    rows = []
    for i in range(n_tx):
        is_ring = rng.random() < 0.25
        if is_ring:
            rc, rm, rd, ri = ring_specs[rng.integers(0, num_rings)]
            card_id = rng.choice(rc)
            merchant_id = rng.choice(rm)
            device_id = rng.choice(rd)
            ip_address = rng.choice(ri)
            fraud_p = 0.55 + 0.4 * rng.random()
        else:
            card_id = rng.choice(cards)
            merchant_id = rng.choice(merchants)
            device_id = rng.choice(devices)
            ip_address = rng.choice(ips)
            fraud_p = 0.03 + 0.02 * rng.random()

        is_fraud = int(rng.random() < fraud_p)
        amount = float(np.round(rng.lognormal(mean=3.4, sigma=0.7), 2))
        ts = start_time + timedelta(minutes=int(rng.integers(0, 60 * 24 * 30)))
        rows.append({
            "transaction_id": f"tx_{i}",
            "card_id": card_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "ip_address": ip_address,
            "amount": amount,
            "timestamp": ts.isoformat(),
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/mock_transactions.csv")
    parser.add_argument("--n_tx", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_mock_transactions(args.n_tx, args.seed)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

