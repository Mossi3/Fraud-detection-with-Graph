from typing import Optional
import pandas as pd


def load_relational_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {
        "transaction_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip",
        "amount",
        "timestamp",
        "label",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.astype(
        {
            "transaction_id": "string",
            "card_id": "string",
            "merchant_id": "string",
            "device_id": "string",
            "ip": "string",
            "amount": "float64",
            "label": "int64",
        }
    )
    return df

