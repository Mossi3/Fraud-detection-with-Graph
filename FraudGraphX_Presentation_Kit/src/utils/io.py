import pandas as pd
def read_parquet(path: str) -> pd.DataFrame: return pd.read_parquet(path)
def save_parquet(df: pd.DataFrame, path: str): df.to_parquet(path, index=False)
