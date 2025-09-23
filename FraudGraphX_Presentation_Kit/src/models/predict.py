import argparse, joblib
from ..utils.io import read_parquet
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input', required=True); ap.add_argument('--model_path', required=True); ap.add_argument('--out', required=True); a=ap.parse_args()
    model=joblib.load(a.model_path); df=read_parquet(a.input)
    drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
    X=df.drop(columns=[c for c in drop if c in df.columns]).fillna(0.0)
    import pandas as pd; out=df[['transaction_id']].copy(); out['score']=model.predict_proba(X)[:,1]; out.to_csv(a.out,index=False); print(out.head())
