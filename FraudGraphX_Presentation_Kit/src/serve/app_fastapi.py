from fastapi import FastAPI
import joblib, numpy as np, pandas as pd
from ..utils.io import read_parquet
app = FastAPI(title='FraudGraphX API')
_model=None; _feat_cols=None
def load(model_path='models/xgb_final.joblib', example='data/processed/transactions_with_features.parquet'):
    global _model,_feat_cols
    if _model is None:
        _model=joblib.load(model_path); df=read_parquet(example)
        drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
        _feat_cols=[c for c in df.columns if c not in drop]
    return _model,_feat_cols
@app.get('/health') 
def health(): return {'status':'ok'}
@app.post('/score')
def score(txn: dict):
    model, cols = load(); x=np.array([[txn.get(c,0.0) for c in cols]], dtype=float)
    return {'score': float(model.predict_proba(x)[0,1])}
