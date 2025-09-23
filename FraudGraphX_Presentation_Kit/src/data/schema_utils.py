import pandas as pd
from datetime import datetime, timedelta
LABELS=['fraud','is_fraud','class','label','target']
AMTS=['amount','amt','transaction_amount']
TIMES=['timestamp','time','datetime','date']
def infer_cols(df: pd.DataFrame):
    cols = { 'label':None,'amount':None,'timestamp':None }
    low = {c:c.lower() for c in df.columns}
    def find(cands):
        for c in cands:
            for k,v in low.items():
                if v==c: return k
        return None
    cols['label']=find(LABELS); cols['amount']=find(AMTS); cols['timestamp']=find(TIMES)
    return cols
def synthesize(df: pd.DataFrame):
    import numpy as np
    cols = infer_cols(df)
    out = df.copy()
    out['amount'] = pd.to_numeric(out[cols['amount']], errors='coerce').fillna(0.0) if cols['amount'] else 10.0
    if cols['timestamp']:
        out['timestamp'] = pd.to_datetime(out[cols['timestamp']], errors='coerce')
        mask = out['timestamp'].isna()
        if mask.any():
            base = datetime(2024,1,1)
            out.loc[mask,'timestamp'] = [base+timedelta(minutes=i) for i in range(mask.sum())]
    else:
        base = datetime(2024,1,1)
        out['timestamp'] = [base+timedelta(minutes=i) for i in range(len(out))]
    if cols['label']:
        y = pd.to_numeric(out[cols['label']], errors='coerce').fillna(0).astype(int)
        out['fraud'] = (y>0).astype(int)
    else:
        cutoff = out['amount'] if isinstance(out['amount'], float) else out['amount'].quantile(0.99)
        if isinstance(cutoff, float):
            out['fraud'] = (out['amount']>=cutoff).astype(int)
        else:
            out['fraud'] = (out['amount']>=cutoff).astype(int)
    if 'transaction_id' not in out.columns:
        out['transaction_id'] = range(len(out))
    for col, pref, mod in [('card_id','card_',1000),('merchant_id','m_',50),('device_id','d_',200),('ip','10.0.0.',255)]:
        if col not in out.columns: out[col] = [f"{pref}{i%mod}" for i in range(len(out))]
    return out
