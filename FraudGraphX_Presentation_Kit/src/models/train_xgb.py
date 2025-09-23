import argparse, joblib, xgboost as xgb
from sklearn.model_selection import train_test_split
from ..utils.io import read_parquet
from ..utils.metrics import basic_scores
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--model_path', default='models/xgb_final.joblib'); a=ap.parse_args()
    df=read_parquet(a.input); y=df['fraud'].astype(int)
    drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
    X=df.drop(columns=[c for c in drop if c in df.columns]).fillna(0.0)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=0.2,shuffle=False)
    ratio=(ytr==0).sum()/max((ytr==1).sum(),1)
    model=xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=ratio,eval_metric='aucpr')
    model.fit(Xtr,ytr,eval_set=[(Xv,yv)],verbose=False); joblib.dump(model,a.model_path)
    import numpy as np; yp=model.predict_proba(Xv)[:,1]; print('Validation:', basic_scores(yv, yp))
