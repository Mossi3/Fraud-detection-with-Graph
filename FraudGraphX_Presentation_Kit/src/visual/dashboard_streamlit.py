import streamlit as st, joblib, matplotlib.pyplot as plt
from ..utils.io import read_parquet
st.set_page_config(page_title='FraudGraphX Dashboard', layout='wide')
st.title('FraudGraphX — Presentation Dashboard')
data_path = st.text_input('Processed data path', 'data/processed/transactions_with_features.parquet')
model_path = st.text_input('Model path', 'models/xgb_final.joblib')
if st.button('Load'):
    df=read_parquet(data_path); st.success(f'Loaded {len(df):,} rows')
    model=joblib.load(model_path)
    drop=['fraud','transaction_id','timestamp','card_id','merchant_id','device_id','ip']
    X=df.drop(columns=[c for c in drop if c in df.columns]).fillna(0.0)
    proba=model.predict_proba(X)[:,1]; df['score']=proba
    st.subheader('Top Alerts'); st.dataframe(df.sort_values('score', ascending=False).head(20))
    fig=plt.figure(); plt.hist(df['score'], bins=50); plt.title('Score Distribution'); st.pyplot(fig)
    tx_id = st.number_input('Transaction ID to inspect', int(df['transaction_id'].min()), int(df['transaction_id'].max()), int(df['transaction_id'].iloc[0]))
    st.subheader('Transaction Details'); st.json(df[df['transaction_id']==tx_id].iloc[0].to_dict())
