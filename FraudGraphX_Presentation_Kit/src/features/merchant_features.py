import pandas as pd
def merchant_agg(df: pd.DataFrame)->pd.DataFrame:
    agg=df.groupby('merchant_id')['amount'].agg(['mean','max','count']).reset_index()
    agg.columns=['merchant_id','m_amt_mean','m_amt_max','m_txn_count']
    return df[['transaction_id','merchant_id']].merge(agg, on='merchant_id', how='left')
