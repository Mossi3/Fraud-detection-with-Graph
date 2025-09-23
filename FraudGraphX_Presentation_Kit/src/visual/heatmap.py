import argparse, os
from ..utils.io import read_parquet
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); a=ap.parse_args()
    os.makedirs(a.out, exist_ok=True); df=read_parquet(a.data)
    df['hour']=df['timestamp'].dt.hour; df['amount_bin']=pd.qcut(df['amount'], q=10, duplicates='drop')
    pv=df.pivot_table(index='hour', columns='amount_bin', values='fraud', aggfunc='mean')
    plt.figure(); sns.heatmap(pv); plt.title('Fraud Rate by Hour vs Amount'); plt.tight_layout(); plt.savefig(os.path.join(a.out,'fraud_heatmap_hour_amount.png'))
    num=df.select_dtypes(include=['number']).corr()
    plt.figure(); sns.heatmap(num); plt.title('Feature Correlation'); plt.tight_layout(); plt.savefig(os.path.join(a.out,'feature_correlation_heatmap.png'))
