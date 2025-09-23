
import sys
sys.path.append('/workspace/fraud_detection_graph')

from models.graph_builder import FraudGraphBuilder
import pandas as pd
import numpy as np
import torch

def demo_fraud_prediction():
    print("🔍 FRAUD PREDICTION DEMO")
    print("=" * 40)
    
    # Load data
    builder = FraudGraphBuilder()
    try:
        hetero_data, _ = builder.load_graph()
        print("✅ Graph loaded successfully")
        
        # Show graph statistics
        print(f"📊 Graph Statistics:")
        print(f"   Cards: {hetero_data['card'].num_nodes:,}")
        print(f"   Merchants: {hetero_data['merchant'].num_nodes:,}")
        print(f"   Devices: {hetero_data['device'].num_nodes:,}")
        print(f"   IPs: {hetero_data['ip'].num_nodes:,}")
        print(f"   Transactions: {len(hetero_data.transaction_labels):,}")
        
        fraud_rate = hetero_data.transaction_labels.float().mean().item()
        print(f"   Fraud Rate: {fraud_rate:.3%}")
        
        # Show some sample transactions
        transactions = pd.read_csv('data/transactions.csv')
        print(f"\n📋 Sample Transactions:")
        print(transactions[['transaction_id', 'card_id', 'merchant_id', 'amount', 'is_fraud']].head(10).to_string(index=False))
        
        # Show fraud examples
        fraud_transactions = transactions[transactions['is_fraud'] == 1].head(5)
        print(f"\n🚨 Sample Fraud Transactions:")
        print(fraud_transactions[['transaction_id', 'card_id', 'merchant_id', 'amount', 'fraud_type']].to_string(index=False))
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_fraud_prediction()
