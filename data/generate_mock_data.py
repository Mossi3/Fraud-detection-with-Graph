import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json

class FraudDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define fraud rings and patterns
        self.fraud_rings = {
            'ring_1': {
                'cards': [f'card_{i:06d}' for i in range(1000, 1010)],
                'merchants': [f'merchant_{i:04d}' for i in range(200, 210)],
                'devices': [f'device_{i:06d}' for i in range(5000, 5010)],
                'ips': [f'192.168.{i}.{j}' for i in range(1, 3) for j in range(1, 6)]
            },
            'ring_2': {
                'cards': [f'card_{i:06d}' for i in range(2000, 2015)],
                'merchants': [f'merchant_{i:04d}' for i in range(300, 315)],
                'devices': [f'device_{i:06d}' for i in range(6000, 6015)],
                'ips': [f'10.0.{i}.{j}' for i in range(1, 4) for j in range(1, 6)]
            },
            'ring_3': {
                'cards': [f'card_{i:06d}' for i in range(3000, 3020)],
                'merchants': [f'merchant_{i:04d}' for i in range(400, 420)],
                'devices': [f'device_{i:06d}' for i in range(7000, 7020)],
                'ips': [f'172.16.{i}.{j}' for i in range(1, 5) for j in range(1, 6)]
            }
        }
        
        # Normal entities
        self.normal_cards = [f'card_{i:06d}' for i in range(10000, 20000)]
        self.normal_merchants = [f'merchant_{i:04d}' for i in range(1000, 2000)]
        self.normal_devices = [f'device_{i:06d}' for i in range(10000, 20000)]
        self.normal_ips = [f'203.{i}.{j}.{k}' for i in range(1, 10) for j in range(1, 10) for k in range(1, 10)]
        
    def generate_transactions(self, n_transactions=50000) -> pd.DataFrame:
        """Generate mock credit card transactions with fraud patterns"""
        transactions = []
        
        # Generate normal transactions (90%)
        n_normal = int(n_transactions * 0.9)
        for i in range(n_normal):
            transaction = self._generate_normal_transaction()
            transactions.append(transaction)
        
        # Generate fraud transactions (10%)
        n_fraud = n_transactions - n_normal
        for i in range(n_fraud):
            transaction = self._generate_fraud_transaction()
            transactions.append(transaction)
        
        # Shuffle transactions
        random.shuffle(transactions)
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    def _generate_normal_transaction(self) -> Dict:
        """Generate a normal transaction"""
        card = random.choice(self.normal_cards)
        merchant = random.choice(self.normal_merchants)
        device = random.choice(self.normal_devices)
        ip = random.choice(self.normal_ips)
        
        return {
            'transaction_id': f'txn_{random.randint(100000, 999999)}',
            'card_id': card,
            'merchant_id': merchant,
            'device_id': device,
            'ip_address': ip,
            'amount': round(np.random.lognormal(3, 1), 2),
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
            'category': random.choice(['groceries', 'gas', 'restaurant', 'retail', 'online']),
            'country': random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
            'is_fraud': 0,
            'fraud_ring': None
        }
    
    def _generate_fraud_transaction(self) -> Dict:
        """Generate a fraud transaction"""
        ring_name = random.choice(list(self.fraud_rings.keys()))
        ring = self.fraud_rings[ring_name]
        
        # Use entities from the same fraud ring
        card = random.choice(ring['cards'])
        merchant = random.choice(ring['merchants'])
        device = random.choice(ring['devices'])
        ip = random.choice(ring['ips'])
        
        # Fraud transactions tend to be larger amounts
        amount = round(np.random.lognormal(5, 1.5), 2)
        
        return {
            'transaction_id': f'txn_{random.randint(100000, 999999)}',
            'card_id': card,
            'merchant_id': merchant,
            'device_id': device,
            'ip_address': ip,
            'amount': amount,
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
            'category': random.choice(['electronics', 'jewelry', 'cash_advance', 'online']),
            'country': random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
            'is_fraud': 1,
            'fraud_ring': ring_name
        }
    
    def build_transaction_graph(self, transactions_df: pd.DataFrame) -> nx.Graph:
        """Build a heterogeneous graph from transactions"""
        G = nx.Graph()
        
        # Add nodes
        for _, row in transactions_df.iterrows():
            # Transaction node
            txn_id = row['transaction_id']
            G.add_node(txn_id, 
                      type='transaction',
                      amount=row['amount'],
                      timestamp=row['timestamp'],
                      is_fraud=row['is_fraud'],
                      fraud_ring=row['fraud_ring'])
            
            # Entity nodes
            G.add_node(row['card_id'], type='card')
            G.add_node(row['merchant_id'], type='merchant')
            G.add_node(row['device_id'], type='device')
            G.add_node(row['ip_address'], type='ip')
            
            # Add edges
            G.add_edge(txn_id, row['card_id'])
            G.add_edge(txn_id, row['merchant_id'])
            G.add_edge(txn_id, row['device_id'])
            G.add_edge(txn_id, row['ip_address'])
        
        return G
    
    def generate_entity_features(self, transactions_df: pd.DataFrame) -> Dict:
        """Generate features for entities based on transaction history"""
        features = {}
        
        # Card features
        card_features = {}
        for card in transactions_df['card_id'].unique():
            card_txns = transactions_df[transactions_df['card_id'] == card]
            card_features[card] = {
                'total_transactions': len(card_txns),
                'total_amount': card_txns['amount'].sum(),
                'avg_amount': card_txns['amount'].mean(),
                'fraud_rate': card_txns['is_fraud'].mean(),
                'unique_merchants': card_txns['merchant_id'].nunique(),
                'unique_devices': card_txns['device_id'].nunique(),
                'unique_ips': card_txns['ip_address'].nunique(),
                'time_span_days': (card_txns['timestamp'].max() - card_txns['timestamp'].min()).days
            }
        
        # Merchant features
        merchant_features = {}
        for merchant in transactions_df['merchant_id'].unique():
            merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant]
            merchant_features[merchant] = {
                'total_transactions': len(merchant_txns),
                'total_amount': merchant_txns['amount'].sum(),
                'avg_amount': merchant_txns['amount'].mean(),
                'fraud_rate': merchant_txns['is_fraud'].mean(),
                'unique_cards': merchant_txns['card_id'].nunique(),
                'unique_devices': merchant_txns['device_id'].nunique(),
                'unique_ips': merchant_txns['ip_address'].nunique()
            }
        
        features['cards'] = card_features
        features['merchants'] = merchant_features
        
        return features

def create_mock_dataset():
    """Create and save mock dataset"""
    generator = FraudDataGenerator()
    
    # Generate transactions
    print("Generating transactions...")
    transactions_df = generator.generate_transactions(50000)
    
    # Build graph
    print("Building transaction graph...")
    graph = generator.build_transaction_graph(transactions_df)
    
    # Generate features
    print("Generating entity features...")
    features = generator.generate_entity_features(transactions_df)
    
    # Save data
    transactions_df.to_csv('/workspace/data/transactions.csv', index=False)
    
    # Save graph
    nx.write_gpickle(graph, '/workspace/data/transaction_graph.pkl')
    
    # Save features
    with open('/workspace/data/entity_features.json', 'w') as f:
        json.dump(features, f, indent=2, default=str)
    
    # Save fraud rings info
    with open('/workspace/data/fraud_rings.json', 'w') as f:
        json.dump(generator.fraud_rings, f, indent=2)
    
    print(f"Dataset created with {len(transactions_df)} transactions")
    print(f"Fraud rate: {transactions_df['is_fraud'].mean():.2%}")
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    return transactions_df, graph, features

if __name__ == "__main__":
    create_mock_dataset()