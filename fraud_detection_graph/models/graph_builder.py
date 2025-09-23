"""
Graph Builder for Fraud Detection
Constructs bipartite and heterogeneous graphs from transaction data
to model relationships between cards, merchants, devices, and IPs.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
from datetime import datetime
import pickle

class FraudGraphBuilder:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.node_mappings = {}
        self.reverse_mappings = {}
        
    def load_data(self, data_path: str = '/workspace/fraud_detection_graph/data/') -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        data = {}
        
        # Load CSV files
        for file_type in ['cards', 'merchants', 'devices', 'ips', 'transactions']:
            df = pd.read_csv(f'{data_path}{file_type}.csv')
            data[file_type] = df
            print(f"Loaded {file_type}: {len(df)} records")
        
        # Load fraud rings
        with open(f'{data_path}fraud_rings.json', 'r') as f:
            data['fraud_rings'] = json.load(f)
            
        return data
    
    def create_node_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
        """Create node features for each entity type"""
        features = {}
        
        # Card features
        card_features = []
        cards_df = data['cards'].copy()
        
        # Encode categorical features
        if 'card_type' not in self.label_encoders:
            self.label_encoders['card_type'] = LabelEncoder()
            cards_df['card_type_encoded'] = self.label_encoders['card_type'].fit_transform(cards_df['card_type'])
        else:
            cards_df['card_type_encoded'] = self.label_encoders['card_type'].transform(cards_df['card_type'])
        
        # Normalize dates
        cards_df['issue_date'] = pd.to_datetime(cards_df['issue_date'])
        cards_df['days_since_issue'] = (datetime.now() - cards_df['issue_date']).dt.days
        
        card_feature_cols = ['card_type_encoded', 'credit_limit', 'customer_age', 
                           'customer_income', 'risk_score', 'days_since_issue']
        card_features = cards_df[card_feature_cols].fillna(0).values
        
        if not hasattr(self.scaler, 'mean_'):  # First time fitting
            card_features = self.scaler.fit_transform(card_features)
        else:
            card_features = self.scaler.transform(card_features)
            
        features['card'] = torch.FloatTensor(card_features)
        
        # Merchant features
        merchants_df = data['merchants'].copy()
        
        # Encode categorical features
        for col in ['category', 'location_state', 'risk_level']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                merchants_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(merchants_df[col])
            else:
                merchants_df[f'{col}_encoded'] = self.label_encoders[col].transform(merchants_df[col])
        
        merchant_feature_cols = ['category_encoded', 'location_state_encoded', 'risk_level_encoded',
                               'avg_transaction_amount', 'fraud_history_count']
        merchant_features = merchants_df[merchant_feature_cols].fillna(0).values
        merchant_features = StandardScaler().fit_transform(merchant_features)
        features['merchant'] = torch.FloatTensor(merchant_features)
        
        # Device features
        devices_df = data['devices'].copy()
        
        for col in ['device_type', 'os_type', 'browser']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                devices_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(devices_df[col])
            else:
                devices_df[f'{col}_encoded'] = self.label_encoders[col].transform(devices_df[col])
        
        devices_df['first_seen'] = pd.to_datetime(devices_df['first_seen'])
        devices_df['last_seen'] = pd.to_datetime(devices_df['last_seen'])
        devices_df['days_since_first_seen'] = (datetime.now() - devices_df['first_seen']).dt.days
        devices_df['days_since_last_seen'] = (datetime.now() - devices_df['last_seen']).dt.days
        devices_df['is_suspicious_int'] = devices_df['is_suspicious'].astype(int)
        
        device_feature_cols = ['device_type_encoded', 'os_type_encoded', 'browser_encoded',
                             'transaction_count', 'days_since_first_seen', 'days_since_last_seen',
                             'is_suspicious_int']
        device_features = devices_df[device_feature_cols].fillna(0).values
        device_features = StandardScaler().fit_transform(device_features)
        features['device'] = torch.FloatTensor(device_features)
        
        # IP features
        ips_df = data['ips'].copy()
        
        for col in ['country', 'region', 'isp']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                ips_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(ips_df[col])
            else:
                ips_df[f'{col}_encoded'] = self.label_encoders[col].transform(ips_df[col])
        
        ips_df['first_seen'] = pd.to_datetime(ips_df['first_seen'])
        ips_df['last_seen'] = pd.to_datetime(ips_df['last_seen'])
        ips_df['days_since_first_seen'] = (datetime.now() - ips_df['first_seen']).dt.days
        ips_df['days_since_last_seen'] = (datetime.now() - ips_df['last_seen']).dt.days
        ips_df['is_vpn_int'] = ips_df['is_vpn'].astype(int)
        ips_df['is_tor_int'] = ips_df['is_tor'].astype(int)
        
        ip_feature_cols = ['country_encoded', 'region_encoded', 'isp_encoded',
                         'is_vpn_int', 'is_tor_int', 'reputation_score',
                         'days_since_first_seen', 'days_since_last_seen']
        ip_features = ips_df[ip_feature_cols].fillna(0).values
        ip_features = StandardScaler().fit_transform(ip_features)
        features['ip'] = torch.FloatTensor(ip_features)
        
        return features
    
    def create_node_mappings(self, data: Dict[str, pd.DataFrame]) -> None:
        """Create mappings from entity IDs to node indices"""
        self.node_mappings = {
            'card': {card_id: idx for idx, card_id in enumerate(data['cards']['card_id'])},
            'merchant': {merchant_id: idx for idx, merchant_id in enumerate(data['merchants']['merchant_id'])},
            'device': {device_id: idx for idx, device_id in enumerate(data['devices']['device_id'])},
            'ip': {ip_addr: idx for idx, ip_addr in enumerate(data['ips']['ip_address'])}
        }
        
        # Create reverse mappings
        self.reverse_mappings = {
            entity_type: {idx: entity_id for entity_id, idx in mapping.items()}
            for entity_type, mapping in self.node_mappings.items()
        }
    
    def build_bipartite_graph(self, data: Dict[str, pd.DataFrame]) -> nx.Graph:
        """Build a bipartite graph with cards and merchants"""
        G = nx.Graph()
        
        # Add card nodes
        cards = [(f"card_{card_id}", {'type': 'card', 'bipartite': 0}) 
                for card_id in data['cards']['card_id']]
        G.add_nodes_from(cards)
        
        # Add merchant nodes
        merchants = [(f"merchant_{merchant_id}", {'type': 'merchant', 'bipartite': 1}) 
                    for merchant_id in data['merchants']['merchant_id']]
        G.add_nodes_from(merchants)
        
        # Add edges from transactions
        transactions = data['transactions']
        for _, txn in transactions.iterrows():
            card_node = f"card_{txn['card_id']}"
            merchant_node = f"merchant_{txn['merchant_id']}"
            
            if G.has_edge(card_node, merchant_node):
                # Update edge attributes
                G[card_node][merchant_node]['weight'] += 1
                G[card_node][merchant_node]['total_amount'] += txn['amount']
                G[card_node][merchant_node]['fraud_count'] += int(txn['is_fraud'])
            else:
                # Create new edge
                G.add_edge(card_node, merchant_node, 
                          weight=1, 
                          total_amount=txn['amount'],
                          fraud_count=int(txn['is_fraud']))
        
        return G
    
    def build_heterogeneous_graph(self, data: Dict[str, pd.DataFrame]) -> HeteroData:
        """Build a heterogeneous graph with all entity types"""
        hetero_data = HeteroData()
        
        # Create node mappings
        self.create_node_mappings(data)
        
        # Add node features
        node_features = self.create_node_features(data)
        for node_type, features in node_features.items():
            hetero_data[node_type].x = features
            hetero_data[node_type].num_nodes = features.size(0)
        
        # Create edge indices and attributes from transactions
        transactions = data['transactions']
        
        # Initialize edge lists
        edge_indices = {
            ('card', 'transacts_with', 'merchant'): [[], []],
            ('card', 'uses', 'device'): [[], []],
            ('card', 'from_ip', 'ip'): [[], []],
            ('device', 'at_merchant', 'merchant'): [[], []],
            ('ip', 'at_merchant', 'merchant'): [[], []],
            ('device', 'has_ip', 'ip'): [[], []]
        }
        
        edge_attrs = {
            ('card', 'transacts_with', 'merchant'): [],
            ('card', 'uses', 'device'): [],
            ('card', 'from_ip', 'ip'): [],
            ('device', 'at_merchant', 'merchant'): [],
            ('ip', 'at_merchant', 'merchant'): [],
            ('device', 'has_ip', 'ip'): []
        }
        
        # Process transactions to build edges
        for _, txn in transactions.iterrows():
            card_idx = self.node_mappings['card'][txn['card_id']]
            merchant_idx = self.node_mappings['merchant'][txn['merchant_id']]
            device_idx = self.node_mappings['device'][txn['device_id']]
            ip_idx = self.node_mappings['ip'][txn['ip_address']]
            
            # Card-Merchant edge
            edge_indices[('card', 'transacts_with', 'merchant')][0].append(card_idx)
            edge_indices[('card', 'transacts_with', 'merchant')][1].append(merchant_idx)
            edge_attrs[('card', 'transacts_with', 'merchant')].append([
                txn['amount'], int(txn['is_fraud']), txn['response_time_ms']
            ])
            
            # Card-Device edge
            edge_indices[('card', 'uses', 'device')][0].append(card_idx)
            edge_indices[('card', 'uses', 'device')][1].append(device_idx)
            edge_attrs[('card', 'uses', 'device')].append([
                txn['amount'], int(txn['is_fraud'])
            ])
            
            # Card-IP edge
            edge_indices[('card', 'from_ip', 'ip')][0].append(card_idx)
            edge_indices[('card', 'from_ip', 'ip')][1].append(ip_idx)
            edge_attrs[('card', 'from_ip', 'ip')].append([
                txn['amount'], int(txn['is_fraud'])
            ])
            
            # Device-Merchant edge
            edge_indices[('device', 'at_merchant', 'merchant')][0].append(device_idx)
            edge_indices[('device', 'at_merchant', 'merchant')][1].append(merchant_idx)
            edge_attrs[('device', 'at_merchant', 'merchant')].append([
                txn['amount'], int(txn['is_fraud'])
            ])
            
            # IP-Merchant edge
            edge_indices[('ip', 'at_merchant', 'merchant')][0].append(ip_idx)
            edge_indices[('ip', 'at_merchant', 'merchant')][1].append(merchant_idx)
            edge_attrs[('ip', 'at_merchant', 'merchant')].append([
                txn['amount'], int(txn['is_fraud'])
            ])
            
            # Device-IP edge
            edge_indices[('device', 'has_ip', 'ip')][0].append(device_idx)
            edge_indices[('device', 'has_ip', 'ip')][1].append(ip_idx)
            edge_attrs[('device', 'has_ip', 'ip')].append([
                txn['amount'], int(txn['is_fraud'])
            ])
        
        # Convert to tensors and add to hetero_data
        for edge_type, indices in edge_indices.items():
            if len(indices[0]) > 0:
                hetero_data[edge_type].edge_index = torch.tensor(indices, dtype=torch.long)
                hetero_data[edge_type].edge_attr = torch.tensor(edge_attrs[edge_type], dtype=torch.float)
        
        # Add labels for supervised learning (transaction-level fraud detection)
        transaction_labels = []
        transaction_node_pairs = []
        
        for _, txn in transactions.iterrows():
            card_idx = self.node_mappings['card'][txn['card_id']]
            merchant_idx = self.node_mappings['merchant'][txn['merchant_id']]
            transaction_labels.append(int(txn['is_fraud']))
            transaction_node_pairs.append([card_idx, merchant_idx])
        
        hetero_data.transaction_labels = torch.tensor(transaction_labels, dtype=torch.long)
        hetero_data.transaction_pairs = torch.tensor(transaction_node_pairs, dtype=torch.long)
        
        return hetero_data
    
    def add_fraud_ring_labels(self, hetero_data: HeteroData, data: Dict) -> HeteroData:
        """Add fraud ring labels to nodes"""
        # Initialize fraud ring labels
        for node_type in ['card', 'merchant', 'device', 'ip']:
            num_nodes = hetero_data[node_type].num_nodes
            hetero_data[node_type].fraud_ring_id = torch.full((num_nodes,), -1, dtype=torch.long)
            hetero_data[node_type].is_fraud_ring_member = torch.zeros(num_nodes, dtype=torch.long)
        
        # Assign fraud ring labels
        for ring_idx, ring in enumerate(data['fraud_rings']):
            # Mark cards in fraud ring
            for card_id in ring['cards']:
                if card_id in self.node_mappings['card']:
                    node_idx = self.node_mappings['card'][card_id]
                    hetero_data['card'].fraud_ring_id[node_idx] = ring_idx
                    hetero_data['card'].is_fraud_ring_member[node_idx] = 1
            
            # Mark merchants in fraud ring
            for merchant_id in ring['merchants']:
                if merchant_id in self.node_mappings['merchant']:
                    node_idx = self.node_mappings['merchant'][merchant_id]
                    hetero_data['merchant'].fraud_ring_id[node_idx] = ring_idx
                    hetero_data['merchant'].is_fraud_ring_member[node_idx] = 1
            
            # Mark devices in fraud ring
            for device_id in ring['devices']:
                if device_id in self.node_mappings['device']:
                    node_idx = self.node_mappings['device'][device_id]
                    hetero_data['device'].fraud_ring_id[node_idx] = ring_idx
                    hetero_data['device'].is_fraud_ring_member[node_idx] = 1
            
            # Mark IPs in fraud ring
            for ip_addr in ring['ips']:
                if ip_addr in self.node_mappings['ip']:
                    node_idx = self.node_mappings['ip'][ip_addr]
                    hetero_data['ip'].fraud_ring_id[node_idx] = ring_idx
                    hetero_data['ip'].is_fraud_ring_member[node_idx] = 1
        
        return hetero_data
    
    def save_graph(self, hetero_data: HeteroData, bipartite_graph: nx.Graph, 
                   filepath: str = '/workspace/fraud_detection_graph/data/') -> None:
        """Save the constructed graphs"""
        # Save heterogeneous graph
        torch.save(hetero_data, f'{filepath}hetero_graph.pt')
        
        # Save bipartite graph
        with open(f'{filepath}bipartite_graph.gpickle', 'wb') as f:
            pickle.dump(bipartite_graph, f)
        
        # Save mappings and encoders
        with open(f'{filepath}node_mappings.pkl', 'wb') as f:
            pickle.dump(self.node_mappings, f)
        
        with open(f'{filepath}label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print(f"Graphs saved to {filepath}")
    
    def load_graph(self, filepath: str = '/workspace/fraud_detection_graph/data/') -> Tuple[HeteroData, nx.Graph]:
        """Load the constructed graphs"""
        hetero_data = torch.load(f'{filepath}hetero_graph.pt', weights_only=False)
        with open(f'{filepath}bipartite_graph.gpickle', 'rb') as f:
            bipartite_graph = pickle.load(f)
        
        with open(f'{filepath}node_mappings.pkl', 'rb') as f:
            self.node_mappings = pickle.load(f)
        
        with open(f'{filepath}label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        return hetero_data, bipartite_graph

if __name__ == "__main__":
    # Build and save graphs
    builder = FraudGraphBuilder()
    
    print("Loading data...")
    data = builder.load_data()
    
    print("Building bipartite graph...")
    bipartite_graph = builder.build_bipartite_graph(data)
    print(f"Bipartite graph: {bipartite_graph.number_of_nodes()} nodes, {bipartite_graph.number_of_edges()} edges")
    
    print("Building heterogeneous graph...")
    hetero_data = builder.build_heterogeneous_graph(data)
    
    print("Adding fraud ring labels...")
    hetero_data = builder.add_fraud_ring_labels(hetero_data, data)
    
    print("Saving graphs...")
    builder.save_graph(hetero_data, bipartite_graph)
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Card nodes: {hetero_data['card'].num_nodes}")
    print(f"Merchant nodes: {hetero_data['merchant'].num_nodes}")
    print(f"Device nodes: {hetero_data['device'].num_nodes}")
    print(f"IP nodes: {hetero_data['ip'].num_nodes}")
    
    for edge_type in hetero_data.edge_types:
        num_edges = hetero_data[edge_type].edge_index.size(1)
        print(f"{edge_type} edges: {num_edges}")
    
    fraud_transactions = hetero_data.transaction_labels.sum().item()
    total_transactions = len(hetero_data.transaction_labels)
    print(f"Fraud transactions: {fraud_transactions}/{total_transactions} ({fraud_transactions/total_transactions:.3f})")