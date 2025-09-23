"""
Advanced heterogeneous graph construction for fraud detection.
Supports multiple node types: transactions, cards, merchants, devices, IPs.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import dgl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeterogeneousGraphBuilder:
    """Build heterogeneous graphs for fraud detection with multiple entity types."""
    
    def __init__(self, include_temporal: bool = True, include_amount_edges: bool = True):
        self.include_temporal = include_temporal
        self.include_amount_edges = include_amount_edges
        self.node_encoders = {}
        self.feature_scalers = {}
        
    def build_graph(self, df: pd.DataFrame, graph_type: str = 'torch_geometric') -> Any:
        """
        Build heterogeneous graph from transaction data.
        
        Args:
            df: Transaction dataframe
            graph_type: 'torch_geometric', 'dgl', or 'networkx'
        
        Returns:
            Graph object of specified type
        """
        logger.info(f"Building {graph_type} heterogeneous graph from {len(df)} transactions")
        
        # Prepare node mappings and features
        node_mappings, node_features = self._prepare_nodes(df)
        
        # Build edges
        edge_indices, edge_features, edge_types = self._build_edges(df, node_mappings)
        
        if graph_type == 'torch_geometric':
            return self._build_torch_geometric_graph(node_mappings, node_features, edge_indices, edge_features, edge_types)
        elif graph_type == 'dgl':
            return self._build_dgl_graph(node_mappings, node_features, edge_indices, edge_types)
        elif graph_type == 'networkx':
            return self._build_networkx_graph(df, node_mappings, node_features, edge_indices, edge_types)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
    
    def _prepare_nodes(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Prepare node mappings and features for all entity types."""
        node_mappings = {}
        node_features = {}
        
        # Transaction nodes
        transaction_features = self._extract_transaction_features(df)
        node_mappings['transaction'] = {txn_id: idx for idx, txn_id in enumerate(df['transaction_id'])}
        node_features['transaction'] = transaction_features
        
        # Card nodes
        card_features = self._extract_card_features(df)
        unique_cards = df['card_id'].unique()
        node_mappings['card'] = {card_id: idx for idx, card_id in enumerate(unique_cards)}
        node_features['card'] = card_features
        
        # Merchant nodes
        merchant_features = self._extract_merchant_features(df)
        unique_merchants = df['merchant_id'].unique()
        node_mappings['merchant'] = {merchant_id: idx for idx, merchant_id in enumerate(unique_merchants)}
        node_features['merchant'] = merchant_features
        
        # Device nodes
        device_features = self._extract_device_features(df)
        unique_devices = df['device_id'].unique()
        node_mappings['device'] = {device_id: idx for idx, device_id in enumerate(unique_devices)}
        node_features['device'] = device_features
        
        # IP nodes
        ip_features = self._extract_ip_features(df)
        unique_ips = df['ip'].unique()
        node_mappings['ip'] = {ip: idx for idx, ip in enumerate(unique_ips)}
        node_features['ip'] = ip_features
        
        return node_mappings, node_features
    
    def _extract_transaction_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract features for transaction nodes."""
        features = []
        
        # Basic transaction features
        features.append(df['amount'].values.reshape(-1, 1))
        features.append(df['hour'].values.reshape(-1, 1))
        features.append(df['day_of_week'].values.reshape(-1, 1))
        features.append(df['velocity_1h'].values.reshape(-1, 1))
        features.append(df['velocity_24h'].values.reshape(-1, 1))
        features.append(df['amount_std_dev'].values.reshape(-1, 1))
        features.append(df['location_risk_score'].values.reshape(-1, 1))
        
        # One-hot encode transaction type
        tx_type_encoder = LabelEncoder()
        tx_types = tx_type_encoder.fit_transform(df['transaction_type'])
        tx_type_onehot = np.eye(len(tx_type_encoder.classes_))[tx_types]
        features.append(tx_type_onehot)
        
        # One-hot encode merchant category
        cat_encoder = LabelEncoder()
        categories = cat_encoder.fit_transform(df['merchant_category'])
        cat_onehot = np.eye(len(cat_encoder.classes_))[categories]
        features.append(cat_onehot)
        
        # Combine all features
        feature_matrix = np.concatenate(features, axis=1)
        
        # Scale features
        if 'transaction' not in self.feature_scalers:
            self.feature_scalers['transaction'] = StandardScaler()
            feature_matrix = self.feature_scalers['transaction'].fit_transform(feature_matrix)
        else:
            feature_matrix = self.feature_scalers['transaction'].transform(feature_matrix)
        
        return torch.tensor(feature_matrix, dtype=torch.float32)
    
    def _extract_card_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract aggregated features for card nodes."""
        card_stats = df.groupby('card_id').agg({
            'amount': ['mean', 'std', 'count', 'sum'],
            'fraud': ['mean', 'sum'],
            'velocity_1h': 'mean',
            'velocity_24h': 'mean',
            'location_risk_score': 'mean'
        }).fillna(0)
        
        # Flatten column names
        card_stats.columns = ['_'.join(col).strip() for col in card_stats.columns]
        
        # Add derived features
        card_stats['fraud_ratio'] = card_stats['fraud_sum'] / card_stats['amount_count']
        card_stats['avg_transaction_size'] = card_stats['amount_sum'] / card_stats['amount_count']
        
        # Scale features
        if 'card' not in self.feature_scalers:
            self.feature_scalers['card'] = StandardScaler()
            scaled_features = self.feature_scalers['card'].fit_transform(card_stats.values)
        else:
            scaled_features = self.feature_scalers['card'].transform(card_stats.values)
        
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _extract_merchant_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract aggregated features for merchant nodes."""
        merchant_stats = df.groupby('merchant_id').agg({
            'amount': ['mean', 'std', 'count', 'sum'],
            'fraud': ['mean', 'sum'],
            'card_id': 'nunique',
            'device_id': 'nunique',
            'ip': 'nunique'
        }).fillna(0)
        
        # Flatten column names
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        
        # Add derived features
        merchant_stats['fraud_ratio'] = merchant_stats['fraud_sum'] / merchant_stats['amount_count']
        merchant_stats['unique_cards_ratio'] = merchant_stats['card_id_nunique'] / merchant_stats['amount_count']
        merchant_stats['unique_devices_ratio'] = merchant_stats['device_id_nunique'] / merchant_stats['amount_count']
        
        # Scale features
        if 'merchant' not in self.feature_scalers:
            self.feature_scalers['merchant'] = StandardScaler()
            scaled_features = self.feature_scalers['merchant'].fit_transform(merchant_stats.values)
        else:
            scaled_features = self.feature_scalers['merchant'].transform(merchant_stats.values)
        
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _extract_device_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract aggregated features for device nodes."""
        device_stats = df.groupby('device_id').agg({
            'amount': ['mean', 'std', 'count'],
            'fraud': ['mean', 'sum'],
            'card_id': 'nunique',
            'merchant_id': 'nunique',
            'ip': 'nunique'
        }).fillna(0)
        
        # Flatten column names
        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]
        
        # Add derived features
        device_stats['fraud_ratio'] = device_stats['fraud_sum'] / device_stats['amount_count']
        device_stats['merchant_diversity'] = device_stats['merchant_id_nunique'] / device_stats['amount_count']
        
        # Scale features
        if 'device' not in self.feature_scalers:
            self.feature_scalers['device'] = StandardScaler()
            scaled_features = self.feature_scalers['device'].fit_transform(device_stats.values)
        else:
            scaled_features = self.feature_scalers['device'].transform(device_stats.values)
        
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _extract_ip_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract aggregated features for IP nodes."""
        ip_stats = df.groupby('ip').agg({
            'amount': ['mean', 'std', 'count'],
            'fraud': ['mean', 'sum'],
            'card_id': 'nunique',
            'merchant_id': 'nunique',
            'device_id': 'nunique'
        }).fillna(0)
        
        # Flatten column names
        ip_stats.columns = ['_'.join(col).strip() for col in ip_stats.columns]
        
        # Add derived features
        ip_stats['fraud_ratio'] = ip_stats['fraud_sum'] / ip_stats['amount_count']
        ip_stats['entity_diversity'] = (ip_stats['card_id_nunique'] + 
                                       ip_stats['merchant_id_nunique'] + 
                                       ip_stats['device_id_nunique']) / ip_stats['amount_count']
        
        # Scale features
        if 'ip' not in self.feature_scalers:
            self.feature_scalers['ip'] = StandardScaler()
            scaled_features = self.feature_scalers['ip'].fit_transform(ip_stats.values)
        else:
            scaled_features = self.feature_scalers['ip'].transform(ip_stats.values)
        
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _build_edges(self, df: pd.DataFrame, node_mappings: Dict) -> Tuple[Dict, Dict, List]:
        """Build edges between different node types."""
        edge_indices = defaultdict(list)
        edge_features = defaultdict(list)
        edge_types = []
        
        # Transaction-Card edges
        for _, row in df.iterrows():
            txn_idx = node_mappings['transaction'][row['transaction_id']]
            card_idx = node_mappings['card'][row['card_id']]
            
            edge_indices[('transaction', 'uses', 'card')].append([txn_idx, card_idx])
            edge_features[('transaction', 'uses', 'card')].append([row['amount'], row['velocity_1h']])
        
        # Transaction-Merchant edges
        for _, row in df.iterrows():
            txn_idx = node_mappings['transaction'][row['transaction_id']]
            merchant_idx = node_mappings['merchant'][row['merchant_id']]
            
            edge_indices[('transaction', 'at', 'merchant')].append([txn_idx, merchant_idx])
            edge_features[('transaction', 'at', 'merchant')].append([row['amount'], row['location_risk_score']])
        
        # Transaction-Device edges
        for _, row in df.iterrows():
            txn_idx = node_mappings['transaction'][row['transaction_id']]
            device_idx = node_mappings['device'][row['device_id']]
            
            edge_indices[('transaction', 'from', 'device')].append([txn_idx, device_idx])
            edge_features[('transaction', 'from', 'device')].append([row['velocity_24h']])
        
        # Transaction-IP edges
        for _, row in df.iterrows():
            txn_idx = node_mappings['transaction'][row['transaction_id']]
            ip_idx = node_mappings['ip'][row['ip']]
            
            edge_indices[('transaction', 'via', 'ip')].append([txn_idx, ip_idx])
            edge_features[('transaction', 'via', 'ip')].append([row['location_risk_score']])
        
        # Card-Device co-occurrence edges (same card used on multiple devices)
        card_device_pairs = df.groupby(['card_id', 'device_id']).size().reset_index(name='count')
        for _, row in card_device_pairs.iterrows():
            card_idx = node_mappings['card'][row['card_id']]
            device_idx = node_mappings['device'][row['device_id']]
            
            edge_indices[('card', 'used_on', 'device')].append([card_idx, device_idx])
            edge_features[('card', 'used_on', 'device')].append([row['count']])
        
        # Convert to tensors
        for edge_type, indices in edge_indices.items():
            edge_indices[edge_type] = torch.tensor(indices, dtype=torch.long).t()
            if edge_type in edge_features:
                edge_features[edge_type] = torch.tensor(edge_features[edge_type], dtype=torch.float32)
        
        return dict(edge_indices), dict(edge_features), list(edge_indices.keys())
    
    def _build_torch_geometric_graph(self, node_mappings: Dict, node_features: Dict, 
                                   edge_indices: Dict, edge_features: Dict, edge_types: List) -> HeteroData:
        """Build PyTorch Geometric heterogeneous graph."""
        data = HeteroData()
        
        # Add node features
        for node_type, features in node_features.items():
            data[node_type].x = features
            data[node_type].num_nodes = features.shape[0]
        
        # Add edges
        for edge_type, indices in edge_indices.items():
            data[edge_type].edge_index = indices
            if edge_type in edge_features:
                data[edge_type].edge_attr = edge_features[edge_type]
        
        # Convert to undirected for some edge types
        transform = ToUndirected()
        data = transform(data)
        
        return data
    
    def _build_dgl_graph(self, node_mappings: Dict, node_features: Dict, 
                        edge_indices: Dict, edge_types: List) -> dgl.DGLGraph:
        """Build DGL heterogeneous graph."""
        graph_data = {}
        
        for edge_type, indices in edge_indices.items():
            src_type, rel_type, dst_type = edge_type
            src_nodes = indices[0].numpy()
            dst_nodes = indices[1].numpy()
            graph_data[edge_type] = (src_nodes, dst_nodes)
        
        # Create heterogeneous graph
        g = dgl.heterograph(graph_data)
        
        # Add node features
        for node_type, features in node_features.items():
            g.nodes[node_type].data['feat'] = features
        
        return g
    
    def _build_networkx_graph(self, df: pd.DataFrame, node_mappings: Dict, 
                             node_features: Dict, edge_indices: Dict, edge_types: List) -> nx.Graph:
        """Build NetworkX graph for visualization and analysis."""
        G = nx.Graph()
        
        # Add nodes with features
        for node_type, mapping in node_mappings.items():
            for node_id, idx in mapping.items():
                G.add_node(f"{node_type}_{node_id}", 
                          node_type=node_type, 
                          features=node_features[node_type][idx].numpy())
        
        # Add edges
        for edge_type, indices in edge_indices.items():
            src_type, rel_type, dst_type = edge_type
            src_mapping = node_mappings[src_type]
            dst_mapping = node_mappings[dst_type]
            
            for i in range(indices.shape[1]):
                src_idx = indices[0, i].item()
                dst_idx = indices[1, i].item()
                
                # Find original IDs
                src_id = list(src_mapping.keys())[src_idx]
                dst_id = list(dst_mapping.keys())[dst_idx]
                
                G.add_edge(f"{src_type}_{src_id}", f"{dst_type}_{dst_id}", 
                          edge_type=rel_type)
        
        return G

def main():
    """Example usage of the graph builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build heterogeneous fraud detection graph")
    parser.add_argument('--input', type=str, required=True, help='Input transaction CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output graph file prefix')
    parser.add_argument('--graph_type', type=str, default='torch_geometric', 
                       choices=['torch_geometric', 'dgl', 'networkx'], help='Graph library to use')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Build graph
    builder = HeterogeneousGraphBuilder()
    graph = builder.build_graph(df, graph_type=args.graph_type)
    
    # Save graph
    if args.graph_type == 'torch_geometric':
        torch.save(graph, f"{args.output}_torch_geometric.pt")
    elif args.graph_type == 'networkx':
        nx.write_gpickle(graph, f"{args.output}_networkx.gpickle")
    
    logger.info(f"Graph saved to {args.output}_{args.graph_type}")

if __name__ == "__main__":
    main()