"""
Advanced Graph Construction Module for Fraud Detection
Supports heterogeneous graphs with multiple node and edge types
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for graph construction"""
    # Node types
    include_cards: bool = True
    include_merchants: bool = True
    include_devices: bool = True
    include_ips: bool = True
    include_accounts: bool = True
    
    # Edge types
    card_merchant_edges: bool = True
    device_ip_edges: bool = True
    card_device_edges: bool = True
    merchant_device_edges: bool = True
    account_card_edges: bool = True
    
    # Temporal features
    time_window_hours: int = 24
    temporal_decay: float = 0.9
    
    # Graph properties
    directed: bool = True
    weighted: bool = True
    self_loops: bool = False

class HeterogeneousGraphBuilder:
    """
    Builds heterogeneous graphs for fraud detection with multiple node and edge types
    """
    
    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.graph = nx.MultiDiGraph() if self.config.directed else nx.MultiGraph()
        self.node_features = {}
        self.edge_features = {}
        
    def add_transaction_data(self, df: pd.DataFrame) -> 'HeterogeneousGraphBuilder':
        """Add transaction data to build the graph"""
        logger.info(f"Adding {len(df)} transactions to graph")
        
        # Add nodes
        self._add_nodes(df)
        
        # Add edges
        self._add_edges(df)
        
        # Add temporal features
        if self.config.time_window_hours > 0:
            self._add_temporal_features(df)
            
        return self
    
    def _add_nodes(self, df: pd.DataFrame):
        """Add nodes of different types to the graph"""
        
        if self.config.include_cards:
            cards = df['card_id'].unique()
            for card in cards:
                self.graph.add_node(card, node_type='card')
                
        if self.config.include_merchants:
            merchants = df['merchant_id'].unique()
            for merchant in merchants:
                self.graph.add_node(merchant, node_type='merchant')
                
        if self.config.include_devices:
            devices = df['device_id'].unique()
            for device in devices:
                self.graph.add_node(device, node_type='device')
                
        if self.config.include_ips:
            ips = df['ip'].unique()
            for ip in ips:
                self.graph.add_node(ip, node_type='ip')
                
        if self.config.include_accounts and 'account_id' in df.columns:
            accounts = df['account_id'].unique()
            for account in accounts:
                self.graph.add_node(account, node_type='account')
    
    def _add_edges(self, df: pd.DataFrame):
        """Add edges between different node types"""
        
        for _, row in df.iterrows():
            timestamp = row.get('timestamp', 0)
            amount = row.get('amount', 1.0)
            fraud_label = row.get('fraud', 0)
            
            # Card-Merchant edges
            if self.config.card_merchant_edges:
                edge_weight = self._calculate_edge_weight(amount, fraud_label, timestamp)
                self.graph.add_edge(
                    row['card_id'], 
                    row['merchant_id'],
                    edge_type='card_merchant',
                    weight=edge_weight,
                    timestamp=timestamp,
                    amount=amount,
                    fraud=fraud_label
                )
            
            # Device-IP edges
            if self.config.device_ip_edges:
                self.graph.add_edge(
                    row['device_id'],
                    row['ip'],
                    edge_type='device_ip',
                    weight=1.0,
                    timestamp=timestamp
                )
            
            # Card-Device edges
            if self.config.card_device_edges:
                self.graph.add_edge(
                    row['card_id'],
                    row['device_id'],
                    edge_type='card_device',
                    weight=edge_weight,
                    timestamp=timestamp
                )
            
            # Merchant-Device edges
            if self.config.merchant_device_edges:
                self.graph.add_edge(
                    row['merchant_id'],
                    row['device_id'],
                    edge_type='merchant_device',
                    weight=edge_weight,
                    timestamp=timestamp
                )
            
            # Account-Card edges (if account data available)
            if self.config.account_card_edges and 'account_id' in df.columns:
                self.graph.add_edge(
                    row['account_id'],
                    row['card_id'],
                    edge_type='account_card',
                    weight=1.0,
                    timestamp=timestamp
                )
    
    def _calculate_edge_weight(self, amount: float, fraud_label: int, timestamp: int) -> float:
        """Calculate edge weight based on transaction characteristics"""
        base_weight = np.log(amount + 1)  # Log-scale for amount
        
        # Boost weight for fraud transactions
        fraud_boost = 1.5 if fraud_label == 1 else 1.0
        
        # Temporal decay
        if self.config.temporal_decay < 1.0:
            # Simple temporal decay - in practice, you'd use actual timestamps
            temporal_factor = self.config.temporal_decay
        else:
            temporal_factor = 1.0
            
        return base_weight * fraud_boost * temporal_factor
    
    def _add_temporal_features(self, df: pd.DataFrame):
        """Add temporal features to nodes and edges"""
        
        # Calculate temporal features for each node type
        for node_type in ['card', 'merchant', 'device', 'ip']:
            if node_type in ['card', 'merchant', 'device', 'ip']:
                node_col = f"{node_type}_id" if node_type != 'ip' else 'ip'
                if node_col in df.columns:
                    self._add_node_temporal_features(df, node_col, node_type)
    
    def _add_node_temporal_features(self, df: pd.DataFrame, node_col: str, node_type: str):
        """Add temporal features for a specific node type"""
        
        for node in df[node_col].unique():
            node_data = df[df[node_col] == node]
            
            # Calculate temporal features
            features = {
                'transaction_count': len(node_data),
                'fraud_rate': node_data['fraud'].mean(),
                'avg_amount': node_data['amount'].mean(),
                'max_amount': node_data['amount'].max(),
                'min_amount': node_data['amount'].min(),
                'amount_std': node_data['amount'].std(),
                'unique_merchants': node_data['merchant_id'].nunique() if 'merchant_id' in node_data.columns else 0,
                'unique_devices': node_data['device_id'].nunique() if 'device_id' in node_data.columns else 0,
                'unique_ips': node_data['ip'].nunique() if 'ip' in node_data.columns else 0
            }
            
            # Add features to node
            for feature_name, feature_value in features.items():
                if pd.notna(feature_value):
                    self.graph.nodes[node][feature_name] = float(feature_value)
    
    def get_subgraph_by_type(self, node_type: str) -> nx.Graph:
        """Extract subgraph containing only nodes of specified type"""
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == node_type]
        return self.graph.subgraph(nodes)
    
    def get_edges_by_type(self, edge_type: str) -> List[Tuple]:
        """Get all edges of specified type"""
        return [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == edge_type]
    
    def calculate_centrality_measures(self) -> Dict[str, Dict]:
        """Calculate various centrality measures for fraud detection"""
        
        centrality_measures = {}
        
        # Degree centrality
        centrality_measures['degree'] = nx.degree_centrality(self.graph)
        
        # Betweenness centrality
        centrality_measures['betweenness'] = nx.betweenness_centrality(self.graph, weight='weight')
        
        # Closeness centrality
        centrality_measures['closeness'] = nx.closeness_centrality(self.graph, distance='weight')
        
        # PageRank
        centrality_measures['pagerank'] = nx.pagerank(self.graph, weight='weight')
        
        # Eigenvector centrality
        try:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(self.graph, weight='weight')
        except:
            centrality_measures['eigenvector'] = {}
        
        return centrality_measures
    
    def detect_suspicious_patterns(self) -> Dict[str, List]:
        """Detect suspicious patterns in the graph"""
        
        suspicious_patterns = {
            'high_degree_nodes': [],
            'high_betweenness_nodes': [],
            'suspicious_edges': [],
            'isolated_fraud': []
        }
        
        # High degree nodes (potential fraud rings)
        degree_centrality = nx.degree_centrality(self.graph)
        high_degree_threshold = np.percentile(list(degree_centrality.values()), 95)
        
        for node, centrality in degree_centrality.items():
            if centrality > high_degree_threshold:
                suspicious_patterns['high_degree_nodes'].append({
                    'node': node,
                    'degree_centrality': centrality,
                    'node_type': self.graph.nodes[node].get('node_type', 'unknown')
                })
        
        # High betweenness nodes (potential intermediaries)
        betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight')
        high_betweenness_threshold = np.percentile(list(betweenness_centrality.values()), 95)
        
        for node, centrality in betweenness_centrality.items():
            if centrality > high_betweenness_threshold:
                suspicious_patterns['high_betweenness_nodes'].append({
                    'node': node,
                    'betweenness_centrality': centrality,
                    'node_type': self.graph.nodes[node].get('node_type', 'unknown')
                })
        
        # Suspicious edges (high weight fraud edges)
        for u, v, d in self.graph.edges(data=True):
            if d.get('fraud', 0) == 1 and d.get('weight', 0) > 2.0:
                suspicious_patterns['suspicious_edges'].append({
                    'source': u,
                    'target': v,
                    'weight': d.get('weight', 0),
                    'edge_type': d.get('edge_type', 'unknown'),
                    'amount': d.get('amount', 0)
                })
        
        return suspicious_patterns
    
    def to_networkx(self) -> nx.Graph:
        """Return the constructed NetworkX graph"""
        return self.graph
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.config.directed else nx.is_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph) if self.config.directed else nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'node_types': {},
            'edge_types': {}
        }
        
        # Count node types
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edge types
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        return stats

def build_fraud_graph(df: pd.DataFrame, config: GraphConfig = None) -> HeterogeneousGraphBuilder:
    """
    Convenience function to build a fraud detection graph from transaction data
    
    Args:
        df: Transaction dataframe with columns: card_id, merchant_id, device_id, ip, amount, fraud, timestamp
        config: Graph configuration
    
    Returns:
        HeterogeneousGraphBuilder with constructed graph
    """
    builder = HeterogeneousGraphBuilder(config)
    builder.add_transaction_data(df)
    return builder