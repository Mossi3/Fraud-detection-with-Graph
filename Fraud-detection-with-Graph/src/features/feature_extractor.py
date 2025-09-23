"""
Feature extraction module for graph-based fraud detection.
Extracts node, edge, and subgraph features for machine learning models.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger


class GraphFeatureExtractor:
    """
    Extracts features from fraud detection graphs.
    Includes structural, temporal, and behavioral features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_node_features(self, graph: nx.MultiDiGraph, node_id: str) -> np.ndarray:
        """Extract comprehensive features for a single node"""
        features = []
        
        # Basic node attributes
        node_data = graph.nodes[node_id]
        entity_type = node_data.get("entity_type", "unknown")
        
        # Structural features
        structural_features = self._extract_structural_features(graph, node_id)
        features.extend(structural_features)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(graph, node_id)
        features.extend(temporal_features)
        
        # Behavioral features
        behavioral_features = self._extract_behavioral_features(graph, node_id)
        features.extend(behavioral_features)
        
        # Entity-specific features
        if entity_type == "card":
            card_features = self._extract_card_features(graph, node_id)
            features.extend(card_features)
        elif entity_type == "merchant":
            merchant_features = self._extract_merchant_features(graph, node_id)
            features.extend(merchant_features)
        elif entity_type == "transaction":
            transaction_features = self._extract_transaction_features(graph, node_id)
            features.extend(transaction_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_structural_features(self, graph: nx.MultiDiGraph, node_id: str) -> List[float]:
        """Extract graph structural features"""
        features = []
        
        # Degree features
        in_degree = graph.in_degree(node_id)
        out_degree = graph.out_degree(node_id)
        total_degree = in_degree + out_degree
        
        features.extend([in_degree, out_degree, total_degree])
        
        # Centrality measures
        try:
            # PageRank
            pagerank = nx.pagerank(graph, alpha=0.85).get(node_id, 0)
            features.append(pagerank)
            
            # Betweenness centrality (sample for large graphs)
            if graph.number_of_nodes() < 1000:
                betweenness = nx.betweenness_centrality(graph).get(node_id, 0)
            else:
                betweenness = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes())).get(node_id, 0)
            features.append(betweenness)
            
            # Closeness centrality
            if nx.is_strongly_connected(graph):
                closeness = nx.closeness_centrality(graph).get(node_id, 0)
            else:
                closeness = 0
            features.append(closeness)
            
        except Exception as e:
            logger.warning(f"Error computing centrality measures: {e}")
            features.extend([0, 0, 0])
        
        # Local clustering coefficient
        try:
            clustering = nx.clustering(graph.to_undirected()).get(node_id, 0)
            features.append(clustering)
        except:
            features.append(0)
        
        # Neighbor diversity
        neighbors = list(graph.neighbors(node_id)) + list(graph.predecessors(node_id))
        neighbor_types = [graph.nodes[n].get("entity_type", "unknown") for n in neighbors]
        type_diversity = len(set(neighbor_types)) / max(len(neighbor_types), 1)
        features.append(type_diversity)
        
        # Triangle count
        undirected = graph.to_undirected()
        triangles = sum(1 for _ in nx.triangles(undirected, [node_id]).values()) / 3
        features.append(triangles)
        
        return features
    
    def _extract_temporal_features(self, graph: nx.MultiDiGraph, node_id: str) -> List[float]:
        """Extract temporal pattern features"""
        features = []
        
        # Get timestamps of connected transactions
        timestamps = []
        for neighbor in graph.neighbors(node_id):
            if graph.nodes[neighbor].get("entity_type") == "transaction":
                ts_str = graph.nodes[neighbor].get("timestamp")
                if ts_str:
                    timestamps.append(datetime.fromisoformat(ts_str))
        
        for pred in graph.predecessors(node_id):
            if graph.nodes[pred].get("entity_type") == "transaction":
                ts_str = graph.nodes[pred].get("timestamp")
                if ts_str:
                    timestamps.append(datetime.fromisoformat(ts_str))
        
        if timestamps:
            timestamps.sort()
            
            # Activity frequency features
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            features.append(len(timestamps) / max(time_span, 1))  # transactions per hour
            
            # Time between transactions statistics
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                             for i in range(len(timestamps)-1)]  # minutes
                features.extend([
                    np.mean(time_diffs),
                    np.std(time_diffs),
                    np.min(time_diffs),
                    np.max(time_diffs),
                    np.median(time_diffs)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Hour of day distribution
            hours = [ts.hour for ts in timestamps]
            hour_dist = np.histogram(hours, bins=24, range=(0, 24))[0]
            hour_entropy = -np.sum((hour_dist/len(hours)) * np.log(hour_dist/len(hours) + 1e-10))
            features.append(hour_entropy)
            
            # Unusual time activity (late night: 0-6 AM)
            late_night_ratio = sum(1 for h in hours if 0 <= h < 6) / len(hours)
            features.append(late_night_ratio)
            
        else:
            features.extend([0] * 8)
        
        return features
    
    def _extract_behavioral_features(self, graph: nx.MultiDiGraph, node_id: str) -> List[float]:
        """Extract behavioral pattern features"""
        features = []
        
        # Velocity patterns
        velocity_patterns = 0
        for neighbor in graph.neighbors(node_id):
            if graph.nodes[neighbor].get("entity_type") == "velocity_pattern":
                velocity_patterns += 1
        features.append(velocity_patterns)
        
        # Shared resources (devices, IPs)
        shared_devices = 0
        shared_ips = 0
        
        for _, _, edge_data in graph.edges(node_id, data=True):
            if edge_data.get("relationship_type") == "shares_device":
                shared_devices += 1
            elif edge_data.get("relationship_type") == "shares_ip":
                shared_ips += 1
        
        features.extend([shared_devices, shared_ips])
        
        # Risk score aggregation
        risk_scores = []
        for _, _, edge_data in graph.edges(node_id, data=True):
            if "risk_score" in edge_data:
                risk_scores.append(edge_data["risk_score"])
        
        if risk_scores:
            features.extend([
                np.mean(risk_scores),
                np.max(risk_scores),
                np.sum(risk_scores)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Connection to known fraud
        fraud_connections = 0
        fraud_distance = float('inf')
        
        # BFS to find nearest fraud
        visited = {node_id}
        queue = [(node_id, 0)]
        
        while queue and fraud_distance == float('inf'):
            current, dist = queue.pop(0)
            
            for neighbor in list(graph.neighbors(current)) + list(graph.predecessors(current)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    if graph.nodes[neighbor].get("is_fraud", False):
                        fraud_connections += 1
                        fraud_distance = min(fraud_distance, dist + 1)
                    
                    if dist < 3:  # Limit search depth
                        queue.append((neighbor, dist + 1))
        
        features.extend([
            fraud_connections,
            1.0 / (fraud_distance + 1) if fraud_distance < float('inf') else 0
        ])
        
        return features
    
    def _extract_card_features(self, graph: nx.MultiDiGraph, node_id: str) -> List[float]:
        """Extract card-specific features"""
        features = []
        
        # Transaction statistics
        transactions = []
        merchants = set()
        
        for neighbor in graph.neighbors(node_id):
            if graph.nodes[neighbor].get("entity_type") == "transaction":
                amount = graph.nodes[neighbor].get("amount", 0)
                transactions.append(amount)
                
                # Find merchant
                for merchant in graph.neighbors(neighbor):
                    if graph.nodes[merchant].get("entity_type") == "merchant":
                        merchants.add(merchant)
        
        if transactions:
            features.extend([
                len(transactions),  # transaction count
                np.mean(transactions),
                np.std(transactions),
                np.min(transactions),
                np.max(transactions),
                np.sum(transactions)
            ])
        else:
            features.extend([0] * 6)
        
        # Merchant diversity
        features.append(len(merchants))
        
        # Device and IP counts
        devices = sum(1 for n in graph.neighbors(node_id) 
                     if graph.nodes[n].get("entity_type") == "device")
        ips = sum(1 for n in graph.neighbors(node_id) 
                 if graph.nodes[n].get("entity_type") == "ip_address")
        
        features.extend([devices, ips])
        
        return features
    
    def _extract_merchant_features(self, graph: nx.MultiDiGraph, node_id: str) -> List[float]:
        """Extract merchant-specific features"""
        features = []
        
        # Transaction and card statistics
        transactions = []
        unique_cards = set()
        fraud_transactions = 0
        
        for pred in graph.predecessors(node_id):
            if graph.nodes[pred].get("entity_type") == "transaction":
                amount = graph.nodes[pred].get("amount", 0)
                transactions.append(amount)
                
                if graph.nodes[pred].get("is_fraud", False):
                    fraud_transactions += 1
                
                # Find card
                for card in graph.predecessors(pred):
                    if graph.nodes[card].get("entity_type") == "card":
                        unique_cards.add(card)
        
        if transactions:
            features.extend([
                len(transactions),
                np.mean(transactions),
                np.std(transactions),
                fraud_transactions / len(transactions)  # fraud rate
            ])
        else:
            features.extend([0] * 4)
        
        features.append(len(unique_cards))
        
        # Customer loyalty (repeat customers)
        card_transaction_counts = defaultdict(int)
        for pred in graph.predecessors(node_id):
            if graph.nodes[pred].get("entity_type") == "transaction":
                for card in graph.predecessors(pred):
                    if graph.nodes[card].get("entity_type") == "card":
                        card_transaction_counts[card] += 1
        
        if card_transaction_counts:
            repeat_rate = sum(1 for count in card_transaction_counts.values() if count > 1) / len(card_transaction_counts)
            avg_transactions_per_card = np.mean(list(card_transaction_counts.values()))
        else:
            repeat_rate = 0
            avg_transactions_per_card = 0
        
        features.extend([repeat_rate, avg_transactions_per_card])
        
        return features
    
    def _extract_transaction_features(self, graph: nx.MultiDiGraph, node_id: str) -> List[float]:
        """Extract transaction-specific features"""
        features = []
        
        node_data = graph.nodes[node_id]
        
        # Basic transaction features
        amount = node_data.get("amount", 0)
        features.append(amount)
        
        # Time-based features
        if "timestamp" in node_data:
            ts = datetime.fromisoformat(node_data["timestamp"])
            features.extend([
                ts.hour,
                ts.weekday(),
                ts.day,
                int(ts.weekday() >= 5)  # is weekend
            ])
        else:
            features.extend([0] * 4)
        
        # Context features (card and merchant history)
        card_history_features = []
        merchant_history_features = []
        
        for pred in graph.predecessors(node_id):
            if graph.nodes[pred].get("entity_type") == "card":
                card_features = self._extract_card_features(graph, pred)
                card_history_features = card_features[:6]  # transaction statistics
        
        for succ in graph.neighbors(node_id):
            if graph.nodes[succ].get("entity_type") == "merchant":
                merchant_features = self._extract_merchant_features(graph, succ)
                merchant_history_features = merchant_features[:4]  # merchant statistics
        
        features.extend(card_history_features if card_history_features else [0] * 6)
        features.extend(merchant_history_features if merchant_history_features else [0] * 4)
        
        return features
    
    def extract_subgraph_features(self, graph: nx.MultiDiGraph, 
                                 center_nodes: List[str], 
                                 max_depth: int = 2) -> Dict[str, Any]:
        """Extract features from a subgraph around center nodes"""
        # Get subgraph
        nodes_to_include = set(center_nodes)
        
        for _ in range(max_depth):
            new_nodes = set()
            for node in nodes_to_include:
                if node in graph:
                    new_nodes.update(graph.neighbors(node))
                    new_nodes.update(graph.predecessors(node))
            nodes_to_include.update(new_nodes)
        
        subgraph = graph.subgraph(nodes_to_include)
        
        # Extract subgraph statistics
        features = {
            "num_nodes": subgraph.number_of_nodes(),
            "num_edges": subgraph.number_of_edges(),
            "density": nx.density(subgraph),
            "avg_clustering": nx.average_clustering(subgraph.to_undirected()),
        }
        
        # Entity type distribution
        entity_counts = defaultdict(int)
        for node in subgraph.nodes():
            entity_type = subgraph.nodes[node].get("entity_type", "unknown")
            entity_counts[entity_type] += 1
        
        features["entity_distribution"] = dict(entity_counts)
        
        # Fraud rate in subgraph
        fraud_count = sum(1 for _, data in subgraph.nodes(data=True) 
                         if data.get("is_fraud", False))
        features["fraud_rate"] = fraud_count / max(subgraph.number_of_nodes(), 1)
        
        # Pattern detection
        features["has_rings"] = self._detect_rings(subgraph)
        features["has_chains"] = self._detect_chains(subgraph)
        
        return features
    
    def _detect_rings(self, graph: nx.MultiDiGraph) -> bool:
        """Detect ring patterns in the graph"""
        try:
            cycles = list(nx.simple_cycles(graph))
            return len(cycles) > 0
        except:
            return False
    
    def _detect_chains(self, graph: nx.MultiDiGraph) -> bool:
        """Detect chain patterns in the graph"""
        # Look for long paths
        for node in graph.nodes():
            if graph.in_degree(node) == 0:  # potential start of chain
                path_lengths = nx.single_source_shortest_path_length(graph, node)
                if max(path_lengths.values()) >= 4:
                    return True
        return False
    
    def create_feature_matrix(self, graph: nx.MultiDiGraph, 
                            node_ids: List[str]) -> np.ndarray:
        """Create feature matrix for multiple nodes"""
        features = []
        
        for node_id in node_ids:
            if node_id in graph:
                node_features = self.extract_node_features(graph, node_id)
                features.append(node_features)
            else:
                logger.warning(f"Node {node_id} not found in graph")
                features.append(np.zeros(len(self.feature_names)))
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        if not self.feature_names:
            self.feature_names = [
                # Structural features
                "in_degree", "out_degree", "total_degree",
                "pagerank", "betweenness", "closeness",
                "clustering_coefficient", "neighbor_diversity", "triangles",
                
                # Temporal features
                "transaction_frequency", 
                "time_diff_mean", "time_diff_std", "time_diff_min", 
                "time_diff_max", "time_diff_median",
                "hour_entropy", "late_night_ratio",
                
                # Behavioral features
                "velocity_patterns", "shared_devices", "shared_ips",
                "risk_score_mean", "risk_score_max", "risk_score_sum",
                "fraud_connections", "fraud_proximity",
                
                # Entity-specific features (varies by type)
                # Add more as needed
            ]
        
        return self.feature_names