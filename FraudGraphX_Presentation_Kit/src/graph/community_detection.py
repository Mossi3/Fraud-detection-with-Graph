"""
Community Detection Algorithms for Fraud Ring Detection
Implements Louvain, Leiden, and other community detection methods
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning("Leiden algorithm not available. Install leidenalg and igraph for advanced community detection.")

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some clustering methods will be disabled.")

logger = logging.getLogger(__name__)

class CommunityDetector:
    """
    Community detection for fraud ring identification
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.communities = {}
        self.community_stats = {}
        
    def detect_louvain_communities(self, resolution: float = 1.0, 
                                 random_state: int = 42) -> Dict[int, List]:
        """
        Detect communities using the Louvain algorithm
        
        Args:
            resolution: Resolution parameter for modularity optimization
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping community ID to list of nodes
        """
        logger.info("Detecting communities using Louvain algorithm")
        
        # Convert to undirected graph for community detection
        if self.graph.is_directed():
            undirected_graph = self.graph.to_undirected()
        else:
            undirected_graph = self.graph.copy()
        
        # Run Louvain algorithm
        communities = nx.community.louvain_communities(
            undirected_graph, 
            resolution=resolution, 
            seed=random_state
        )
        
        # Convert to dictionary format
        community_dict = {}
        for i, community in enumerate(communities):
            community_dict[i] = list(community)
        
        self.communities['louvain'] = community_dict
        self._calculate_community_stats('louvain')
        
        return community_dict
    
    def detect_leiden_communities(self, resolution: float = 1.0, 
                                random_state: int = 42) -> Dict[int, List]:
        """
        Detect communities using the Leiden algorithm
        
        Args:
            resolution: Resolution parameter for modularity optimization
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping community ID to list of nodes
        """
        if not LEIDEN_AVAILABLE:
            logger.warning("Leiden algorithm not available. Falling back to Louvain.")
            return self.detect_louvain_communities(resolution, random_state)
        
        logger.info("Detecting communities using Leiden algorithm")
        
        # Convert NetworkX graph to igraph
        ig_graph = ig.Graph.from_networkx(self.graph)
        
        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig_graph, 
            leidenalg.ModularityVertexPartition,
            resolution_parameter=resolution,
            seed=random_state
        )
        
        # Convert to dictionary format
        community_dict = {}
        for i, community in enumerate(partition):
            community_dict[i] = [ig_graph.vs[node]['name'] for node in community]
        
        self.communities['leiden'] = community_dict
        self._calculate_community_stats('leiden')
        
        return community_dict
    
    def detect_fraud_rings(self, fraud_labels: Dict[str, int], 
                          min_community_size: int = 3,
                          fraud_threshold: float = 0.3) -> Dict[int, Dict]:
        """
        Detect fraud rings based on community structure and fraud labels
        
        Args:
            fraud_labels: Dictionary mapping node to fraud label (0/1)
            min_community_size: Minimum size for a community to be considered
            fraud_threshold: Minimum fraud rate to consider a community suspicious
            
        Returns:
            Dictionary with fraud ring information
        """
        logger.info("Detecting fraud rings from communities")
        
        fraud_rings = {}
        
        for method, communities in self.communities.items():
            fraud_rings[method] = {}
            
            for comm_id, nodes in communities.items():
                if len(nodes) < min_community_size:
                    continue
                
                # Calculate fraud statistics for this community
                fraud_nodes = [node for node in nodes if fraud_labels.get(node, 0) == 1]
                fraud_rate = len(fraud_nodes) / len(nodes)
                
                if fraud_rate >= fraud_threshold:
                    fraud_rings[method][comm_id] = {
                        'nodes': nodes,
                        'fraud_nodes': fraud_nodes,
                        'fraud_rate': fraud_rate,
                        'size': len(nodes),
                        'fraud_count': len(fraud_nodes),
                        'suspicious_score': self._calculate_suspicious_score(nodes, fraud_labels)
                    }
        
        return fraud_rings
    
    def _calculate_suspicious_score(self, nodes: List[str], fraud_labels: Dict[str, int]) -> float:
        """
        Calculate suspicious score for a community based on various factors
        """
        if not nodes:
            return 0.0
        
        # Fraud rate
        fraud_rate = sum(fraud_labels.get(node, 0) for node in nodes) / len(nodes)
        
        # Connectivity within community
        internal_edges = 0
        total_possible_edges = len(nodes) * (len(nodes) - 1) / 2
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if self.graph.has_edge(node1, node2) or self.graph.has_edge(node2, node1):
                    internal_edges += 1
        
        connectivity = internal_edges / total_possible_edges if total_possible_edges > 0 else 0
        
        # Node type diversity (if available)
        node_types = [self.graph.nodes[node].get('node_type', 'unknown') for node in nodes]
        type_diversity = len(set(node_types)) / len(node_types) if node_types else 0
        
        # Combined suspicious score
        suspicious_score = (
            0.5 * fraud_rate +
            0.3 * connectivity +
            0.2 * type_diversity
        )
        
        return suspicious_score
    
    def _calculate_community_stats(self, method: str):
        """Calculate statistics for detected communities"""
        
        communities = self.communities[method]
        
        stats = {
            'num_communities': len(communities),
            'avg_community_size': np.mean([len(comm) for comm in communities.values()]),
            'max_community_size': max([len(comm) for comm in communities.values()]) if communities else 0,
            'min_community_size': min([len(comm) for comm in communities.values()]) if communities else 0,
            'community_sizes': [len(comm) for comm in communities.values()],
            'modularity': self._calculate_modularity(communities)
        }
        
        self.community_stats[method] = stats
    
    def _calculate_modularity(self, communities: Dict[int, List]) -> float:
        """Calculate modularity of the community structure"""
        
        if self.graph.is_directed():
            # For directed graphs, use directed modularity
            return nx.community.modularity(self.graph, communities.values())
        else:
            # For undirected graphs
            undirected_graph = self.graph.to_undirected()
            return nx.community.modularity(undirected_graph, communities.values())
    
    def analyze_community_overlap(self) -> Dict[str, float]:
        """
        Analyze overlap between different community detection methods
        """
        overlap_stats = {}
        
        methods = list(self.communities.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                overlap = self._calculate_method_overlap(method1, method2)
                overlap_stats[f"{method1}_vs_{method2}"] = overlap
        
        return overlap_stats
    
    def _calculate_method_overlap(self, method1: str, method2: str) -> float:
        """Calculate overlap between two community detection methods"""
        
        comm1 = self.communities[method1]
        comm2 = self.communities[method2]
        
        total_overlap = 0
        total_nodes = 0
        
        for comm1_nodes in comm1.values():
            for comm2_nodes in comm2.values():
                overlap = len(set(comm1_nodes) & set(comm2_nodes))
                total_overlap += overlap
                total_nodes += len(set(comm1_nodes) | set(comm2_nodes))
        
        return total_overlap / total_nodes if total_nodes > 0 else 0.0
    
    def get_community_features(self, communities: Dict[int, List]) -> pd.DataFrame:
        """
        Extract features for each community
        """
        
        features = []
        
        for comm_id, nodes in communities.items():
            # Basic community features
            comm_features = {
                'community_id': comm_id,
                'size': len(nodes),
                'node_types': [self.graph.nodes[node].get('node_type', 'unknown') for node in nodes],
                'fraud_rate': 0.0,  # Will be calculated if fraud labels are available
                'avg_degree': 0.0,
                'internal_edges': 0,
                'external_edges': 0
            }
            
            # Calculate degree statistics
            degrees = [self.graph.degree(node) for node in nodes]
            comm_features['avg_degree'] = np.mean(degrees) if degrees else 0
            comm_features['max_degree'] = np.max(degrees) if degrees else 0
            comm_features['min_degree'] = np.min(degrees) if degrees else 0
            
            # Calculate edge statistics
            internal_edges = 0
            external_edges = 0
            
            for node in nodes:
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor in nodes:
                        internal_edges += 1
                    else:
                        external_edges += 1
            
            comm_features['internal_edges'] = internal_edges
            comm_features['external_edges'] = external_edges
            comm_features['internal_ratio'] = internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
            
            features.append(comm_features)
        
        return pd.DataFrame(features)
    
    def visualize_communities(self, communities: Dict[int, List], 
                            fraud_labels: Dict[str, int] = None,
                            save_path: str = None) -> None:
        """
        Visualize detected communities
        """
        
        plt.figure(figsize=(15, 10))
        
        # Create subplot
        ax = plt.subplot(111)
        
        # Color nodes by community
        node_colors = {}
        for comm_id, nodes in communities.items():
            color = plt.cm.Set3(comm_id % 12)  # Use different colors for different communities
            for node in nodes:
                node_colors[node] = color
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw graph
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=[node_colors.get(node, 'lightgray') for node in self.graph.nodes()],
                              node_size=50, alpha=0.7)
        
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=0.5)
        
        # Highlight fraud nodes
        if fraud_labels:
            fraud_nodes = [node for node, label in fraud_labels.items() if label == 1]
            nx.draw_networkx_nodes(self.graph, pos, 
                                 nodelist=fraud_nodes,
                                 node_color='red', node_size=100, alpha=0.8)
        
        plt.title(f"Community Detection Results ({len(communities)} communities)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class FraudRingAnalyzer:
    """
    Advanced analysis of detected fraud rings
    """
    
    def __init__(self, communities: Dict[int, List], graph: nx.Graph):
        self.communities = communities
        self.graph = graph
        self.ring_analysis = {}
    
    def analyze_ring_patterns(self, fraud_labels: Dict[str, int]) -> Dict[str, any]:
        """
        Analyze patterns in detected fraud rings
        """
        
        analysis = {
            'ring_statistics': {},
            'node_patterns': {},
            'edge_patterns': {},
            'temporal_patterns': {},
            'anomaly_scores': {}
        }
        
        for comm_id, nodes in self.communities.items():
            if len(nodes) < 3:  # Skip small communities
                continue
            
            ring_analysis = self._analyze_single_ring(comm_id, nodes, fraud_labels)
            analysis['ring_statistics'][comm_id] = ring_analysis
        
        return analysis
    
    def _analyze_single_ring(self, comm_id: int, nodes: List[str], 
                           fraud_labels: Dict[str, int]) -> Dict[str, any]:
        """Analyze a single fraud ring"""
        
        # Basic statistics
        fraud_nodes = [node for node in nodes if fraud_labels.get(node, 0) == 1]
        fraud_rate = len(fraud_nodes) / len(nodes)
        
        # Node type analysis
        node_types = [self.graph.nodes[node].get('node_type', 'unknown') for node in nodes]
        type_counts = Counter(node_types)
        
        # Edge analysis
        internal_edges = 0
        external_edges = 0
        edge_types = []
        
        for node in nodes:
            neighbors = list(self.graph.neighbors(node))
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(node, neighbor, default={})
                edge_type = edge_data.get('edge_type', 'unknown')
                edge_types.append(edge_type)
                
                if neighbor in nodes:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        edge_type_counts = Counter(edge_types)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_ring_anomaly_score(nodes, fraud_labels)
        
        return {
            'size': len(nodes),
            'fraud_rate': fraud_rate,
            'fraud_count': len(fraud_nodes),
            'node_types': dict(type_counts),
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'edge_types': dict(edge_type_counts),
            'anomaly_score': anomaly_score,
            'connectivity': internal_edges / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }
    
    def _calculate_ring_anomaly_score(self, nodes: List[str], 
                                   fraud_labels: Dict[str, int]) -> float:
        """Calculate anomaly score for a fraud ring"""
        
        # Fraud rate component
        fraud_rate = sum(fraud_labels.get(node, 0) for node in nodes) / len(nodes)
        
        # Connectivity component
        internal_edges = 0
        total_possible = len(nodes) * (len(nodes) - 1) / 2
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if self.graph.has_edge(node1, node2) or self.graph.has_edge(node2, node1):
                    internal_edges += 1
        
        connectivity = internal_edges / total_possible if total_possible > 0 else 0
        
        # Node type diversity component
        node_types = [self.graph.nodes[node].get('node_type', 'unknown') for node in nodes]
        type_diversity = len(set(node_types)) / len(node_types) if node_types else 0
        
        # Combined anomaly score
        anomaly_score = (
            0.4 * fraud_rate +
            0.3 * connectivity +
            0.3 * type_diversity
        )
        
        return anomaly_score
    
    def detect_ring_leaders(self, fraud_labels: Dict[str, int]) -> Dict[int, List[str]]:
        """
        Detect potential ring leaders based on centrality and fraud status
        """
        
        ring_leaders = {}
        
        for comm_id, nodes in self.communities.items():
            if len(nodes) < 3:
                continue
            
            # Calculate centrality measures for nodes in this community
            subgraph = self.graph.subgraph(nodes)
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(subgraph)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            
            # Combine centrality measures
            combined_scores = {}
            for node in nodes:
                score = (
                    0.4 * degree_centrality.get(node, 0) +
                    0.6 * betweenness_centrality.get(node, 0)
                )
                combined_scores[node] = score
            
            # Sort by combined score and fraud status
            sorted_nodes = sorted(nodes, key=lambda x: (
                fraud_labels.get(x, 0),  # Fraud nodes first
                combined_scores.get(x, 0)  # Then by centrality
            ), reverse=True)
            
            # Top 20% as potential leaders
            num_leaders = max(1, len(sorted_nodes) // 5)
            ring_leaders[comm_id] = sorted_nodes[:num_leaders]
        
        return ring_leaders
    
    def generate_ring_report(self, fraud_labels: Dict[str, int]) -> str:
        """
        Generate a comprehensive report on detected fraud rings
        """
        
        report = []
        report.append("=== FRAUD RING DETECTION REPORT ===\n")
        
        # Overall statistics
        total_nodes = sum(len(nodes) for nodes in self.communities.values())
        total_fraud_nodes = sum(fraud_labels.get(node, 0) for nodes in self.communities.values() for node in nodes)
        
        report.append(f"Total communities detected: {len(self.communities)}")
        report.append(f"Total nodes in communities: {total_nodes}")
        report.append(f"Total fraud nodes: {total_fraud_nodes}")
        report.append(f"Overall fraud rate: {total_fraud_nodes/total_nodes:.3f}\n")
        
        # Individual ring analysis
        for comm_id, nodes in self.communities.items():
            if len(nodes) < 3:
                continue
            
            fraud_nodes = [node for node in nodes if fraud_labels.get(node, 0) == 1]
            fraud_rate = len(fraud_nodes) / len(nodes)
            
            report.append(f"--- Community {comm_id} ---")
            report.append(f"Size: {len(nodes)} nodes")
            report.append(f"Fraud rate: {fraud_rate:.3f}")
            report.append(f"Fraud nodes: {len(fraud_nodes)}")
            
            # Node type breakdown
            node_types = [self.graph.nodes[node].get('node_type', 'unknown') for node in nodes]
            type_counts = Counter(node_types)
            report.append(f"Node types: {dict(type_counts)}")
            
            report.append("")
        
        return "\n".join(report)