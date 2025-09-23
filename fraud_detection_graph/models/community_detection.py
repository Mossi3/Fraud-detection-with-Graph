"""
Community Detection for Fraud Ring Identification
Uses Louvain algorithm and other methods to detect fraud rings in the graph.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import community as community_louvain
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import torch
from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/workspace/fraud_detection_graph')
from models.graph_builder import FraudGraphBuilder

class FraudRingDetector:
    """Detect fraud rings using various community detection algorithms"""
    
    def __init__(self):
        self.communities = {}
        self.community_stats = {}
        self.fraud_rings = []
        
    def load_graph_data(self, data_path: str = '/workspace/fraud_detection_graph/data/'):
        """Load graph and transaction data"""
        builder = FraudGraphBuilder()
        
        # Load processed data
        self.data = builder.load_data(data_path)
        self.hetero_data, self.bipartite_graph = builder.load_graph(data_path)
        
        # Load node mappings
        import pickle
        with open(f'{data_path}node_mappings.pkl', 'rb') as f:
            self.node_mappings = pickle.load(f)
        
        print(f"Loaded graph with {self.bipartite_graph.number_of_nodes()} nodes and {self.bipartite_graph.number_of_edges()} edges")
        
    def create_projection_graphs(self) -> Dict[str, nx.Graph]:
        """Create projection graphs for different entity types"""
        projections = {}
        
        # Card-Card projection (cards connected through merchants)
        card_nodes = [n for n, d in self.bipartite_graph.nodes(data=True) if d['type'] == 'card']
        merchant_nodes = [n for n, d in self.bipartite_graph.nodes(data=True) if d['type'] == 'merchant']
        
        # Create card projection
        card_graph = nx.Graph()
        card_graph.add_nodes_from(card_nodes)
        
        # Connect cards that share merchants
        merchant_cards = defaultdict(list)
        for card in card_nodes:
            for merchant in self.bipartite_graph.neighbors(card):
                merchant_cards[merchant].append(card)
        
        for merchant, cards in merchant_cards.items():
            for i in range(len(cards)):
                for j in range(i+1, len(cards)):
                    card1, card2 = cards[i], cards[j]
                    if card_graph.has_edge(card1, card2):
                        card_graph[card1][card2]['weight'] += 1
                        card_graph[card1][card2]['shared_merchants'].append(merchant)
                    else:
                        card_graph.add_edge(card1, card2, weight=1, shared_merchants=[merchant])
        
        projections['card'] = card_graph
        
        # Merchant-Merchant projection (merchants connected through cards)
        merchant_graph = nx.Graph()
        merchant_graph.add_nodes_from(merchant_nodes)
        
        card_merchants = defaultdict(list)
        for merchant in merchant_nodes:
            for card in self.bipartite_graph.neighbors(merchant):
                card_merchants[card].append(merchant)
        
        for card, merchants in card_merchants.items():
            for i in range(len(merchants)):
                for j in range(i+1, len(merchants)):
                    merchant1, merchant2 = merchants[i], merchants[j]
                    if merchant_graph.has_edge(merchant1, merchant2):
                        merchant_graph[merchant1][merchant2]['weight'] += 1
                        merchant_graph[merchant1][merchant2]['shared_cards'].append(card)
                    else:
                        merchant_graph.add_edge(merchant1, merchant2, weight=1, shared_cards=[card])
        
        projections['merchant'] = merchant_graph
        
        return projections
    
    def detect_communities_louvain(self, graph: nx.Graph, resolution: float = 1.0) -> Dict[Any, int]:
        """Detect communities using Louvain algorithm"""
        try:
            # Ensure graph has weights
            for u, v, d in graph.edges(data=True):
                if 'weight' not in d:
                    d['weight'] = 1.0
            
            # Apply Louvain algorithm
            communities = community_louvain.best_partition(graph, resolution=resolution)
            
            return communities
        except Exception as e:
            print(f"Error in Louvain detection: {e}")
            return {}
    
    def detect_communities_embedding_clustering(self, embeddings: torch.Tensor, 
                                              node_ids: List[str], method: str = 'kmeans',
                                              n_clusters: int = None) -> Dict[str, int]:
        """Detect communities using clustering on node embeddings"""
        embeddings_np = embeddings.detach().cpu().numpy()
        
        if method == 'kmeans':
            if n_clusters is None:
                n_clusters = min(20, len(node_ids) // 10)  # Heuristic
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(embeddings_np)
        
        # Create community dictionary
        communities = {node_id: int(label) for node_id, label in zip(node_ids, cluster_labels)}
        
        return communities
    
    def analyze_communities(self, communities: Dict[Any, int], graph: nx.Graph,
                          entity_type: str = 'card') -> Dict[int, Dict]:
        """Analyze detected communities for fraud patterns"""
        community_analysis = defaultdict(lambda: {
            'nodes': [],
            'size': 0,
            'fraud_score': 0.0,
            'total_transactions': 0,
            'fraud_transactions': 0,
            'avg_transaction_amount': 0.0,
            'density': 0.0,
            'is_suspicious': False
        })
        
        # Group nodes by community
        for node, community_id in communities.items():
            community_analysis[community_id]['nodes'].append(node)
        
        # Analyze each community
        for community_id, info in community_analysis.items():
            nodes = info['nodes']
            info['size'] = len(nodes)
            
            # Calculate subgraph density
            if len(nodes) > 1:
                subgraph = graph.subgraph(nodes)
                possible_edges = len(nodes) * (len(nodes) - 1) / 2
                actual_edges = subgraph.number_of_edges()
                info['density'] = actual_edges / possible_edges if possible_edges > 0 else 0
            
            # Analyze transaction patterns
            if entity_type == 'card':
                self._analyze_card_community(info, nodes)
            elif entity_type == 'merchant':
                self._analyze_merchant_community(info, nodes)
            
            # Determine if community is suspicious
            info['is_suspicious'] = self._is_community_suspicious(info)
        
        return dict(community_analysis)
    
    def _analyze_card_community(self, info: Dict, card_nodes: List[str]) -> None:
        """Analyze card community for fraud patterns"""
        # Extract card IDs from node names
        card_ids = [node.replace('card_', '') for node in card_nodes]
        
        # Get transactions for these cards
        community_transactions = self.data['transactions'][
            self.data['transactions']['card_id'].isin(card_ids)
        ]
        
        if len(community_transactions) > 0:
            info['total_transactions'] = len(community_transactions)
            info['fraud_transactions'] = community_transactions['is_fraud'].sum()
            info['fraud_score'] = info['fraud_transactions'] / info['total_transactions']
            info['avg_transaction_amount'] = community_transactions['amount'].mean()
            
            # Additional patterns
            info['unique_merchants'] = community_transactions['merchant_id'].nunique()
            info['unique_devices'] = community_transactions['device_id'].nunique()
            info['unique_ips'] = community_transactions['ip_address'].nunique()
            
            # Time analysis
            community_transactions['timestamp'] = pd.to_datetime(community_transactions['timestamp'])
            info['time_span_hours'] = (
                community_transactions['timestamp'].max() - 
                community_transactions['timestamp'].min()
            ).total_seconds() / 3600
    
    def _analyze_merchant_community(self, info: Dict, merchant_nodes: List[str]) -> None:
        """Analyze merchant community for fraud patterns"""
        merchant_ids = [node.replace('merchant_', '') for node in merchant_nodes]
        
        community_transactions = self.data['transactions'][
            self.data['transactions']['merchant_id'].isin(merchant_ids)
        ]
        
        if len(community_transactions) > 0:
            info['total_transactions'] = len(community_transactions)
            info['fraud_transactions'] = community_transactions['is_fraud'].sum()
            info['fraud_score'] = info['fraud_transactions'] / info['total_transactions']
            info['avg_transaction_amount'] = community_transactions['amount'].mean()
            
            info['unique_cards'] = community_transactions['card_id'].nunique()
            info['unique_devices'] = community_transactions['device_id'].nunique()
            info['unique_ips'] = community_transactions['ip_address'].nunique()
    
    def _is_community_suspicious(self, info: Dict) -> bool:
        """Determine if a community shows suspicious fraud patterns"""
        suspicious_criteria = [
            info['fraud_score'] > 0.1,  # High fraud rate
            info['size'] >= 3,  # Minimum size for ring
            info['density'] > 0.3,  # High connectivity
            info.get('unique_merchants', 1) / info['size'] < 0.5,  # Cards sharing merchants
            info.get('time_span_hours', float('inf')) < 24,  # Activity in short time window
        ]
        
        return sum(suspicious_criteria) >= 3  # At least 3 criteria met
    
    def identify_fraud_rings(self, min_ring_size: int = 3) -> List[Dict]:
        """Identify potential fraud rings from communities"""
        fraud_rings = []
        
        # Create projection graphs
        projections = self.create_projection_graphs()
        
        # Detect communities in each projection
        for entity_type, graph in projections.items():
            if graph.number_of_nodes() == 0:
                continue
                
            print(f"Detecting communities in {entity_type} projection...")
            
            # Apply Louvain algorithm with different resolutions
            for resolution in [0.5, 1.0, 1.5]:
                communities = self.detect_communities_louvain(graph, resolution=resolution)
                
                if not communities:
                    continue
                
                # Analyze communities
                community_analysis = self.analyze_communities(communities, graph, entity_type)
                
                # Extract fraud rings
                for community_id, info in community_analysis.items():
                    if (info['is_suspicious'] and 
                        info['size'] >= min_ring_size and
                        info['fraud_score'] > 0.05):
                        
                        ring = {
                            'ring_id': f"{entity_type}_ring_{len(fraud_rings)}",
                            'detection_method': f'louvain_resolution_{resolution}',
                            'entity_type': entity_type,
                            'nodes': info['nodes'],
                            'size': info['size'],
                            'fraud_score': info['fraud_score'],
                            'total_transactions': info['total_transactions'],
                            'fraud_transactions': info['fraud_transactions'],
                            'density': info['density'],
                            'avg_transaction_amount': info['avg_transaction_amount'],
                            'community_id': community_id
                        }
                        
                        fraud_rings.append(ring)
        
        # Remove duplicate rings (similar node sets)
        fraud_rings = self._deduplicate_rings(fraud_rings)
        
        self.fraud_rings = fraud_rings
        return fraud_rings
    
    def _deduplicate_rings(self, rings: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """Remove duplicate fraud rings based on node overlap"""
        unique_rings = []
        
        for ring in rings:
            is_duplicate = False
            ring_nodes = set(ring['nodes'])
            
            for existing_ring in unique_rings:
                existing_nodes = set(existing_ring['nodes'])
                
                # Calculate Jaccard similarity
                intersection = len(ring_nodes & existing_nodes)
                union = len(ring_nodes | existing_nodes)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > similarity_threshold:
                    is_duplicate = True
                    # Keep the ring with higher fraud score
                    if ring['fraud_score'] > existing_ring['fraud_score']:
                        unique_rings.remove(existing_ring)
                        unique_rings.append(ring)
                    break
            
            if not is_duplicate:
                unique_rings.append(ring)
        
        return unique_rings
    
    def evaluate_ring_detection(self, detected_rings: List[Dict], 
                               true_rings: List[Dict]) -> Dict[str, float]:
        """Evaluate fraud ring detection performance"""
        if not true_rings or not detected_rings:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Convert rings to node sets for comparison
        true_ring_sets = []
        for ring in true_rings:
            ring_nodes = set()
            for entity_type in ['cards', 'merchants', 'devices', 'ips']:
                if entity_type in ring:
                    for entity in ring[entity_type]:
                        ring_nodes.add(f"{entity_type[:-1]}_{entity}")
            true_ring_sets.append(ring_nodes)
        
        detected_ring_sets = [set(ring['nodes']) for ring in detected_rings]
        
        # Calculate precision and recall
        true_positives = 0
        for detected_set in detected_ring_sets:
            for true_set in true_ring_sets:
                # Consider a match if Jaccard similarity > 0.5
                intersection = len(detected_set & true_set)
                union = len(detected_set | true_set)
                if union > 0 and intersection / union > 0.5:
                    true_positives += 1
                    break
        
        precision = true_positives / len(detected_rings) if detected_rings else 0
        recall = true_positives / len(true_rings) if true_rings else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detected_rings': len(detected_rings),
            'true_rings': len(true_rings),
            'true_positives': true_positives
        }
    
    def save_results(self, filepath: str = '/workspace/fraud_detection_graph/data/detected_fraud_rings.json'):
        """Save detected fraud rings"""
        results = {
            'fraud_rings': self.fraud_rings,
            'detection_timestamp': pd.Timestamp.now().isoformat(),
            'total_rings_detected': len(self.fraud_rings),
            'summary_stats': self._compute_summary_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Saved {len(self.fraud_rings)} detected fraud rings to {filepath}")
    
    def _compute_summary_stats(self) -> Dict:
        """Compute summary statistics of detected rings"""
        if not self.fraud_rings:
            return {}
        
        sizes = [ring['size'] for ring in self.fraud_rings]
        fraud_scores = [ring['fraud_score'] for ring in self.fraud_rings]
        
        return {
            'avg_ring_size': np.mean(sizes),
            'max_ring_size': np.max(sizes),
            'min_ring_size': np.min(sizes),
            'avg_fraud_score': np.mean(fraud_scores),
            'max_fraud_score': np.max(fraud_scores),
            'total_fraudulent_transactions': sum(ring['fraud_transactions'] for ring in self.fraud_rings),
            'total_transactions_in_rings': sum(ring['total_transactions'] for ring in self.fraud_rings)
        }

if __name__ == "__main__":
    import pickle
    
    # Initialize detector
    detector = FraudRingDetector()
    
    # Load data
    print("Loading graph data...")
    detector.load_graph_data()
    
    # Detect fraud rings
    print("Detecting fraud rings...")
    detected_rings = detector.identify_fraud_rings(min_ring_size=3)
    
    print(f"Detected {len(detected_rings)} potential fraud rings")
    
    # Evaluate against ground truth
    true_rings = detector.data['fraud_rings']
    evaluation = detector.evaluate_ring_detection(detected_rings, true_rings)
    
    print(f"Evaluation Results:")
    print(f"Precision: {evaluation['precision']:.3f}")
    print(f"Recall: {evaluation['recall']:.3f}")
    print(f"F1-Score: {evaluation['f1']:.3f}")
    
    # Save results
    detector.save_results()
    
    # Print some example rings
    print(f"\nTop 5 detected fraud rings:")
    sorted_rings = sorted(detected_rings, key=lambda x: x['fraud_score'], reverse=True)[:5]
    for i, ring in enumerate(sorted_rings):
        print(f"{i+1}. {ring['ring_id']}: {ring['size']} nodes, fraud score: {ring['fraud_score']:.3f}")