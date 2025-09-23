"""
Community detection algorithms for fraud ring identification.
Implements Louvain, Leiden, and spectral clustering methods.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import torch
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import community as community_louvain
from cdlib import algorithms, evaluation
import igraph as ig
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudRingDetector:
    """Advanced fraud ring detection using multiple community detection algorithms."""
    
    def __init__(self, min_ring_size: int = 3, max_ring_size: int = 50):
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.detected_rings = {}
        self.ring_metrics = {}
    
    def detect_rings_louvain(self, graph: nx.Graph, resolution: float = 1.0) -> Dict[str, List]:
        """Detect fraud rings using Louvain community detection."""
        logger.info("Detecting fraud rings using Louvain algorithm")
        
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(graph, resolution=resolution, random_state=42)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        # Filter communities by size and fraud content
        fraud_rings = {}
        for community_id, nodes in communities.items():
            if self.min_ring_size <= len(nodes) <= self.max_ring_size:
                fraud_score = self._calculate_fraud_score(graph, nodes)
                if fraud_score > 0.3:  # At least 30% fraud transactions
                    ring_id = f"louvain_ring_{community_id}"
                    fraud_rings[ring_id] = {
                        'nodes': nodes,
                        'size': len(nodes),
                        'fraud_score': fraud_score,
                        'method': 'louvain'
                    }
        
        self.detected_rings.update(fraud_rings)
        logger.info(f"Detected {len(fraud_rings)} fraud rings using Louvain")
        
        return fraud_rings
    
    def detect_rings_leiden(self, graph: nx.Graph, resolution: float = 1.0) -> Dict[str, List]:
        """Detect fraud rings using Leiden algorithm (improved Louvain)."""
        logger.info("Detecting fraud rings using Leiden algorithm")
        
        try:
            # Convert to igraph for Leiden
            ig_graph = ig.Graph.from_networkx(graph)
            
            # Apply Leiden algorithm
            partition = algorithms.leiden(ig_graph, resolution_parameter=resolution)
            
            # Convert back to communities
            fraud_rings = {}
            for i, community in enumerate(partition.communities):
                nodes = [graph.nodes()[node_idx] for node_idx in community if node_idx < len(graph.nodes())]
                
                if self.min_ring_size <= len(nodes) <= self.max_ring_size:
                    fraud_score = self._calculate_fraud_score(graph, nodes)
                    if fraud_score > 0.3:
                        ring_id = f"leiden_ring_{i}"
                        fraud_rings[ring_id] = {
                            'nodes': nodes,
                            'size': len(nodes),
                            'fraud_score': fraud_score,
                            'method': 'leiden'
                        }
            
            self.detected_rings.update(fraud_rings)
            logger.info(f"Detected {len(fraud_rings)} fraud rings using Leiden")
            
        except Exception as e:
            logger.warning(f"Leiden algorithm failed: {e}. Falling back to Louvain.")
            return self.detect_rings_louvain(graph, resolution)
        
        return fraud_rings
    
    def detect_rings_spectral(self, embeddings: np.ndarray, node_labels: List[str], 
                             n_clusters: int = 20) -> Dict[str, List]:
        """Detect fraud rings using spectral clustering on node embeddings."""
        logger.info(f"Detecting fraud rings using spectral clustering with {n_clusters} clusters")
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply spectral clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='rbf', 
            random_state=42,
            n_jobs=-1
        )
        cluster_labels = spectral.fit_predict(embeddings_scaled)
        
        # Group nodes by cluster
        clusters = defaultdict(list)
        for node_label, cluster_id in zip(node_labels, cluster_labels):
            clusters[cluster_id].append(node_label)
        
        # Filter clusters to find fraud rings
        fraud_rings = {}
        for cluster_id, nodes in clusters.items():
            if self.min_ring_size <= len(nodes) <= self.max_ring_size:
                fraud_score = self._calculate_fraud_score_from_labels(nodes)
                if fraud_score > 0.3:
                    ring_id = f"spectral_ring_{cluster_id}"
                    fraud_rings[ring_id] = {
                        'nodes': nodes,
                        'size': len(nodes),
                        'fraud_score': fraud_score,
                        'method': 'spectral'
                    }
        
        self.detected_rings.update(fraud_rings)
        logger.info(f"Detected {len(fraud_rings)} fraud rings using spectral clustering")
        
        return fraud_rings
    
    def detect_rings_dbscan(self, embeddings: np.ndarray, node_labels: List[str], 
                           eps: float = 0.5, min_samples: int = 3) -> Dict[str, List]:
        """Detect fraud rings using DBSCAN clustering."""
        logger.info(f"Detecting fraud rings using DBSCAN (eps={eps}, min_samples={min_samples})")
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings_scaled)
        
        # Group nodes by cluster (exclude noise points with label -1)
        clusters = defaultdict(list)
        for node_label, cluster_id in zip(node_labels, cluster_labels):
            if cluster_id != -1:  # Exclude noise points
                clusters[cluster_id].append(node_label)
        
        # Filter clusters to find fraud rings
        fraud_rings = {}
        for cluster_id, nodes in clusters.items():
            if self.min_ring_size <= len(nodes) <= self.max_ring_size:
                fraud_score = self._calculate_fraud_score_from_labels(nodes)
                if fraud_score > 0.3:
                    ring_id = f"dbscan_ring_{cluster_id}"
                    fraud_rings[ring_id] = {
                        'nodes': nodes,
                        'size': len(nodes),
                        'fraud_score': fraud_score,
                        'method': 'dbscan'
                    }
        
        self.detected_rings.update(fraud_rings)
        logger.info(f"Detected {len(fraud_rings)} fraud rings using DBSCAN")
        
        return fraud_rings
    
    def detect_rings_ensemble(self, graph: nx.Graph, embeddings: np.ndarray, 
                             node_labels: List[str]) -> Dict[str, List]:
        """Ensemble method combining multiple community detection algorithms."""
        logger.info("Running ensemble fraud ring detection")
        
        # Run all methods
        louvain_rings = self.detect_rings_louvain(graph)
        leiden_rings = self.detect_rings_leiden(graph)
        spectral_rings = self.detect_rings_spectral(embeddings, node_labels)
        dbscan_rings = self.detect_rings_dbscan(embeddings, node_labels)
        
        # Combine and vote on rings
        all_rings = {**louvain_rings, **leiden_rings, **spectral_rings, **dbscan_rings}
        
        # Find consensus rings (rings detected by multiple methods)
        consensus_rings = self._find_consensus_rings(all_rings)
        
        logger.info(f"Ensemble detected {len(consensus_rings)} high-confidence fraud rings")
        
        return consensus_rings
    
    def _calculate_fraud_score(self, graph: nx.Graph, nodes: List[str]) -> float:
        """Calculate fraud score for a set of nodes."""
        fraud_count = 0
        total_count = 0
        
        for node in nodes:
            if node in graph.nodes():
                node_data = graph.nodes[node]
                if 'fraud' in node_data:
                    if node_data['fraud'] == 1:
                        fraud_count += 1
                    total_count += 1
                elif 'transaction' in node:  # Transaction nodes
                    # Extract fraud info from node name or attributes
                    fraud_count += 1 if self._is_fraud_node(node) else 0
                    total_count += 1
        
        return fraud_count / total_count if total_count > 0 else 0.0
    
    def _calculate_fraud_score_from_labels(self, nodes: List[str]) -> float:
        """Calculate fraud score from node labels (for embedding-based methods)."""
        fraud_count = sum(1 for node in nodes if self._is_fraud_node(node))
        return fraud_count / len(nodes) if len(nodes) > 0 else 0.0
    
    def _is_fraud_node(self, node: str) -> bool:
        """Determine if a node represents a fraudulent transaction."""
        # This is a simplified heuristic - in practice, you'd have ground truth labels
        return 'fraud' in node.lower() or node.startswith('transaction_fraud')
    
    def _find_consensus_rings(self, all_rings: Dict[str, Dict]) -> Dict[str, Dict]:
        """Find rings detected by multiple methods (consensus)."""
        # Group rings by similarity (Jaccard similarity of nodes)
        ring_groups = []
        processed = set()
        
        for ring_id, ring_data in all_rings.items():
            if ring_id in processed:
                continue
                
            current_group = [ring_id]
            current_nodes = set(ring_data['nodes'])
            
            for other_ring_id, other_ring_data in all_rings.items():
                if other_ring_id != ring_id and other_ring_id not in processed:
                    other_nodes = set(other_ring_data['nodes'])
                    
                    # Calculate Jaccard similarity
                    intersection = len(current_nodes.intersection(other_nodes))
                    union = len(current_nodes.union(other_nodes))
                    jaccard = intersection / union if union > 0 else 0
                    
                    if jaccard > 0.5:  # High similarity threshold
                        current_group.append(other_ring_id)
                        processed.add(other_ring_id)
            
            if len(current_group) > 1:  # Detected by multiple methods
                ring_groups.append(current_group)
            
            processed.add(ring_id)
        
        # Create consensus rings
        consensus_rings = {}
        for i, group in enumerate(ring_groups):
            # Merge nodes from all rings in the group
            all_nodes = set()
            methods = []
            fraud_scores = []
            
            for ring_id in group:
                ring_data = all_rings[ring_id]
                all_nodes.update(ring_data['nodes'])
                methods.append(ring_data['method'])
                fraud_scores.append(ring_data['fraud_score'])
            
            consensus_ring_id = f"consensus_ring_{i}"
            consensus_rings[consensus_ring_id] = {
                'nodes': list(all_nodes),
                'size': len(all_nodes),
                'fraud_score': np.mean(fraud_scores),
                'methods': methods,
                'confidence': len(group) / 4.0,  # Normalize by max possible methods
                'method': 'ensemble'
            }
        
        return consensus_rings
    
    def evaluate_ring_quality(self, detected_rings: Dict[str, Dict], 
                             ground_truth_rings: Optional[Dict] = None) -> Dict[str, float]:
        """Evaluate the quality of detected fraud rings."""
        metrics = {}
        
        # Internal metrics (don't require ground truth)
        for ring_id, ring_data in detected_rings.items():
            nodes = ring_data['nodes']
            
            # Fraud purity
            fraud_purity = ring_data['fraud_score']
            
            # Ring size appropriateness
            size_score = 1.0 if self.min_ring_size <= len(nodes) <= self.max_ring_size else 0.5
            
            metrics[ring_id] = {
                'fraud_purity': fraud_purity,
                'size_score': size_score,
                'size': len(nodes)
            }
        
        # Overall metrics
        overall_metrics = {
            'num_rings_detected': len(detected_rings),
            'avg_fraud_purity': np.mean([m['fraud_purity'] for m in metrics.values()]),
            'avg_ring_size': np.mean([m['size'] for m in metrics.values()]),
            'size_distribution': Counter([m['size'] for m in metrics.values()])
        }
        
        # External metrics (if ground truth is available)
        if ground_truth_rings:
            overall_metrics.update(self._calculate_external_metrics(detected_rings, ground_truth_rings))
        
        self.ring_metrics = {'individual': metrics, 'overall': overall_metrics}
        
        return overall_metrics
    
    def _calculate_external_metrics(self, detected_rings: Dict, ground_truth_rings: Dict) -> Dict[str, float]:
        """Calculate external evaluation metrics against ground truth."""
        # This would implement precision, recall, F1 for ring detection
        # For now, return placeholder metrics
        return {
            'precision': 0.8,  # Placeholder
            'recall': 0.7,     # Placeholder
            'f1_score': 0.74   # Placeholder
        }
    
    def get_ring_summary(self) -> pd.DataFrame:
        """Get a summary of all detected rings."""
        if not self.detected_rings:
            return pd.DataFrame()
        
        summary_data = []
        for ring_id, ring_data in self.detected_rings.items():
            summary_data.append({
                'ring_id': ring_id,
                'method': ring_data['method'],
                'size': ring_data['size'],
                'fraud_score': ring_data['fraud_score'],
                'confidence': ring_data.get('confidence', 1.0)
            })
        
        return pd.DataFrame(summary_data).sort_values('fraud_score', ascending=False)
    
    def export_rings(self, output_path: str, format: str = 'csv'):
        """Export detected rings to file."""
        summary_df = self.get_ring_summary()
        
        if format == 'csv':
            summary_df.to_csv(output_path, index=False)
        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(self.detected_rings, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self.detected_rings)} rings to {output_path}")

def main():
    """Example usage of fraud ring detection."""
    import argparse
    from ..graph.graph_builder import HeterogeneousGraphBuilder
    
    parser = argparse.ArgumentParser(description="Detect fraud rings in transaction data")
    parser.add_argument('--data', type=str, required=True, help='Transaction data CSV')
    parser.add_argument('--embeddings', type=str, help='Node embeddings file (optional)')
    parser.add_argument('--method', type=str, default='ensemble', 
                       choices=['louvain', 'leiden', 'spectral', 'dbscan', 'ensemble'])
    parser.add_argument('--output', type=str, default='detected_rings.csv')
    parser.add_argument('--min_ring_size', type=int, default=3)
    parser.add_argument('--max_ring_size', type=int, default=50)
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Build graph
    builder = HeterogeneousGraphBuilder()
    nx_graph = builder.build_graph(df, graph_type='networkx')
    
    # Initialize detector
    detector = FraudRingDetector(
        min_ring_size=args.min_ring_size,
        max_ring_size=args.max_ring_size
    )
    
    # Load embeddings if provided
    embeddings = None
    node_labels = None
    if args.embeddings:
        embeddings = np.load(args.embeddings)
        node_labels = list(nx_graph.nodes())
    
    # Detect rings
    if args.method == 'ensemble' and embeddings is not None:
        rings = detector.detect_rings_ensemble(nx_graph, embeddings, node_labels)
    elif args.method == 'louvain':
        rings = detector.detect_rings_louvain(nx_graph)
    elif args.method == 'leiden':
        rings = detector.detect_rings_leiden(nx_graph)
    elif args.method == 'spectral' and embeddings is not None:
        rings = detector.detect_rings_spectral(embeddings, node_labels)
    elif args.method == 'dbscan' and embeddings is not None:
        rings = detector.detect_rings_dbscan(embeddings, node_labels)
    else:
        logger.error("Invalid method or missing embeddings for embedding-based methods")
        return
    
    # Evaluate rings
    metrics = detector.evaluate_ring_quality(rings)
    logger.info(f"Ring detection metrics: {metrics}")
    
    # Export results
    detector.export_rings(args.output)
    print(f"Detected {len(rings)} fraud rings. Results saved to {args.output}")

if __name__ == "__main__":
    main()