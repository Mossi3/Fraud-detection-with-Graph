"""
Community detection algorithms for fraud ring identification.
Implements Louvain, spectral clustering, and custom fraud-specific algorithms.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import community as community_louvain
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import igraph as ig
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class CommunityDetector:
    """
    Detects communities and fraud rings in transaction graphs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.communities = {}
        self.fraud_rings = []
        self.ring_scores = {}
        
    def detect_communities(self, graph: nx.Graph, 
                         method: str = "louvain") -> Dict[str, int]:
        """
        Detect communities using specified method.
        
        Args:
            graph: NetworkX graph
            method: Algorithm to use (louvain, spectral, label_propagation, etc.)
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        logger.info(f"Detecting communities using {method} algorithm")
        
        if method == "louvain":
            communities = self._louvain_communities(graph)
        elif method == "spectral":
            communities = self._spectral_communities(graph)
        elif method == "label_propagation":
            communities = self._label_propagation_communities(graph)
        elif method == "infomap":
            communities = self._infomap_communities(graph)
        elif method == "walktrap":
            communities = self._walktrap_communities(graph)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.communities = communities
        self._evaluate_communities(graph, communities)
        
        return communities
    
    def _louvain_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Louvain algorithm for community detection"""
        # Convert to undirected if needed
        if graph.is_directed():
            graph = graph.to_undirected()
        
        # Apply Louvain
        partition = community_louvain.best_partition(graph)
        
        return partition
    
    def _spectral_communities(self, graph: nx.Graph, 
                            n_clusters: Optional[int] = None) -> Dict[str, int]:
        """Spectral clustering for community detection"""
        # Create adjacency matrix
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        adj_matrix = nx.adjacency_matrix(graph).todense()
        
        # Estimate number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._estimate_n_clusters(adj_matrix)
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(adj_matrix)
        
        # Map back to node IDs
        communities = {node: int(labels[idx]) for node, idx in node_to_idx.items()}
        
        return communities
    
    def _label_propagation_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Label propagation algorithm"""
        communities_generator = nx.algorithms.community.label_propagation_communities(graph)
        
        communities = {}
        for idx, community in enumerate(communities_generator):
            for node in community:
                communities[node] = idx
                
        return communities
    
    def _infomap_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Infomap algorithm using igraph"""
        # Convert to igraph
        ig_graph = ig.Graph.from_networkx(graph)
        
        # Run Infomap
        communities_ig = ig_graph.community_infomap()
        
        # Map back to original node IDs
        communities = {}
        nodes = list(graph.nodes())
        for idx, membership in enumerate(communities_ig.membership):
            communities[nodes[idx]] = membership
            
        return communities
    
    def _walktrap_communities(self, graph: nx.Graph, steps: int = 4) -> Dict[str, int]:
        """Walktrap algorithm using igraph"""
        # Convert to igraph
        ig_graph = ig.Graph.from_networkx(graph)
        
        # Run Walktrap
        communities_ig = ig_graph.community_walktrap(steps=steps)
        clusters = communities_ig.as_clustering()
        
        # Map back to original node IDs
        communities = {}
        nodes = list(graph.nodes())
        for idx, membership in enumerate(clusters.membership):
            communities[nodes[idx]] = membership
            
        return communities
    
    def _estimate_n_clusters(self, adj_matrix: np.ndarray) -> int:
        """Estimate optimal number of clusters using eigenvalues"""
        # Compute eigenvalues of Laplacian
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1).A1)
        laplacian = degree_matrix - adj_matrix
        
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)
        
        # Find spectral gap
        gaps = np.diff(eigenvalues)
        n_clusters = np.argmax(gaps[:20]) + 1  # Look at first 20 gaps
        
        return max(2, min(n_clusters, int(np.sqrt(adj_matrix.shape[0]))))
    
    def _evaluate_communities(self, graph: nx.Graph, 
                            communities: Dict[str, int]) -> Dict[str, float]:
        """Evaluate quality of detected communities"""
        metrics = {}
        
        # Modularity
        partition_list = defaultdict(list)
        for node, comm in communities.items():
            partition_list[comm].append(node)
        
        modularity = nx.algorithms.community.modularity(
            graph, partition_list.values()
        )
        metrics['modularity'] = modularity
        
        # Coverage
        intra_edges = 0
        total_edges = graph.number_of_edges()
        
        for u, v in graph.edges():
            if communities.get(u) == communities.get(v):
                intra_edges += 1
                
        metrics['coverage'] = intra_edges / max(total_edges, 1)
        
        # Number of communities
        metrics['n_communities'] = len(set(communities.values()))
        
        # Average community size
        comm_sizes = Counter(communities.values())
        metrics['avg_community_size'] = np.mean(list(comm_sizes.values()))
        
        logger.info(f"Community detection metrics: {metrics}")
        
        return metrics
    
    def detect_fraud_rings(self, graph: nx.Graph,
                          fraud_labels: Dict[str, bool],
                          min_ring_size: int = 3,
                          fraud_threshold: float = 0.5) -> List[Set[str]]:
        """
        Detect fraud rings based on community structure and fraud labels.
        
        Args:
            graph: NetworkX graph
            fraud_labels: Dictionary mapping node IDs to fraud status
            min_ring_size: Minimum size for a fraud ring
            fraud_threshold: Minimum fraction of fraudulent nodes in a ring
            
        Returns:
            List of fraud rings (sets of node IDs)
        """
        logger.info("Detecting fraud rings")
        
        # First detect communities
        if not self.communities:
            self.detect_communities(graph)
        
        # Analyze each community for fraud patterns
        community_nodes = defaultdict(list)
        for node, comm_id in self.communities.items():
            community_nodes[comm_id].append(node)
        
        fraud_rings = []
        
        for comm_id, nodes in community_nodes.items():
            if len(nodes) < min_ring_size:
                continue
                
            # Calculate fraud rate in community
            fraud_count = sum(1 for node in nodes 
                            if fraud_labels.get(node, False))
            fraud_rate = fraud_count / len(nodes)
            
            if fraud_rate >= fraud_threshold:
                # This is a potential fraud ring
                ring = set(nodes)
                
                # Refine ring by removing non-suspicious nodes
                refined_ring = self._refine_fraud_ring(graph, ring, fraud_labels)
                
                if len(refined_ring) >= min_ring_size:
                    fraud_rings.append(refined_ring)
                    
                    # Calculate ring score
                    score = self._calculate_ring_score(graph, refined_ring, fraud_labels)
                    self.ring_scores[frozenset(refined_ring)] = score
        
        # Detect rings using alternative methods
        pattern_rings = self._detect_pattern_based_rings(graph, fraud_labels)
        fraud_rings.extend(pattern_rings)
        
        # Remove duplicate rings
        unique_rings = []
        seen = set()
        
        for ring in fraud_rings:
            ring_frozen = frozenset(ring)
            if ring_frozen not in seen:
                seen.add(ring_frozen)
                unique_rings.append(ring)
        
        self.fraud_rings = unique_rings
        logger.info(f"Detected {len(unique_rings)} fraud rings")
        
        return unique_rings
    
    def _refine_fraud_ring(self, graph: nx.Graph,
                          ring: Set[str],
                          fraud_labels: Dict[str, bool]) -> Set[str]:
        """Refine fraud ring by removing peripheral non-fraudulent nodes"""
        refined_ring = set()
        
        # Create subgraph
        subgraph = graph.subgraph(ring)
        
        # Keep fraudulent nodes and highly connected nodes
        for node in ring:
            if fraud_labels.get(node, False):
                refined_ring.add(node)
            else:
                # Check connectivity to fraudulent nodes
                fraud_neighbors = sum(1 for neighbor in subgraph.neighbors(node)
                                    if fraud_labels.get(neighbor, False))
                
                if fraud_neighbors >= 2:  # Connected to multiple fraudulent nodes
                    refined_ring.add(node)
        
        return refined_ring
    
    def _calculate_ring_score(self, graph: nx.Graph,
                            ring: Set[str],
                            fraud_labels: Dict[str, bool]) -> float:
        """Calculate suspiciousness score for a fraud ring"""
        subgraph = graph.subgraph(ring)
        
        # Factors for scoring
        size_score = min(len(ring) / 10, 1.0)  # Larger rings are more suspicious
        
        fraud_rate = sum(1 for node in ring if fraud_labels.get(node, False)) / len(ring)
        
        density = nx.density(subgraph)
        
        # Check for suspicious patterns
        pattern_score = 0
        
        # Circular money flow
        try:
            cycles = list(nx.simple_cycles(subgraph))
            if cycles:
                pattern_score += 0.3
        except:
            pass
        
        # Star pattern (one central node)
        degrees = dict(subgraph.degree())
        max_degree = max(degrees.values())
        if max_degree > len(ring) * 0.7:
            pattern_score += 0.2
        
        # Calculate final score
        score = (size_score * 0.2 + 
                fraud_rate * 0.4 + 
                density * 0.2 + 
                pattern_score * 0.2)
        
        return score
    
    def _detect_pattern_based_rings(self, graph: nx.Graph,
                                  fraud_labels: Dict[str, bool]) -> List[Set[str]]:
        """Detect fraud rings based on specific patterns"""
        rings = []
        
        # Pattern 1: Circular transaction chains
        rings.extend(self._detect_circular_chains(graph, fraud_labels))
        
        # Pattern 2: Hub-and-spoke patterns
        rings.extend(self._detect_hub_patterns(graph, fraud_labels))
        
        # Pattern 3: Bipartite collusion patterns
        rings.extend(self._detect_bipartite_patterns(graph, fraud_labels))
        
        # Pattern 4: Temporal burst patterns
        rings.extend(self._detect_temporal_bursts(graph, fraud_labels))
        
        return rings
    
    def _detect_circular_chains(self, graph: nx.Graph,
                              fraud_labels: Dict[str, bool],
                              min_cycle_length: int = 3) -> List[Set[str]]:
        """Detect circular transaction chains"""
        rings = []
        
        try:
            # Find all simple cycles
            cycles = nx.simple_cycles(graph)
            
            for cycle in cycles:
                if len(cycle) >= min_cycle_length:
                    # Check if cycle contains fraudulent nodes
                    fraud_count = sum(1 for node in cycle 
                                    if fraud_labels.get(node, False))
                    
                    if fraud_count > 0:
                        rings.append(set(cycle))
                        
        except Exception as e:
            logger.warning(f"Error detecting circular chains: {e}")
        
        return rings
    
    def _detect_hub_patterns(self, graph: nx.Graph,
                           fraud_labels: Dict[str, bool],
                           min_spoke_fraud_rate: float = 0.3) -> List[Set[str]]:
        """Detect hub-and-spoke fraud patterns"""
        rings = []
        
        # Find high-degree nodes
        degrees = dict(graph.degree())
        avg_degree = np.mean(list(degrees.values()))
        std_degree = np.std(list(degrees.values()))
        
        hubs = [node for node, degree in degrees.items()
                if degree > avg_degree + 2 * std_degree]
        
        for hub in hubs:
            # Get hub's neighbors
            neighbors = set(graph.neighbors(hub))
            
            # Check fraud rate among neighbors
            fraud_neighbors = sum(1 for n in neighbors 
                                if fraud_labels.get(n, False))
            
            if fraud_neighbors / len(neighbors) >= min_spoke_fraud_rate:
                ring = neighbors.union({hub})
                rings.append(ring)
        
        return rings
    
    def _detect_bipartite_patterns(self, graph: nx.Graph,
                                 fraud_labels: Dict[str, bool]) -> List[Set[str]]:
        """Detect bipartite collusion patterns"""
        rings = []
        
        # Look for dense bipartite subgraphs between entity types
        node_types = nx.get_node_attributes(graph, 'entity_type')
        
        # Group nodes by type
        type_groups = defaultdict(list)
        for node, node_type in node_types.items():
            type_groups[node_type].append(node)
        
        # Check bipartite patterns between different types
        type_pairs = [
            ('card', 'merchant'),
            ('card', 'device'),
            ('device', 'ip_address')
        ]
        
        for type1, type2 in type_pairs:
            if type1 in type_groups and type2 in type_groups:
                # Find dense connections
                for nodes1 in self._find_dense_groups(graph, type_groups[type1]):
                    for nodes2 in self._find_dense_groups(graph, type_groups[type2]):
                        # Check if there's significant cross-connection
                        cross_edges = sum(1 for n1 in nodes1 for n2 in nodes2
                                        if graph.has_edge(n1, n2))
                        
                        potential_edges = len(nodes1) * len(nodes2)
                        if cross_edges / potential_edges > 0.5:
                            ring = set(nodes1).union(set(nodes2))
                            
                            # Check for fraud
                            fraud_count = sum(1 for n in ring 
                                            if fraud_labels.get(n, False))
                            
                            if fraud_count > 0:
                                rings.append(ring)
        
        return rings
    
    def _detect_temporal_bursts(self, graph: nx.Graph,
                              fraud_labels: Dict[str, bool]) -> List[Set[str]]:
        """Detect temporal burst patterns indicating coordinated fraud"""
        rings = []
        
        # Get temporal information
        timestamps = nx.get_node_attributes(graph, 'timestamp')
        
        # Group nodes by time windows
        from datetime import datetime, timedelta
        
        time_windows = defaultdict(list)
        window_size = timedelta(hours=1)
        
        for node, ts_str in timestamps.items():
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    window = ts.replace(minute=0, second=0, microsecond=0)
                    time_windows[window].append(node)
                except:
                    pass
        
        # Find bursts
        for window, nodes in time_windows.items():
            if len(nodes) > 10:  # Significant activity
                # Check if these nodes form a connected component
                subgraph = graph.subgraph(nodes)
                components = list(nx.connected_components(subgraph.to_undirected()))
                
                for component in components:
                    if len(component) > 5:
                        # Check fraud rate
                        fraud_count = sum(1 for n in component 
                                        if fraud_labels.get(n, False))
                        
                        if fraud_count / len(component) > 0.3:
                            rings.append(set(component))
        
        return rings
    
    def _find_dense_groups(self, graph: nx.Graph,
                         nodes: List[str],
                         min_size: int = 3) -> List[List[str]]:
        """Find densely connected groups within a node set"""
        if len(nodes) < min_size:
            return []
        
        # Create subgraph
        subgraph = graph.subgraph(nodes)
        
        # Use DBSCAN on adjacency matrix
        adj_matrix = nx.adjacency_matrix(subgraph).todense()
        
        clustering = DBSCAN(eps=0.3, min_samples=min_size-1, metric='precomputed')
        labels = clustering.fit_predict(1 - adj_matrix)  # Use dissimilarity
        
        # Extract groups
        groups = defaultdict(list)
        node_list = list(nodes)
        
        for idx, label in enumerate(labels):
            if label != -1:  # Not noise
                groups[label].append(node_list[idx])
        
        return list(groups.values())
    
    def analyze_ring_characteristics(self, graph: nx.Graph,
                                   ring: Set[str]) -> Dict[str, Any]:
        """Analyze characteristics of a fraud ring"""
        subgraph = graph.subgraph(ring)
        
        analysis = {
            'size': len(ring),
            'density': nx.density(subgraph),
            'clustering_coefficient': nx.average_clustering(subgraph.to_undirected()),
            'diameter': nx.diameter(subgraph.to_undirected()) if nx.is_connected(subgraph.to_undirected()) else -1,
        }
        
        # Node type distribution
        node_types = nx.get_node_attributes(subgraph, 'entity_type')
        type_dist = Counter(node_types.values())
        analysis['node_type_distribution'] = dict(type_dist)
        
        # Centrality analysis
        degrees = dict(subgraph.degree())
        analysis['avg_degree'] = np.mean(list(degrees.values()))
        analysis['max_degree'] = max(degrees.values())
        
        # Find central nodes
        pagerank = nx.pagerank(subgraph)
        central_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
        analysis['central_nodes'] = central_nodes
        
        # Pattern detection
        patterns = []
        
        # Check for cycles
        try:
            if list(nx.simple_cycles(subgraph)):
                patterns.append('circular_flow')
        except:
            pass
        
        # Check for star pattern
        if analysis['max_degree'] > len(ring) * 0.6:
            patterns.append('hub_and_spoke')
        
        # Check for bipartite structure
        if nx.is_bipartite(subgraph.to_undirected()):
            patterns.append('bipartite')
        
        analysis['patterns'] = patterns
        
        # Risk score
        if frozenset(ring) in self.ring_scores:
            analysis['risk_score'] = self.ring_scores[frozenset(ring)]
        else:
            analysis['risk_score'] = self._calculate_ring_score(graph, ring, {})
        
        return analysis
    
    def export_communities(self, output_path: str, format: str = 'json'):
        """Export detected communities to file"""
        import json
        
        data = {
            'communities': self.communities,
            'fraud_rings': [list(ring) for ring in self.fraud_rings],
            'ring_scores': {str(list(k)): v for k, v in self.ring_scores.items()}
        }
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported communities to {output_path}")