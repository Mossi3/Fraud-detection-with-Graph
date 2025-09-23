import json
import csv
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import random

class GraphFraudDetector:
    """Graph-based fraud detection using community detection and network analysis"""
    
    def __init__(self):
        self.transactions = []
        self.graph = {}
        self.node_features = {}
        self.communities = {}
        self.fraud_rings = {}
        self.load_data()
        self.build_graph()
        self.detect_communities()
    
    def load_data(self):
        """Load transaction data and fraud rings"""
        # Load transactions
        with open('/workspace/data/transactions.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['amount'] = float(row['amount'])
                row['is_fraud'] = int(row['is_fraud'])
                self.transactions.append(row)
        
        # Load fraud rings
        with open('/workspace/data/fraud_rings.json', 'r') as f:
            self.fraud_rings = json.load(f)
    
    def build_graph(self):
        """Build heterogeneous graph from transactions"""
        # Initialize graph structure
        self.graph = {
            'nodes': {},
            'edges': []
        }
        
        # Add nodes and edges
        for txn in self.transactions:
            txn_id = txn['transaction_id']
            
            # Add transaction node
            self.graph['nodes'][txn_id] = {
                'type': 'transaction',
                'amount': txn['amount'],
                'is_fraud': txn['is_fraud'],
                'fraud_ring': txn['fraud_ring'],
                'timestamp': txn['timestamp']
            }
            
            # Add entity nodes
            entities = {
                'card': txn['card_id'],
                'merchant': txn['merchant_id'],
                'device': txn['device_id'],
                'ip': txn['ip_address']
            }
            
            for entity_type, entity_id in entities.items():
                if entity_id not in self.graph['nodes']:
                    self.graph['nodes'][entity_id] = {
                        'type': entity_type,
                        'transactions': [],
                        'fraud_count': 0,
                        'total_amount': 0
                    }
                
                # Update entity statistics
                self.graph['nodes'][entity_id]['transactions'].append(txn_id)
                if txn['is_fraud']:
                    self.graph['nodes'][entity_id]['fraud_count'] += 1
                self.graph['nodes'][entity_id]['total_amount'] += txn['amount']
                
                # Add edge
                self.graph['edges'].append({
                    'source': txn_id,
                    'target': entity_id,
                    'type': f'txn_to_{entity_type}'
                })
        
        # Calculate node features
        self._calculate_node_features()
    
    def _calculate_node_features(self):
        """Calculate features for each node"""
        for node_id, node_data in self.graph['nodes'].items():
            if node_data['type'] == 'transaction':
                continue
            
            transactions = node_data['transactions']
            fraud_count = node_data['fraud_count']
            total_amount = node_data['total_amount']
            
            features = {
                'transaction_count': len(transactions),
                'fraud_count': fraud_count,
                'fraud_rate': fraud_count / len(transactions) if transactions else 0,
                'total_amount': total_amount,
                'avg_amount': total_amount / len(transactions) if transactions else 0,
                'degree': 0  # Will be calculated based on edges
            }
            
            # Calculate degree (number of connections)
            degree = 0
            for edge in self.graph['edges']:
                if edge['source'] == node_id or edge['target'] == node_id:
                    degree += 1
            features['degree'] = degree
            
            self.node_features[node_id] = features
    
    def detect_communities(self):
        """Detect communities using simplified community detection algorithm"""
        # Group nodes by their connections
        communities = defaultdict(set)
        visited = set()
        
        # Start with fraud rings as seed communities
        for ring_name, ring_data in self.fraud_rings.items():
            community_id = f"fraud_ring_{ring_name}"
            for card in ring_data['cards']:
                communities[community_id].add(card)
            for merchant in ring_data['merchants']:
                communities[community_id].add(merchant)
            for device in ring_data['devices']:
                communities[community_id].add(device)
            for ip in ring_data['ips']:
                communities[community_id].add(ip)
        
        # Add normal communities based on connectivity
        community_counter = 0
        for node_id in self.graph['nodes']:
            if node_id in visited:
                continue
            
            # Find connected components
            community = self._find_connected_component(node_id, visited)
            if len(community) > 1:  # Only consider communities with multiple nodes
                community_id = f"normal_community_{community_counter}"
                communities[community_id] = community
                community_counter += 1
        
        self.communities = dict(communities)
    
    def _find_connected_component(self, start_node: str, visited: Set[str]) -> Set[str]:
        """Find all nodes connected to start_node"""
        component = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            # Find neighbors
            for edge in self.graph['edges']:
                if edge['source'] == node and edge['target'] not in visited:
                    stack.append(edge['target'])
                elif edge['target'] == node and edge['source'] not in visited:
                    stack.append(edge['source'])
        
        return component
    
    def calculate_community_features(self) -> Dict:
        """Calculate features for each community"""
        community_features = {}
        
        for community_id, nodes in self.communities.items():
            features = {
                'size': len(nodes),
                'fraud_rate': 0,
                'total_transactions': 0,
                'total_amount': 0,
                'node_types': defaultdict(int),
                'suspicious_score': 0
            }
            
            fraud_count = 0
            total_transactions = 0
            total_amount = 0
            
            for node_id in nodes:
                if node_id in self.graph['nodes']:
                    node_data = self.graph['nodes'][node_id]
                    features['node_types'][node_data['type']] += 1
                    
                    if node_data['type'] == 'transaction':
                        total_transactions += 1
                        total_amount += node_data['amount']
                        if node_data['is_fraud']:
                            fraud_count += 1
                    else:
                        # Entity node
                        if node_id in self.node_features:
                            features['total_transactions'] += self.node_features[node_id]['transaction_count']
                            features['total_amount'] += self.node_features[node_id]['total_amount']
                            fraud_count += self.node_features[node_id]['fraud_count']
            
            features['fraud_rate'] = fraud_count / max(total_transactions, 1)
            features['total_transactions'] = total_transactions
            features['total_amount'] = total_amount
            
            # Calculate suspicious score
            suspicious_score = 0
            if features['fraud_rate'] > 0.5:
                suspicious_score += 0.5
            elif features['fraud_rate'] > 0.2:
                suspicious_score += 0.3
            
            if features['size'] > 20:  # Large communities might be suspicious
                suspicious_score += 0.2
            
            if 'fraud_ring' in community_id:
                suspicious_score += 0.3
            
            features['suspicious_score'] = min(suspicious_score, 1.0)
            community_features[community_id] = features
        
        return community_features
    
    def detect_fraud_rings_graph(self) -> Dict:
        """Detect fraud rings using graph analysis"""
        fraud_rings_detected = {}
        
        for community_id, features in self.calculate_community_features().items():
            if features['suspicious_score'] > 0.5:
                # Extract entities from this community
                community_nodes = self.communities[community_id]
                
                cards = []
                merchants = []
                devices = []
                ips = []
                
                for node_id in community_nodes:
                    if node_id in self.graph['nodes']:
                        node_type = self.graph['nodes'][node_id]['type']
                        if node_type == 'card':
                            cards.append(node_id)
                        elif node_type == 'merchant':
                            merchants.append(node_id)
                        elif node_type == 'device':
                            devices.append(node_id)
                        elif node_type == 'ip':
                            ips.append(node_id)
                
                fraud_rings_detected[community_id] = {
                    'cards': cards,
                    'merchants': merchants,
                    'devices': devices,
                    'ips': ips,
                    'suspicious_score': features['suspicious_score'],
                    'fraud_rate': features['fraud_rate'],
                    'size': features['size']
                }
        
        return fraud_rings_detected
    
    def predict_transaction_fraud(self, transaction: Dict) -> Dict:
        """Predict fraud for a transaction using graph features"""
        # Find communities that contain transaction entities
        relevant_communities = []
        
        entities = {
            'card': transaction['card_id'],
            'merchant': transaction['merchant_id'],
            'device': transaction['device_id'],
            'ip': transaction['ip_address']
        }
        
        for entity_type, entity_id in entities.items():
            for community_id, nodes in self.communities.items():
                if entity_id in nodes:
                    relevant_communities.append(community_id)
        
        # Calculate fraud score based on community features
        fraud_score = 0
        
        if relevant_communities:
            community_features = self.calculate_community_features()
            for community_id in relevant_communities:
                if community_id in community_features:
                    features = community_features[community_id]
                    fraud_score += features['suspicious_score'] * 0.3
                    fraud_score += features['fraud_rate'] * 0.4
        
        # Add individual entity features
        for entity_type, entity_id in entities.items():
            if entity_id in self.node_features:
                features = self.node_features[entity_id]
                fraud_score += features['fraud_rate'] * 0.1
        
        # Amount-based scoring
        amount = transaction['amount']
        if amount > 1000:
            fraud_score += 0.2
        elif amount > 500:
            fraud_score += 0.1
        
        fraud_score = min(fraud_score, 1.0)
        
        return {
            'fraud_probability': fraud_score,
            'is_fraud': fraud_score > 0.5,
            'relevant_communities': relevant_communities,
            'confidence': abs(fraud_score - 0.5) * 2
        }
    
    def get_graph_statistics(self) -> Dict:
        """Get overall graph statistics"""
        stats = {
            'total_nodes': len(self.graph['nodes']),
            'total_edges': len(self.graph['edges']),
            'node_types': defaultdict(int),
            'communities': len(self.communities),
            'fraud_rings_detected': 0
        }
        
        # Count node types
        for node_data in self.graph['nodes'].values():
            stats['node_types'][node_data['type']] += 1
        
        # Count detected fraud rings
        fraud_rings = self.detect_fraud_rings_graph()
        stats['fraud_rings_detected'] = len(fraud_rings)
        
        return stats
    
    def visualize_graph_structure(self) -> str:
        """Create a simple text-based visualization of the graph structure"""
        visualization = "Graph Structure Visualization\n"
        visualization += "=" * 50 + "\n\n"
        
        # Node type distribution
        stats = self.get_graph_statistics()
        visualization += "Node Distribution:\n"
        for node_type, count in stats['node_types'].items():
            visualization += f"  {node_type}: {count}\n"
        
        visualization += f"\nTotal Edges: {stats['total_edges']}\n"
        visualization += f"Communities Detected: {stats['communities']}\n"
        visualization += f"Fraud Rings Detected: {stats['fraud_rings_detected']}\n\n"
        
        # Top communities by suspicious score
        community_features = self.calculate_community_features()
        sorted_communities = sorted(
            community_features.items(),
            key=lambda x: x[1]['suspicious_score'],
            reverse=True
        )[:5]
        
        visualization += "Top Suspicious Communities:\n"
        for community_id, features in sorted_communities:
            visualization += f"  {community_id}: suspicious_score={features['suspicious_score']:.3f}, fraud_rate={features['fraud_rate']:.3f}\n"
        
        return visualization

# Initialize graph fraud detector
graph_detector = GraphFraudDetector()

if __name__ == "__main__":
    # Test the graph detector
    print("Graph Fraud Detection System")
    print("=" * 40)
    
    # Get graph statistics
    stats = graph_detector.get_graph_statistics()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Communities detected: {stats['communities']}")
    print(f"Fraud rings detected: {stats['fraud_rings_detected']}")
    
    # Test transaction prediction
    test_transaction = {
        'transaction_id': 'test_001',
        'card_id': 'card_001000',  # From fraud ring 1
        'merchant_id': 'merchant_0200',  # From fraud ring 1
        'device_id': 'device_005000',  # From fraud ring 1
        'ip_address': '192.168.1.1',  # From fraud ring 1
        'amount': 1500.0,
        'timestamp': '2024-01-15T10:30:00',
        'category': 'electronics',
        'country': 'US',
        'is_fraud': 1,
        'fraud_ring': 'ring_1'
    }
    
    prediction = graph_detector.predict_transaction_fraud(test_transaction)
    print(f"\nTest prediction: {prediction}")
    
    # Save graph data
    graph_data = {
        'nodes': graph_detector.graph['nodes'],
        'edges': graph_detector.graph['edges'],
        'communities': graph_detector.communities,
        'node_features': graph_detector.node_features,
        'statistics': stats
    }
    
    with open('/workspace/data/graph_data.json', 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)
    
    print("\nGraph data saved to /workspace/data/graph_data.json")
    
    # Print visualization
    print("\n" + graph_detector.visualize_graph_structure())