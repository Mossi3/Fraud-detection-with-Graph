import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import DBSCAN
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class GraphFraudDetector:
    """Graph-based fraud detection system"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_features = {}
        self.node_labels = {}
        self.fraud_rings = []
        self.communities = {}
        
    def build_transaction_graph(self, df):
        """Build transaction graph from dataframe"""
        print("Building transaction graph...")
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes for cards, merchants, devices, IPs
        for idx, row in df.iterrows():
            card_id = row['Card_ID']
            merchant_id = row['Merchant_ID']
            device_id = row['Device_ID']
            ip_address = row['IP_Address']
            
            # Add nodes with features
            self.graph.add_node(card_id, node_type='card', 
                              features=self._extract_card_features(row))
            self.graph.add_node(merchant_id, node_type='merchant',
                              features=self._extract_merchant_features(row))
            self.graph.add_node(device_id, node_type='device',
                              features=self._extract_device_features(row))
            self.graph.add_node(ip_address, node_type='ip',
                              features=self._extract_ip_features(row))
            
            # Add edges between connected entities
            self.graph.add_edge(card_id, merchant_id, 
                              weight=row['Amount'], 
                              timestamp=row['Time'],
                              fraud_label=row['Class'])
            self.graph.add_edge(card_id, device_id, 
                              weight=1, 
                              timestamp=row['Time'],
                              fraud_label=row['Class'])
            self.graph.add_edge(device_id, ip_address, 
                              weight=1, 
                              timestamp=row['Time'],
                              fraud_label=row['Class'])
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _extract_card_features(self, row):
        """Extract features for card nodes"""
        return np.array([
            row['Amount'],
            row['V1'], row['V2'], row['V3'], row['V4'], row['V5'],
            row['Hour'], row['Day'],
            row.get('Country_encoded', 0),
            row.get('Device_encoded', 0)
        ])
    
    def _extract_merchant_features(self, row):
        """Extract features for merchant nodes"""
        return np.array([
            row['Amount'],
            row['V6'], row['V7'], row['V8'], row['V9'], row['V10'],
            row['Hour'], row['Day'],
            row.get('Merchant_encoded', 0),
            row.get('Country_encoded', 0)
        ])
    
    def _extract_device_features(self, row):
        """Extract features for device nodes"""
        return np.array([
            row['V11'], row['V12'], row['V13'], row['V14'], row['V15'],
            row['Hour'], row['Day'],
            row.get('Device_encoded', 0),
            row.get('IP_Country_encoded', 0),
            1  # Device indicator
        ])
    
    def _extract_ip_features(self, row):
        """Extract features for IP nodes"""
        return np.array([
            row['V16'], row['V17'], row['V18'], row['V19'], row['V20'],
            row['Hour'], row['Day'],
            row.get('IP_Country_encoded', 0),
            row.get('Country_encoded', 0),
            1  # IP indicator
        ])
    
    def detect_fraud_rings(self, min_ring_size=3):
        """Detect fraud rings using community detection"""
        print("Detecting fraud rings...")
        
        # Use Louvain community detection
        communities = community_louvain.best_partition(self.graph)
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node, community_id in communities.items():
            community_groups[community_id].append(node)
        
        # Identify potential fraud rings
        fraud_rings = []
        for community_id, nodes in community_groups.items():
            if len(nodes) >= min_ring_size:
                # Check if community has high fraud rate
                fraud_count = 0
                total_transactions = 0
                
                for node in nodes:
                    if self.graph.nodes[node].get('node_type') == 'card':
                        # Count fraud transactions for this card
                        for neighbor in self.graph.neighbors(node):
                            edge_data = self.graph[node][neighbor]
                            if 'fraud_label' in edge_data:
                                total_transactions += 1
                                if edge_data['fraud_label'] == 1:
                                    fraud_count += 1
                
                if total_transactions > 0 and fraud_count / total_transactions > 0.3:
                    fraud_rings.append({
                        'community_id': community_id,
                        'nodes': nodes,
                        'fraud_rate': fraud_count / total_transactions,
                        'size': len(nodes)
                    })
        
        self.fraud_rings = fraud_rings
        print(f"Detected {len(fraud_rings)} potential fraud rings")
        
        return fraud_rings
    
    def detect_suspicious_patterns(self):
        """Detect suspicious patterns in the graph"""
        suspicious_patterns = []
        
        # Pattern 1: Cards with many different merchants (card testing)
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('node_type') == 'card':
                merchants = [n for n in self.graph.neighbors(node) 
                           if self.graph.nodes[n].get('node_type') == 'merchant']
                if len(merchants) > 10:  # Threshold for suspicious activity
                    suspicious_patterns.append({
                        'pattern': 'card_testing',
                        'node': node,
                        'score': len(merchants),
                        'description': f'Card {node} used at {len(merchants)} different merchants'
                    })
        
        # Pattern 2: Devices with many different cards (device compromise)
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('node_type') == 'device':
                cards = [n for n in self.graph.neighbors(node) 
                        if self.graph.nodes[n].get('node_type') == 'card']
                if len(cards) > 5:  # Threshold for suspicious activity
                    suspicious_patterns.append({
                        'pattern': 'device_compromise',
                        'node': node,
                        'score': len(cards),
                        'description': f'Device {node} used by {len(cards)} different cards'
                    })
        
        # Pattern 3: IP addresses with many different devices (IP compromise)
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('node_type') == 'ip':
                devices = [n for n in self.graph.neighbors(node) 
                          if self.graph.nodes[n].get('node_type') == 'device']
                if len(devices) > 3:  # Threshold for suspicious activity
                    suspicious_patterns.append({
                        'pattern': 'ip_compromise',
                        'node': node,
                        'score': len(devices),
                        'description': f'IP {node} used by {len(devices)} different devices'
                    })
        
        return suspicious_patterns

class GraphSAGEFraudDetector(nn.Module):
    """GraphSAGE-based fraud detector"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2):
        super(GraphSAGEFraudDetector, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)

class GATFraudDetector(nn.Module):
    """Graph Attention Network fraud detector"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_heads=4, num_layers=2):
        super(GATFraudDetector, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.3))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.3))
        
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.3))
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)

class GCNFraudDetector(nn.Module):
    """Graph Convolutional Network fraud detector"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2):
        super(GCNFraudDetector, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)

class GraphModelTrainer:
    """Trainer for graph-based models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def prepare_graph_data(self, graph, node_features, node_labels):
        """Prepare graph data for PyTorch Geometric"""
        # Convert NetworkX graph to PyTorch Geometric format
        edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        
        # Prepare node features
        feature_matrix = []
        labels = []
        
        for node in graph.nodes():
            if node in node_features:
                feature_matrix.append(node_features[node])
                labels.append(node_labels.get(node, 0))
        
        x = torch.tensor(feature_matrix, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def train(self, data, epochs=100, lr=0.01):
        """Train the graph model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        data = data.to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index)
            loss = criterion(out.squeeze(), data.y)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def evaluate(self, data):
        """Evaluate the model"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data.x, data.edge_index)
            predictions = (predictions > 0.5).float()
            
            accuracy = (predictions.squeeze() == data.y).float().mean()
            return accuracy.item()

def visualize_fraud_rings(graph, fraud_rings, title="Fraud Ring Detection"):
    """Visualize detected fraud rings"""
    plt.figure(figsize=(15, 10))
    
    # Create subgraph with fraud ring nodes
    fraud_nodes = set()
    for ring in fraud_rings:
        fraud_nodes.update(ring['nodes'])
    
    subgraph = graph.subgraph(fraud_nodes)
    
    # Layout
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    # Color nodes by type
    node_colors = []
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get('node_type', 'unknown')
        if node_type == 'card':
            node_colors.append('red')
        elif node_type == 'merchant':
            node_colors.append('blue')
        elif node_type == 'device':
            node_colors.append('green')
        elif node_type == 'ip':
            node_colors.append('orange')
        else:
            node_colors.append('gray')
    
    # Draw the graph
    nx.draw(subgraph, pos, 
            node_color=node_colors,
            node_size=100,
            with_labels=False,
            edge_color='gray',
            alpha=0.7)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cards'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Merchants'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Devices'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='IPs')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('/workspace/visualization/fraud_rings.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap_matrix(graph, node_types=['card', 'merchant', 'device', 'ip']):
    """Create heatmap matrix showing connections between node types"""
    matrix = np.zeros((len(node_types), len(node_types)))
    
    for i, type1 in enumerate(node_types):
        for j, type2 in enumerate(node_types):
            count = 0
            for node1 in graph.nodes():
                if graph.nodes[node1].get('node_type') == type1:
                    for node2 in graph.neighbors(node1):
                        if graph.nodes[node2].get('node_type') == type2:
                            count += 1
            matrix[i][j] = count
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
                xticklabels=node_types, 
                yticklabels=node_types,
                annot=True, 
                fmt='d',
                cmap='YlOrRd')
    plt.title('Connection Matrix Between Node Types')
    plt.xlabel('Target Node Type')
    plt.ylabel('Source Node Type')
    plt.tight_layout()
    plt.savefig('/workspace/visualization/connection_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return matrix

if __name__ == "__main__":
    # Test the graph-based fraud detection
    from data_processor import FraudDataProcessor
    
    # Load and preprocess data
    processor = FraudDataProcessor()
    df = processor.load_data()
    df_processed = processor.preprocess_data(df)
    
    # Build graph
    graph_detector = GraphFraudDetector()
    graph = graph_detector.build_transaction_graph(df_processed)
    
    # Detect fraud rings
    fraud_rings = graph_detector.detect_fraud_rings()
    
    # Detect suspicious patterns
    suspicious_patterns = graph_detector.detect_suspicious_patterns()
    
    print(f"\nDetected {len(fraud_rings)} fraud rings:")
    for i, ring in enumerate(fraud_rings[:5]):  # Show first 5
        print(f"Ring {i+1}: {ring['size']} nodes, fraud rate: {ring['fraud_rate']:.2f}")
    
    print(f"\nDetected {len(suspicious_patterns)} suspicious patterns:")
    for pattern in suspicious_patterns[:5]:  # Show first 5
        print(f"- {pattern['description']}")
    
    # Visualize results
    visualize_fraud_rings(graph, fraud_rings)
    create_heatmap_matrix(graph)
    
    # Test GNN models
    print("\nTesting GNN models...")
    
    # Prepare node features and labels
    node_features = {}
    node_labels = {}
    
    for node in graph.nodes():
        if 'features' in graph.nodes[node]:
            node_features[node] = graph.nodes[node]['features']
            # Simple label: 1 if node is connected to fraud, 0 otherwise
            fraud_connected = False
            for neighbor in graph.neighbors(node):
                for edge_data in graph[node][neighbor].values():
                    if edge_data.get('fraud_label', 0) == 1:
                        fraud_connected = True
                        break
            node_labels[node] = 1 if fraud_connected else 0
    
    # Test different GNN architectures
    input_dim = len(list(node_features.values())[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'GraphSAGE': GraphSAGEFraudDetector(input_dim),
        'GAT': GATFraudDetector(input_dim),
        'GCN': GCNFraudDetector(input_dim)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trainer = GraphModelTrainer(model, device)
        
        # Prepare data
        data = trainer.prepare_graph_data(graph, node_features, node_labels)
        
        # Train model
        trainer.train(data, epochs=20)
        
        # Evaluate
        accuracy = trainer.evaluate(data)
        print(f"{name} Accuracy: {accuracy:.4f}")