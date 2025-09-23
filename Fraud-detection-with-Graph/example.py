#!/usr/bin/env python3
"""
Example usage of the Fraud Detection Graph System
"""

import os
import sys
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph.graph_builder import GraphBuilder, Transaction
from features.feature_extractor import GraphFeatureExtractor
from models.community_detection import CommunityDetector
from visualization.graph_visualizer import FraudGraphVisualizer


def create_sample_transactions():
    """Create a small set of sample transactions for demonstration"""
    transactions = []
    base_time = datetime.now() - timedelta(hours=24)
    
    # Normal transactions
    for i in range(20):
        txn = Transaction(
            transaction_id=f"normal_{i}",
            card_id=f"card_{i % 5}",  # 5 different cards
            merchant_id=f"merchant_{i % 3}",  # 3 different merchants
            amount=random.uniform(10, 200),
            timestamp=base_time + timedelta(hours=i),
            device_id=f"device_{i % 4}",
            ip_address=f"192.168.1.{i % 10}",
            is_fraud=False
        )
        transactions.append(txn)
    
    # Create a fraud ring - multiple cards using same device
    fraud_device = "device_fraud"
    fraud_ip = "10.0.0.1"
    
    for i in range(5):
        for j in range(3):  # 3 transactions per card
            txn = Transaction(
                transaction_id=f"fraud_{i}_{j}",
                card_id=f"fraud_card_{i}",
                merchant_id="suspicious_merchant",
                amount=random.uniform(500, 1000),
                timestamp=base_time + timedelta(hours=20 + i, minutes=j*10),
                device_id=fraud_device,  # All using same device
                ip_address=fraud_ip,     # All from same IP
                is_fraud=True
            )
            transactions.append(txn)
    
    return transactions


def main():
    print("Fraud Detection Graph System - Example")
    print("=" * 50)
    
    # Create sample transactions
    print("\n1. Creating sample transactions...")
    transactions = create_sample_transactions()
    print(f"   Created {len(transactions)} transactions")
    print(f"   - Normal: {sum(1 for t in transactions if not t.is_fraud)}")
    print(f"   - Fraudulent: {sum(1 for t in transactions if t.is_fraud)}")
    
    # Build graph
    print("\n2. Building transaction graph...")
    graph_builder = GraphBuilder()
    fraud_graph = graph_builder.build_from_transactions(transactions)
    
    stats = graph_builder.get_graph_statistics()
    print(f"   Graph statistics:")
    print(f"   - Nodes: {stats['total_nodes']}")
    print(f"   - Edges: {stats['total_edges']}")
    print(f"   - Density: {stats['density']:.4f}")
    
    # Extract features
    print("\n3. Extracting features...")
    feature_extractor = GraphFeatureExtractor()
    
    # Get features for a sample transaction node
    sample_node = "txn_fraud_0_0"
    if sample_node in fraud_graph.graph:
        features = feature_extractor.extract_node_features(fraud_graph.graph, sample_node)
        print(f"   Feature vector length: {len(features)}")
        print(f"   Sample features for {sample_node}:")
        print(f"   - Degree: {features[0]}")
        print(f"   - PageRank: {features[3]:.6f}")
    
    # Detect communities
    print("\n4. Detecting communities...")
    detector = CommunityDetector()
    communities = detector.detect_communities(fraud_graph.graph, method="louvain")
    
    print(f"   Found {len(set(communities.values()))} communities")
    
    # Detect fraud rings
    print("\n5. Detecting fraud rings...")
    fraud_labels = {}
    for node_id, data in fraud_graph.graph.nodes(data=True):
        fraud_labels[node_id] = data.get('is_fraud', False)
    
    fraud_rings = detector.detect_fraud_rings(
        fraud_graph.graph,
        fraud_labels,
        min_ring_size=3,
        fraud_threshold=0.5
    )
    
    print(f"   Detected {len(fraud_rings)} fraud rings")
    
    for i, ring in enumerate(fraud_rings):
        print(f"   Ring {i+1}: {len(ring)} members")
        # Show some members
        sample_members = list(ring)[:5]
        for member in sample_members:
            node_type = fraud_graph.graph.nodes[member].get('entity_type', 'unknown')
            print(f"     - {member} ({node_type})")
        if len(ring) > 5:
            print(f"     ... and {len(ring) - 5} more")
    
    # Create visualization
    print("\n6. Creating visualization...")
    os.makedirs("visualizations", exist_ok=True)
    
    visualizer = FraudGraphVisualizer()
    
    # Visualize the first fraud ring if found
    if fraud_rings:
        output_path = "visualizations/example_fraud_ring.html"
        visualizer.visualize_fraud_ring(
            fraud_graph.graph,
            fraud_rings[0],
            fraud_labels,
            output_path,
            interactive=True
        )
        print(f"   Saved interactive visualization to: {output_path}")
    
    # Create overall network visualization
    output_path = "visualizations/example_network.html"
    visualizer.create_pyvis_network(
        fraud_graph.graph,
        fraud_rings,
        output_path
    )
    print(f"   Saved network visualization to: {output_path}")
    
    print("\n✓ Example completed successfully!")
    print("\nTo view visualizations, open the HTML files in visualizations/ directory")


if __name__ == "__main__":
    main()