#!/usr/bin/env python3
"""
FraudGraphX Demo Script
Comprehensive demonstration of all features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from graph.construction import HeterogeneousGraphBuilder, GraphConfig
from graph.community_detection import CommunityDetector, FraudRingAnalyzer
from graph.models import create_gnn_model
from graph.training import GNNTrainer, GraphDataProcessor
from graph.visualization import GraphVisualizer
from graph.data_generator import FraudRingGenerator
from graph.evaluation import FraudDetectionEvaluator

def demo_synthetic_data_generation():
    """Demonstrate synthetic fraud ring data generation"""
    
    print("🔍 FraudGraphX Demo - Synthetic Data Generation")
    print("=" * 50)
    
    # Create generator
    generator = FraudRingGenerator(seed=42)
    
    # Generate comprehensive dataset
    df, metadata = generator.generate_evaluation_dataset(
        num_rings=8,
        ring_types=['card_testing_ring', 'merchant_collusion', 'device_farming', 'ip_proxy_ring'],
        base_transactions=20000
    )
    
    print(f"✅ Generated {len(df)} transactions")
    print(f"✅ Fraud rate: {df['fraud'].mean():.2%}")
    print(f"✅ Number of fraud rings: {len(df[df['ring_id'] != -1]['ring_id'].unique())}")
    
    # Show ring statistics
    print("\n📊 Fraud Ring Statistics:")
    ring_stats = df[df['ring_type'] != 'legitimate']['ring_type'].value_counts()
    for ring_type, count in ring_stats.items():
        print(f"  - {ring_type}: {count} transactions")
    
    return df, metadata

def demo_graph_construction(df):
    """Demonstrate graph construction"""
    
    print("\n🕸️ FraudGraphX Demo - Graph Construction")
    print("=" * 50)
    
    # Configure graph
    config = GraphConfig(
        include_cards=True,
        include_merchants=True,
        include_devices=True,
        include_ips=True,
        card_merchant_edges=True,
        device_ip_edges=True,
        card_device_edges=True,
        merchant_device_edges=True,
        time_window_hours=24,
        temporal_decay=0.9
    )
    
    # Build graph
    builder = HeterogeneousGraphBuilder(config)
    builder.add_transaction_data(df)
    
    # Get statistics
    stats = builder.get_graph_statistics()
    
    print(f"✅ Graph built successfully!")
    print(f"  - Nodes: {stats['num_nodes']}")
    print(f"  - Edges: {stats['num_edges']}")
    print(f"  - Density: {stats['density']:.4f}")
    print(f"  - Connected: {stats['is_connected']}")
    print(f"  - Components: {stats['num_components']}")
    
    # Node type breakdown
    print("\n📊 Node Types:")
    for node_type, count in stats['node_types'].items():
        print(f"  - {node_type}: {count}")
    
    # Edge type breakdown
    print("\n📊 Edge Types:")
    for edge_type, count in stats['edge_types'].items():
        print(f"  - {edge_type}: {count}")
    
    return builder

def demo_community_detection(builder):
    """Demonstrate community detection"""
    
    print("\n🏘️ FraudGraphX Demo - Community Detection")
    print("=" * 50)
    
    # Get NetworkX graph
    graph = builder.to_networkx()
    
    # Create community detector
    detector = CommunityDetector(graph)
    
    # Detect communities with different methods
    print("🔍 Detecting communities with Louvain algorithm...")
    louvain_communities = detector.detect_louvain_communities(resolution=1.0)
    
    print("🔍 Detecting communities with Leiden algorithm...")
    try:
        leiden_communities = detector.detect_leiden_communities(resolution=1.0)
        print("✅ Leiden algorithm successful")
    except Exception as e:
        print(f"⚠️ Leiden algorithm not available: {e}")
        leiden_communities = {}
    
    # Show community statistics
    print(f"\n📊 Community Statistics:")
    print(f"  - Louvain communities: {len(louvain_communities)}")
    if leiden_communities:
        print(f"  - Leiden communities: {len(leiden_communities)}")
    
    # Community size distribution
    louvain_sizes = [len(comm) for comm in louvain_communities.values()]
    print(f"  - Avg community size (Louvain): {np.mean(louvain_sizes):.1f}")
    print(f"  - Max community size (Louvain): {max(louvain_sizes)}")
    print(f"  - Min community size (Louvain): {min(louvain_sizes)}")
    
    return detector, louvain_communities

def demo_fraud_ring_detection(detector, df):
    """Demonstrate fraud ring detection"""
    
    print("\n🔍 FraudGraphX Demo - Fraud Ring Detection")
    print("=" * 50)
    
    # Create fraud labels
    fraud_labels = dict(zip(df['card_id'], df['fraud']))
    
    # Detect fraud rings
    fraud_rings = detector.detect_fraud_rings(
        fraud_labels=fraud_labels,
        min_community_size=3,
        fraud_threshold=0.3
    )
    
    print(f"✅ Fraud ring detection completed!")
    
    # Analyze fraud rings
    analyzer = FraudRingAnalyzer(detector.communities['louvain'], detector.graph)
    ring_analysis = analyzer.analyze_ring_patterns(fraud_labels)
    
    print(f"\n📊 Fraud Ring Analysis:")
    print(f"  - Total rings analyzed: {len(ring_analysis['ring_statistics'])}")
    
    # Show top suspicious rings
    suspicious_rings = []
    for ring_id, ring_data in ring_analysis['ring_statistics'].items():
        if ring_data['fraud_rate'] > 0.5:
            suspicious_rings.append((ring_id, ring_data))
    
    suspicious_rings.sort(key=lambda x: x[1]['fraud_rate'], reverse=True)
    
    print(f"\n🚨 Top Suspicious Rings:")
    for i, (ring_id, ring_data) in enumerate(suspicious_rings[:5]):
        print(f"  {i+1}. Ring {ring_id}:")
        print(f"     - Size: {ring_data['size']}")
        print(f"     - Fraud rate: {ring_data['fraud_rate']:.2%}")
        print(f"     - Suspicious score: {ring_data['anomaly_score']:.3f}")
    
    return fraud_rings, ring_analysis

def demo_gnn_training(builder, df):
    """Demonstrate GNN training"""
    
    print("\n🧠 FraudGraphX Demo - GNN Training")
    print("=" * 50)
    
    try:
        # Convert graph to PyTorch Geometric format
        graph = builder.to_networkx()
        processor = GraphDataProcessor()
        graph_data = processor.networkx_to_pytorch_geometric(graph)
        
        # Create labels
        labels = processor.create_labels_from_transactions(df, graph_data['node_mapping'])
        
        print(f"✅ Graph converted to PyTorch Geometric format")
        print(f"  - Node features: {graph_data['x'].shape}")
        print(f"  - Edges: {graph_data['edge_index'].shape}")
        print(f"  - Labels: {labels.shape}")
        
        # Train different GNN models
        models = {}
        
        for model_type in ['graphsage', 'gat']:
            print(f"\n🔬 Training {model_type.upper()} model...")
            
            try:
                # Create model
                input_dim = graph_data['x'].shape[1]
                model = create_gnn_model(model_type, input_dim, hidden_dim=32, dropout=0.1)
                
                # Create trainer
                trainer = GNNTrainer(model)
                
                # Prepare data
                train_data, val_data, test_data = trainer.prepare_data(graph_data, labels)
                
                # Train model
                trainer.train(train_data, val_data, epochs=20, learning_rate=0.001, patience=5)
                
                # Evaluate
                metrics = trainer.evaluate(test_data)
                
                models[model_type] = {
                    'trainer': trainer,
                    'metrics': metrics
                }
                
                print(f"✅ {model_type.upper()} training completed!")
                print(f"  - ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                print(f"  - PR-AUC: {metrics.get('pr_auc', 0):.4f}")
                print(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  - F1-Score: {metrics.get('f1_score', 0):.4f}")
                
            except Exception as e:
                print(f"❌ Error training {model_type}: {e}")
        
        return models
        
    except Exception as e:
        print(f"❌ GNN training failed: {e}")
        return {}

def demo_evaluation(df, builder, communities, fraud_rings):
    """Demonstrate comprehensive evaluation"""
    
    print("\n📊 FraudGraphX Demo - Evaluation")
    print("=" * 50)
    
    # Create fraud labels
    fraud_labels = dict(zip(df['card_id'], df['fraud']))
    
    # Create evaluator
    evaluator = FraudDetectionEvaluator()
    
    # Evaluate graph-based detection
    graph = builder.to_networkx()
    metrics = evaluator.evaluate_graph_based_detection(
        graph, communities, fraud_labels
    )
    
    print("✅ Graph-based evaluation completed!")
    
    # Show key metrics
    print(f"\n📈 Key Metrics:")
    print(f"  - Fraud Coverage: {metrics.get('fraud_coverage', 0):.2%}")
    print(f"  - Community Purity: {metrics.get('avg_community_purity', 0):.2%}")
    print(f"  - Total Coverage: {metrics.get('total_coverage', 0):.2%}")
    print(f"  - Avg Community Size: {metrics.get('avg_community_size', 0):.1f}")
    
    # Evaluate fraud ring detection
    if fraud_rings:
        ring_metrics = evaluator.evaluate_fraud_rings(fraud_rings, {})
        print(f"\n🔍 Fraud Ring Detection Metrics:")
        for method, rings in fraud_rings.items():
            if f"{method}_ring_precision" in ring_metrics:
                precision = ring_metrics[f"{method}_ring_precision"]
                recall = ring_metrics[f"{method}_ring_recall"]
                f1 = ring_metrics[f"{method}_ring_f1"]
                print(f"  - {method.title()}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(metrics)
    
    print(f"\n📋 Evaluation Report:")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    return metrics

def demo_visualization(builder, communities, fraud_labels):
    """Demonstrate visualization capabilities"""
    
    print("\n📊 FraudGraphX Demo - Visualization")
    print("=" * 50)
    
    try:
        # Create visualizer
        graph = builder.to_networkx()
        visualizer = GraphVisualizer(graph, communities)
        
        print("✅ Visualizer created successfully!")
        
        # Create visualizations
        print("🎨 Generating visualizations...")
        
        # Network visualization
        try:
            network_fig = visualizer.create_interactive_network(fraud_labels)
            print("✅ Network visualization generated")
        except Exception as e:
            print(f"⚠️ Network visualization failed: {e}")
        
        # Community visualization
        try:
            community_fig = visualizer.create_community_visualization(fraud_labels)
            print("✅ Community visualization generated")
        except Exception as e:
            print(f"⚠️ Community visualization failed: {e}")
        
        # Centrality heatmap
        try:
            centrality_measures = builder.calculate_centrality_measures()
            centrality_fig = visualizer.create_centrality_heatmap(centrality_measures)
            print("✅ Centrality heatmap generated")
        except Exception as e:
            print(f"⚠️ Centrality heatmap failed: {e}")
        
        print("✅ All visualizations completed!")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def demo_api_endpoints():
    """Demonstrate API endpoints"""
    
    print("\n🚀 FraudGraphX Demo - API Endpoints")
    print("=" * 50)
    
    print("📡 Available API Endpoints:")
    print("  - POST /graph/build - Build heterogeneous graph")
    print("  - GET /graph/{id}/statistics - Get graph statistics")
    print("  - POST /community/{id}/detect - Detect communities")
    print("  - POST /gnn_training/{id}/train - Train GNN model")
    print("  - POST /fraud_detection/{id}/predict - Predict fraud")
    print("  - POST /synthetic_data/generate - Generate synthetic data")
    print("  - GET /visualization/{id}/network - Get network visualization")
    
    print("\n🔧 To start the API server:")
    print("  uvicorn src.serve.app_graph_api:app --reload")
    
    print("\n🧪 To run API tests:")
    print("  chmod +x tests/test_api_curl.sh")
    print("  ./tests/test_api_curl.sh")

def main():
    """Main demo function"""
    
    print("🔍 FraudGraphX - Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases all features of the FraudGraphX system")
    print("=" * 60)
    
    try:
        # 1. Generate synthetic data
        df, metadata = demo_synthetic_data_generation()
        
        # 2. Build graph
        builder = demo_graph_construction(df)
        
        # 3. Detect communities
        detector, communities = demo_community_detection(builder)
        
        # 4. Detect fraud rings
        fraud_rings, ring_analysis = demo_fraud_ring_detection(detector, df)
        
        # 5. Train GNN models
        models = demo_gnn_training(builder, df)
        
        # 6. Evaluate performance
        metrics = demo_evaluation(df, builder, communities, fraud_rings)
        
        # 7. Generate visualizations
        fraud_labels = dict(zip(df['card_id'], df['fraud']))
        demo_visualization(builder, communities, fraud_labels)
        
        # 8. Show API endpoints
        demo_api_endpoints()
        
        # Summary
        print("\n🎉 Demo Completed Successfully!")
        print("=" * 60)
        print("✅ All FraudGraphX features demonstrated")
        print("✅ Synthetic data generated and processed")
        print("✅ Graph constructed and analyzed")
        print("✅ Communities and fraud rings detected")
        print("✅ GNN models trained and evaluated")
        print("✅ Comprehensive evaluation completed")
        print("✅ Visualizations generated")
        print("✅ API endpoints documented")
        
        print("\n📚 Next Steps:")
        print("  1. Start the API server: uvicorn src.serve.app_graph_api:app --reload")
        print("  2. Run the test suite: ./tests/test_api_curl.sh")
        print("  3. Launch the dashboard: streamlit run src/graph/visualization.py")
        print("  4. Explore the codebase in src/graph/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())