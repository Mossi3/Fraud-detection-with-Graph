#!/usr/bin/env python3
"""
FraudGraphX - Comprehensive Graph-based Fraud Detection System
Main script that demonstrates the complete pipeline
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from graph.construction import HeterogeneousGraphBuilder, GraphConfig
from graph.community_detection import CommunityDetector, FraudRingAnalyzer
from graph.models import create_gnn_model
from graph.training import GNNTrainer, GraphDataProcessor, train_gnn_pipeline
from graph.visualization import GraphVisualizer
from graph.data_generator import FraudRingGenerator
from graph.evaluation import FraudDetectionEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudGraphXPipeline:
    """
    Complete pipeline for graph-based fraud detection
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.graph = None
        self.communities = {}
        self.fraud_rings = {}
        self.model = None
        self.evaluator = FraudDetectionEvaluator()
        self.results = {}
        
    def run_complete_pipeline(self, df: pd.DataFrame, 
                            generate_synthetic: bool = False) -> Dict[str, Any]:
        """
        Run the complete fraud detection pipeline
        
        Args:
            df: Transaction dataframe
            generate_synthetic: Whether to generate synthetic data for testing
            
        Returns:
            Dictionary with all results
        """
        
        logger.info("Starting FraudGraphX pipeline")
        
        # Step 1: Generate synthetic data if requested
        if generate_synthetic:
            logger.info("Generating synthetic fraud ring data")
            generator = FraudRingGenerator(seed=42)
            df, metadata = generator.generate_evaluation_dataset(
                num_rings=8,
                ring_types=['card_testing_ring', 'merchant_collusion', 'device_farming', 'ip_proxy_ring'],
                base_transactions=20000
            )
            self.results['synthetic_metadata'] = metadata
        
        # Step 2: Build heterogeneous graph
        logger.info("Building heterogeneous graph")
        self._build_graph(df)
        
        # Step 3: Detect communities
        logger.info("Detecting communities")
        self._detect_communities()
        
        # Step 4: Detect fraud rings
        logger.info("Detecting fraud rings")
        self._detect_fraud_rings(df)
        
        # Step 5: Train GNN model
        logger.info("Training GNN model")
        self._train_gnn_model(df)
        
        # Step 6: Evaluate performance
        logger.info("Evaluating performance")
        self._evaluate_performance(df)
        
        # Step 7: Generate visualizations
        logger.info("Generating visualizations")
        self._generate_visualizations(df)
        
        logger.info("Pipeline completed successfully")
        return self.results
    
    def _build_graph(self, df: pd.DataFrame):
        """Build heterogeneous graph from transaction data"""
        
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
        self.graph = builder
        
        # Store statistics
        stats = builder.get_graph_statistics()
        self.results['graph_statistics'] = stats
        
        logger.info(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    def _detect_communities(self):
        """Detect communities using multiple methods"""
        
        nx_graph = self.graph.to_networkx()
        detector = CommunityDetector(nx_graph)
        
        # Detect communities with different methods
        louvain_communities = detector.detect_louvain_communities(resolution=1.0)
        self.communities['louvain'] = louvain_communities
        
        try:
            leiden_communities = detector.detect_leiden_communities(resolution=1.0)
            self.communities['leiden'] = leiden_communities
        except Exception as e:
            logger.warning(f"Leiden algorithm not available: {e}")
        
        # Store community statistics
        self.results['community_statistics'] = detector.community_stats
        
        logger.info(f"Communities detected: {len(louvain_communities)} (Louvain)")
    
    def _detect_fraud_rings(self, df: pd.DataFrame):
        """Detect fraud rings from communities"""
        
        # Create fraud labels
        fraud_labels = {}
        if 'fraud' in df.columns and 'card_id' in df.columns:
            fraud_labels = dict(zip(df['card_id'], df['fraud']))
        
        nx_graph = self.graph.to_networkx()
        detector = CommunityDetector(nx_graph)
        
        # Detect fraud rings for each community method
        for method, communities in self.communities.items():
            detector.communities = {method: communities}
            rings = detector.detect_fraud_rings(
                fraud_labels=fraud_labels,
                min_community_size=3,
                fraud_threshold=0.3
            )
            self.fraud_rings[method] = rings.get(method, {})
        
        # Analyze ring patterns
        analyzer = FraudRingAnalyzer(self.communities['louvain'], nx_graph)
        ring_analysis = analyzer.analyze_ring_patterns(fraud_labels)
        self.results['ring_analysis'] = ring_analysis
        
        logger.info(f"Fraud rings detected: {sum(len(rings) for rings in self.fraud_rings.values())}")
    
    def _train_gnn_model(self, df: pd.DataFrame):
        """Train GNN model for fraud detection"""
        
        try:
            # Convert graph to PyTorch Geometric format
            nx_graph = self.graph.to_networkx()
            processor = GraphDataProcessor()
            graph_data = processor.networkx_to_pytorch_geometric(nx_graph)
            
            # Create labels
            labels = processor.create_labels_from_transactions(df, graph_data['node_mapping'])
            
            # Train different GNN models
            models = {}
            
            for model_type in ['graphsage', 'gat']:
                try:
                    logger.info(f"Training {model_type} model")
                    
                    # Create model
                    input_dim = graph_data['x'].shape[1]
                    model = create_gnn_model(model_type, input_dim, hidden_dim=32, dropout=0.1)
                    
                    # Create trainer
                    trainer = GNNTrainer(model)
                    
                    # Prepare data
                    train_data, val_data, test_data = trainer.prepare_data(graph_data, labels)
                    
                    # Train model
                    trainer.train(train_data, val_data, epochs=50, learning_rate=0.001, patience=10)
                    
                    # Evaluate
                    metrics = trainer.evaluate(test_data)
                    
                    models[model_type] = {
                        'trainer': trainer,
                        'metrics': metrics
                    }
                    
                    logger.info(f"{model_type} model trained - AUC: {metrics.get('roc_auc', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_type} model: {e}")
            
            self.model = models
            self.results['model_metrics'] = {k: v['metrics'] for k, v in models.items()}
            
        except Exception as e:
            logger.error(f"Error in GNN training: {e}")
    
    def _evaluate_performance(self, df: pd.DataFrame):
        """Evaluate overall performance"""
        
        # Create fraud labels
        fraud_labels = {}
        if 'fraud' in df.columns and 'card_id' in df.columns:
            fraud_labels = dict(zip(df['card_id'], df['fraud']))
        
        # Evaluate graph-based detection
        nx_graph = self.graph.to_networkx()
        metrics = self.evaluator.evaluate_graph_based_detection(
            nx_graph, self.communities['louvain'], fraud_labels
        )
        
        # Evaluate fraud ring detection
        if self.fraud_rings:
            ring_metrics = self.evaluator.evaluate_fraud_rings(
                self.fraud_rings, {}  # No ground truth rings for now
            )
            metrics.update(ring_metrics)
        
        self.results['evaluation_metrics'] = metrics
        
        # Generate report
        report = self.evaluator.generate_evaluation_report(metrics)
        self.results['evaluation_report'] = report
        
        logger.info("Performance evaluation completed")
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualizations"""
        
        try:
            # Create fraud labels
            fraud_labels = {}
            if 'fraud' in df.columns and 'card_id' in df.columns:
                fraud_labels = dict(zip(df['card_id'], df['fraud']))
            
            # Create visualizer
            nx_graph = self.graph.to_networkx()
            visualizer = GraphVisualizer(nx_graph, self.communities['louvain'])
            
            # Generate visualizations
            visualizations = {}
            
            # Network visualization
            try:
                network_fig = visualizer.create_interactive_network(fraud_labels)
                visualizations['network'] = "Network visualization generated"
            except Exception as e:
                logger.warning(f"Network visualization failed: {e}")
            
            # Community visualization
            try:
                community_fig = visualizer.create_community_visualization(fraud_labels)
                visualizations['communities'] = "Community visualization generated"
            except Exception as e:
                logger.warning(f"Community visualization failed: {e}")
            
            # Fraud ring analysis
            if self.fraud_rings:
                try:
                    ring_fig = visualizer.create_fraud_ring_analysis(self.fraud_rings)
                    visualizations['fraud_rings'] = "Fraud ring analysis generated"
                except Exception as e:
                    logger.warning(f"Fraud ring visualization failed: {e}")
            
            # Temporal analysis
            try:
                temporal_fig = visualizer.create_temporal_analysis(df)
                visualizations['temporal'] = "Temporal analysis generated"
            except Exception as e:
                logger.warning(f"Temporal visualization failed: {e}")
            
            self.results['visualizations'] = visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def save_results(self, output_dir: str = "results"):
        """Save all results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(output_dir, "fraud_detection_results.json")
        
        # Convert non-serializable objects
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save evaluation report
        if 'evaluation_report' in self.results:
            report_file = os.path.join(output_dir, "evaluation_report.txt")
            with open(report_file, 'w') as f:
                f.write(self.results['evaluation_report'])
            logger.info(f"Evaluation report saved to {report_file}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="FraudGraphX - Graph-based Fraud Detection")
    parser.add_argument("--input", "-i", help="Input CSV file with transaction data")
    parser.add_argument("--output", "-o", default="results", help="Output directory for results")
    parser.add_argument("--synthetic", "-s", action="store_true", help="Generate synthetic data for testing")
    parser.add_argument("--config", "-c", help="Configuration file (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Load data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
    elif args.synthetic:
        logger.info("Generating synthetic data")
        generator = FraudRingGenerator(seed=42)
        df, metadata = generator.generate_evaluation_dataset(
            num_rings=8,
            ring_types=['card_testing_ring', 'merchant_collusion', 'device_farming', 'ip_proxy_ring'],
            base_transactions=20000
        )
    else:
        logger.error("No input data provided. Use --input or --synthetic")
        return 1
    
    # Create pipeline
    pipeline = FraudGraphXPipeline(config)
    
    # Run pipeline
    try:
        results = pipeline.run_complete_pipeline(df, generate_synthetic=args.synthetic)
        
        # Save results
        pipeline.save_results(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("FRAUDGRAPHX PIPELINE SUMMARY")
        print("="*60)
        
        if 'graph_statistics' in results:
            stats = results['graph_statistics']
            print(f"Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        if 'community_statistics' in results:
            comm_stats = results['community_statistics']
            if 'louvain' in comm_stats:
                print(f"Communities: {comm_stats['louvain']['num_communities']} detected")
        
        if 'evaluation_metrics' in results:
            metrics = results['evaluation_metrics']
            print(f"Fraud Coverage: {metrics.get('fraud_coverage', 0):.2%}")
            print(f"Community Purity: {metrics.get('avg_community_purity', 0):.2%}")
        
        if 'model_metrics' in results:
            model_metrics = results['model_metrics']
            for model_type, metrics in model_metrics.items():
                print(f"{model_type.upper()} AUC: {metrics.get('roc_auc', 0):.4f}")
        
        print(f"\nResults saved to: {args.output}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())