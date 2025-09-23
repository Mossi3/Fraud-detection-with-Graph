#!/usr/bin/env python3
"""
Main entry point for the Fraud Detection Graph System.
Provides CLI interface for various operations.
"""

import click
import sys
import os
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


@click.group()
def cli():
    """Fraud Detection Graph System - Command Line Interface"""
    pass


@cli.command()
@click.option('--num-transactions', default=50000, help='Number of transactions to generate')
@click.option('--output', default='data/training_data.csv', help='Output file path')
def generate_data(num_transactions, output):
    """Generate synthetic fraud detection dataset"""
    from utils.data_generator import TransactionDataGenerator
    
    logger.info(f"Generating {num_transactions} transactions...")
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(num_transactions)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    generator.save_dataset(df, output)
    
    logger.info(f"Dataset saved to {output}")
    logger.info(f"Total transactions: {len(df)}")
    logger.info(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")


@cli.command()
@click.option('--data', default='data/training_data.csv', help='Training data path')
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--lr', default=0.001, help='Learning rate')
def train(data, epochs, lr):
    """Train fraud detection models"""
    import train as training_module
    
    config = {
        'epochs': epochs,
        'learning_rate': lr,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.5
    }
    
    trainer = training_module.FraudDetectionTrainer(config)
    trainer.prepare_data(data)
    trainer.initialize_models()
    trainer.train_all_models()


@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
def serve(host, port):
    """Start the fraud detection API server"""
    import uvicorn
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "api.fraud_api:app",
        host=host,
        port=port,
        reload=False
    )


@cli.command()
def test():
    """Run all tests"""
    import subprocess
    
    logger.info("Running tests...")
    
    # Run Python tests
    result = subprocess.run(['pytest', 'tests/', '-v'], capture_output=True)
    print(result.stdout.decode())
    
    # Run API tests
    if os.path.exists('tests/test_api.sh'):
        logger.info("\nRunning API tests...")
        subprocess.run(['bash', 'tests/test_api.sh'])


@cli.command()
@click.argument('transaction_file')
@click.option('--visualize', is_flag=True, help='Generate visualization')
def analyze(transaction_file, visualize):
    """Analyze a transaction file for fraud patterns"""
    from graph.graph_builder import GraphBuilder, Transaction
    from models.community_detection import CommunityDetector
    from visualization.graph_visualizer import FraudGraphVisualizer
    import pandas as pd
    
    logger.info(f"Analyzing {transaction_file}...")
    
    # Load transactions
    df = pd.read_csv(transaction_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert to Transaction objects
    transactions = []
    for _, row in df.iterrows():
        txn = Transaction(
            transaction_id=row['transaction_id'],
            card_id=row['card_id'],
            merchant_id=row['merchant_id'],
            amount=row['amount'],
            timestamp=row['timestamp'],
            device_id=row.get('device_id'),
            ip_address=row.get('ip_address'),
            is_fraud=row.get('is_fraud', False)
        )
        transactions.append(txn)
    
    # Build graph
    graph_builder = GraphBuilder()
    fraud_graph = graph_builder.build_from_transactions(transactions)
    
    # Get statistics
    stats = graph_builder.get_graph_statistics()
    logger.info(f"Graph statistics: {stats}")
    
    # Detect communities
    detector = CommunityDetector()
    communities = detector.detect_communities(fraud_graph.graph)
    
    # Detect fraud rings
    fraud_labels = {f"txn_{row['transaction_id']}": row.get('is_fraud', False) 
                   for _, row in df.iterrows()}
    fraud_rings = detector.detect_fraud_rings(fraud_graph.graph, fraud_labels)
    
    logger.info(f"Detected {len(fraud_rings)} fraud rings")
    
    if visualize and fraud_rings:
        visualizer = FraudGraphVisualizer()
        output_path = "visualizations/fraud_analysis.html"
        os.makedirs("visualizations", exist_ok=True)
        
        visualizer.create_pyvis_network(
            fraud_graph.graph,
            fraud_rings,
            output_path
        )
        logger.info(f"Visualization saved to {output_path}")


@cli.command()
def benchmark():
    """Run performance benchmarks"""
    from tests.test_scenarios import FraudScenarioGenerator
    
    logger.info("Running fraud detection benchmarks...")
    
    generator = FraudScenarioGenerator()
    results = generator.run_all_scenarios()
    
    logger.info("\nBenchmark Results:")
    for scenario, metrics in results.items():
        logger.info(f"\n{scenario}:")
        logger.info(f"  Detection Rate: {metrics['detection_rate']:.1%}")
        logger.info(f"  Avg Processing Time: {metrics['avg_processing_time']:.1f}ms")


@cli.command()
def info():
    """Display system information"""
    import torch
    import networkx as nx
    import platform
    
    print("\n=== Fraud Detection Graph System ===")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"NetworkX: {nx.__version__}")
    print(f"Platform: {platform.platform()}")
    
    # Check for trained models
    models_dir = "models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        print(f"\nTrained Models: {len(models)}")
        for model in models:
            print(f"  - {model}")
    
    print("\nFor help, run: python main.py --help")


if __name__ == "__main__":
    cli()