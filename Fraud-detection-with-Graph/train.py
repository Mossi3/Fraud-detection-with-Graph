"""
Training script for Graph Neural Network fraud detection models.
Trains GraphSAGE and GAT models on transaction graph data.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph.graph_builder import GraphBuilder, Transaction
from features.feature_extractor import GraphFeatureExtractor
from models.gnn_models import GraphSAGE, GAT, HybridGNN
from utils.data_generator import TransactionDataGenerator


class FraudDetectionTrainer:
    """Trainer for fraud detection GNN models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.graph_builder = GraphBuilder(config.get('graph_config', {}))
        self.feature_extractor = GraphFeatureExtractor(config.get('feature_config', {}))
        
        self.models = {}
        self.optimizers = {}
        self.best_scores = {}
        
    def prepare_data(self, data_path: str = None):
        """Prepare data for training"""
        logger.info("Preparing data...")
        
        if data_path and os.path.exists(data_path):
            # Load existing data
            generator = TransactionDataGenerator()
            df = generator.load_dataset(data_path)
        else:
            # Generate synthetic data
            logger.info("Generating synthetic dataset...")
            generator = TransactionDataGenerator()
            df = generator.generate_dataset(
                num_transactions=self.config.get('num_transactions', 50000)
            )
            
            # Save generated data
            os.makedirs('data', exist_ok=True)
            generator.save_dataset(df, 'data/training_data.csv')
        
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
                location=(row.get('location_lat'), row.get('location_lon')),
                is_fraud=row['is_fraud']
            )
            transactions.append(txn)
        
        # Build graph
        logger.info("Building transaction graph...")
        self.fraud_graph = self.graph_builder.build_from_transactions(transactions)
        
        # Extract features
        logger.info("Extracting features...")
        node_ids = list(self.fraud_graph.graph.nodes())
        feature_matrix = self.feature_extractor.create_feature_matrix(
            self.fraud_graph.graph, node_ids
        )
        
        # Get labels
        labels = []
        for node_id in node_ids:
            node_data = self.fraud_graph.graph.nodes[node_id]
            is_fraud = node_data.get('is_fraud', False)
            labels.append(1 if is_fraud else 0)
        
        labels = np.array(labels)
        
        # Create edge index for PyTorch Geometric
        edges = list(self.fraud_graph.graph.edges())
        edge_index = torch.LongTensor(edges).t().contiguous()
        
        # Create PyTorch Geometric data object
        self.data = Data(
            x=torch.FloatTensor(feature_matrix),
            edge_index=edge_index,
            y=torch.LongTensor(labels)
        )
        
        # Create train/val/test splits
        num_nodes = len(node_ids)
        indices = np.arange(num_nodes)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42
        )
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        
        logger.info(f"Data prepared - Nodes: {num_nodes}, Edges: {len(edges)}")
        logger.info(f"Train: {self.train_mask.sum()}, Val: {self.val_mask.sum()}, "
                   f"Test: {self.test_mask.sum()}")
        logger.info(f"Fraud rate: {labels.mean():.2%}")
        
    def initialize_models(self):
        """Initialize GNN models"""
        logger.info("Initializing models...")
        
        num_features = self.data.x.shape[1]
        
        # GraphSAGE model
        self.models['graphsage'] = GraphSAGE(
            num_features=num_features,
            hidden_dim=self.config.get('hidden_dim', 128),
            output_dim=2,
            num_layers=self.config.get('num_layers', 3),
            dropout=self.config.get('dropout', 0.5)
        ).to(self.device)
        
        # GAT model
        self.models['gat'] = GAT(
            num_features=num_features,
            hidden_dim=self.config.get('hidden_dim', 128),
            output_dim=2,
            num_layers=self.config.get('num_layers', 3),
            num_heads=self.config.get('num_heads', 8),
            dropout=self.config.get('dropout', 0.5)
        ).to(self.device)
        
        # Hybrid model
        self.models['hybrid'] = HybridGNN(
            num_features=num_features,
            hidden_dim=self.config.get('hidden_dim', 128),
            output_dim=2,
            num_layers=self.config.get('num_layers', 3),
            dropout=self.config.get('dropout', 0.5)
        ).to(self.device)
        
        # Initialize optimizers
        for model_name, model in self.models.items():
            self.optimizers[model_name] = optim.Adam(
                model.parameters(),
                lr=self.config.get('learning_rate', 0.001),
                weight_decay=self.config.get('weight_decay', 0.0001)
            )
            self.best_scores[model_name] = 0
            
    def train_epoch(self, model, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        optimizer.zero_grad()
        
        # Move data to device
        data = self.data.to(self.device)
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Calculate loss only on training nodes
        loss = criterion(out[self.train_mask], data.y[self.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = out.argmax(dim=1)
        train_acc = (pred[self.train_mask] == data.y[self.train_mask]).float().mean()
        
        return loss.item(), train_acc.item()
    
    def evaluate(self, model, mask):
        """Evaluate model performance"""
        model.eval()
        
        with torch.no_grad():
            data = self.data.to(self.device)
            out = model(data.x, data.edge_index)
            
            # Get predictions
            pred = out.argmax(dim=1)
            probas = torch.softmax(out, dim=1)[:, 1]
            
            # Calculate metrics
            accuracy = (pred[mask] == data.y[mask]).float().mean().item()
            
            # Get numpy arrays for sklearn metrics
            y_true = data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            y_proba = probas[mask].cpu().numpy()
            
            # Calculate additional metrics
            auc_score = roc_auc_score(y_true, y_proba)
            
            # Find optimal threshold using PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_threshold_idx = np.argmax(f1_scores[:-1])
            best_threshold = thresholds[best_threshold_idx]
            
            # Get predictions with optimal threshold
            y_pred_optimal = (y_proba >= best_threshold).astype(int)
            
            # Classification report
            report = classification_report(
                y_true, y_pred_optimal,
                target_names=['Normal', 'Fraud'],
                output_dict=True
            )
            
        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': report['Fraud']['precision'],
            'recall': report['Fraud']['recall'],
            'f1': report['Fraud']['f1-score'],
            'threshold': best_threshold
        }
    
    def train_model(self, model_name: str):
        """Train a specific model"""
        logger.info(f"\nTraining {model_name} model...")
        
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        
        # Class weights for imbalanced data
        class_counts = torch.bincount(self.data.y)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config.get('epochs', 100)):
            # Train
            train_loss, train_acc = self.train_epoch(model, optimizer, criterion)
            
            # Validate
            val_metrics = self.evaluate(model, self.val_mask)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:03d} - "
                    f"Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}, "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                torch.save(
                    model.state_dict(),
                    f"models/{model_name}_best.pt"
                )
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.get('patience', 20):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f"models/{model_name}_best.pt"))
        
        # Final evaluation
        test_metrics = self.evaluate(model, self.test_mask)
        self.best_scores[model_name] = test_metrics['f1']
        
        logger.info(f"\n{model_name} Test Performance:")
        logger.info(f"  AUC: {test_metrics['auc']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")
        logger.info(f"  Optimal Threshold: {test_metrics['threshold']:.4f}")
        
        return test_metrics
    
    def train_all_models(self):
        """Train all models"""
        os.makedirs('models', exist_ok=True)
        
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.train_model(model_name)
        
        # Save results
        with open('models/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comparison
        logger.info("\n" + "="*50)
        logger.info("Model Comparison:")
        logger.info("="*50)
        
        for model_name, metrics in results.items():
            logger.info(
                f"{model_name:12} - "
                f"F1: {metrics['f1']:.4f}, "
                f"AUC: {metrics['auc']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}"
            )
        
        # Save final models
        for model_name, model in self.models.items():
            torch.save(model, f"models/{model_name}.pt")
        
        logger.info(f"\nModels saved to models/ directory")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection GNN models')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'num_transactions': 50000,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': 0.5,
            'weight_decay': 0.0001,
            'patience': 20,
            'num_heads': 8
        }
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(config)
    
    # Prepare data
    trainer.prepare_data(args.data)
    
    # Initialize models
    trainer.initialize_models()
    
    # Train models
    results = trainer.train_all_models()
    
    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()