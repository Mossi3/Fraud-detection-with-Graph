"""
Graph Neural Network Training Module
Comprehensive training pipeline for fraud detection GNNs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import json
import os

logger = logging.getLogger(__name__)

class GNNTrainer:
    """
    Comprehensive trainer for Graph Neural Network fraud detection models
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []
        self.best_val_auc = 0.0
        self.best_model_state = None
        
        # Metrics
        self.metrics = {}
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for training"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def prepare_data(self, graph_data: Dict[str, torch.Tensor], 
                    labels: torch.Tensor,
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    random_state: int = 42) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare data for training, validation, and testing
        
        Args:
            graph_data: Dictionary containing graph data (x, edge_index, etc.)
            labels: Fraud labels
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        
        logger.info("Preparing data for training")
        
        # Split data
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            indices, test_size=test_size + val_size, 
            random_state=random_state, stratify=labels
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_size/(test_size + val_size),
            random_state=random_state, stratify=labels[temp_idx]
        )
        
        # Prepare data splits
        train_data = {
            'x': graph_data['x'][train_idx],
            'edge_index': graph_data['edge_index'],
            'labels': labels[train_idx],
            'indices': train_idx
        }
        
        val_data = {
            'x': graph_data['x'][val_idx],
            'edge_index': graph_data['edge_index'],
            'labels': labels[val_idx],
            'indices': val_idx
        }
        
        test_data = {
            'x': graph_data['x'][test_idx],
            'edge_index': graph_data['edge_index'],
            'labels': labels[test_idx],
            'indices': test_idx
        }
        
        logger.info(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return train_data, val_data, test_data
    
    def train_epoch(self, train_data: Dict, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_data: Training data
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, average_auc)
        """
        
        self.model.train()
        total_loss = 0.0
        total_auc = 0.0
        num_batches = 0
        
        # Move data to device
        x = train_data['x'].to(self.device)
        edge_index = train_data['edge_index'].to(self.device)
        labels = train_data['labels'].to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(x, edge_index)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            if len(np.unique(labels_np)) > 1:  # Check if both classes present
                auc = roc_auc_score(labels_np, predictions)
            else:
                auc = 0.5  # Random performance if only one class
            
            total_loss += loss.item()
            total_auc += auc
            num_batches += 1
        
        return total_loss / num_batches, total_auc / num_batches
    
    def validate_epoch(self, val_data: Dict, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            val_data: Validation data
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, average_auc)
        """
        
        self.model.eval()
        total_loss = 0.0
        total_auc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # Move data to device
            x = val_data['x'].to(self.device)
            edge_index = val_data['edge_index'].to(self.device)
            labels = val_data['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(x, edge_index)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels.float())
            
            # Calculate metrics
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            if len(np.unique(labels_np)) > 1:
                auc = roc_auc_score(labels_np, predictions)
            else:
                auc = 0.5
            
            total_loss += loss.item()
            total_auc += auc
            num_batches += 1
        
        return total_loss / num_batches, total_auc / num_batches
    
    def train(self, train_data: Dict, val_data: Dict,
              epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 10,
              save_best: bool = True) -> Dict[str, List]:
        """
        Train the model
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            save_best: Whether to save best model
            
        Returns:
            Training history
        """
        
        logger.info(f"Starting training for {epochs} epochs")
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        # Early stopping
        best_val_auc = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            train_loss, train_auc = self.train_epoch(train_data, optimizer, criterion)
            
            # Validate
            val_loss, val_auc = self.validate_epoch(val_data, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                if save_best:
                    self.best_model_state = self.model.state_dict().copy()
                    self.best_val_auc = val_auc
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with validation AUC: {self.best_val_auc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_aucs': self.train_aucs,
            'val_aucs': self.val_aucs
        }
    
    def evaluate(self, test_data: Dict) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        logger.info("Evaluating model on test data")
        
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            x = test_data['x'].to(self.device)
            edge_index = test_data['edge_index'].to(self.device)
            labels = test_data['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(x, edge_index)
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Calculate metrics
            metrics = {}
            
            # ROC AUC
            if len(np.unique(labels_np)) > 1:
                metrics['roc_auc'] = roc_auc_score(labels_np, predictions)
            else:
                metrics['roc_auc'] = 0.5
            
            # PR AUC
            metrics['pr_auc'] = average_precision_score(labels_np, predictions)
            
            # Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(labels_np, predictions)
            
            # Find optimal threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Binary predictions
            binary_predictions = (predictions >= optimal_threshold).astype(int)
            
            # Additional metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = accuracy_score(labels_np, binary_predictions)
            metrics['precision'] = precision_score(labels_np, binary_predictions, zero_division=0)
            metrics['recall'] = recall_score(labels_np, binary_predictions, zero_division=0)
            metrics['f1_score'] = f1_score(labels_np, binary_predictions, zero_division=0)
            metrics['optimal_threshold'] = optimal_threshold
            
            self.metrics = metrics
            
            logger.info(f"Test Metrics: {metrics}")
            
            return metrics
    
    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC plot
        axes[1].plot(self.train_aucs, label='Train AUC', color='blue')
        axes[1].plot(self.val_aucs, label='Validation AUC', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Training and Validation AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, test_data: Dict, save_path: str = None) -> None:
        """
        Plot precision-recall curve
        """
        
        self.model.eval()
        
        with torch.no_grad():
            x = test_data['x'].to(self.device)
            edge_index = test_data['edge_index'].to(self.device)
            labels = test_data['labels'].to(self.device)
            
            outputs = self.model(x, edge_index)
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            precision, recall, thresholds = precision_recall_curve(labels_np, predictions)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {self.metrics.get("pr_auc", 0):.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'metrics': self.metrics,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_aucs': self.train_aucs,
                'val_aucs': self.val_aucs
            }
        }
        
        torch.save(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model
        
        Args:
            path: Path to load the model from
        """
        
        model_data = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.metrics = model_data.get('metrics', {})
        
        training_history = model_data.get('training_history', {})
        self.train_losses = training_history.get('train_losses', [])
        self.val_losses = training_history.get('val_losses', [])
        self.train_aucs = training_history.get('train_aucs', [])
        self.val_aucs = training_history.get('val_aucs', [])
        
        logger.info(f"Model loaded from {path}")

class GraphDataProcessor:
    """
    Process graph data for GNN training
    """
    
    @staticmethod
    def networkx_to_pytorch_geometric(graph: nx.Graph, 
                                    node_features: Dict[str, np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """
        Convert NetworkX graph to PyTorch Geometric format
        
        Args:
            graph: NetworkX graph
            node_features: Optional node features
            
        Returns:
            Dictionary with PyTorch tensors
        """
        
        # Get node mapping
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge index
        edge_list = []
        edge_weights = []
        
        for edge in graph.edges(data=True):
            src, dst, data = edge
            edge_list.append([node_to_idx[src], node_to_idx[dst]])
            
            # Add edge weight if available
            weight = data.get('weight', 1.0)
            edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Create node features
        if node_features is not None:
            # Use provided features
            feature_dim = len(next(iter(node_features.values())))
            x = torch.zeros(len(nodes), feature_dim)
            
            for node, features in node_features.items():
                if node in node_to_idx:
                    x[node_to_idx[node]] = torch.tensor(features, dtype=torch.float)
        else:
            # Create basic features (degree, node type, etc.)
            feature_dim = 10  # Adjust based on your needs
            x = torch.zeros(len(nodes), feature_dim)
            
            for node in nodes:
                idx = node_to_idx[node]
                
                # Basic features
                x[idx, 0] = graph.degree(node)  # Degree
                x[idx, 1] = len(graph.neighbors(node))  # Number of neighbors
                
                # Node type encoding (if available)
                node_type = graph.nodes[node].get('node_type', 'unknown')
                type_encoding = hash(node_type) % 8  # Simple encoding
                x[idx, 2] = type_encoding
                
                # Additional features can be added here
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'node_mapping': node_to_idx
        }
    
    @staticmethod
    def create_labels_from_transactions(df: pd.DataFrame, 
                                      node_mapping: Dict[str, int],
                                      label_column: str = 'fraud') -> torch.Tensor:
        """
        Create labels tensor from transaction data
        
        Args:
            df: Transaction dataframe
            node_mapping: Mapping from node names to indices
            label_column: Column name for labels
            
        Returns:
            Labels tensor
        """
        
        # Create labels for each node
        labels = torch.zeros(len(node_mapping))
        
        # Aggregate labels by node (e.g., card_id)
        if 'card_id' in df.columns and label_column in df.columns:
            node_labels = df.groupby('card_id')[label_column].mean()
            
            for node, label in node_labels.items():
                if node in node_mapping:
                    labels[node_mapping[node]] = label
        
        return labels

def train_gnn_pipeline(df: pd.DataFrame, 
                       model_type: str = 'graphsage',
                       epochs: int = 100,
                       learning_rate: float = 0.001,
                       save_path: str = 'models/gnn_model.pth') -> GNNTrainer:
    """
    Complete pipeline for training GNN models
    
    Args:
        df: Transaction dataframe
        model_type: Type of GNN model
        epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Path to save the model
        
    Returns:
        Trained GNNTrainer instance
    """
    
    logger.info(f"Starting GNN training pipeline with {model_type}")
    
    # Build graph
    from src.graph.construction import HeterogeneousGraphBuilder, GraphConfig
    
    config = GraphConfig()
    builder = HeterogeneousGraphBuilder(config)
    builder.add_transaction_data(df)
    graph = builder.to_networkx()
    
    # Convert to PyTorch Geometric format
    processor = GraphDataProcessor()
    graph_data = processor.networkx_to_pytorch_geometric(graph)
    
    # Create labels
    labels = processor.create_labels_from_transactions(df, graph_data['node_mapping'])
    
    # Create model
    from src.graph.models import create_gnn_model
    
    input_dim = graph_data['x'].shape[1]
    model = create_gnn_model(model_type, input_dim)
    
    # Create trainer
    trainer = GNNTrainer(model)
    
    # Prepare data
    train_data, val_data, test_data = trainer.prepare_data(graph_data, labels)
    
    # Train model
    trainer.train(train_data, val_data, epochs=epochs, learning_rate=learning_rate)
    
    # Evaluate
    metrics = trainer.evaluate(test_data)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.save_model(save_path)
    
    # Plot results
    trainer.plot_training_history(f"{save_path}_training_history.png")
    trainer.plot_precision_recall_curve(test_data, f"{save_path}_pr_curve.png")
    
    logger.info(f"Training completed. Final metrics: {metrics}")
    
    return trainer