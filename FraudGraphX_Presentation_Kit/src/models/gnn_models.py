"""
Advanced Graph Neural Network models for fraud detection.
Implements GraphSAGE, GAT, and heterogeneous GNN variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
import dgl
import dgl.nn as dglnn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE for fraud detection."""
    
    def __init__(self, metadata: Tuple, hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, aggr: str = 'mean'):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projections for each node type
        self.node_projections = nn.ModuleDict()
        for node_type in metadata[0]:
            # These will be set dynamically based on input features
            self.node_projections[node_type] = None
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src_type, rel_type, dst_type = edge_type
                conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Output layers
        self.fraud_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Ring detection head
        self.ring_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32)  # Embedding for clustering
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Project input features to hidden dimension
        for node_type, x in x_dict.items():
            if self.node_projections[node_type] is None:
                self.node_projections[node_type] = Linear(x.size(-1), self.hidden_dim)
            x_dict[node_type] = self.node_projections[node_type](x)
        
        # Apply GraphSAGE layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Get transaction embeddings for classification
        transaction_embeddings = x_dict.get('transaction')
        
        outputs = {}
        if transaction_embeddings is not None:
            outputs['fraud_logits'] = self.fraud_classifier(transaction_embeddings)
            outputs['ring_embeddings'] = self.ring_detector(transaction_embeddings)
        
        # Add embeddings for all node types
        outputs['node_embeddings'] = x_dict
        
        return outputs

class HeteroGAT(nn.Module):
    """Heterogeneous Graph Attention Network for fraud detection."""
    
    def __init__(self, metadata: Tuple, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projections
        self.node_projections = nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_projections[node_type] = None
        
        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            in_dim = hidden_dim
            out_dim = hidden_dim // num_heads if i < num_layers - 1 else hidden_dim
            
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATConv(
                    in_dim, out_dim, heads=num_heads, 
                    dropout=dropout, concat=(i < num_layers - 1)
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Output layers
        self.fraud_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.ring_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32)
        )
        
        # Attention weights for interpretability
        self.attention_weights = {}
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Project input features
        for node_type, x in x_dict.items():
            if self.node_projections[node_type] is None:
                self.node_projections[node_type] = Linear(x.size(-1), self.hidden_dim)
            x_dict[node_type] = self.node_projections[node_type](x)
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        # Get transaction embeddings
        transaction_embeddings = x_dict.get('transaction')
        
        outputs = {}
        if transaction_embeddings is not None:
            outputs['fraud_logits'] = self.fraud_classifier(transaction_embeddings)
            outputs['ring_embeddings'] = self.ring_detector(transaction_embeddings)
        
        outputs['node_embeddings'] = x_dict
        
        return outputs

class DualChannelGNN(nn.Module):
    """Dual-channel GNN that separates homophilic and heterophilic patterns."""
    
    def __init__(self, metadata: Tuple, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        
        # Homophilic channel (similar nodes connect)
        self.homo_gnn = HeteroGraphSAGE(metadata, hidden_dim, num_layers)
        
        # Heterophilic channel (different nodes connect - fraud patterns)
        self.hetero_gnn = HeteroGAT(metadata, hidden_dim, num_layers)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Get embeddings from both channels
        homo_out = self.homo_gnn(x_dict, edge_index_dict)
        hetero_out = self.hetero_gnn(x_dict, edge_index_dict)
        
        # Fuse embeddings
        if 'transaction' in homo_out['node_embeddings']:
            homo_emb = homo_out['node_embeddings']['transaction']
            hetero_emb = hetero_out['node_embeddings']['transaction']
            
            fused_emb = self.fusion(torch.cat([homo_emb, hetero_emb], dim=1))
            fraud_logits = self.classifier(fused_emb)
            
            return {
                'fraud_logits': fraud_logits,
                'homo_embeddings': homo_emb,
                'hetero_embeddings': hetero_emb,
                'fused_embeddings': fused_emb,
                'ring_embeddings': homo_out.get('ring_embeddings')
            }
        
        return homo_out

class FraudGNNTrainer:
    """Training and evaluation utilities for fraud detection GNNs."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_metrics = []
    
    def setup_training(self, lr: float = 0.001, weight_decay: float = 1e-4):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
    
    def train_epoch(self, data: HeteroData, train_mask: torch.Tensor, 
                   class_weights: Optional[torch.Tensor] = None) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        train_mask = train_mask.to(self.device)
        
        # Forward pass
        out = self.model(data.x_dict, data.edge_index_dict)
        fraud_logits = out['fraud_logits']
        
        # Get labels
        labels = data['transaction'].y[train_mask]
        predictions = fraud_logits[train_mask]
        
        # Compute loss with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: HeteroData, eval_mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            eval_mask = eval_mask.to(self.device)
            
            out = self.model(data.x_dict, data.edge_index_dict)
            fraud_logits = out['fraud_logits']
            
            labels = data['transaction'].y[eval_mask].cpu().numpy()
            predictions = fraud_logits[eval_mask].cpu().numpy()
            probs = F.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
            
            # Calculate metrics
            auc_roc = roc_auc_score(labels, probs)
            precision, recall, _ = precision_recall_curve(labels, probs)
            auc_pr = auc(recall, precision)
            
            # Accuracy
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = (pred_labels == labels).mean()
            
            return {
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'accuracy': accuracy,
                'precision': precision[-1],  # Final precision
                'recall': recall[-1]  # Final recall
            }
    
    def train(self, data: HeteroData, train_mask: torch.Tensor, 
              val_mask: torch.Tensor, epochs: int = 100, 
              early_stopping_patience: int = 15) -> Dict[str, List]:
        """Full training loop with early stopping."""
        
        # Calculate class weights for imbalanced data
        labels = data['transaction'].y[train_mask].cpu().numpy()
        class_counts = np.bincount(labels)
        class_weights = torch.tensor(len(labels) / (2 * class_counts), dtype=torch.float32)
        
        logger.info(f"Training with class weights: {class_weights}")
        
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(data, train_mask, class_weights)
            self.train_losses.append(train_loss)
            
            # Validation
            val_metrics = self.evaluate(data, val_mask)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['auc_pr'])
            
            # Early stopping
            if val_metrics['auc_pr'] > best_val_auc:
                best_val_auc = val_metrics['auc_pr']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gnn_model.pt')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val AUC-PR: {val_metrics['auc_pr']:.4f}, "
                          f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pt'))
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_val_auc_pr': best_val_auc
        }

def create_model(model_type: str, metadata: Tuple, **kwargs) -> nn.Module:
    """Factory function to create GNN models."""
    
    if model_type.lower() == 'graphsage':
        return HeteroGraphSAGE(metadata, **kwargs)
    elif model_type.lower() == 'gat':
        return HeteroGAT(metadata, **kwargs)
    elif model_type.lower() == 'dual_channel':
        return DualChannelGNN(metadata, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    """Example training script."""
    import argparse
    from ..graph.graph_builder import HeterogeneousGraphBuilder
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Train GNN for fraud detection")
    parser.add_argument('--data', type=str, required=True, help='Transaction data CSV')
    parser.add_argument('--model', type=str, default='graphsage', 
                       choices=['graphsage', 'gat', 'dual_channel'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Load data and build graph
    df = pd.read_csv(args.data)
    builder = HeterogeneousGraphBuilder()
    data = builder.build_graph(df, graph_type='torch_geometric')
    
    # Add labels to transaction nodes
    data['transaction'].y = torch.tensor(df['fraud'].values, dtype=torch.long)
    
    # Create train/val splits
    n_transactions = len(df)
    indices = torch.randperm(n_transactions)
    train_size = int(0.7 * n_transactions)
    val_size = int(0.15 * n_transactions)
    
    train_mask = torch.zeros(n_transactions, dtype=torch.bool)
    val_mask = torch.zeros(n_transactions, dtype=torch.bool)
    test_mask = torch.zeros(n_transactions, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create model
    model = create_model(args.model, data.metadata(), 
                        hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    # Train model
    trainer = FraudGNNTrainer(model)
    trainer.setup_training(lr=args.lr)
    
    results = trainer.train(data, train_mask, val_mask, epochs=args.epochs)
    
    # Final evaluation
    test_metrics = trainer.evaluate(data, test_mask)
    logger.info(f"Test Results: {test_metrics}")

if __name__ == "__main__":
    main()