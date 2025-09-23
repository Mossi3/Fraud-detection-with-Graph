"""
Graph Neural Network Models for Fraud Detection
Implements GraphSAGE and GAT models for heterogeneous graph fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE model for fraud detection"""
    
    def __init__(self, node_types: List[str], edge_types: List[Tuple], 
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create heterogeneous convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                if i == 0:
                    # First layer - adapt to hidden dimension
                    conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)
                else:
                    conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Node-level prediction heads
        self.node_classifiers = nn.ModuleDict()
        for node_type in node_types:
            self.node_classifiers[node_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)  # Binary classification
            )
        
        # Edge-level prediction for transactions
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary fraud classification
        )
        
        # Risk scoring head
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple, torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Node-level predictions
        node_predictions = {}
        for node_type, classifier in self.node_classifiers.items():
            if node_type in x_dict:
                node_predictions[f'{node_type}_fraud_prob'] = classifier(x_dict[node_type])
        
        # Risk scores
        risk_scores = {}
        for node_type in x_dict:
            risk_scores[f'{node_type}_risk_score'] = self.risk_scorer(x_dict[node_type]).squeeze()
        
        return {
            'embeddings': x_dict,
            'node_predictions': node_predictions,
            'risk_scores': risk_scores
        }
    
    def predict_transaction_fraud(self, embeddings: Dict[str, torch.Tensor], 
                                 transaction_pairs: torch.Tensor) -> torch.Tensor:
        """Predict fraud for specific transactions"""
        card_embeddings = embeddings['card'][transaction_pairs[:, 0]]
        merchant_embeddings = embeddings['merchant'][transaction_pairs[:, 1]]
        
        # Concatenate embeddings
        edge_embeddings = torch.cat([card_embeddings, merchant_embeddings], dim=1)
        
        # Predict fraud probability
        fraud_logits = self.edge_classifier(edge_embeddings)
        return fraud_logits

class HeteroGAT(nn.Module):
    """Heterogeneous Graph Attention Network for fraud detection"""
    
    def __init__(self, node_types: List[str], edge_types: List[Tuple],
                 hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, 
                 dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create heterogeneous GAT convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                if i == 0:
                    conv_dict[edge_type] = GATConv((-1, -1), hidden_dim // num_heads, 
                                                 heads=num_heads, dropout=dropout)
                else:
                    conv_dict[edge_type] = GATConv((-1, -1), hidden_dim // num_heads, 
                                                 heads=num_heads, dropout=dropout)
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Attention aggregation
        self.attention_weights = nn.ModuleDict()
        for node_type in node_types:
            self.attention_weights[node_type] = nn.Linear(hidden_dim, 1)
        
        # Node-level prediction heads
        self.node_classifiers = nn.ModuleDict()
        for node_type in node_types:
            self.node_classifiers[node_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            )
        
        # Edge-level prediction
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Risk scoring
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple, torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        
        # Apply GAT convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # GAT already applies dropout and activation
        
        # Apply attention aggregation
        for node_type in x_dict:
            if node_type in self.attention_weights:
                attention = torch.softmax(self.attention_weights[node_type](x_dict[node_type]), dim=0)
                x_dict[node_type] = x_dict[node_type] * attention
        
        # Node-level predictions
        node_predictions = {}
        for node_type, classifier in self.node_classifiers.items():
            if node_type in x_dict:
                node_predictions[f'{node_type}_fraud_prob'] = classifier(x_dict[node_type])
        
        # Risk scores
        risk_scores = {}
        for node_type in x_dict:
            risk_scores[f'{node_type}_risk_score'] = self.risk_scorer(x_dict[node_type]).squeeze()
        
        return {
            'embeddings': x_dict,
            'node_predictions': node_predictions,
            'risk_scores': risk_scores
        }
    
    def predict_transaction_fraud(self, embeddings: Dict[str, torch.Tensor], 
                                 transaction_pairs: torch.Tensor) -> torch.Tensor:
        """Predict fraud for specific transactions"""
        card_embeddings = embeddings['card'][transaction_pairs[:, 0]]
        merchant_embeddings = embeddings['merchant'][transaction_pairs[:, 1]]
        
        edge_embeddings = torch.cat([card_embeddings, merchant_embeddings], dim=1)
        fraud_logits = self.edge_classifier(edge_embeddings)
        return fraud_logits

class FraudDetectionTrainer:
    """Training and evaluation utilities for fraud detection models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
    
    def prepare_data(self, hetero_data: HeteroData, test_size: float = 0.2, 
                    val_size: float = 0.1) -> Tuple[Dict, Dict, Dict]:
        """Prepare training, validation, and test data"""
        
        # Split transaction indices
        total_transactions = len(hetero_data.transaction_labels)
        indices = np.arange(total_transactions)
        
        train_idx, temp_idx = train_test_split(indices, test_size=test_size + val_size, 
                                             stratify=hetero_data.transaction_labels, 
                                             random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size/(test_size + val_size), 
                                           stratify=hetero_data.transaction_labels[temp_idx], 
                                           random_state=42)
        
        # Create data splits
        train_data = {
            'transaction_pairs': hetero_data.transaction_pairs[train_idx],
            'transaction_labels': hetero_data.transaction_labels[train_idx],
            'indices': train_idx
        }
        
        val_data = {
            'transaction_pairs': hetero_data.transaction_pairs[val_idx],
            'transaction_labels': hetero_data.transaction_labels[val_idx],
            'indices': val_idx
        }
        
        test_data = {
            'transaction_pairs': hetero_data.transaction_pairs[test_idx],
            'transaction_labels': hetero_data.transaction_labels[test_idx],
            'indices': test_idx
        }
        
        return train_data, val_data, test_data
    
    def train_epoch(self, hetero_data: HeteroData, train_data: Dict, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        # Move data to device
        x_dict = {key: x.to(self.device) for key, x in hetero_data.x_dict.items()}
        edge_index_dict = {key: edge_index.to(self.device) 
                          for key, edge_index in hetero_data.edge_index_dict.items()}
        
        # Forward pass
        outputs = self.model(x_dict, edge_index_dict)
        
        # Predict transaction fraud
        transaction_logits = self.model.predict_transaction_fraud(
            outputs['embeddings'], train_data['transaction_pairs'].to(self.device)
        )
        
        # Compute loss
        labels = train_data['transaction_labels'].to(self.device)
        loss = criterion(transaction_logits, labels)
        
        # Add node-level losses (fraud ring detection)
        node_loss = 0
        for node_type in ['card', 'merchant', 'device', 'ip']:
            if f'{node_type}_fraud_prob' in outputs['node_predictions']:
                node_labels = hetero_data[node_type].is_fraud_ring_member.to(self.device)
                node_pred = outputs['node_predictions'][f'{node_type}_fraud_prob']
                node_loss += criterion(node_pred, node_labels) * 0.1  # Weight node losses less
        
        total_loss = loss + node_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, hetero_data: HeteroData, eval_data: Dict) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            x_dict = {key: x.to(self.device) for key, x in hetero_data.x_dict.items()}
            edge_index_dict = {key: edge_index.to(self.device) 
                              for key, edge_index in hetero_data.edge_index_dict.items()}
            
            # Forward pass
            outputs = self.model(x_dict, edge_index_dict)
            
            # Predict transaction fraud
            transaction_logits = self.model.predict_transaction_fraud(
                outputs['embeddings'], eval_data['transaction_pairs'].to(self.device)
            )
            
            # Get predictions and probabilities
            probs = F.softmax(transaction_logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(transaction_logits, dim=1).cpu().numpy()
            labels = eval_data['transaction_labels'].numpy()
            
            # Calculate metrics
            auc_roc = roc_auc_score(labels, probs)
            precision, recall, _ = precision_recall_curve(labels, probs)
            auc_pr = auc(recall, precision)
            
            accuracy = (preds == labels).mean()
            
            # Calculate fraud ring detection metrics
            ring_metrics = {}
            for node_type in ['card', 'merchant', 'device', 'ip']:
                if f'{node_type}_fraud_prob' in outputs['node_predictions']:
                    node_probs = F.softmax(outputs['node_predictions'][f'{node_type}_fraud_prob'], dim=1)[:, 1]
                    node_labels = hetero_data[node_type].is_fraud_ring_member.cpu().numpy()
                    
                    if len(np.unique(node_labels)) > 1:  # Only if we have both classes
                        ring_auc = roc_auc_score(node_labels, node_probs.cpu().numpy())
                        ring_metrics[f'{node_type}_ring_auc'] = ring_auc
        
        metrics = {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            **ring_metrics
        }
        
        return metrics
    
    def train(self, hetero_data: HeteroData, num_epochs: int = 100, 
             lr: float = 0.01, weight_decay: float = 1e-4) -> Dict[str, List]:
        """Full training loop"""
        
        # Prepare data
        train_data, val_data, test_data = self.prepare_data(hetero_data)
        
        # Setup optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_auc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(hetero_data, train_data, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(hetero_data, val_data)
            self.val_losses.append(train_loss)  # For plotting purposes
            self.metrics_history.append(val_metrics)
            
            # Early stopping
            if val_metrics['auc_roc'] > best_val_auc:
                best_val_auc = val_metrics['auc_roc']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/workspace/fraud_detection_graph/models/best_model.pt')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation on test set
        self.model.load_state_dict(torch.load('/workspace/fraud_detection_graph/models/best_model.pt'))
        test_metrics = self.evaluate(hetero_data, test_data)
        
        print(f"\nFinal Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'test_metrics': test_metrics,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }

def create_model(hetero_data: HeteroData, model_type: str = 'graphsage', 
                hidden_dim: int = 128) -> nn.Module:
    """Create a GNN model based on the heterogeneous graph"""
    
    node_types = list(hetero_data.x_dict.keys())
    edge_types = list(hetero_data.edge_index_dict.keys())
    
    if model_type.lower() == 'graphsage':
        model = HeteroGraphSAGE(node_types, edge_types, hidden_dim=hidden_dim)
    elif model_type.lower() == 'gat':
        model = HeteroGAT(node_types, edge_types, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

if __name__ == "__main__":
    # Load the heterogeneous graph
    import sys
    sys.path.append('/workspace/fraud_detection_graph')
    from models.graph_builder import FraudGraphBuilder
    
    builder = FraudGraphBuilder()
    hetero_data, _ = builder.load_graph()
    
    print("Creating GraphSAGE model...")
    model_sage = create_model(hetero_data, 'graphsage', hidden_dim=64)
    trainer_sage = FraudDetectionTrainer(model_sage)
    
    print("Training GraphSAGE...")
    results_sage = trainer_sage.train(hetero_data, num_epochs=50, lr=0.01)
    
    print("\nCreating GAT model...")
    model_gat = create_model(hetero_data, 'gat', hidden_dim=64)
    trainer_gat = FraudDetectionTrainer(model_gat)
    
    print("Training GAT...")
    results_gat = trainer_gat.train(hetero_data, num_epochs=50, lr=0.01)
    
    # Compare models
    print(f"\nModel Comparison:")
    print(f"GraphSAGE Test AUC-ROC: {results_sage['test_metrics']['auc_roc']:.4f}")
    print(f"GAT Test AUC-ROC: {results_gat['test_metrics']['auc_roc']:.4f}")
    
    # Save results
    torch.save({
        'graphsage_results': results_sage,
        'gat_results': results_gat,
        'hetero_data': hetero_data
    }, '/workspace/fraud_detection_graph/models/training_results.pt')