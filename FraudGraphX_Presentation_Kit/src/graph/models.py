"""
Graph Neural Network Models for Fraud Detection
Implements GraphSAGE, GAT, and other GNN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GraphSAGEFraudDetector(nn.Module):
    """
    GraphSAGE-based fraud detection model
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GraphSAGE layers"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling and classification
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.classifier(x)
        
        return x

class GATFraudDetector(nn.Module):
    """
    Graph Attention Network-based fraud detection model
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT layers"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling and classification
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.classifier(x)
        
        return x

class HeteroGNNFraudDetector(nn.Module):
    """
    Heterogeneous Graph Neural Network for fraud detection
    Handles multiple node and edge types
    """
    
    def __init__(self, node_types: List[str], edge_types: List[Tuple[str, str, str]], 
                 input_dims: Dict[str, int], hidden_dim: int = 64, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict()
        for node_type in node_types:
            self.node_embeddings[node_type] = nn.Linear(input_dims[node_type], hidden_dim)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        """Forward pass through heterogeneous GNN"""
        
        # Node type embeddings
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.node_embeddings[node_type](x)
        
        # Heterogeneous convolutions
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.relu(h) for key, h in h_dict.items()}
        
        # Global pooling across all node types
        global_features = []
        for node_type, h in h_dict.items():
            global_features.append(torch.mean(h, dim=0))
        
        # Concatenate and classify
        x = torch.cat(global_features, dim=0)
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.classifier(x)
        
        return x

class GraphTransformerFraudDetector(nn.Module):
    """
    Graph Transformer-based fraud detection model
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph Transformer layers
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        self.convs.append(TransformerConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through Graph Transformer layers"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling and classification
        x = torch.mean(x, dim=0, keepdim=True)  # Global mean pooling
        x = self.classifier(x)
        
        return x

class GraphFraudEnsemble(nn.Module):
    """
    Ensemble of different GNN models for robust fraud detection
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        # Individual models
        self.graphsage = GraphSAGEFraudDetector(input_dim, hidden_dim, output_dim, dropout=dropout)
        self.gat = GATFraudDetector(input_dim, hidden_dim, output_dim, dropout=dropout)
        self.transformer = GraphTransformerFraudDetector(input_dim, hidden_dim, output_dim, dropout=dropout)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        
        # Get predictions from each model
        sage_out = self.graphsage(x, edge_index)
        gat_out = self.gat(x, edge_index)
        transformer_out = self.transformer(x, edge_index)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = weights[0] * sage_out + weights[1] * gat_out + weights[2] * transformer_out
        
        # Final classification
        combined_features = torch.cat([sage_out, gat_out, transformer_out], dim=1)
        final_out = self.classifier(combined_features)
        
        return final_out

def create_gnn_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create GNN models
    
    Args:
        model_type: Type of GNN model ('graphsage', 'gat', 'transformer', 'ensemble')
        input_dim: Input feature dimension
        **kwargs: Additional model parameters
    
    Returns:
        PyTorch GNN model
    """
    
    models = {
        'graphsage': GraphSAGEFraudDetector,
        'gat': GATFraudDetector,
        'transformer': GraphTransformerFraudDetector,
        'ensemble': GraphFraudEnsemble
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](input_dim, **kwargs)