"""
Graph Neural Network models for fraud detection.
Implements GraphSAGE and Graph Attention Networks (GAT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for fraud detection.
    Supports heterogeneous graphs and multiple aggregation methods.
    """
    
    def __init__(self, 
                 num_features: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 aggregator: str = "mean"):
        super(GraphSAGE, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        
        # Additional layers for graph-level prediction
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Node embeddings
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # For graph-level prediction
        if batch is not None:
            # Combine mean and max pooling
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            x = self.graph_pred_linear(x)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings without final classification layer"""
        with torch.no_grad():
            for i in range(self.num_layers - 1):
                x = self.convs[i](x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network for fraud detection.
    Uses attention mechanism to weight neighbor importance.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.5,
                 attention_dropout: float = 0.2):
        super(GAT, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(num_features, hidden_dim // num_heads,
                   heads=num_heads, dropout=attention_dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                       heads=num_heads, dropout=attention_dropout)
            )
        
        # Output layer
        self.convs.append(
            GATConv(hidden_dim, output_dim,
                   heads=1, concat=False, dropout=attention_dropout)
        )
        
        # Graph-level prediction
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass, returns predictions and attention weights"""
        attention_weights = []
        
        # Apply GAT layers
        for i in range(self.num_layers - 1):
            x, (edge_index_i, alpha) = self.convs[i](x, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x, (edge_index_i, alpha) = self.convs[-1](x, edge_index, return_attention_weights=True)
        attention_weights.append(alpha)
        
        # For graph-level prediction
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            x = self.graph_pred_linear(x)
        
        return x, attention_weights
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights for interpretability"""
        with torch.no_grad():
            _, attention_weights = self.forward(x, edge_index)
        return attention_weights


class HybridGNN(nn.Module):
    """
    Hybrid model combining GraphSAGE and GAT for robust fraud detection.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        super(HybridGNN, self).__init__()
        
        # GraphSAGE branch
        self.sage = GraphSAGE(
            num_features, hidden_dim // 2, hidden_dim // 2,
            num_layers, dropout
        )
        
        # GAT branch
        self.gat = GAT(
            num_features, hidden_dim // 2, hidden_dim // 2,
            num_layers, num_heads=4, dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass combining both models"""
        # Get embeddings from both models
        sage_out = self.sage(x, edge_index, batch)
        gat_out, _ = self.gat(x, edge_index, batch)
        
        # Concatenate and fuse
        combined = torch.cat([sage_out, gat_out], dim=-1)
        out = self.fusion(combined)
        
        return out


class FraudGNN(nn.Module):
    """
    Specialized GNN for fraud detection with multiple node types.
    Handles heterogeneous graphs and incorporates domain knowledge.
    """
    
    def __init__(self,
                 node_type_features: Dict[str, int],
                 edge_types: List[str],
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        super(FraudGNN, self).__init__()
        
        self.node_types = list(node_type_features.keys())
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Linear(feat_dim, hidden_dim)
            for node_type, feat_dim in node_type_features.items()
        })
        
        # Edge type specific convolutions
        self.convs = nn.ModuleDict()
        for edge_type in edge_types:
            conv_list = nn.ModuleList()
            for _ in range(num_layers):
                conv_list.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs[edge_type] = conv_list
        
        # Attention for edge type aggregation
        self.edge_attention = nn.Linear(hidden_dim * len(edge_types), len(edge_types))
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Fraud pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 fraud pattern types
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for heterogeneous graph.
        
        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge indices
        """
        # Embed nodes by type
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_embeddings:
                h_dict[node_type] = self.node_embeddings[node_type](x)
                h_dict[node_type] = F.relu(h_dict[node_type])
        
        # Apply convolutions for each edge type
        edge_outputs = []
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type in self.convs:
                # Get source and target node types
                src_type, dst_type = edge_type.split("_to_")
                
                h = h_dict[src_type]
                for conv in self.convs[edge_type]:
                    h = conv(h, edge_index)
                    h = F.relu(h)
                    h = F.dropout(h, p=0.5, training=self.training)
                
                edge_outputs.append(h)
        
        # Aggregate edge-type specific outputs
        if edge_outputs:
            stacked = torch.stack(edge_outputs, dim=1)
            attention_input = stacked.view(stacked.size(0), -1)
            attention_weights = F.softmax(self.edge_attention(attention_input), dim=1)
            
            # Weighted sum
            aggregated = torch.sum(
                stacked * attention_weights.unsqueeze(-1),
                dim=1
            )
        else:
            aggregated = h_dict[self.node_types[0]]
        
        # Classification and pattern detection
        fraud_scores = self.classifier(aggregated)
        pattern_scores = self.pattern_detector(aggregated)
        
        return {
            "fraud_scores": fraud_scores,
            "pattern_scores": pattern_scores,
            "node_embeddings": aggregated
        }


class GraphDataset(torch.utils.data.Dataset):
    """
    Dataset class for graph data.
    """
    
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs
        
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


def create_data_loaders(train_graphs: List[Data],
                       val_graphs: List[Data],
                       test_graphs: List[Data],
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training"""
    train_dataset = GraphDataset(train_graphs)
    val_dataset = GraphDataset(val_graphs)
    test_dataset = GraphDataset(test_graphs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader