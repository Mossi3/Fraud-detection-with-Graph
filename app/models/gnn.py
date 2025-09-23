from typing import Dict, Tuple
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroSAGE(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        hidden_channels: int = 64,
        dropout: float = 0.2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Project node features to common hidden dim
        self.lin_in = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_channels)
            for node_type, in_dim in in_dims.items()
        })

        # Build stacked HeteroConv layers. We will set edge types dynamically in forward
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Placeholder HeteroConv; will be assigned with SAGEConv modules per edge type in forward
            self.convs.append(HeteroConv({}, aggr="mean"))

        # Classifier for card nodes
        self.card_classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x_dict, edge_index_dict):
        # Initial projection
        h_dict = {k: self.act(self.lin_in[k](x)) for k, x in x_dict.items()}
        h_dict = {k: self.dropout(v) for k, v in h_dict.items()}

        # Build conv layers with per-edge SAGEConv using dynamic input dims (-1)
        # We construct conv modules lazily on first forward
        for layer_idx in range(self.num_layers):
            if len(self.convs[layer_idx].convs) == 0:
                self.convs[layer_idx].convs = torch.nn.ModuleDict({
                    edge_type: SAGEConv((-1, -1), self.hidden_channels)
                    for edge_type in edge_index_dict.keys()
                })

            h_dict = self.convs[layer_idx](h_dict, edge_index_dict)
            if layer_idx < self.num_layers - 1:
                h_dict = {k: self.dropout(self.act(v)) for k, v in h_dict.items()}

        logits_card = self.card_classifier(h_dict["card"])
        return logits_card, h_dict

