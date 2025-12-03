
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class IGAP(torch.nn.Module):
    """
    The I-GAP model, consisting of a GCN encoder and a simple MLP decoder.
    This version incorporates edge weights and a feature projection layer.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(IGAP, self).__init__()
        # Project 2D coords to a higher dimensional feature space
        self.feature_proj = torch.nn.Linear(in_channels, 128)
        
        # GCN layers that support edge weights
        self.conv1 = GCNConv(128, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Partitioner (Decoder)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # Project features
        x = F.relu(self.feature_proj(x))
        
        # Encoder with edge weights
        x = self.conv1(x, edge_index, edge_weight.squeeze())
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight.squeeze())
        x = F.relu(x)

        # Decoder
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.softmax(x, dim=-1)
