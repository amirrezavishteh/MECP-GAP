
import torch
import numpy as np
import scipy.spatial
from torch_geometric.data import Data

def generate_data(num_nodes=100, grid_size=10):
    """
    Generates a simulated Radio Access Network (RAN) using Delaunay triangulation
    and a gravity model for edge weights.
    """
    # 1. Nodes (gNBs): Generate random (x, y) coordinates
    node_coords = np.random.rand(num_nodes, 2) * grid_size
    x = torch.tensor(node_coords, dtype=torch.float)

    # 2. Edges (Connections): Use Delaunay Triangulation
    tri = scipy.spatial.Delaunay(node_coords)
    edges = np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
    edge_index = torch.tensor(np.sort(edges), dtype=torch.long).t().contiguous()

    # 3. Weights (Handover Traffic): Use an inverse distance gravity model
    row, col = edge_index
    dist = torch.norm(x[row] - x[col], p=2, dim=1)
    edge_attr = 1.0 / (dist + 1e-5) # Add epsilon to avoid division by zero
    edge_attr = edge_attr.view(-1, 1)

    # 4. Packaging: Convert to PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data
