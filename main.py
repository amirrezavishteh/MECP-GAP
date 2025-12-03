
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# Step 1: Data Generation (Simulating the RAN)
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

# Step 2: Model Architecture (Designing I-GAP)
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

# Step 3: The Custom Loss Function
def custom_loss(output, data, alpha=1.0, beta=1.0):
    """
    Implements the custom loss function from the paper (Equation 9) using vectorized operations.
    Loss = alpha * EdgeCut_Loss + beta * LoadBalance_Loss
    """
    # 1. Load Balancing Loss (Vectorized)
    # Eq 8: sum((sum(x_ik) - N/P)^2)
    partition_sizes = torch.sum(output, dim=0) # (P)
    target_size = data.num_nodes / output.size(1)
    load_balance_loss = torch.sum((partition_sizes - target_size) ** 2)

    # 2. Edge Cut Loss (Vectorized)
    # Eq 7: sum( X * (1-X)^T . W )
    # We want to penalize edges (u,v) where output[u] and output[v] are DIFFERENT.
    # Similarity = output[u] dot output[v].
    # Loss contribution = weight * (1 - Similarity)

    # Get source and target nodes from edge_index
    u, v = data.edge_index

    # Calculate dot product for every edge pair efficiently
    # (Edges, P) * (Edges, P) -> sum over P -> (Edges)
    # Gather the probability vectors for all source nodes u and target nodes v
    prob_u = output[u]
    prob_v = output[v]

    # Dot product: sum(u_k * v_k)
    similarity = torch.sum(prob_u * prob_v, dim=1)

    # Total Edge Cut Loss = Sum( weight_uv * (1 - similarity_uv) )
    # Ensure edge_attr is squeezed to shape (Edges,)
    weights = data.edge_attr.squeeze()
    edge_cut_loss = torch.sum(weights * (1 - similarity))

    return alpha * edge_cut_loss + beta * load_balance_loss

# Step 4: The Training Loop
def train(model, data, optimizer, epochs=200):
    """
    The main training loop for the I-GAP model.
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass
        output = model(data.x, data.edge_index, data.edge_attr)
        
        # 2. Loss Calculation
        loss = custom_loss(output, data)
        
        # 3. Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

# Step 5: Visualization & Inference
def visualize(model, data):
    """
    Visualizes the graph partitioning results.
    """
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_attr)
        pred = output.argmax(dim=1)

    plt.figure(figsize=(10, 10))
    plt.scatter(data.x[:, 0], data.x[:, 1], c=pred, cmap='viridis', s=50)
    
    # Plot edges
    for i in range(data.edge_index.size(1)):
        u, v = data.edge_index[0, i], data.edge_index[1, i]
        plt.plot([data.x[u, 0], data.x[v, 0]], [data.x[u, 1], data.x[v, 1]], 'k-', alpha=0.3)
        
    plt.title("MECP-GAP Partitioning Results")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()

# Main execution block
if __name__ == '__main__':
    # --- Parameters ---
    NUM_NODES = 200
    GRID_SIZE = 100
    NUM_PARTITIONS = 4  # Number of MEC servers
    HIDDEN_CHANNELS = 16
    LEARNING_RATE = 0.01
    EPOCHS = 200

    # 1. Generate Data
    data = generate_data(num_nodes=NUM_NODES, grid_size=GRID_SIZE)

    # 2. Initialize Model
    model = IGAP(in_channels=data.num_node_features, 
                 hidden_channels=HIDDEN_CHANNELS, 
                 out_channels=NUM_PARTITIONS)

    # 3. Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train the model
    train(model, data, optimizer, epochs=EPOCHS)

    # 5. Visualize the results
    visualize(model, data)

