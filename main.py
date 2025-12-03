
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# Step 1: Data Generation (Simulating the RAN)
def generate_data(num_nodes=100, grid_size=10):
    """
    Generates a simulated Radio Access Network (RAN) using Delaunay triangulation.
    """
    # 1. Nodes (gNBs): Generate random (x, y) coordinates
    node_coords = np.random.rand(num_nodes, 2) * grid_size

    # 2. Edges (Connections): Use Delaunay Triangulation
    tri = scipy.spatial.Delaunay(node_coords)
    edges = np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
    
    # Create a symmetric adjacency list
    edge_index = torch.tensor(np.sort(edges), dtype=torch.long).t().contiguous()

    # 3. Weights (Handover Traffic): Generate random weights for edges
    edge_attr = torch.rand(edge_index.size(1), 1)

    # 4. Packaging: Convert to PyG Data object
    x = torch.tensor(node_coords, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

# Step 2: Model Architecture (Designing I-GAP)
class IGAP(torch.nn.Module):
    """
    The I-GAP model, consisting of a GraphSAGE encoder and a simple MLP decoder.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(IGAP, self).__init__()
        # Part A: The Graph Embedder (Encoder)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Part B: The Partitioner (Decoder)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Encoder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Decoder
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.softmax(x, dim=-1)

# Step 3: The Custom Loss Function
def custom_loss(output, data, alpha=1.0, beta=1.0):
    """
    Implements the custom loss function from the paper (Equation 9).
    Loss = alpha * EdgeCut_Loss + beta * LoadBalance_Loss
    """
    # 1. Calculate Edge Cut Loss
    edge_cut_loss = 0
    for i in range(data.edge_index.size(1)):
        u, v = data.edge_index[0, i], data.edge_index[1, i]
        # Penalize if connected nodes are in different partitions
        edge_cut_loss += torch.sum(1 - torch.sum(output[u] * output[v])) * data.edge_attr[i]
        
    # 2. Calculate Load Balancing Loss
    partition_sizes = torch.sum(output, dim=0)
    mean_size = torch.mean(partition_sizes)
    load_balance_loss = torch.sum((partition_sizes - mean_size)**2)

    # 3. Combine
    total_loss = (alpha * edge_cut_loss) + (beta * load_balance_loss)
    
    return total_loss

# Step 4: The Training Loop
def train(model, data, optimizer, epochs=200):
    """
    The main training loop for the I-GAP model.
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass
        output = model(data.x, data.edge_index)
        
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
        output = model(data.x, data.edge_index)
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

