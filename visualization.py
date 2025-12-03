
import torch
import matplotlib.pyplot as plt

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
