import torch
import numpy as np

def evaluate_metrics(model, data):
    """
    Calculates the Edge Cut and Load Balance metrics from the paper.
    """
    model.eval()
    with torch.no_grad():
        # Get model probabilities and final hard partition labels
        output = model(data.x, data.edge_index, data.edge_attr)
        pred_labels = output.argmax(dim=1)

    # --- Metric 1: Edge Cut (Minimizing Handover Cost) ---
    # Formula (Eq 6): Sum of weights of edges connecting different partitions.
    row, col = data.edge_index
    weights = data.edge_attr.squeeze()

    # Find edges where the source node and target node are in DIFFERENT partitions
    mask_cut = pred_labels[row] != pred_labels[col]
    
    # Sum the weights of these cut edges
    # Note: PyG represents undirected graphs with bidirectional edges (u->v and v->u).
    # We divide by 2 to avoid double-counting the same physical edge.
    edge_cut_score = torch.sum(weights[mask_cut]).item() / 2

    # --- Metric 2: Load Balancing (Server Fairness) ---
    # Formula (Eq 2): Variance-like measure: sum((N_k - N/P)^2)
    num_partitions = output.size(1)
    
    # Count how many nodes are in each partition
    # bincount gives us [Count_Partition0, Count_Partition1, ...]
    partition_counts = torch.bincount(pred_labels, minlength=num_partitions).float()
    
    # The ideal size (N / P)
    target_size = data.num_nodes / num_partitions
    
    # Calculate the variance metric from the paper
    load_balance_score = torch.sum((partition_counts - target_size)**2).item()

    # --- Print Report ---
    print("\n" + "="*40)
    print("      MECP-GAP VERIFICATION REPORT      ")
    print("="*40)
    print(f"Total Nodes: {data.num_nodes}")
    print(f"Total Edges: {data.edge_index.size(1) // 2}")
    print("-" * 40)
    print(f"1. Edge Cut Cost (Lower is better):  {edge_cut_score:.4f}")
    print(f"2. Load Balance Score (Lower is better): {load_balance_score:.4f}")
    print("-" * 40)
    print("Partition Distribution (Ideally equal):")
    print(f"Counts: {partition_counts.tolist()}")
    print(f"Ideal:  {target_size}")
    print("="*40 + "\n")

    return edge_cut_score, load_balance_score

def compare_with_random(data, num_partitions):
    """Calculates metrics for a completely random partition."""
    # Generate random labels
    random_labels = torch.randint(0, num_partitions, (data.num_nodes,))
    
    # Calculate Random Edge Cut
    row, col = data.edge_index
    weights = data.edge_attr.squeeze()
    mask_cut = random_labels[row] != random_labels[col]
    random_cut = torch.sum(weights[mask_cut]).item() / 2
    
    print(f"Random Baseline Edge Cut: {random_cut:.4f}")
    return random_cut
