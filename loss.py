
import torch

def custom_loss(output, data, alpha=1.0, beta=1.0):
    """
    Implements the custom loss function from the paper (Equation 9) using vectorized operations.
    Loss = alpha * EdgeCut_Loss + beta * LoadBalance_Loss
    
    Returns the total loss, and the unweighted individual loss components.
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

    total_loss = alpha * edge_cut_loss + beta * load_balance_loss
    
    return total_loss, edge_cut_loss, load_balance_loss
