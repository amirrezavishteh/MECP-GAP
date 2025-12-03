import torch
from data_generation import generate_data
from model import IGAP
from training import train
from visualization import visualize
from verification import evaluate_metrics, compare_with_random

# Main execution block
if __name__ == '__main__':
    # --- Parameters from Paper ---
    NUM_NODES = 200       # Section V-C
    GRID_SIZE = 100       # Arbitrary, but fine
    NUM_PARTITIONS = 4
    HIDDEN_CHANNELS = 128 # Section V-A (Embedding Dim)
    LEARNING_RATE = 0.001  # Lowered to address instability
    EPOCHS = 200          # Section V-C (Convergence shown ~50 epochs)
    
    # Loss weights from Section V-C
    ALPHA = 0.001
    BETA = 1.0

    # 1. Generate Data
    data = generate_data(num_nodes=NUM_NODES, grid_size=GRID_SIZE)

    # 2. Initialize Model
    model = IGAP(in_channels=data.num_node_features,
                 hidden_channels=HIDDEN_CHANNELS,
                 out_channels=NUM_PARTITIONS)

    # 3. Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train the model
    train(model, data, optimizer, epochs=EPOCHS, alpha=ALPHA, beta=BETA)

    # 5. Verify and evaluate the results
    print("\n--- Verifying final results ---")
    evaluate_metrics(model, data)
    print("\n--- Comparing with random baseline ---")
    compare_with_random(data, NUM_PARTITIONS)


    # 6. Visualize the results
    visualize(model, data)