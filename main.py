import torch
from data_generation import generate_data
from model import IGAP
from training import train
from visualization import visualize

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