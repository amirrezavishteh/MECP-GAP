# MECP-GAP Implementation

This project provides a complete implementation of the MECP-GAP (Mobile Edge Computing Placement - Graph Attention Partitioning) article's methodology using Python and PyTorch Geometric. It simulates a Radio Access Network (RAN) and partitions it into optimal Mobile Edge Computing (MEC) server regions, aiming to balance load and minimize handover traffic.

## Introduction

The MECP-GAP approach leverages Graph Neural Networks (GNNs) to solve the NP-hard problem of MEC server placement in cellular networks. By representing the RAN as a graph, where base stations are nodes and connections represent user traffic, the model learns an optimal partitioning strategy.

## Key Features

*   **Simulated RAN Data Generation:** Generates synthetic base station coordinates and connections using Delaunay Triangulation, with random edge weights simulating handover traffic.
*   **I-GAP Model Architecture:** Implements a hybrid GNN model consisting of:
    *   **GraphSAGE Encoder:** Learns contextual embeddings for each base station based on spatial coordinates and weighted neighbors.
    *   **MLP Decoder:** Translates embeddings into a probability distribution for MEC server assignments using a Softmax activation.
*   **Custom Loss Function:** Implements a unique loss function (Equation 9 from the paper) that combines:
    *   **Edge Cut Loss:** Penalizes partitioning strongly connected base stations into different MEC regions.
    *   **Load Balancing Loss:** Ensures an equitable distribution of base stations (and thus traffic) across all MEC servers.
*   **Training Loop:** Standard PyTorch training pipeline with an Adam optimizer.
*   **Visualization:** Plots the simulated RAN with base stations colored according to their assigned MEC server, demonstrating the learned partitions.

## Technologies Used

*   **Python 3.x**
*   **PyTorch:** Core deep learning framework.
*   **PyTorch Geometric (PyG):** For efficient graph data handling and GNN layers (e.g., `SAGEConv`).
*   **NumPy:** Numerical operations, especially for coordinate generation.
*   **SciPy:** Specifically `scipy.spatial.Delaunay` for realistic network topology generation.
*   **Matplotlib:** For visualizing the partitioning results.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MECP-GAP-Implementation.git
    cd MECP-GAP-Implementation
    ```
    *(Note: Replace `https://github.com/your-username/MECP-GAP-Implementation.git` with the actual repository URL if this project is hosted.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    source venv/bin/activate # On macOS/Linux
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the MECP-GAP simulation and observe the partitioning:

```bash
python main.py
```

The script will print the training loss for every 20 epochs. After training, a Matplotlib window will pop up, displaying a scatter plot of the simulated base stations. Each base station will be colored according to the MEC server it has been assigned to, with connecting lines representing the network edges.

## Code Structure

The project is organized into a modular structure to improve readability and maintainability:

*   **`main.py`**: The main entry point of the application. It orchestrates the entire pipeline by importing modules, setting parameters, and calling the necessary functions in order.
*   **`data_generation.py`**: Contains the `generate_data` function, which is responsible for creating the simulated Radio Access Network (RAN) graph data, including node coordinates, edges, and traffic-based edge weights.
*   **`model.py`**: Defines the `IGAP` GNN model class. This includes the feature projection layer, the GCN-based encoder, and the MLP-based decoder.
*   **`loss.py`**: Contains the `custom_loss` function, which implements the vectorized, mobility-aware loss function combining Edge Cut and Load Balancing penalties.
*   **`training.py`**: Contains the `train` function, which handles the model training loop, including the forward pass, loss calculation, and backpropagation.
*   **`visualization.py`**: Contains the `visualize` function, responsible for running the model in evaluation mode and generating the final plot of the partitioned graph using Matplotlib.

## Results

You should expect to see the simulated grid of base stations partitioned into distinct, colored regions. Ideally, the boundaries between these regions will occur in areas with fewer connections or lower handover traffic, and the regions themselves will be roughly balanced in terms of the number of assigned base stations.

## Customization

You can modify the simulation parameters by editing the `if __name__ == '__main__':` block in `main.py`:

*   `NUM_NODES`: Number of base stations to simulate.
*   `GRID_SIZE`: The spatial extent of the simulation area.
*   `NUM_PARTITIONS`: The number of MEC servers (i.e., the number of partitions).
*   `HIDDEN_CHANNELS`: Dimension of the hidden layers in the GNN.
*   `LEARNING_RATE`: Learning rate for the Adam optimizer.
*   `EPOCHS`: Number of training epochs.
*   `alpha`, `beta`: Weights for the Edge Cut and Load Balancing loss components (currently hardcoded within `custom_loss`, but can be exposed as parameters).

Feel free to experiment with these parameters to observe their effect on the partitioning results.
