# MECP-GAP: Graph Neural Network for Mobile Edge Computing Placement

This project implements the methodology from the "Mobility-Aware MEC Placement in Vehicular Networks" paper (or similar concepts), using a Graph Neural Network (GNN) to solve the complex task of partitioning a cellular network into optimal regions for Mobile Edge Computing (MEC) servers.

The primary goal is to **minimize handover costs** (Edge Cut) while ensuring that no single server is overloaded (Load Balancing).

## How It Works

The problem is modeled as a graph partitioning task. The network of cellular base stations is converted into a graph, and a GNN is trained to assign each base station (node) to a specific MEC server (partition).

### 1. File Structure
- **`main.py`**: The central script that runs the entire pipeline.
- **`data_generation.py`**: Creates a simulated cellular network graph using Delaunay triangulation to connect nearby nodes.
- **`model.py`**: Defines the `IGAP` GNN architecture, which uses `GraphSAGE` layers to learn node embeddings.
- **`loss.py`**: Contains the crucial `custom_loss` function.
- **`training.py`**: Implements the model training loop.
- **`verification.py`**: Calculates the final performance metrics after training.
- **`visualization.py`**: Generates a visual plot of the final partitions.

### 2. The Loss Function

The model's behavior is guided by a custom loss function with two competing objectives, controlled by the `ALPHA` and `BETA` hyperparameters:

- **`Edge Cut Loss` (controlled by `ALPHA`)**: This loss penalizes the model for assigning two well-connected nodes to *different* partitions. Minimizing this is the primary goal, as it directly reduces the handover traffic between servers.

- **`Load Balance Loss` (controlled by `BETA`)**: This loss penalizes the model for creating partitions of unequal size. It ensures that the workload is distributed fairly across all MEC servers, preventing any single server from becoming a bottleneck.

The final loss is a weighted sum: `Total Loss = ALPHA * EdgeCut_Loss + BETA * LoadBalance_Loss`.

## How to Run

### 1. Installation
First, ensure you have a Python environment. Then, install the required packages.

```bash
# Clone the repository (if you haven't already)
git clone <repository_url>
cd <repository_directory>

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution
Run the main script from your terminal:

```bash
python main.py
```

The script will execute the full pipeline: generate data, train the model, verify the results, and save a visualization.

## Understanding the Output

The script produces three key outputs: the Training Log, the Verification Report, and a visualization image.

### 1. Training Log
During training, a log is printed every 20 epochs. This shows the live values of the loss components.

```
Epoch 000: Total Loss: 5669.3423 | Edge Cut: 126.0868 | Load Balance: 5669.2163
...
Epoch 180: Total Loss: 0.1583 | Edge Cut: 158.2757 | Load Balance: 0.0000
```
- **Edge Cut**: The unweighted loss from inter-partition connections.
- **Load Balance**: The unweighted loss from partition size imbalance.

Notice in the example above, the `Load Balance` loss quickly drops to zero, while the `Edge Cut` loss stays high. This indicates the model is prioritizing balancing the partitions perfectly at the expense of creating costly edge cuts. This is a sign that the `ALPHA` and `BETA` weights may need tuning.

### 2. Verification Report
After training, a final report card for the model is printed.

```
========================================
      MECP-GAP VERIFICATION REPORT
========================================
Total Nodes: 200
Total Edges: 577
----------------------------------------
1. Edge Cut Cost (Lower is better):  28.4069
2. Load Balance Score (Lower is better): 2778.0000
----------------------------------------
Partition Distribution (Ideally equal):
Counts: [60.0, 63.0, 72.0, 5.0]
Ideal:  50.0
========================================
```
- **Edge Cut Cost**: The final, real "handover cost". This is the sum of weights of all edges connecting different partitions. **A lower score is better.**
- **Load Balance Score**: A measure of fairness. A score of `0.0` would mean every server has the exact same number of nodes. **A lower score is better.** In the example, the score of `2778.0` is very high because the partitions are unbalanced (`[60, 63, 72, 5]`), far from the ideal of `50` nodes per partition.

### 3. Random Baseline
```
--- Comparing with random baseline ---
Random Baseline Edge Cut: 75.1567
```
This number shows the Edge Cut cost you would get from a completely random assignment. Your model's `Edge Cut Cost` (`28.4069` in this case) should be **significantly lower** than the random baseline. If it is, your model has successfully learned to create meaningful, low-cost partitions.

### 4. Visualization
The script generates a file named `partition_visualization.png`. This image shows the graph, where each node is colored according to its assigned MEC server. This provides a clear, visual confirmation of the partitions described in the verification report.

## Customization & Tuning
You can modify the simulation and model parameters in the `if __name__ == '__main__':` block of `main.py`.

The most important parameters for tuning are `ALPHA` and `BETA`.

- **To prioritize minimizing handover cost (Edge Cut)**: Increase `ALPHA` relative to `BETA`. For example, `ALPHA = 0.1`, `BETA = 1.0`.
- **To prioritize server fairness (Load Balance)**: Increase `BETA` relative to `ALPHA`. For example, `ALPHA = 0.001`, `BETA = 1.0`.

The poor load balance in the example output (`Load Balance Score: 2778.0`) suggests that the balance between `ALPHA` and `BETA` is not optimal. The model found it easy to reduce the `LoadBalance` term to zero during training, even though the final `Load Balance Score` was high. This is because the loss is calculated on *soft* assignments (probabilities), while the final score is on *hard* assignments (the final decision). A slight adjustment to the hyperparameters may be needed to achieve a better trade-off.
