
from loss import custom_loss

def train(model, data, optimizer, epochs=200, alpha=1.0, beta=1.0):
    """
    The main training loop for the I-GAP model.
    """
    model.train()
    print("Starting training...")
    print(f"Hyperparameters: ALPHA={alpha}, BETA={beta}")
    print("-" * 40)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass
        output = model(data.x, data.edge_index, data.edge_attr)
        
        # 2. Loss Calculation
        loss, edge_cut_loss, load_balance_loss = custom_loss(output, data, alpha=alpha, beta=beta)
        
        # 3. Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: Total Loss: {loss.item():.4f} | "
                  f"Edge Cut: {edge_cut_loss.item():.4f} | "
                  f"Load Balance: {load_balance_loss.item():.4f}")
