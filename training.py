
from loss import custom_loss

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
