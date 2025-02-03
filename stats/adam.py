import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple dataset (10 features, 1 output)
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 1)   # 100 target values

# Define a simple linear model
model = nn.Linear(10, 1)  

# Define the loss function (Mean Squared Error for regression)
criterion = nn.MSELoss()

# Define Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()  # Reset gradients
    y_pred = model(X)  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')
