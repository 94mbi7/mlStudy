import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(100, 10)  
y = torch.randn(100, 1)  

model = nn.Linear(10, 1)  

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()  
    y_pred = model(X)  
    loss = criterion(y_pred, y) 
    loss.backward() 
    optimizer.step() 

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')
