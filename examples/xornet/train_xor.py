#!/usr/bin/env python3
"""Ultra-minimal XOR training - every line uses code you built!"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tinytorch.nn as nn
import tinytorch.nn.functional as F  
import tinytorch.optim as optim
from tinytorch.core.tensor import Tensor
from tinytorch.core.training import MeanSquaredError

# XOR network - you built every component!
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)  # You built Linear!
        self.output = nn.Linear(4, 1)  # You built Linear!
    
    def forward(self, x):
        x = F.relu(self.hidden(x))     # You built ReLU!
        return self.output(x)

# XOR data
X = Tensor(np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32))
y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float32))

# Training setup - you built everything!
model = XORNet()
optimizer = optim.SGD(model.parameters(), learning_rate=0.1)  # You built SGD!
loss_fn = MeanSquaredError()                                  # You built MSE!

# Training loop - you built every operation!
for epoch in range(1000):
    inputs = X     # Data tensors don't need gradients
    targets = y    # Labels never need gradients 
    
    outputs = model(inputs)           # You built forward pass!
    loss = loss_fn(outputs, targets)  # You built MSE loss!
    
    loss.backward()                   # You built backprop!
    optimizer.step()                  # You built parameter updates!
    optimizer.zero_grad()             # You built gradient clearing!
    
    if epoch % 200 == 0:
        loss_val = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

# Test - you built inference!
print("\nXOR Results:")
for i in range(4):
    test_input = Tensor(X.data[i:i+1])  # You built Tensor!
    prediction = model(test_input)
    print(f"{X.data[i]} -> {prediction.data[0,0]:.3f} (target: {y.data[i,0]})")