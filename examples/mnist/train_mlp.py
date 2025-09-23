#!/usr/bin/env python3
"""Ultra-minimal MNIST MLP - every line uses code you built!"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tinytorch.nn as nn
import tinytorch.nn.functional as F
import tinytorch.optim as optim
from tinytorch.core.tensor import Tensor
from tinytorch.core.training import CrossEntropyLoss

# MNIST MLP - you built every component!
class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # You built Linear!
        self.fc2 = nn.Linear(128, 64)   # You built Linear!
        self.fc3 = nn.Linear(64, 10)    # You built Linear!
    
    def forward(self, x):
        x = F.flatten(x, start_dim=1)   # You built flatten!
        x = F.relu(self.fc1(x))         # You built ReLU!
        x = F.relu(self.fc2(x))         # You built ReLU!
        return self.fc3(x)

# Sample MNIST-like data (28x28 images, 10 classes)
batch_size, num_samples = 32, 1000
X = np.random.randn(num_samples, 28, 28).astype(np.float32)
y = np.random.randint(0, 10, (num_samples,)).astype(np.int64)

# Training setup - you built everything!
model = MNIST_MLP()
optimizer = optim.Adam(model.parameters(), learning_rate=0.001)  # You built Adam!
loss_fn = CrossEntropyLoss()                                     # You built CrossEntropy!

print("Training MNIST MLP...")
# Training loop - you built every operation!
for epoch in range(20):
    total_loss = 0
    for i in range(0, num_samples, batch_size):
        # Get batch
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        inputs = Tensor(batch_X)    # You built Tensor!
        targets = Tensor(batch_y)   # You built Tensor!
        
        outputs = model(inputs)               # You built forward pass!
        loss = loss_fn(outputs, targets)      # You built CrossEntropy!
        
        loss.backward()                       # You built backprop!
        optimizer.step()                      # You built Adam updates!
        optimizer.zero_grad()                 # You built gradient clearing!
        
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
    
    print(f"Epoch {epoch+1}: Avg Loss = {total_loss/(num_samples//batch_size):.4f}")

print("✅ MNIST MLP trained successfully!")