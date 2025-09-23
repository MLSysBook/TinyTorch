#!/usr/bin/env python3
"""Ultra-minimal CIFAR-10 CNN - every line uses code you built!"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tinytorch.nn as nn
import tinytorch.nn.functional as F
import tinytorch.optim as optim
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

# CIFAR-10 CNN - you built every component!
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)    # You built Conv2d!
        self.conv2 = nn.Conv2d(32, 64, 5)   # You built Conv2d!
        self.fc1 = nn.Linear(1600, 256)     # You built Linear!
        self.fc2 = nn.Linear(256, 10)       # You built Linear!
    
    def forward(self, x):
        x = F.relu(self.conv1(x))           # You built ReLU + Conv2d!
        x = F.max_pool2d(x, 2)              # You built max_pool2d!
        x = F.relu(self.conv2(x))           # You built ReLU + Conv2d!
        x = F.max_pool2d(x, 2)              # You built max_pool2d!
        x = F.flatten(x, start_dim=1)       # You built flatten!
        x = F.relu(self.fc1(x))             # You built ReLU + Linear!
        return self.fc2(x)

# Real CIFAR-10 data using DataLoader you built!
train_dataset = CIFAR10Dataset(train=True)        # You built CIFAR10Dataset!
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # You built DataLoader!

# Training setup - you built everything!
model = CIFAR_CNN()
optimizer = optim.Adam(model.parameters(), learning_rate=0.001)  # You built Adam!
loss_fn = CrossEntropyLoss()                                     # You built CrossEntropy!

print("Training CIFAR-10 CNN on real data...")
# Training loop - you built every operation!
for epoch in range(10):
    total_loss = 0
    batch_count = 0
    
    for batch_X, batch_y in train_loader:        # You built DataLoader iteration!
        # DataLoader returns Tensors ready to use
        outputs = model(batch_X)                 # You built forward pass!
        loss = loss_fn(outputs, batch_y)         # You built CrossEntropy!
        
        loss.backward()                          # You built backprop through CNN!
        optimizer.step()                         # You built Adam updates!
        optimizer.zero_grad()                    # You built gradient clearing!
        
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        batch_count += 1
        
        if batch_count >= 50:  # Train on subset for demo
            break
    
    print(f"Epoch {epoch+1}: Avg Loss = {total_loss/batch_count:.4f}")

print("✅ CIFAR-10 CNN trained successfully!")