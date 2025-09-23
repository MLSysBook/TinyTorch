#!/usr/bin/env python3
"""
Modern CIFAR-10 CNN Training Example - PyTorch-like API

This example demonstrates the clean PyTorch-compatible API introduced in 
TinyTorch's simplification. Students implement core algorithms while using
familiar, professional interfaces.

Compare this file to train_working_cnn.py to see the dramatic simplification!
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Clean PyTorch-like imports
import tinytorch.nn as nn
import tinytorch.nn.functional as F
import tinytorch.optim as optim
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

class ModernCNN(nn.Module):
    """
    CNN using modern PyTorch-like API.
    
    Notice how clean this is compared to the old API:
    - Inherits from nn.Module (automatic parameter registration)
    - Uses nn.Conv2d and nn.Linear (familiar naming)
    - Uses F.relu and F.flatten (functional interface)
    - __call__ method works automatically
    """
    
    def __init__(self):
        super().__init__()  # Initialize Module base class
        print("🏗️ Initializing modern CNN architecture...")
        
        # Layers are automatically registered as parameters!
        self.conv1 = nn.Conv2d(3, 32, (3, 3))    # You built this convolution!
        self.conv2 = nn.Conv2d(32, 64, (3, 3))   # Multi-channel convolution
        self.fc1 = nn.Linear(2304, 128)          # You built this dense layer!
        self.fc2 = nn.Linear(128, 10)            # Output layer
        
        print("✅ Modern CNN initialized - parameters auto-registered!")
        print(f"📊 Total parameters: {len(list(self.parameters()))}")
    
    def forward(self, x):
        """Forward pass using functional interface."""
        # First conv block: Conv -> ReLU -> Pool
        x = F.relu(self.conv1(x))        # Your convolution + activation
        x = F.max_pool2d(x, (2, 2))      # Pooling operation
        
        # Second conv block: Conv -> ReLU -> Pool  
        x = F.relu(self.conv2(x))        # More feature extraction
        x = F.max_pool2d(x, (2, 2))      # More downsampling
        
        # Classifier: Flatten -> Linear -> ReLU -> Linear
        x = F.flatten(x)                 # Flatten for dense layers
        x = F.relu(self.fc1(x))          # Hidden layer
        x = self.fc2(x)                  # Output logits
        
        return x

def train_modern_cnn():
    """Train CNN using modern PyTorch-like API."""
    print("🚀 Training CIFAR-10 CNN with Modern API")
    print("=" * 50)
    
    # Create model - notice the clean instantiation
    model = ModernCNN()
    
    # Create optimizer - automatic parameter collection!
    optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
    criterion = CrossEntropyLoss()
    
    # Load data
    print("📦 Loading CIFAR-10 data...")
    train_dataset = CIFAR10Dataset(train=True, download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    print("🏃 Starting training...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Convert to Variables for gradient tracking
            if not isinstance(data, Variable):
                data = Variable(data, requires_grad=False)
            if not isinstance(targets, Variable):
                targets = Variable(targets, requires_grad=False)
            
            # Forward pass - clean and simple!
            outputs = model(data)  # model(x) calls model.forward(x)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Statistics - extract scalar value from Variable -> Tensor -> numpy scalar
            if hasattr(loss.data, 'data'):
                # loss.data is a Tensor, so get its numpy data
                loss_value = loss.data.data.item() if hasattr(loss.data.data, 'item') else float(loss.data.data)
            else:
                # loss.data is already numpy
                loss_value = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            epoch_loss += loss_value
            
            # Calculate accuracy - extract data from Variable -> Tensor -> numpy array
            if hasattr(outputs.data, 'data'):
                output_data = outputs.data.data
            else:
                output_data = outputs.data
            predicted = np.argmax(output_data, axis=1)
            if hasattr(targets.data, 'data'):
                target_data = targets.data.data
            else:
                target_data = targets.data
            epoch_correct += np.sum(predicted == target_data)
            epoch_total += len(target_data)
            
            # Progress update
            if batch_idx % 50 == 0:
                accuracy = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0.0
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {epoch_loss/(batch_idx+1):.4f}, "
                      f"Accuracy: {accuracy:.2f}%")
        
        # Epoch summary
        epoch_time = time.time() - start_time
        epoch_accuracy = 100. * epoch_correct / epoch_total
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"📊 Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print("-" * 50)
    
    print("🎉 Training completed!")
    return model

def compare_apis():
    """Show the difference between old and new APIs."""
    print("🔍 API Comparison:")
    print("=" * 60)
    
    print("❌ OLD API (Complex):")
    print("from tinytorch.core.layers import Dense")
    print("from tinytorch.core.spatial import MultiChannelConv2D")  
    print("sys.path.insert(0, 'modules/source/06_spatial')")
    print("from spatial_dev import flatten, MaxPool2D")
    print("# Manual parameter management...")
    print("# Manual weight initialization...")
    print("# No automatic registration...")
    print()
    
    print("✅ NEW API (Clean):")
    print("import tinytorch.nn as nn")
    print("import tinytorch.nn.functional as F")
    print("import tinytorch.optim as optim")
    print()
    print("class CNN(nn.Module):")
    print("    def __init__(self):")
    print("        super().__init__()")
    print("        self.conv1 = nn.Conv2d(3, 32, (3, 3))  # Auto-registered!")
    print("        self.fc1 = nn.Linear(800, 10)          # Auto-registered!")
    print("    ")
    print("    def forward(self, x):")
    print("        x = F.relu(self.conv1(x))")
    print("        x = F.flatten(x)")
    print("        return self.fc1(x)")
    print()
    print("model = CNN()")
    print("optimizer = optim.Adam(model.parameters())  # Auto-collected!")

if __name__ == "__main__":
    print("🔥 TinyTorch Modern API Example")
    print("Building real ML systems with clean, familiar interfaces")
    print()
    
    # Show API comparison
    compare_apis()
    print()
    
    # Train the model
    try:
        model = train_modern_cnn()
        print("🎯 Success! Your CNN implementation works with PyTorch-like API!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 This shows where the implementation needs completion.")
    
    print()
    print("🎓 Educational Value:")
    print("- You implemented Conv2d, Linear, ReLU, Adam optimizer")
    print("- Infrastructure provides clean PyTorch-compatible API")
    print("- Focus on algorithms, not boilerplate!")
    print("- Professional development patterns from day one")