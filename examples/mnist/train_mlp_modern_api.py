#!/usr/bin/env python3
"""
MNIST MLP Training with Modern PyTorch-like API

This example demonstrates training a simple Multi-Layer Perceptron (MLP) 
on MNIST digits using TinyTorch's clean, modern API that mirrors PyTorch.

Students learn the fundamentals of neural networks with professional patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import tinytorch.nn as nn
import tinytorch.nn.functional as F
import tinytorch.optim as optim
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.training import CrossEntropyLoss

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST classification.
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden1: 128 neurons with ReLU
    - Hidden2: 64 neurons with ReLU  
    - Output: 10 classes (digits 0-9)
    """
    
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        # Flatten input if needed: (batch, 28, 28) -> (batch, 784)
        x = F.flatten(x, start_dim=1)
        
        # Forward pass through layers
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)  # No activation here - CrossEntropy handles it
        return x

def create_sample_mnist_data():
    """Create sample MNIST-like data for demonstration."""
    print("📊 Creating sample MNIST data...")
    
    # Create simple synthetic data that mimics MNIST
    # In real use, you'd load actual MNIST data
    batch_size = 32
    X = np.random.randn(batch_size, 784).astype(np.float32) * 0.1
    y = np.random.randint(0, 10, batch_size).astype(np.int64)
    
    print("✅ Sample MNIST data created")
    print(f"   Input shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Label range: {y.min()}-{y.max()}")
    
    return X, y

def train_mlp():
    """Train MLP using modern API."""
    print("🚀 Training MNIST MLP with Modern API")
    print("=" * 50)
    
    # Create model and optimizer - notice how clean this is!
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), learning_rate=0.001)
    criterion = CrossEntropyLoss()
    
    print(f"🧠 Created MLP with {len(list(model.parameters()))} parameter tensors")
    
    # Create sample data (in real use, load actual MNIST)
    X, y = create_sample_mnist_data()
    
    # Training loop
    print("🏃 Starting training...")
    num_epochs = 50
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Mini-batch training (process all data as one batch for simplicity)
        inputs = Variable(Tensor(X), requires_grad=False)
        targets = Variable(Tensor(y.astype(np.float32)), requires_grad=False)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate accuracy
        if hasattr(outputs.data, 'data'):
            output_data = outputs.data.data
        else:
            output_data = outputs.data
        predicted = np.argmax(output_data, axis=1)
        
        if hasattr(targets.data, 'data'):
            target_data = targets.data.data
        else:
            target_data = targets.data
        
        correct = np.sum(predicted == target_data.astype(np.int64))
        total = len(target_data)
        accuracy = 100. * correct / total
        
        # Extract loss value
        if hasattr(loss.data, 'data'):
            loss_value = loss.data.data.item() if hasattr(loss.data.data, 'item') else float(loss.data.data)
        else:
            loss_value = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        
        # Progress update
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs}, Loss: {loss_value:.4f}, Accuracy: {accuracy:.1f}%")
    
    print("✅ Training completed!")
    
    # Final test
    print("\n🧪 Testing MLP")
    print("=" * 30)
    
    # Test on the same data (in real use, use separate test set)
    test_output = model(inputs)
    if hasattr(test_output.data, 'data'):
        test_predictions = np.argmax(test_output.data.data, axis=1)
    else:
        test_predictions = np.argmax(test_output.data, axis=1)
    
    test_accuracy = 100. * np.sum(test_predictions == target_data.astype(np.int64)) / len(target_data)
    print(f"📊 Final Test Accuracy: {test_accuracy:.1f}%")
    
    print("\n✨ Key Insight: Clean APIs don't reduce educational value!")
    print("   Students still implement core algorithms while using professional patterns.")

def show_api_comparison():
    """Show side-by-side API comparison."""
    print("🔍 API Comparison - MNIST MLP")
    print("=" * 50)
    print("❌ OLD API:")
    print("from tinytorch.core.layers import Dense")
    print("from tinytorch.core.activations import ReLU")
    print("# Manual parameter collection for optimizer...")
    print("# Manual forward pass implementation...")
    print("# No automatic parameter registration...")
    print()
    print("✅ NEW API:")
    print("import tinytorch.nn as nn")
    print("import tinytorch.nn.functional as F")
    print("import tinytorch.optim as optim")
    print()
    print("class SimpleMLP(nn.Module):")
    print("    def __init__(self):")
    print("        super().__init__()")
    print("        self.hidden1 = nn.Linear(784, 128)  # Auto-registered!")
    print("        self.hidden2 = nn.Linear(128, 64)   # Auto-registered!")
    print("        self.output = nn.Linear(64, 10)     # Auto-registered!")
    print("    ")
    print("    def forward(self, x):")
    print("        x = F.flatten(x, start_dim=1)")
    print("        x = F.relu(self.hidden1(x))")
    print("        x = F.relu(self.hidden2(x))")
    print("        return self.output(x)")
    print()
    print("model = SimpleMLP()")
    print("optimizer = optim.Adam(model.parameters())  # Auto-collected!")
    print()

if __name__ == "__main__":
    show_api_comparison()
    train_mlp()