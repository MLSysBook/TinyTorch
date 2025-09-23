#!/usr/bin/env python3
"""
XOR Network Training with Modern PyTorch-like API

This example demonstrates the clean, modern TinyTorch API for solving
the classic XOR problem. Compare to train_xor_network.py to see the 
dramatic simplification while maintaining full educational value.

Students implement core algorithms but use professional interfaces.
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
from tinytorch.core.training import MeanSquaredError as MSELoss

class XORNet(nn.Module):
    """
    XOR Network using modern PyTorch-like API.
    
    This demonstrates how clean the new API is:
    - Inherits from nn.Module for automatic parameter registration
    - Uses nn.Linear for fully connected layers  
    - Uses F.relu for activation
    - Parameters are automatically collected for optimizer
    """
    
    def __init__(self):
        super().__init__()
        print("🧠 Creating XOR neural network...")
        
        # Hidden layer: 2 inputs -> 4 hidden units (you built this!)
        self.hidden = nn.Linear(2, 4)
        
        # Output layer: 4 hidden -> 1 output  
        self.output = nn.Linear(4, 1)
        
        print(f"✅ XORNet created with {len(list(self.parameters()))} parameters")
    
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.hidden(x))  # Hidden layer + activation
        x = self.output(x)          # Output layer (no activation for regression)
        return x

def create_xor_dataset():
    """Create the XOR dataset."""
    print("📊 Creating XOR dataset...")
    
    # XOR truth table
    X = np.array([
        [0, 0],  # 0 XOR 0 = 0
        [0, 1],  # 0 XOR 1 = 1  
        [1, 0],  # 1 XOR 0 = 1
        [1, 1]   # 1 XOR 1 = 0
    ], dtype=np.float32)
    
    y = np.array([
        [0],     # 0
        [1],     # 1
        [1],     # 1  
        [0]      # 0
    ], dtype=np.float32)
    
    print("✅ XOR dataset created")
    print("📋 Truth table:")
    for i in range(len(X)):
        print(f"   {X[i]} -> {y[i]}")
    
    return X, y

def train_xor_network():
    """Train XOR network using modern API."""
    print("🚀 Training XOR Network with Modern API")
    print("=" * 50)
    
    # Create model and optimizer - notice how clean this is!
    model = XORNet()
    optimizer = optim.SGD(model.parameters(), learning_rate=0.1)  # Auto parameter collection!
    criterion = MSELoss()
    
    # Create dataset
    X, y = create_xor_dataset()
    
    # Training loop
    print("🏃 Starting training...")
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # Train on entire dataset (batch size = 4)
        for i in range(len(X)):
            # Convert to Variables
            inputs = Variable(Tensor(X[i:i+1]), requires_grad=False)
            targets = Variable(Tensor(y[i:i+1]), requires_grad=False)
            
            # Forward pass - clean and simple!
            outputs = model(inputs)  # model(x) calls model.forward(x) automatically
            loss = criterion(outputs, targets)
            
            # Backward pass  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        
        # Progress update
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch:4d}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("✅ Training completed!")
    return model

def test_xor_network(model):
    """Test the trained network on XOR truth table."""
    print("🧪 Testing XOR Network")
    print("=" * 30)
    
    X, y = create_xor_dataset()
    
    print("📊 Results:")
    print("Input  | Target | Predicted | Correct?")
    print("-------|--------|-----------|----------")
    
    all_correct = True
    for i in range(len(X)):
        inputs = Variable(Tensor(X[i:i+1]), requires_grad=False)
        
        # Forward pass
        output = model(inputs)
        predicted = output.data[0, 0]
        target = y[i, 0]
        
        # Binary classification threshold
        predicted_binary = 1 if predicted > 0.5 else 0
        correct = "✅" if abs(predicted_binary - target) < 0.1 else "❌"
        
        if abs(predicted_binary - target) >= 0.1:
            all_correct = False
        
        print(f"{X[i]}  |   {target}    |   {predicted:.3f}   | {correct}")
    
    print("=" * 30)
    if all_correct:
        print("🎉 Perfect! XOR network learned the pattern!")
    else:
        print("⚠️  Network needs more training or different architecture")
    
    return all_correct

def compare_apis():
    """Show the API improvement."""
    print("🔍 API Comparison - XOR Network")
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
    print("class XORNet(nn.Module):")
    print("    def __init__(self):")
    print("        super().__init__()")
    print("        self.hidden = nn.Linear(2, 4)  # Auto-registered!")
    print("        self.output = nn.Linear(4, 1)  # Auto-registered!")
    print("    ")
    print("    def forward(self, x):")
    print("        x = F.relu(self.hidden(x))")
    print("        return self.output(x)")
    print()
    print("model = XORNet()")
    print("optimizer = optim.SGD(model.parameters())  # Auto-collected!")

if __name__ == "__main__":
    print("🔥 TinyTorch Modern API - XOR Example")
    print("Learning nonlinear patterns with clean, professional interfaces")
    print()
    
    # Show API comparison
    compare_apis()
    print()
    
    # Train and test
    try:
        model = train_xor_network()
        success = test_xor_network(model)
        
        if success:
            print()
            print("🎓 Educational Achievement:")
            print("- You implemented Linear layers (matrix multiplication + bias)")
            print("- You implemented ReLU activation (nonlinearity)")  
            print("- You implemented SGD optimizer (gradient descent)")
            print("- Infrastructure provides clean PyTorch-compatible API")
            print("- Result: Perfect XOR classification!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 This shows where the implementation needs completion.")
    
    print()
    print("✨ Key Insight: Clean APIs don't reduce educational value!")
    print("   Students still implement core algorithms while using professional patterns.")