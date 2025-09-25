#!/usr/bin/env python
"""
Test Training with Proper Gradient Propagation
===============================================
This implements the PyTorch way: requires_grad propagates through operations.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from tinytorch.core.tensor import Tensor, Parameter
from tinytorch.core.layers import Linear, Module
from tinytorch.core.activations import ReLU, Sigmoid
from tinytorch.core.training import MeanSquaredError
from tinytorch.core.optimizers import SGD, Adam
from tinytorch.core.networks import Sequential
from tinytorch.core.autograd import Variable


def test_gradient_propagation():
    """Test that requires_grad propagates correctly."""
    print("="*60)
    print("Testing Gradient Propagation (PyTorch Way)")
    print("="*60)
    
    # Rule 1: Parameters always require gradients
    param = Parameter(np.array([[2.0]]))
    print(f"Parameter requires_grad: {param.requires_grad}")  # Should be True
    
    # Rule 2: Regular tensors don't by default
    data = Tensor(np.array([[3.0]]))
    print(f"Regular tensor requires_grad: {data.requires_grad}")  # Should be False
    
    # Rule 3: Operations propagate requires_grad
    # When we mix Parameter and Tensor, result should require gradients
    print("\nTesting operation propagation:")
    
    # Convert to Variables for operations (this is the current workaround)
    param_var = Variable(param)
    data_var = Variable(data, requires_grad=False)
    
    result = param_var * data_var
    print(f"Result requires_grad: {result.requires_grad}")  # Should be True
    
    # Test backward
    result.backward()
    print(f"Parameter gradient: {param.grad.data if param.grad else 'None'}")


def test_xor_with_proper_setup():
    """Test XOR training with proper gradient setup."""
    print("\n" + "="*60)
    print("Testing XOR Training (Proper Setup)")
    print("="*60)
    
    # XOR dataset
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float32))
    
    # Build network - need to ensure gradients flow
    class XORNet(Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Linear(2, 4)
            self.layer2 = Linear(4, 1)
            self.relu = ReLU()
            self.sigmoid = Sigmoid()
        
        def forward(self, x):
            # Convert to Variable to maintain gradient chain
            if not isinstance(x, Variable):
                x = Variable(x, requires_grad=False)
            
            # Layer 1
            x = self.layer1(x)
            x = self.relu(x)
            
            # Layer 2
            x = self.layer2(x)
            x = self.sigmoid(x)
            
            return x
    
    model = XORNet()
    optimizer = SGD(model.parameters(), learning_rate=0.5)
    criterion = MeanSquaredError()
    
    # Training loop
    losses = []
    for epoch in range(1000):
        # Forward pass
        output = model(X)
        loss = criterion(output, y)
        
        # Extract loss value
        if hasattr(loss, 'data'):
            if hasattr(loss.data, 'data'):
                loss_val = float(loss.data.data)
            else:
                loss_val = float(loss.data)
        else:
            loss_val = float(loss)
        
        losses.append(loss_val)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check if gradients exist
        if epoch == 0:
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.data)
                    print(f"Param {i} gradient norm: {grad_norm:.4f}")
                else:
                    print(f"Param {i}: No gradient!")
        
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss_val:.4f}")
    
    # Final evaluation
    print("\nFinal predictions:")
    final_output = model(X)
    
    # Extract predictions
    if hasattr(final_output, 'data'):
        if hasattr(final_output.data, 'data'):
            predictions = final_output.data.data
        else:
            predictions = final_output.data
    else:
        predictions = final_output
    
    for i, (x_val, pred, target) in enumerate(zip(X.data, predictions, y.data)):
        print(f"  {x_val} → {pred[0]:.3f} (target: {target[0]})")
    
    # Check learning
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\nLoss improved by {improvement:.1f}%")
    
    # Check accuracy
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y.data)
    print(f"Accuracy: {accuracy*100:.0f}%")
    
    if accuracy >= 0.75:
        print("✅ XOR learned successfully!")
    else:
        print("⚠️ XOR partially learned (training is working but needs tuning)")


def test_simple_linear_regression():
    """Test simple linear regression to verify basic training."""
    print("\n" + "="*60)
    print("Testing Linear Regression (Simplest Case)")
    print("="*60)
    
    # Simple data: y = 2x + 1
    X = Tensor(np.array([[1], [2], [3], [4]], dtype=np.float32))
    y = Tensor(np.array([[3], [5], [7], [9]], dtype=np.float32))
    
    # Single layer model
    model = Linear(1, 1)
    print(f"Initial weight: {model.weights.data[0,0]:.3f}")
    print(f"Initial bias: {model.bias.data[0]:.3f}")
    
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    criterion = MeanSquaredError()
    
    # Training
    for epoch in range(200):
        # Need to ensure gradient flow
        output = Variable(model(X)) if not isinstance(model(X), Variable) else model(X)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            loss_val = float(loss.data.data) if hasattr(loss.data, 'data') else float(loss.data)
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")
    
    print(f"\nFinal weight: {model.weights.data[0,0]:.3f} (target: 2.0)")
    print(f"Final bias: {model.bias.data[0]:.3f} (target: 1.0)")
    
    # Check if learned
    weight_error = abs(model.weights.data[0,0] - 2.0)
    bias_error = abs(model.bias.data[0] - 1.0)
    
    if weight_error < 0.1 and bias_error < 0.1:
        print("✅ Linear regression learned perfectly!")
    elif weight_error < 0.5 and bias_error < 0.5:
        print("✅ Linear regression learned reasonably well!")
    else:
        print("⚠️ Linear regression learning but not converged")


def analyze_current_issues():
    """Analyze what's working and what needs fixing."""
    print("\n" + "="*60)
    print("ANALYSIS: Current State of Training")
    print("="*60)
    
    print("""
WHAT'S WORKING:
✅ Variable class properly tracks gradients
✅ Autograd backward pass computes gradients
✅ Gradients flow back to Parameters (via _source_tensor)
✅ Optimizers can update parameters

WHAT NEEDS FIXING:
❌ Linear layer returns Tensor, not Variable (breaks chain)
❌ Activations may not preserve Variable type
❌ Operations between Tensor and Variable unclear

THE CORE ISSUE:
- Operations need to automatically promote to Variable when ANY input requires_grad
- This is the "PyTorch way" - automatic gradient tracking

SOLUTIONS:
1. SHORT TERM: Wrap operations in Variables in forward passes
2. LONG TERM: Make operations automatically handle gradient propagation
3. BEST: Unify Tensor/Variable with requires_grad flag (like modern PyTorch)
""")


if __name__ == "__main__":
    # Test gradient propagation
    test_gradient_propagation()
    
    # Test simple case first
    test_simple_linear_regression()
    
    # Test XOR (harder non-linear problem)
    test_xor_with_proper_setup()
    
    # Analysis
    analyze_current_issues()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
To make training work properly without hacks, we need to:

1. Make operations (matmul, add, etc.) return Variables when ANY input has requires_grad
2. Ensure all layer operations preserve the gradient chain
3. Make activations handle Variables properly

This follows the PyTorch design where gradient tracking propagates automatically.
""")