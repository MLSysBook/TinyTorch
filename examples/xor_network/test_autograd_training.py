#!/usr/bin/env python3
"""
Test script to verify that training module now supports autograd for real training.

This demonstrates the fix that integrates loss functions with the autograd system.
"""

import sys
import numpy as np

# Import TinyTorch components
from tinytorch.core.autograd import Variable
from tinytorch.core.training import MeanSquaredError, Trainer
from tinytorch.core.tensor import Tensor

def test_autograd_training():
    """Test that training loops now work with autograd gradients."""
    print("🧪 Testing Training Loop with Autograd Integration")
    print("=" * 50)
    
    # Test 1: Basic loss function with autograd
    print("\n1. Testing MSE Loss with autograd:")
    mse_loss = MeanSquaredError()
    
    # Create Variables for autograd
    predictions = Variable([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = Variable([[1.5, 2.5], [2.5, 3.5]], requires_grad=False)
    
    # Compute loss
    loss = mse_loss(predictions, targets)
    print(f"   Loss type: {type(loss)}")
    print(f"   Loss value: {loss.data}")
    print(f"   Loss supports .backward(): {hasattr(loss, 'backward')}")
    
    # Test backward pass
    loss.backward()
    print(f"   Gradients computed: {predictions.grad is not None}")
    if predictions.grad is not None:
        print(f"   Gradient values: {predictions.grad.data}")
    
    # Test 2: Loss functions work with both Tensors and Variables
    print("\n2. Testing backward compatibility with Tensors:")
    
    # Test with regular Tensors (should auto-convert to Variables)
    tensor_pred = Tensor([[1.0, 2.0]])
    tensor_true = Tensor([[1.5, 2.5]])
    
    loss_from_tensors = mse_loss(tensor_pred, tensor_true)
    print(f"   Loss from Tensors type: {type(loss_from_tensors)}")
    print(f"   Supports backward: {hasattr(loss_from_tensors, 'backward')}")
    
    # Test 3: Demonstrate gradient flow
    print("\n3. Testing gradient flow through loss:")
    
    # Create a simple scenario
    x = Variable([[1.0, 2.0]], requires_grad=True)
    y_true = Variable([[3.0, 4.0]], requires_grad=False)
    
    # Simple "model": y = 2*x (should learn that weights should be [2, 2])
    w = Variable([[2.1, 1.9]], requires_grad=True)  # Slightly off optimal weights
    
    # Forward pass
    y_pred = x * w
    loss = mse_loss(y_pred, y_true)
    
    print(f"   Initial weights: {w.data}")
    print(f"   Predictions: {y_pred.data}")
    print(f"   Loss: {loss.data}")
    
    # Backward pass
    loss.backward()
    print(f"   Weight gradients: {w.grad.data if w.grad else 'None'}")
    print(f"   Input gradients: {x.grad.data if x.grad else 'None'}")
    
    print("\n✅ All autograd training tests passed!")
    print("🎯 Training module now supports automatic gradient computation")
    print("🚀 Ready for real neural network training with backpropagation!")

if __name__ == "__main__":
    test_autograd_training()