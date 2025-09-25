#!/usr/bin/env python
"""
Test Training Solution - Verify PyTorch-inspired fixes work
===========================================================
This tests the proper solution using the fixed TinyTorch architecture.
"""

import numpy as np
import sys
import os

# Add the modules to path for testing
sys.path.insert(0, 'modules/02_tensor')
sys.path.insert(0, 'modules/06_autograd')
sys.path.insert(0, 'modules/04_layers')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable
from layers_dev import Linear


class SimpleReLU:
    """Simple ReLU activation for Variables."""
    
    def __call__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x)
        
        # Forward pass
        relu_data = np.maximum(0, x.data.data)
        
        # Backward pass
        def grad_fn(grad_output):
            grad = (x.data.data > 0) * grad_output.data.data
            x.backward(Variable(grad))
        
        return Variable(relu_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


class SimpleSigmoid:
    """Simple Sigmoid activation for Variables."""
    
    def __call__(self, x):
        if not isinstance(x, Variable):
            x = Variable(x)
        
        # Forward pass
        sig_data = 1.0 / (1.0 + np.exp(-np.clip(x.data.data, -500, 500)))
        
        # Backward pass
        def grad_fn(grad_output):
            grad = sig_data * (1 - sig_data) * grad_output.data.data
            x.backward(Variable(grad))
        
        return Variable(sig_data, requires_grad=x.requires_grad, grad_fn=grad_fn)


class SimpleMSE:
    """Simple MSE loss for Variables."""
    
    def __call__(self, pred, target):
        if not isinstance(pred, Variable):
            pred = Variable(pred)
        if not isinstance(target, Variable):
            target = Variable(target, requires_grad=False)
        
        # Forward: MSE = mean((pred - target)^2)
        diff = pred - target
        squared = diff * diff
        
        # Manual mean
        n = squared.data.data.size
        loss_val = np.mean(squared.data.data)
        
        # Backward
        def grad_fn(grad_output=Variable(1.0)):
            # Gradient: 2 * (pred - target) / n
            grad = 2.0 * (pred.data.data - target.data.data) / n
            pred.backward(Variable(grad))
        
        return Variable(loss_val, requires_grad=True, grad_fn=grad_fn)


class SimpleSGD:
    """Simple SGD optimizer."""
    
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad.data


def test_linear_regression():
    """Test simple linear regression to verify gradient flow."""
    print("="*60)
    print("TESTING LINEAR REGRESSION WITH FIXED ARCHITECTURE")
    print("="*60)
    
    # Simple linear regression: y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y = np.array([[3.0], [5.0], [7.0], [9.0]], dtype=np.float32)
    
    # Create model
    model = Linear(1, 1)
    print(f"Initial: weight={model.weights.data[0,0]:.3f}, bias={model.bias.data[0]:.3f}")
    
    # Training setup
    optimizer = SimpleSGD(model.parameters(), lr=0.01)
    criterion = SimpleMSE()
    
    # Training loop
    for epoch in range(200):
        # Forward pass
        output = model(Tensor(X))
        loss = criterion(output, Tensor(y))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients are flowing
        if epoch == 0:
            print("Gradient check:")
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.data)
                    print(f"  Parameter {i}: grad_norm = {grad_norm:.4f}")
                else:
                    print(f"  Parameter {i}: NO GRADIENT!")
        
        # Update
        optimizer.step()
        
        if epoch % 50 == 0:
            loss_val = float(loss.data.data)
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")
    
    print(f"Final: weight={model.weights.data[0,0]:.3f}, bias={model.bias.data[0]:.3f}")
    print(f"Target: weight=2.000, bias=1.000")
    
    # Verify convergence
    w_err = abs(model.weights.data[0,0] - 2.0)
    b_err = abs(model.bias.data[0] - 1.0)
    
    if w_err < 0.1 and b_err < 0.1:
        print("✅ Linear regression converged correctly!")
        return True
    else:
        print("❌ Linear regression failed to converge")
        return False


def test_xor_training():
    """Test XOR training with multiple layers."""
    print("\n" + "="*60)
    print("TESTING XOR TRAINING WITH FIXED ARCHITECTURE")
    print("="*60)
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Network
    layer1 = Linear(2, 8)
    layer2 = Linear(8, 1)
    relu = SimpleReLU()
    sigmoid = SimpleSigmoid()
    
    # Training setup
    params = layer1.parameters() + layer2.parameters()
    optimizer = SimpleSGD(params, lr=0.5)
    criterion = SimpleMSE()
    
    print(f"Total parameters: {len(params)}")
    
    # Training loop
    for epoch in range(500):
        # Forward pass
        h1 = layer1(Tensor(X))
        h1_relu = relu(h1)
        h2 = layer2(h1_relu)
        output = sigmoid(h2)
        
        # Loss
        loss = criterion(output, Tensor(y))
        loss_val = float(loss.data.data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients are flowing (first epoch only)
        if epoch == 0:
            print("Gradient check:")
            grad_count = 0
            for i, param in enumerate(params):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.data)
                    print(f"  Parameter {i}: grad_norm = {grad_norm:.4f}")
                    grad_count += 1
                else:
                    print(f"  Parameter {i}: NO GRADIENT!")
            
            if grad_count == len(params):
                print("✅ All parameters have gradients!")
            else:
                print(f"❌ Only {grad_count}/{len(params)} parameters have gradients!")
        
        # Update
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss_val:.4f}")
    
    # Test final predictions
    print("\nFinal predictions:")
    h1 = layer1(Tensor(X))
    h1_relu = relu(h1)
    h2 = layer2(h1_relu)
    predictions = sigmoid(h2)
    
    pred_vals = predictions.data.data
    for x_val, pred, target in zip(X, pred_vals, y):
        print(f"  {x_val} → {pred[0]:.3f} (target: {target[0]})")
    
    # Check accuracy
    binary_preds = (pred_vals > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y)
    print(f"\nAccuracy: {accuracy*100:.0f}%")
    
    if accuracy >= 0.75:
        print("✅ XOR training successful!")
        return True
    else:
        print("❌ XOR training failed")
        return False


if __name__ == "__main__":
    print("TESTING TINYTORCH TRAINING SOLUTION")
    print("Based on PyTorch's lessons learned from Variable/Tensor separation")
    print()
    
    # Test simple case first
    linear_success = test_linear_regression()
    
    # Test complex case
    xor_success = test_xor_training()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Linear Regression: {'✅ PASS' if linear_success else '❌ FAIL'}")
    print(f"XOR Training:      {'✅ PASS' if xor_success else '❌ FAIL'}")
    
    if linear_success and xor_success:
        print("\n🎉 ALL TESTS PASSED! Training now works properly!")
        print("\nKey architectural insights:")
        print("1. Variables maintain gradient connections to Parameters via _source_tensor")
        print("2. Linear layers convert Parameters to Variables in forward pass")
        print("3. Matrix multiplication works through Variable.__matmul__")
        print("4. Gradients flow from Variables back to Parameters for optimizer updates")
        
    else:
        print("\n⚠️ Some tests failed. Architecture needs more fixes.")
    
    print("\nThis solution preserves the educational Tensor/Variable separation")
    print("while enabling proper gradient flow for neural network training.")