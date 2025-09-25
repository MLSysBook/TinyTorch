#!/usr/bin/env python
"""
Simple Training Test - Minimal test to verify fixes
==================================================
"""

import numpy as np
import sys

# Import the classes we need directly
sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable, add, multiply, matmul

def simple_linear_test():
    """Test simple linear transformation with Variables."""
    print("Testing simple linear transformation...")
    
    # Data: y = 2x + 1
    X = Variable(np.array([[1.0], [2.0]], dtype=np.float32))
    y_target = np.array([[3.0], [5.0]], dtype=np.float32)
    
    # Parameters - make sure both are 2D for matmul
    weight = Parameter(np.array([[0.5]], dtype=np.float32))   # Shape (1,1) - 2D
    bias = Parameter(np.array([[0.0]], dtype=np.float32))     # Shape (1,1) - 2D
    
    print(f"Shapes: X={X.data.shape}, weight={weight.shape}, bias={bias.shape}")
    print(f"Initial: weight={weight.data[0,0]:.3f}, bias={bias.data[0,0]:.3f}")
    
    # Convert parameters to Variables
    weight_var = Variable(weight)
    bias_var = Variable(bias)
    
    print(f"weight_var.data.data shape: {weight_var.data.data.shape}")
    print(f"X.data.data shape: {X.data.data.shape}")
    
    # Forward pass: y = X @ weight + bias
    output = matmul(X, weight_var)
    output = add(output, bias_var)
    
    print(f"Output: {output.data.data.flatten()}")
    print(f"Target: {y_target.flatten()}")
    
    # Compute loss using Variables for proper gradient flow
    target_var = Variable(y_target, requires_grad=False)
    
    # MSE loss: mean((pred - target)^2)
    diff = output - target_var
    squared_diff = multiply(diff, diff)
    
    # Manual mean (sum / n)
    loss_sum = squared_diff.data.data[0,0] + squared_diff.data.data[1,0]
    loss = Variable(loss_sum / 2, requires_grad=True)
    
    # Set up proper gradient function
    def loss_grad_fn(grad_output):
        # For MSE, gradient w.r.t output = 2 * (pred - target) / n
        pred = output.data.data
        target = y_target
        grad_data = 2.0 * (pred - target) / 2.0  # n=2
        output.backward(Variable(grad_data))
    
    loss._grad_fn = loss_grad_fn
    
    print(f"Loss: {loss.data.data:.3f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Weight gradient: {weight.grad.data if weight.grad else 'None'}")
    print(f"Bias gradient: {bias.grad.data if bias.grad else 'None'}")
    
    if weight.grad is not None and bias.grad is not None:
        print("✅ Gradients computed successfully!")
        return True
    else:
        print("❌ Gradients not computed")
        return False


def test_matmul_variables():
    """Test matrix multiplication between Variables."""
    print("\nTesting Variable matrix multiplication...")
    
    # Create Variables
    a = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    b = Variable(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), requires_grad=True)
    
    print(f"A: {a.data.data}")
    print(f"B: {b.data.data}")
    
    # Matrix multiply
    c = matmul(a, b)
    print(f"C = A @ B: {c.data.data}")
    
    # Expected: [[19, 22], [43, 50]]
    expected = np.array([[19, 22], [43, 50]])
    
    if np.allclose(c.data.data, expected):
        print("✅ Matrix multiplication result correct!")
        
        # Test backward
        c.backward(Variable(np.ones_like(c.data.data)))
        
        if a.grad is not None and b.grad is not None:
            print("✅ Gradients computed for matmul!")
            print(f"A gradient: {a.grad.data.data}")
            print(f"B gradient: {b.grad.data.data}")
            return True
        else:
            print("❌ Gradients not computed for matmul")
            return False
    else:
        print("❌ Matrix multiplication result incorrect")
        return False


if __name__ == "__main__":
    print("SIMPLE TRAINING TEST")
    print("="*50)
    
    # Test matmul first
    matmul_ok = test_matmul_variables()
    
    # Test simple linear
    linear_ok = simple_linear_test()
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(f"Matrix multiplication: {'✅ PASS' if matmul_ok else '❌ FAIL'}")
    print(f"Linear transformation: {'✅ PASS' if linear_ok else '❌ FAIL'}")
    
    if matmul_ok and linear_ok:
        print("\n🎉 Core functionality works!")
        print("Ready for full training tests.")
    else:
        print("\n⚠️ Core functionality needs more fixes.")