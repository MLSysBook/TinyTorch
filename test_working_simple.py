#!/usr/bin/env python
"""
Working Simple Training - Using the gradient flow approach that worked
"""

import numpy as np
import sys

sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable, add, multiply, matmul, subtract

def simple_linear_regression():
    """Simple linear regression using the approach that worked in gradient flow test."""
    print("Testing simple linear regression...")
    
    # Create parameters like in the working gradient test
    weight = Parameter(np.array([[0.5]], dtype=np.float32))  # (1,1)
    bias = Parameter(np.array([[0.0]], dtype=np.float32))    # (1,1)
    
    print(f"Initial: weight={weight.data[0,0]:.3f}, bias={bias.data[0,0]:.3f}")
    
    # Data: simple single example first
    x_data = np.array([[2.0]], dtype=np.float32)  # Input: 2
    y_target = 5.0  # Target: 2*2 + 1 = 5
    
    for epoch in range(10):
        # Convert to Variables (like gradient flow test)
        x = Variable(x_data, requires_grad=False)
        weight_var = Variable(weight)  # This maintains connection to parameter
        bias_var = Variable(bias)
        
        # Forward: y = x @ weight + bias
        output = matmul(x, weight_var)  # (1,1) @ (1,1) = (1,1)
        output = add(output, bias_var)   # (1,1) + (1,1) = (1,1)
        
        # Loss: (output - target)^2
        target_var = Variable(np.array([[y_target]], dtype=np.float32), requires_grad=False)
        diff = subtract(output, target_var)
        loss = multiply(diff, diff)
        
        # Clear gradients
        weight.grad = None
        bias.grad = None
        
        # Backward - this should work like the gradient flow test
        loss.backward(Variable(np.array([[1.0]], dtype=np.float32)))
        
        # Check gradients
        if epoch == 0:
            print(f"  Weight grad: {weight.grad}")
            print(f"  Bias grad: {bias.grad}")
            if weight.grad is None:
                print("  ❌ No gradients flowing!")
                break
        
        # Manual SGD update
        if weight.grad is not None and bias.grad is not None:
            lr = 0.01
            weight.data = weight.data - lr * weight.grad.data
            bias.data = bias.data - lr * bias.grad.data
        
        if epoch % 2 == 0:
            loss_val = loss.data.data[0,0]
            print(f"  Epoch {epoch}: loss={loss_val:.3f}, weight={weight.data[0,0]:.3f}, bias={bias.data[0,0]:.3f}")
    
    # Check final result
    final_w = weight.data[0,0]
    final_b = bias.data[0,0]
    print(f"Final: weight={final_w:.3f}, bias={final_b:.3f}")
    
    # For y = 2x + 1, with x=2, we want weight≈2, bias≈1
    w_err = abs(final_w - 2.0)
    b_err = abs(final_b - 1.0)
    
    if weight.grad is not None:
        print("✅ Gradients are flowing!")
        if w_err < 0.5 and b_err < 0.5:
            print("✅ Parameters converging towards correct values!")
            return True
    
    return False

if __name__ == "__main__":
    print("TESTING SIMPLE APPROACH THAT SHOULD WORK")
    print("="*50)
    
    success = simple_linear_regression()
    
    if success:
        print("\n🎉 Basic training works! Now we can build on this.")
    else:
        print("\n❌ Still not working. Need to debug further.")