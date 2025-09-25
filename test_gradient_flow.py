#!/usr/bin/env python
"""
Test gradient flow step by step
"""

import numpy as np
import sys

sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable, add, multiply, matmul

def test_basic_gradient_flow():
    """Test the most basic gradient flow."""
    print("Testing basic gradient flow...")
    
    # Create a parameter
    param = Parameter(np.array([[2.0]], dtype=np.float32))
    print(f"Parameter: {param.data}, requires_grad: {param.requires_grad}")
    
    # Wrap in Variable
    param_var = Variable(param)
    print(f"Variable: {param_var.data.data}, requires_grad: {param_var.requires_grad}")
    print(f"Source tensor: {param_var._source_tensor}")
    print(f"Source tensor requires_grad: {param_var._source_tensor.requires_grad if param_var._source_tensor else 'None'}")
    
    # Simple operation: y = x * 2
    two = Variable(np.array([[2.0]], dtype=np.float32), requires_grad=False)
    result = multiply(param_var, two)
    print(f"Result: {result.data.data}, requires_grad: {result.requires_grad}")
    
    # Manual backward
    result.backward(Variable(np.array([[1.0]], dtype=np.float32)))
    
    print(f"Parameter gradient after backward: {param.grad}")
    print(f"Parameter_var gradient after backward: {param_var.grad}")
    
    return param.grad is not None

def test_addition_gradient_flow():
    """Test gradient flow through addition."""
    print("\nTesting addition gradient flow...")
    
    # Create parameters
    a = Parameter(np.array([[1.0]], dtype=np.float32))
    b = Parameter(np.array([[2.0]], dtype=np.float32))
    
    # Wrap in Variables
    a_var = Variable(a)
    b_var = Variable(b)
    
    # Add them
    result = add(a_var, b_var)
    print(f"Addition result: {result.data.data}")
    
    # Backward
    result.backward(Variable(np.array([[1.0]], dtype=np.float32)))
    
    print(f"a gradient: {a.grad}")
    print(f"b gradient: {b.grad}")
    
    return a.grad is not None and b.grad is not None

def test_matmul_gradient_flow():
    """Test gradient flow through matrix multiplication."""
    print("\nTesting matmul gradient flow...")
    
    # Create parameters
    a = Parameter(np.array([[1.0, 2.0]], dtype=np.float32))  # (1, 2)
    b = Parameter(np.array([[3.0], [4.0]], dtype=np.float32))  # (2, 1)
    
    # Wrap in Variables
    a_var = Variable(a)
    b_var = Variable(b)
    
    print(f"a shape: {a.shape}, b shape: {b.shape}")
    
    # Matrix multiply
    result = matmul(a_var, b_var)  # Should be (1, 1)
    print(f"Matmul result: {result.data.data}, shape: {result.data.shape}")
    
    # Backward
    result.backward(Variable(np.array([[1.0]], dtype=np.float32)))
    
    print(f"a gradient: {a.grad}")
    print(f"b gradient: {b.grad}")
    
    return a.grad is not None and b.grad is not None

if __name__ == "__main__":
    print("TESTING GRADIENT FLOW STEP BY STEP")
    print("="*50)
    
    basic_ok = test_basic_gradient_flow()
    add_ok = test_addition_gradient_flow()  
    matmul_ok = test_matmul_gradient_flow()
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(f"Basic gradient flow:    {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Addition gradient flow: {'✅ PASS' if add_ok else '❌ FAIL'}")
    print(f"Matmul gradient flow:   {'✅ PASS' if matmul_ok else '❌ FAIL'}")
    
    if basic_ok and add_ok and matmul_ok:
        print("\n🎉 All gradient flow tests passed!")
    else:
        print("\n⚠️ Some gradient flow tests failed.")