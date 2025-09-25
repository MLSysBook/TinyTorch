#!/usr/bin/env python
"""
Debug Tensor/Variable issue
"""

import numpy as np
import sys

sys.path.append('modules/02_tensor')
sys.path.append('modules/06_autograd')

from tensor_dev import Tensor, Parameter
from autograd_dev import Variable

def debug_tensor_variable():
    """Debug the tensor/variable shape issue."""
    print("="*50)
    print("DEBUGGING TENSOR/VARIABLE SHAPE ISSUE")
    print("="*50)
    
    # Create a 2D numpy array
    np_array = np.array([[0.5]], dtype=np.float32)
    print(f"1. Original numpy array shape: {np_array.shape}")
    print(f"   Value: {np_array}")
    
    # Create Parameter (which is a Tensor)
    param = Parameter(np_array)
    print(f"2. Parameter shape: {param.shape}")
    print(f"   Parameter data shape: {param.data.shape}")
    print(f"   Parameter value: {param.data}")
    
    # Create Variable from Parameter
    var = Variable(param)
    print(f"3. Variable data shape: {var.data.shape}")
    print(f"   Variable data.data shape: {var.data.data.shape}")
    print(f"   Variable value: {var.data.data}")
    
    # Check if the issue is in Variable init
    print("\nDebugging Variable init:")
    print(f"   isinstance(param, Tensor): {isinstance(param, Tensor)}")
    print(f"   param type: {type(param)}")
    print(f"   var.data type: {type(var.data)}")
    print(f"   var._source_tensor: {var._source_tensor}")
    
    # Try creating Variable from numpy directly
    var2 = Variable(np_array)
    print(f"4. Variable from numpy shape: {var2.data.shape}")
    print(f"   Variable from numpy data.data shape: {var2.data.data.shape}")

if __name__ == "__main__":
    debug_tensor_variable()