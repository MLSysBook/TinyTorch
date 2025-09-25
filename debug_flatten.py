#!/usr/bin/env python3
"""Debug flatten function with Variables"""

import numpy as np
import sys
import os

# Add TinyTorch to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable  
from tinytorch.core.spatial import flatten

print("🔍 Debug flatten function...")

# Test with Tensor
tensor_input = Tensor(np.random.randn(2, 3, 3).astype(np.float32))
tensor_output = flatten(tensor_input)
print(f"Tensor input type: {type(tensor_input)}")  
print(f"Tensor output type: {type(tensor_output)}")

# Test with Variable
variable_input = Variable(np.random.randn(2, 3, 3).astype(np.float32), requires_grad=True)
variable_output = flatten(variable_input)
print(f"Variable input type: {type(variable_input)}")
print(f"Variable output type: {type(variable_output)}")

print("✅ Flatten type preservation test complete")