#!/usr/bin/env python3
"""Debug MaxPool2D with Variables"""

import numpy as np
import sys
import os

# Add TinyTorch to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tinytorch'))

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable  
from tinytorch.core.spatial import MaxPool2D

print("🔍 Debug MaxPool2D function...")

# Test with Variable
pool = MaxPool2D(pool_size=(2, 2))
variable_input = Variable(np.random.randn(2, 4, 4).astype(np.float32), requires_grad=True)
variable_output = pool(variable_input)

print(f"Variable input type: {type(variable_input)}")
print(f"Variable input shape: {variable_input.shape}")
print(f"Variable output type: {type(variable_output)}")
print(f"Variable output shape: {variable_output.shape}")

print("✅ MaxPool2D type preservation test complete")