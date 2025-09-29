import numpy as np
from tinytorch.core.tensor import Tensor

# Simulate what mse_loss returns
mean_val = np.mean([0.1329])  # Single value
loss = Tensor([mean_val])

print(f"Loss type: {type(loss)}")
print(f"Loss.data: {loss.data}")
print(f"Loss.data type: {type(loss.data)}")

# Check if loss.data has .data attribute
if hasattr(loss.data, 'data'):
    print(f"Loss.data.data exists: {loss.data.data}")
    print(f"Loss.data.data type: {type(loss.data.data)}")

# Proper extraction
if hasattr(loss.data, 'data'):
    # loss.data is a Variable/Tensor with .data
    inner_data = loss.data.data
    if hasattr(inner_data, '__len__') and len(inner_data) > 0:
        loss_val = float(inner_data[0] if len(inner_data) == 1 else inner_data.flat[0])
    else:
        loss_val = float(inner_data)
else:
    # loss.data is numpy array or scalar
    if hasattr(loss.data, '__len__'):
        loss_val = float(loss.data[0] if len(loss.data) > 0 else 0.0)
    else:
        loss_val = float(loss.data)

print(f"\nExtracted loss value: {loss_val}")
