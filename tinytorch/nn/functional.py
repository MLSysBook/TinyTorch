"""
Functional interface for TinyTorch operations.

This module provides function-based implementations of common operations
that can be used independently or within Module classes. This matches
PyTorch's functional interface pattern.

Functions here are stateless - they don't hold parameters, just compute.
"""

import numpy as np
from typing import Tuple


def relu(x):
    """
    Rectified Linear Unit activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with ReLU applied element-wise
        
    Example:
        >>> x = Tensor([-1, 0, 1, 2])
        >>> F.relu(x)  # Returns [0, 0, 1, 2]
    """
    from ..core.tensor import Tensor
    from ..core.autograd import Variable
    
    # Handle both Tensor and Variable inputs
    if hasattr(x, 'data'):
        input_data = x.data
    else:
        input_data = x
    
    # Apply ReLU: max(0, x)
    output_data = np.maximum(0, input_data)
    
    # Preserve input type
    if isinstance(x, Variable):
        # For Variables, preserve gradient tracking
        def relu_grad_fn(grad_output):
            if x.requires_grad:
                # ReLU derivative: 1 where x > 0, 0 elsewhere
                grad_input = grad_output.data * (input_data > 0)
                x.backward(Variable(grad_input))
        
        return Variable(output_data, requires_grad=x.requires_grad, grad_fn=relu_grad_fn)
    else:
        return Tensor(output_data)


def flatten(x, start_dim=1):
    """
    Flatten tensor preserving batch dimension.
    
    Args:
        x: Input tensor with shape (batch_size, ...)
        start_dim: Dimension to start flattening from (default: 1)
        
    Returns:
        Flattened tensor with shape (batch_size, -1)
        
    Example:
        >>> x = Tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])  # (1, 2, 2, 2)
        >>> F.flatten(x)  # Returns shape (1, 8)
    """
    from ..core.tensor import Tensor
    from ..core.autograd import Variable
    
    # Handle both Tensor and Variable inputs
    if hasattr(x, 'data'):
        input_data = x.data
    else:
        input_data = x
    
    # Calculate new shape
    original_shape = input_data.shape
    if start_dim >= len(original_shape):
        raise ValueError(f"start_dim {start_dim} is out of range for tensor with {len(original_shape)} dimensions")
    
    # Keep dimensions before start_dim, flatten the rest
    new_shape = original_shape[:start_dim] + (-1,)
    output_data = input_data.reshape(new_shape)
    
    # Preserve input type
    if isinstance(x, Variable):
        def flatten_grad_fn(grad_output):
            if x.requires_grad:
                # Reshape gradient back to original shape
                grad_input = grad_output.data.reshape(original_shape)
                x.backward(Variable(grad_input))
        
        return Variable(output_data, requires_grad=x.requires_grad, grad_fn=flatten_grad_fn)
    else:
        return Tensor(output_data)


def max_pool2d(x, kernel_size, stride=None):
    """
    Apply 2D max pooling operation.
    
    Args:
        x: Input tensor with shape (..., H, W)
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling (defaults to kernel_size)
        
    Returns:
        Pooled tensor
        
    Example:
        >>> x = Tensor([[[[1, 2, 3, 4]]]])  # (1, 1, 1, 4) 
        >>> F.max_pool2d(x, kernel_size=2)  # Pool 2x2 regions
    """
    from ..core.tensor import Tensor
    from ..core.autograd import Variable
    
    # Handle both Tensor and Variable inputs
    if hasattr(x, 'data'):
        input_data = x.data
    else:
        input_data = x
    
    # Handle kernel_size as int or tuple
    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size
    
    # Default stride to kernel_size (non-overlapping)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        sH = sW = stride
    else:
        sH, sW = stride
    
    # Get input dimensions
    *batch_dims, H, W = input_data.shape
    
    # Calculate output dimensions
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    
    # Initialize output
    output_shape = tuple(batch_dims) + (out_H, out_W)
    output_data = np.zeros(output_shape, dtype=input_data.dtype)
    
    # Apply max pooling
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * sH
            h_end = h_start + kH
            w_start = j * sW
            w_end = w_start + kW
            
            # Extract pooling region and take max
            region = input_data[..., h_start:h_end, w_start:w_end]
            output_data[..., i, j] = np.max(region, axis=(-2, -1))
    
    # Preserve input type
    if isinstance(x, Variable):
        def maxpool_grad_fn(grad_output):
            if x.requires_grad:
                # Simplified gradient - just distribute back
                # In full implementation, would track max locations
                grad_input = np.zeros_like(input_data)
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * sH
                        h_end = h_start + kH
                        w_start = j * sW
                        w_end = w_start + kW
                        grad_input[..., h_start:h_end, w_start:w_end] += grad_output.data[..., i, j, np.newaxis, np.newaxis] / (kH * kW)
                
                x.backward(Variable(grad_input))
        
        return Variable(output_data, requires_grad=x.requires_grad, grad_fn=maxpool_grad_fn)
    else:
        return Tensor(output_data)