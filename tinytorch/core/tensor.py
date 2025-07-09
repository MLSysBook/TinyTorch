"""
Tensor implementation with automatic differentiation support.

This module implements the core Tensor class that serves as the foundation
for all computations in TinyTorch. It includes support for:
- N-dimensional arrays with NumPy backend
- Automatic differentiation (autograd)
- GPU-like operations (CPU implementation)
- Broadcasting and shape manipulation
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Any


class Tensor:
    """
    A tensor with automatic differentiation capabilities.
    
    The Tensor class is the core data structure of TinyTorch, similar to 
    torch.Tensor in PyTorch. It wraps NumPy arrays and adds automatic
    gradient computation for building neural networks.
    
    Attributes:
        data (np.ndarray): The underlying NumPy array containing the data
        grad (Optional[Tensor]): Gradient tensor (same shape as data)
        requires_grad (bool): Whether to track gradients for this tensor
        grad_fn (Optional[Function]): Function that created this tensor (for autograd)
    """
    
    def __init__(
        self, 
        data: Union[np.ndarray, List, float, int],
        requires_grad: bool = False,
        dtype: Optional[np.dtype] = None
    ):
        """
        Initialize a new Tensor.
        
        Args:
            data: The data to store in the tensor
            requires_grad: Whether to track gradients
            dtype: NumPy data type (defaults to float32)
        """
        if dtype is None:
            dtype = np.float32
            
        self.data = np.array(data, dtype=dtype)
        self.grad: Optional['Tensor'] = None
        self.requires_grad = requires_grad
        self.grad_fn = None  # Will be set by autograd functions
        
        # Initialize gradient tensor if needed
        if requires_grad:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the tensor."""
        return self.data.dtype
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim
    
    def __repr__(self) -> str:
        """String representation of the tensor."""
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_info})"
    
    def backward(self, gradient: Optional['Tensor'] = None) -> None:
        """
        Compute gradients using backpropagation.
        
        Args:
            gradient: Gradient flowing back from later operations
        """
        # TODO: Implement backward pass (Chapter 7 - Autograd)
        raise NotImplementedError("Backward pass will be implemented in Chapter 7")
    
    # Mathematical operations (to be implemented)
    def __add__(self, other) -> 'Tensor':
        """Addition operation."""
        # TODO: Implement in Chapter 3
        raise NotImplementedError("Addition will be implemented in Chapter 3")
    
    def __mul__(self, other) -> 'Tensor':
        """Multiplication operation."""
        # TODO: Implement in Chapter 3  
        raise NotImplementedError("Multiplication will be implemented in Chapter 3")
    
    def __matmul__(self, other) -> 'Tensor':
        """Matrix multiplication operation."""
        # TODO: Implement in Chapter 3
        raise NotImplementedError("Matrix multiplication will be implemented in Chapter 3")


def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with zeros."""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with ones.""" 
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad) 