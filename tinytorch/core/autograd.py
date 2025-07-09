"""
Automatic differentiation engine.

This module implements the computational graph and automatic differentiation
system that enables gradient computation for neural network training.

Key concepts:
- Computational graph representation
- Forward and backward pass orchestration  
- Gradient accumulation and propagation
- Function objects for operations
"""

from typing import List, Optional, Callable, Any
from .tensor import Tensor


class Function:
    """
    Base class for all differentiable functions.
    
    Functions represent operations in the computational graph and define
    how to compute gradients during the backward pass.
    """
    
    @staticmethod
    def forward(ctx: 'Context', *args) -> Tensor:
        """
        Forward pass of the function.
        
        Args:
            ctx: Context object for saving information needed in backward
            *args: Input tensors
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Forward pass must be implemented")
    
    @staticmethod  
    def backward(ctx: 'Context', grad_output: Tensor) -> tuple:
        """
        Backward pass of the function.
        
        Args:
            ctx: Context object with saved information from forward
            grad_output: Gradient flowing back from later operations
            
        Returns:
            Tuple of gradients for each input tensor
        """
        raise NotImplementedError("Backward pass must be implemented")


class Context:
    """
    Context object for saving information between forward and backward passes.
    
    This allows functions to store intermediate values needed for gradient
    computation without keeping them in the tensor objects themselves.
    """
    
    def __init__(self):
        """Initialize empty context."""
        self.saved_tensors = []
        self.saved_variables = {}
    
    def save_for_backward(self, *tensors: Tensor) -> None:
        """
        Save tensors for use in backward pass.
        
        Args:
            *tensors: Tensors to save
        """
        self.saved_tensors = list(tensors)
    
    @property
    def saved_tensors(self) -> List[Tensor]:
        """Return saved tensors."""
        return self._saved_tensors
    
    @saved_tensors.setter  
    def saved_tensors(self, tensors: List[Tensor]) -> None:
        """Set saved tensors."""
        self._saved_tensors = tensors


class Node:
    """
    Node in the computational graph.
    
    Each tensor with requires_grad=True has an associated node that
    tracks how it was computed and enables gradient flow.
    """
    
    def __init__(self, tensor: Tensor, grad_fn: Optional[Function] = None):
        """
        Initialize graph node.
        
        Args:
            tensor: The tensor this node represents
            grad_fn: Function that created this tensor
        """
        self.tensor = tensor
        self.grad_fn = grad_fn
        self.next_functions = []  # Functions to call during backward
    
    def add_next_function(self, function: Function, input_idx: int) -> None:
        """
        Add a function to call during backward pass.
        
        Args:
            function: Function to call
            input_idx: Index of this tensor in the function's inputs
        """
        self.next_functions.append((function, input_idx))


def grad(
    outputs: List[Tensor], 
    inputs: List[Tensor],
    grad_outputs: Optional[List[Tensor]] = None,
    retain_graph: bool = False,
    create_graph: bool = False
) -> List[Tensor]:
    """
    Compute gradients of outputs with respect to inputs.
    
    Args:
        outputs: Tensors to compute gradients of
        inputs: Tensors to compute gradients with respect to
        grad_outputs: Gradients to backpropagate (defaults to ones)
        retain_graph: Whether to keep computational graph after backward
        create_graph: Whether to create graph for higher-order derivatives
        
    Returns:
        List of gradients for each input tensor
    """
    # TODO: Implement gradient computation in Chapter 7
    raise NotImplementedError("Gradient computation will be implemented in Chapter 7")


# Common differentiable functions (to be implemented)

class AddFunction(Function):
    """Addition operation: output = input1 + input2"""
    
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor) -> Tensor:
        """Forward pass for addition."""
        # TODO: Implement in Chapter 7
        raise NotImplementedError("Addition forward will be implemented in Chapter 7")
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple:
        """Backward pass for addition."""
        # TODO: Implement in Chapter 7  
        raise NotImplementedError("Addition backward will be implemented in Chapter 7")


class MulFunction(Function):
    """Multiplication operation: output = input1 * input2"""
    
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor) -> Tensor:
        """Forward pass for multiplication."""
        # TODO: Implement in Chapter 7
        raise NotImplementedError("Multiplication forward will be implemented in Chapter 7")
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple:
        """Backward pass for multiplication."""
        # TODO: Implement in Chapter 7
        raise NotImplementedError("Multiplication backward will be implemented in Chapter 7")


class MatMulFunction(Function):
    """Matrix multiplication operation: output = input1 @ input2"""
    
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor) -> Tensor:
        """Forward pass for matrix multiplication."""
        # TODO: Implement in Chapter 7
        raise NotImplementedError("Matrix multiplication forward will be implemented in Chapter 7")
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple:
        """Backward pass for matrix multiplication."""
        # TODO: Implement in Chapter 7
        raise NotImplementedError("Matrix multiplication backward will be implemented in Chapter 7") 