"""
TinyTorch Utility Functions

This module contains utility functions used throughout the TinyTorch system.
Students will implement various utility functions here as part of different projects.
"""

from typing import Any, List, Dict


def format_tensor_shape(shape: tuple) -> str:
    """
    Format a tensor shape tuple for pretty printing.
    
    Args:
        shape: Tuple representing tensor dimensions
        
    Returns:
        Formatted string representation of the shape
    """
    return f"({', '.join(map(str, shape))})"


def validate_tensor_operation(name: str, *tensors) -> None:
    """
    Validate that tensors are compatible for an operation.
    
    Args:
        name: Name of the operation being performed
        *tensors: Variable number of tensor objects to validate
        
    Raises:
        ValueError: If tensors are incompatible
    """
    # TODO: Implement tensor validation logic
    pass


# TODO: Implement hello_tinytorch() function here for the setup project
# 
# def hello_tinytorch() -> str:
#     """
#     Return a greeting message for new TinyTorch users.
#     
#     Returns:
#         A welcoming message string
#     """
#     # Your implementation goes here
#     pass 