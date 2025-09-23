"""
Base Module class for TinyTorch neural network layers.

This module provides the foundational Module class that enables:
- Automatic parameter registration
- Recursive parameter collection
- Clean composition of neural networks
- PyTorch-compatible interface

Students implement the core algorithms while this infrastructure 
provides the clean API patterns they expect.
"""

from typing import Iterator, List


class Module:
    """
    Base class for all neural network modules.
    
    Your models should subclass this class to automatically get:
    - Parameter registration when you set attributes
    - Recursive parameter collection via parameters()
    - Clean callable interface model(x) instead of model.forward(x)
    
    This matches PyTorch's nn.Module interface for familiar patterns.
    """
    
    def __init__(self):
        """Initialize module with parameter and submodule tracking."""
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_parameters', [])
        object.__setattr__(self, '_modules', [])
        object.__setattr__(self, '_initialized', True)
    
    def __setattr__(self, name: str, value):
        """
        Automatically register parameters and submodules.
        
        When you do: self.weight = Parameter(...), it gets auto-registered.
        When you do: self.layer = Linear(...), it gets auto-registered.
        """
        if not hasattr(self, '_initialized'):
            # Still in __init__, use normal assignment
            object.__setattr__(self, name, value)
            return
            
        # Check if this is a Parameter (has requires_grad attribute and is True)
        if hasattr(value, 'requires_grad') and value.requires_grad:
            if value not in self._parameters:
                self._parameters.append(value)
        
        # Check if this is a Module subclass
        elif isinstance(value, Module):
            if value not in self._modules:
                self._modules.append(value)
        
        # Normal attribute assignment
        object.__setattr__(self, name, value)
    
    def parameters(self) -> Iterator:
        """
        Return an iterator over module parameters.
        
        This is used by optimizers to find all trainable parameters:
        optimizer = Adam(model.parameters())
        """
        # Return our direct parameters
        for param in self._parameters:
            yield param
        
        # Recursively collect parameters from submodules
        for module in self._modules:
            for param in module.parameters():
                yield param
    
    def __call__(self, *args, **kwargs):
        """
        Make modules callable: model(x) calls model.forward(x).
        
        This is the standard PyTorch pattern that students expect.
        """
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        Define the forward pass computation.
        
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement forward()")