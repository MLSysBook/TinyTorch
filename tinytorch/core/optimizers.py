# SIMPLIFIED OPTIMIZERS - Module 8: Basic gradient operations from Module 6

__all__ = ['gradient_descent_step', 'SGD', 'Adam']

import numpy as np
import sys
import os
from typing import List, Dict, Any, Optional, Union

# Import our existing components
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import Variable
except ImportError:
    # Create simplified fallback classes for basic gradient operations
    print("Warning: Using simplified classes for basic gradient operations")
    
    class Tensor:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
        
        def __str__(self):
            return f"Tensor({self.data})"
    
    class Variable:
        def __init__(self, data, requires_grad=True):
            if isinstance(data, (int, float)):
                self.data = Tensor([data])
            else:
                self.data = Tensor(data)
            self.requires_grad = requires_grad
            self.grad = None  # Simple gradient storage
        
        def zero_grad(self):
            """Reset gradients to None (basic operation from Module 6)"""
            self.grad = None
        
        def __str__(self):
            return f"Variable({self.data.data})"


def gradient_descent_step(parameter: Variable, learning_rate: float) -> None:
    """
    Perform one step of gradient descent on a parameter.
    
    Basic gradient operation from Module 6.
    """
    if parameter.grad is not None:
        # Get current parameter value and gradient (basic Module 6 operations)
        current_value = parameter.data.data
        gradient_value = parameter.grad.data.data
        
        # Update parameter: new_value = old_value - learning_rate * gradient
        new_value = current_value - learning_rate * gradient_value
        
        # Update parameter data
        parameter.data = Tensor(new_value)


class SGD:
    """
    Simplified SGD Optimizer
    
    Implements basic stochastic gradient descent with optional momentum.
    Uses simple gradient operations from Module 6.
    """
    
    def __init__(self, parameters: List[Variable], learning_rate: float = 0.01, 
                 momentum: float = 0.0):
        """Initialize SGD optimizer with basic parameters."""
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Simple momentum storage (using basic dict)
        self.velocity = {}
        for i, param in enumerate(parameters):
            if self.momentum > 0:
                self.velocity[i] = 0.0  # Initialize velocity to zero
    
    def step(self) -> None:
        """Perform one optimization step using basic gradient operations."""
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Get gradient (basic operation from Module 6)
                gradient = param.grad.data.data
                
                if self.momentum > 0:
                    # Apply momentum (simplified)
                    if i in self.velocity:
                        self.velocity[i] = self.momentum * self.velocity[i] + gradient
                    else:
                        self.velocity[i] = gradient
                    update = self.velocity[i]
                else:
                    # Simple gradient descent (no momentum)
                    update = gradient
                
                # Basic parameter update (like Module 6)
                new_value = param.data.data - self.learning_rate * update
                
                # Simple parameter data update (in-place modification)
                if hasattr(param.data.data, 'item'):
                    # Scalar parameter - create new tensor
                    param.data = Tensor(new_value)
                else:
                    # Array parameter - update in place
                    param.data.data[:] = new_value
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters (basic Module 6 operation)."""
        for param in self.parameters:
            param.zero_grad()


class Adam:
    """
    Simplified Adam Optimizer
    
    Implements a simplified version of Adam algorithm with adaptive learning rates.
    Educational focus on understanding optimization concepts rather than complex implementation.
    """
    
    def __init__(self, parameters: List[Variable], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """Initialize simplified Adam optimizer."""
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Simple moment storage (using basic dict with indices)
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (squared gradients)
        
        # Initialize moments for each parameter
        for i, param in enumerate(parameters):
            self.m[i] = 0.0
            self.v[i] = 0.0
        
        # Step counter for bias correction
        self.t = 0
    
    def step(self) -> None:
        """Perform one optimization step using simplified Adam algorithm."""
        self.t += 1  # Increment step counter
        
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Get gradient (basic operation from Module 6)
                gradient = param.grad.data.data
                
                # Update first moment (momentum)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient
                
                # Update second moment (squared gradients)
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradient * gradient
                
                # Bias correction
                m_corrected = self.m[i] / (1 - self.beta1 ** self.t)
                v_corrected = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Adaptive parameter update
                update = self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
                new_value = param.data.data - update
                
                # Simple parameter data update (like Module 6)
                if hasattr(param.data.data, 'item'):
                    # Scalar parameter - create new tensor
                    param.data = Tensor(new_value)
                else:
                    # Array parameter - update in place
                    param.data.data[:] = new_value
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters (basic Module 6 operation)."""
        for param in self.parameters:
            param.zero_grad()