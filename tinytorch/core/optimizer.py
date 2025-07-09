"""
Optimization algorithms for training neural networks.

This module contains optimizers that update model parameters based on gradients:
- Base Optimizer class
- Stochastic Gradient Descent (SGD)
- Adam optimizer
- Learning rate scheduling
"""

import numpy as np
from typing import List, Optional, Union
from .tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers.
    
    All optimizers should inherit from this class and implement the step() method.
    """
    
    def __init__(self, parameters: List[Tensor], lr: float):
        """
        Initialize the optimizer.
        
        Args:
            parameters: List of parameters to optimize
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr
        self.step_count = 0
    
    def zero_grad(self) -> None:
        """Zero the gradients of all parameters."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.data.fill(0.0)
    
    def step(self) -> None:
        """
        Perform a single optimization step.
        
        This method should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Implements SGD with optional momentum:
    v = momentum * v + lr * grad
    param = param - v
    
    Args:
        parameters: List of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(
        self, 
        parameters: List[Tensor], 
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """Initialize SGD optimizer."""
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers
        self.velocity = []
        for param in parameters:
            self.velocity.append(np.zeros_like(param.data))
    
    def step(self) -> None:
        """
        Perform a single SGD optimization step.
        """
        # TODO: Implement SGD update in Chapter 8
        self.step_count += 1
        raise NotImplementedError("SGD step will be implemented in Chapter 8")


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Implements Adam algorithm with bias correction:
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    
    Args:
        parameters: List of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """Initialize Adam optimizer."""
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.m = []  # First moment (mean)
        self.v = []  # Second moment (uncentered variance)
        
        for param in parameters:
            self.m.append(np.zeros_like(param.data))
            self.v.append(np.zeros_like(param.data))
    
    def step(self) -> None:
        """
        Perform a single Adam optimization step.
        """
        # TODO: Implement Adam update in Chapter 8
        self.step_count += 1
        raise NotImplementedError("Adam step will be implemented in Chapter 8") 