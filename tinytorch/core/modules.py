"""
Neural network modules and layers.

This module contains the building blocks for constructing neural networks:
- Base Module class for all layers
- Linear (fully connected) layers  
- Convolutional layers (Conv2d)
- Pooling layers (MaxPool2d)
- Activation functions
- Model composition utilities
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from .tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    
    All layers and models should inherit from this class. It provides
    the basic infrastructure for parameter management, forward/backward
    passes, and training/evaluation modes.
    """
    
    def __init__(self):
        """Initialize the module."""
        self.training = True
        self._parameters = {}
        self._modules = {}
    
    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass of the module.
        
        This method should be overridden by all subclasses to define
        the computation performed at every call.
        
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Make the module callable."""
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> List[Tensor]:
        """
        Return all parameters of the module.
        
        Returns:
            List of all parameter tensors
        """
        params = []
        for param in self._parameters.values():
            if isinstance(param, Tensor):
                params.append(param)
        
        # Recursively get parameters from submodules
        for module in self._modules.values():
            if isinstance(module, Module):
                params.extend(module.parameters())
                
        return params
    
    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set the module in evaluation mode."""
        return self.train(False)


class Linear(Module):
    """
    Linear (fully connected) layer.
    
    Applies a linear transformation: y = xW^T + b
    
    Args:
        in_features: Size of input features
        out_features: Size of output features  
        bias: Whether to include bias term
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize the linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        
        # Initialize parameters
        # Xavier/Glorot initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.normal(0, std, (out_features, in_features)),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # TODO: Implement matrix multiplication in Chapter 3
        # y = x @ W^T + b
        raise NotImplementedError("Linear forward pass will be implemented in Chapter 3")


class Conv2d(Module):
    """
    2D Convolutional layer.
    
    Applies 2D convolution over input tensor.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding added to input
        bias: Whether to include bias term
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True
    ):
        """Initialize the convolutional layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        self.bias_enabled = bias
        
        # Initialize parameters
        # He initialization for ReLU networks
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = np.sqrt(2.0 / fan_in)
        
        self.weight = Tensor(
            np.random.normal(0, std, (out_channels, in_channels, *self.kernel_size)),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of convolutional layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor after convolution
        """
        # TODO: Implement convolution in Chapter 4
        raise NotImplementedError("Conv2d forward pass will be implemented in Chapter 4")


class MaxPool2d(Module):
    """
    2D Max pooling layer.
    
    Applies 2D max pooling over input tensor.
    
    Args:
        kernel_size: Size of pooling kernel
        stride: Stride of pooling (defaults to kernel_size)
        padding: Padding added to input
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        """Initialize the max pooling layer."""
        super().__init__()
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle stride (default to kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of max pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor after max pooling
        """
        # TODO: Implement max pooling in Chapter 4
        raise NotImplementedError("MaxPool2d forward pass will be implemented in Chapter 4")


class ReLU(Module):
    """
    ReLU activation function.
    
    Applies ReLU: f(x) = max(0, x)
    """
    
    def __init__(self):
        """Initialize ReLU activation."""
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of ReLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after ReLU activation
        """
        # TODO: Implement ReLU in Chapter 3
        raise NotImplementedError("ReLU forward pass will be implemented in Chapter 3")


class Sequential(Module):
    """
    Sequential container for modules.
    
    Modules will be added in the order they are passed in the constructor.
    The forward() method accepts any input and forwards it through each
    module in sequence.
    """
    
    def __init__(self, *modules):
        """
        Initialize sequential container.
        
        Args:
            *modules: Variable number of modules to chain together
        """
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all modules in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all modules
        """
        for module in self._modules.values():
            x = module(x)
        return x 