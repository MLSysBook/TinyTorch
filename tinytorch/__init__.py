__version__ = "0.1.0"

# Import core functionality
from . import core

# Import PyTorch-compatible modules
from . import nn
from . import optim

# Make common components easily accessible
from .core.tensor import Tensor
from .nn import Module

# Export main public API
__all__ = [
    'core',
    'nn', 
    'optim',
    'Tensor',
    'Module'
]
