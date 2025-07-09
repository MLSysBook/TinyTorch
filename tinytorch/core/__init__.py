"""
TinyTorch Core Module

Core components of the TinyTorch ML system including tensors, autograd,
neural network modules, optimizers, and training infrastructure.
"""

from .tensor import Tensor
from .modules import Module, Linear, Conv2d, MaxPool2d
from .optimizer import Optimizer, SGD, Adam
from .trainer import Trainer
from .mlops import (
    DataDriftDetector, 
    ModelMonitor, 
    ModelRegistry, 
    ABTestFramework, 
    ProductionServer
)

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "Module", 
    "Linear",
    "Conv2d", 
    "MaxPool2d",
    "Optimizer",
    "SGD",
    "Adam", 
    "Trainer",
    "DataDriftDetector",
    "ModelMonitor", 
    "ModelRegistry",
    "ABTestFramework",
    "ProductionServer"
] 