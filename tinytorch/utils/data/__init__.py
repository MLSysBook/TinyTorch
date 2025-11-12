"""
TinyTorch Data Loading Utilities

Following torch.utils.data patterns, this module provides:
- Dataset: Base class for all datasets
- DataLoader: Batching and shuffling for training
- Common datasets for learning

This is Module 10 of TinyTorch.
"""

# Import from dataloader module
from .dataloader import *

# Make key classes easily accessible
__all__ = ['Dataset', 'DataLoader', 'SimpleDataset', 'CIFAR10Dataset']