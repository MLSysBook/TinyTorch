"""
Data loading and preprocessing utilities.

This module provides efficient data loading for training neural networks:
- Dataset base class and common datasets (MNIST, CIFAR-10)
- DataLoader for batching and shuffling
- Data transformations and augmentation
- Memory-efficient data pipeline
"""

import numpy as np
from typing import Iterator, Optional, Tuple, List, Callable, Any
from .tensor import Tensor


class Dataset:
    """
    Base class for all datasets.
    
    All datasets should inherit from this class and implement __len__ 
    and __getitem__ methods.
    """
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (data, target) tensors
        """
        raise NotImplementedError("Subclasses must implement __getitem__")


class DataLoader:
    """
    Data loader for iterating over datasets in batches.
    
    Provides batching, shuffling, and parallel loading capabilities.
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data each epoch
        num_workers: Number of worker processes (not implemented yet)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0
    ):
        """Initialize the data loader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Generate indices
        self.indices = list(range(len(dataset)))
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Iterate over batches.
        
        Yields:
            Batches of (data, targets) as tensors
        """
        # TODO: Implement batching in Chapter 6
        raise NotImplementedError("DataLoader iteration will be implemented in Chapter 6")


class Transform:
    """Base class for data transformations."""
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply transformation to data.
        
        Args:
            data: Input data array
            
        Returns:
            Transformed data array
        """
        raise NotImplementedError("Subclasses must implement __call__")


class ToTensor(Transform):
    """Convert numpy array to Tensor."""
    
    def __call__(self, data: np.ndarray) -> Tensor:
        """Convert to tensor."""
        return Tensor(data)


class Normalize(Transform):
    """Normalize data with given mean and standard deviation."""
    
    def __init__(self, mean: float, std: float):
        """
        Initialize normalization transform.
        
        Args:
            mean: Mean for normalization
            std: Standard deviation for normalization
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        return (data - self.mean) / self.std


class Compose(Transform):
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Transform]):
        """
        Initialize composed transform.
        
        Args:
            transforms: List of transforms to apply in sequence
        """
        self.transforms = transforms
    
    def __call__(self, data: np.ndarray) -> Any:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            data = transform(data)
        return data 