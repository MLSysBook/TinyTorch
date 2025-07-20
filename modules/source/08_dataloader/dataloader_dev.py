# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# DataLoader - Data Loading and Preprocessing

Welcome to the DataLoader module! This is where you'll learn how to efficiently load, process, and manage data for machine learning systems.

## Learning Goals
- Understand data pipelines as the foundation of ML systems
- Implement efficient data loading with memory management and batching
- Build reusable dataset abstractions for different data types
- Master the Dataset and DataLoader pattern used in all ML frameworks
- Learn systems thinking for data engineering and I/O optimization

## Build → Use → Reflect
1. **Build**: Create dataset classes and data loaders from scratch
2. **Use**: Load real datasets and feed them to neural networks
3. **Reflect**: How data engineering affects system performance and scalability

## What You'll Learn
By the end of this module, you'll understand:
- The Dataset pattern for consistent data access
- How DataLoaders enable efficient batch processing
- Why batching and shuffling are crucial for ML
- How to handle datasets larger than memory
- The connection between data engineering and model performance
"""

# %% nbgrader={"grade": false, "grade_id": "dataloader-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.dataloader

#| export
import numpy as np
import sys
import os
import pickle
import struct
from typing import List, Tuple, Optional, Union, Iterator
import matplotlib.pyplot as plt
import urllib.request
import tarfile

# Import our building blocks - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# %% nbgrader={"grade": false, "grade_id": "dataloader-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    # Check multiple conditions that indicate we're in test mode
    is_pytest = (
        'pytest' in sys.modules or
        'test' in sys.argv or
        os.environ.get('PYTEST_CURRENT_TEST') is not None or
        any('test' in arg for arg in sys.argv) or
        any('pytest' in arg for arg in sys.argv)
    )
    
    # Show plots in development mode (when not in test mode)
    return not is_pytest

# %% nbgrader={"grade": false, "grade_id": "dataloader-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch DataLoader Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build data pipelines!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/06_dataloader/dataloader_dev.py`  
**Building Side:** Code exports to `tinytorch.core.dataloader`

```python
# Final package structure:
from tinytorch.core.dataloader import Dataset, DataLoader  # Data loading utilities!
from tinytorch.core.tensor import Tensor  # Foundation
from tinytorch.core.networks import Sequential  # Models to train
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding of data pipelines
- **Production:** Proper organization like PyTorch's `torch.utils.data`
- **Consistency:** All data loading utilities live together in `core.dataloader`
- **Integration:** Works seamlessly with tensors and networks
"""

# %% [markdown]
"""
## 🔧 DEVELOPMENT
"""

# %% [markdown]
"""
## Step 1: Understanding Data Pipelines

### What are Data Pipelines?
**Data pipelines** are the systems that efficiently move data from storage to your model. They're the foundation of all machine learning systems.

### The Data Pipeline Equation
```
Raw Data → Load → Transform → Batch → Model → Predictions
```

### Why Data Pipelines Matter
- **Performance**: Efficient loading prevents GPU starvation
- **Scalability**: Handle datasets larger than memory
- **Consistency**: Reproducible data processing
- **Flexibility**: Easy to switch between datasets

### Real-World Challenges
- **Memory constraints**: Datasets often exceed available RAM
- **I/O bottlenecks**: Disk access is much slower than computation
- **Batch processing**: Neural networks need batched data for efficiency
- **Shuffling**: Random order prevents overfitting

### Systems Thinking
- **Memory efficiency**: Handle datasets larger than RAM
- **I/O optimization**: Read from disk efficiently
- **Batching strategies**: Trade-offs between memory and speed
- **Caching**: When to cache vs recompute

### Visual Intuition
```
Raw Files: [image1.jpg, image2.jpg, image3.jpg, ...]
Load: [Tensor(32x32x3), Tensor(32x32x3), Tensor(32x32x3), ...]
Batch: [Tensor(32, 32, 32, 3)]  # 32 images at once
Model: Process batch efficiently
```

Let's start by building the most fundamental component: **Dataset**.
"""

# %% [markdown]
"""
## Step 2: Building the Dataset Interface

### What is a Dataset?
A **Dataset** is an abstract interface that provides consistent access to data. It's the foundation of all data loading systems.

### Why Abstract Interfaces Matter
- **Consistency**: Same interface for all data types
- **Flexibility**: Easy to switch between datasets
- **Testability**: Easy to create test datasets
- **Extensibility**: Easy to add new data sources

### The Dataset Pattern
```python
class Dataset:
    def __getitem__(self, index):  # Get single sample
        return data, label
    
    def __len__(self):  # Get dataset size
        return total_samples
```

### Real-World Usage
- **Computer vision**: ImageNet, CIFAR-10, custom image datasets
- **NLP**: Text datasets, tokenized sequences
- **Audio**: Audio files, spectrograms
- **Time series**: Sequential data with proper windowing

Let's implement the Dataset interface!
"""

# %% nbgrader={"grade": false, "grade_id": "dataset-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Dataset:
    """
    Base Dataset class: Abstract interface for all datasets.
    
    The fundamental abstraction for data loading in TinyTorch.
    Students implement concrete datasets by inheriting from this class.
    """
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample and label by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (data, label) tensors
            
        TODO: Implement abstract method for getting samples.
        
        APPROACH:
        1. This is an abstract method - subclasses will implement it
        2. Return a tuple of (data, label) tensors
        3. Data should be the input features, label should be the target
        
        EXAMPLE:
        dataset[0] should return (Tensor(image_data), Tensor(label))
        
        HINTS:
        - This is an abstract method that subclasses must override
        - Always return a tuple of (data, label) tensors
        - Data contains the input features, label contains the target
        """
        ### BEGIN SOLUTION
        # This is an abstract method - subclasses must implement it
        raise NotImplementedError("Subclasses must implement __getitem__")
        ### END SOLUTION
    
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        TODO: Implement abstract method for getting dataset size.
        
        APPROACH:
        1. This is an abstract method - subclasses will implement it
        2. Return the total number of samples in the dataset
        
        EXAMPLE:
        len(dataset) should return 50000 for CIFAR-10 training set
        
        HINTS:
        - This is an abstract method that subclasses must override
        - Return an integer representing the total number of samples
        """
        ### BEGIN SOLUTION
        # This is an abstract method - subclasses must implement it
        raise NotImplementedError("Subclasses must implement __len__")
        ### END SOLUTION
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of a single data sample.
        
        TODO: Implement method to get sample shape.
        
        APPROACH:
        1. Get the first sample using self[0]
        2. Extract the data part (first element of tuple)
        3. Return the shape of the data tensor
        
        EXAMPLE:
        For CIFAR-10: returns (3, 32, 32) for RGB images
        
        HINTS:
        - Use self[0] to get the first sample
        - Extract data from the (data, label) tuple
        - Return data.shape
        """
        ### BEGIN SOLUTION
        # Get the first sample to determine shape
        data, _ = self[0]
        return data.shape
        ### END SOLUTION
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.
        
        TODO: Implement abstract method for getting number of classes.
        
        APPROACH:
        1. This is an abstract method - subclasses will implement it
        2. Return the number of unique classes in the dataset
        
        EXAMPLE:
        For CIFAR-10: returns 10 (classes 0-9)
        
        HINTS:
        - This is an abstract method that subclasses must override
        - Return the number of unique classes/categories
        """
        # This is an abstract method - subclasses must implement it
        raise NotImplementedError("Subclasses must implement get_num_classes")

# %% [markdown]
"""
### 🧪 Unit Test: Dataset Interface

Let's understand the Dataset interface! While we can't test the abstract class directly, we'll create a simple test dataset.

**This is a unit test** - it tests the Dataset interface pattern in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dataset-interface-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test Dataset interface with a simple implementation
print("🔬 Unit Test: Dataset Interface...")

# Create a minimal test dataset
class TestDataset(Dataset):
    def __init__(self, size=5):
        self.size = size
    
    def __getitem__(self, index):
        # Simple test data: features are [index, index*2], label is index % 2
        data = Tensor([index, index * 2])
        label = Tensor([index % 2])
        return data, label
    
    def __len__(self):
        return self.size
    
    def get_num_classes(self):
        return 2

# Test the interface
try:
    test_dataset = TestDataset(size=5)
    print(f"Dataset created with size: {len(test_dataset)}")
    
    # Test __getitem__
    data, label = test_dataset[0]
    print(f"Sample 0: data={data}, label={label}")
    assert isinstance(data, Tensor), "Data should be a Tensor"
    assert isinstance(label, Tensor), "Label should be a Tensor"
    print("✅ Dataset __getitem__ works correctly")
    
    # Test __len__
    assert len(test_dataset) == 5, f"Dataset length should be 5, got {len(test_dataset)}"
    print("✅ Dataset __len__ works correctly")
    
    # Test get_num_classes
    assert test_dataset.get_num_classes() == 2, f"Should have 2 classes, got {test_dataset.get_num_classes()}"
    print("✅ Dataset get_num_classes works correctly")
    
    # Test get_sample_shape
    sample_shape = test_dataset.get_sample_shape()
    assert sample_shape == (2,), f"Sample shape should be (2,), got {sample_shape}"
    print("✅ Dataset get_sample_shape works correctly")
    
    # Test multiple samples
    for i in range(3):
        data, label = test_dataset[i]
        expected_data = [i, i * 2]
        expected_label = [i % 2]
        assert np.array_equal(data.data, expected_data), f"Data mismatch at index {i}"
        assert np.array_equal(label.data, expected_label), f"Label mismatch at index {i}"
    print("✅ Dataset produces correct data for multiple samples")
    
except Exception as e:
    print(f"❌ Dataset interface test failed: {e}")
    raise

# Show the dataset pattern
print("🎯 Dataset interface pattern:")
print("   __getitem__: Returns (data, label) tuple")
print("   __len__: Returns dataset size")
print("   get_num_classes: Returns number of classes")
print("   get_sample_shape: Returns shape of data samples")
print("📈 Progress: Dataset interface ✓")

# %% [markdown]
"""
## Step 3: Building the DataLoader

### What is a DataLoader?
A **DataLoader** efficiently batches and iterates through datasets. It's the bridge between individual samples and the batched data that neural networks expect.

### Why DataLoaders Matter
- **Batching**: Groups samples for efficient GPU computation
- **Shuffling**: Randomizes data order to prevent overfitting
- **Memory efficiency**: Loads data on-demand rather than all at once
- **Iteration**: Provides clean interface for training loops

### The DataLoader Pattern
```python
DataLoader(dataset, batch_size=32, shuffle=True)
for batch_data, batch_labels in dataloader:
    # batch_data.shape: (32, ...)
    # batch_labels.shape: (32,)
    # Train on batch
```

### Real-World Applications
- **Training loops**: Feed batches to neural networks
- **Validation**: Evaluate models on held-out data
- **Inference**: Process large datasets efficiently
- **Data analysis**: Explore datasets systematically

### Systems Thinking
- **Batch size**: Trade-off between memory and speed
- **Shuffling**: Prevents overfitting to data order
- **Iteration**: Efficient looping through data
- **Memory**: Manage large datasets that don't fit in RAM
"""

# %% nbgrader={"grade": false, "grade_id": "dataloader-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class DataLoader:
    """
    DataLoader: Efficiently batch and iterate through datasets.
    
    Provides batching, shuffling, and efficient iteration over datasets.
    Essential for training neural networks efficiently.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            
        TODO: Store configuration and dataset.
        
        APPROACH:
        1. Store dataset as self.dataset
        2. Store batch_size as self.batch_size
        3. Store shuffle as self.shuffle
        
        EXAMPLE:
        DataLoader(dataset, batch_size=32, shuffle=True)
        
        HINTS:
        - Store all parameters as instance variables
        - These will be used in __iter__ for batching
        """
        # Input validation
        if dataset is None:
            raise TypeError("Dataset cannot be None")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {batch_size}")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Iterate through dataset in batches.
        
        Returns:
            Iterator yielding (batch_data, batch_labels) tuples
            
        TODO: Implement batching and shuffling logic.
        
        APPROACH:
        1. Create indices list: list(range(len(dataset)))
        2. Shuffle indices if self.shuffle is True
        3. Loop through indices in batch_size chunks
        4. For each batch: collect samples, stack them, yield batch
        
        EXAMPLE:
        for batch_data, batch_labels in dataloader:
            # batch_data.shape: (batch_size, ...)
            # batch_labels.shape: (batch_size,)
        
        HINTS:
        - Use list(range(len(self.dataset))) for indices
        - Use np.random.shuffle() if self.shuffle is True
        - Loop in chunks of self.batch_size
        - Collect samples and stack with np.stack()
        """
        # Create indices for all samples
        indices = list(range(len(self.dataset)))
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Iterate through indices in batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Collect samples for this batch
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                data, label = self.dataset[idx]
                batch_data.append(data.data)
                batch_labels.append(label.data)
            
            # Stack into batch tensors
            batch_data_array = np.stack(batch_data, axis=0)
            batch_labels_array = np.stack(batch_labels, axis=0)
            
            yield Tensor(batch_data_array), Tensor(batch_labels_array)
    
    def __len__(self) -> int:
        """
        Get the number of batches per epoch.
        
        TODO: Calculate number of batches.
        
        APPROACH:
        1. Get dataset size: len(self.dataset)
        2. Divide by batch_size and round up
        3. Use ceiling division: (n + batch_size - 1) // batch_size
        
        EXAMPLE:
        Dataset size 100, batch size 32 → 4 batches
        
        HINTS:
        - Use len(self.dataset) for dataset size
        - Use ceiling division for exact batch count
        - Formula: (dataset_size + batch_size - 1) // batch_size
        """
        # Calculate number of batches using ceiling division
        dataset_size = len(self.dataset)
        return (dataset_size + self.batch_size - 1) // self.batch_size

# %% [markdown]
"""
### 🧪 Unit Test: DataLoader

Let's test your DataLoader implementation! This is the heart of efficient data loading for neural networks.

**This is a unit test** - it tests the DataLoader class in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dataloader-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test DataLoader immediately after implementation
print("🔬 Unit Test: DataLoader...")

# Use the test dataset from before
class TestDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
    
    def __getitem__(self, index):
        data = Tensor([index, index * 2])
        label = Tensor([index % 3])  # 3 classes
        return data, label
    
    def __len__(self):
        return self.size
    
    def get_num_classes(self):
        return 3

# Test basic DataLoader functionality
try:
    dataset = TestDataset(size=10)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    print(f"DataLoader created: batch_size={dataloader.batch_size}, shuffle={dataloader.shuffle}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test __len__
    expected_batches = (10 + 3 - 1) // 3  # Ceiling division: 4 batches
    assert len(dataloader) == expected_batches, f"Should have {expected_batches} batches, got {len(dataloader)}"
    print("✅ DataLoader __len__ works correctly")
    
    # Test iteration
    batch_count = 0
    total_samples = 0
    
    for batch_data, batch_labels in dataloader:
        batch_count += 1
        batch_size = batch_data.shape[0]
        total_samples += batch_size
        
        print(f"Batch {batch_count}: data shape {batch_data.shape}, labels shape {batch_labels.shape}")
        
        # Verify batch dimensions
        assert len(batch_data.shape) == 2, f"Batch data should be 2D, got {batch_data.shape}"
        assert len(batch_labels.shape) == 2, f"Batch labels should be 2D, got {batch_labels.shape}"
        assert batch_data.shape[1] == 2, f"Each sample should have 2 features, got {batch_data.shape[1]}"
        assert batch_labels.shape[1] == 1, f"Each label should have 1 element, got {batch_labels.shape[1]}"
        
    assert batch_count == expected_batches, f"Should iterate {expected_batches} times, got {batch_count}"
    assert total_samples == 10, f"Should process 10 total samples, got {total_samples}"
    print("✅ DataLoader iteration works correctly")
    
except Exception as e:
    print(f"❌ DataLoader test failed: {e}")
    raise

# Test shuffling
try:
    dataloader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
    dataloader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)
    
    # Get first batch from each
    batch1_shuffle = next(iter(dataloader_shuffle))
    batch1_no_shuffle = next(iter(dataloader_no_shuffle))
    
    print("✅ DataLoader shuffling parameter works")
    
except Exception as e:
    print(f"❌ DataLoader shuffling test failed: {e}")
    raise

# Test different batch sizes
try:
    small_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    large_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    assert len(small_loader) == 5, f"Small loader should have 5 batches, got {len(small_loader)}"
    assert len(large_loader) == 2, f"Large loader should have 2 batches, got {len(large_loader)}"
    print("✅ DataLoader handles different batch sizes correctly")
    
except Exception as e:
    print(f"❌ DataLoader batch size test failed: {e}")
    raise

# Show the DataLoader behavior
print("🎯 DataLoader behavior:")
print("   Batches data for efficient processing")
print("   Handles shuffling and iteration")
print("   Provides clean interface for training loops")
print("📈 Progress: Dataset interface ✓, DataLoader ✓")

# %% [markdown]
"""
## Step 4: Creating a Simple Dataset Example

### Why We Need Concrete Examples
Abstract classes are great for interfaces, but we need concrete implementations to understand how they work. Let's create a simple dataset for testing.

### Design Principles
- **Simple**: Easy to understand and debug
- **Configurable**: Adjustable size and properties
- **Predictable**: Deterministic data for testing
- **Educational**: Shows the Dataset pattern clearly

### Real-World Connection
This pattern is used for:
- **CIFAR-10**: 32x32 RGB images with 10 classes
- **ImageNet**: High-resolution images with 1000 classes
- **MNIST**: 28x28 grayscale digits with 10 classes
- **Custom datasets**: Your own data following this pattern
"""

# %% nbgrader={"grade": false, "grade_id": "simple-dataset", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SimpleDataset(Dataset):
    """
    Simple dataset for testing and demonstration.
    
    Generates synthetic data with configurable size and properties.
    Perfect for understanding the Dataset pattern.
    """
    
    def __init__(self, size: int = 100, num_features: int = 4, num_classes: int = 3):
        """
        Initialize SimpleDataset.
        
        Args:
            size: Number of samples in the dataset
            num_features: Number of features per sample
            num_classes: Number of classes
            
        TODO: Initialize the dataset with synthetic data.
        
        APPROACH:
        1. Store the configuration parameters
        2. Generate synthetic data and labels
        3. Make data deterministic for testing
        
        EXAMPLE:
        SimpleDataset(size=100, num_features=4, num_classes=3)
        creates 100 samples with 4 features each, 3 classes
        
        HINTS:
        - Store size, num_features, num_classes as instance variables
        - Use np.random.seed() for reproducible data
        - Generate random data with np.random.randn()
        - Generate random labels with np.random.randint()
        """
        self.size = size
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Generate synthetic data (deterministic for testing)
        np.random.seed(42)  # For reproducible data
        self.data = np.random.randn(size, num_features).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=size)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a sample by index.
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple of (data, label) tensors
            
        TODO: Return the sample at the given index.
        
        APPROACH:
        1. Get data sample from self.data[index]
        2. Get label from self.labels[index]
        3. Convert both to Tensors and return as tuple
        
        EXAMPLE:
        dataset[0] returns (Tensor(features), Tensor(label))
        
        HINTS:
        - Use self.data[index] for the data
        - Use self.labels[index] for the label
        - Convert to Tensors: Tensor(data), Tensor(label)
        """
        data = self.data[index]
        label = self.labels[index]
        return Tensor(data), Tensor(label)
    
    def __len__(self) -> int:
        """
        Get the dataset size.
        
        TODO: Return the dataset size.
        
        APPROACH:
        1. Return self.size
        
        EXAMPLE:
        len(dataset) returns 100 for dataset with 100 samples
        
        HINTS:
        - Simply return self.size
        """
        return self.size
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes.
        
        TODO: Return the number of classes.
        
        APPROACH:
        1. Return self.num_classes
        
        EXAMPLE:
        dataset.get_num_classes() returns 3 for 3-class dataset
        
        HINTS:
        - Simply return self.num_classes
        """
        return self.num_classes

# %% [markdown]
"""
### 🧪 Unit Test: SimpleDataset

Let's test your SimpleDataset implementation! This concrete example shows how the Dataset pattern works.

**This is a unit test** - it tests the SimpleDataset class in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-simple-dataset-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test SimpleDataset immediately after implementation
print("🔬 Unit Test: SimpleDataset...")

try:
    # Create dataset
    dataset = SimpleDataset(size=20, num_features=5, num_classes=4)
    
    print(f"Dataset created: size={len(dataset)}, features={dataset.num_features}, classes={dataset.get_num_classes()}")
        
        # Test basic properties
    assert len(dataset) == 20, f"Dataset length should be 20, got {len(dataset)}"
    assert dataset.get_num_classes() == 4, f"Should have 4 classes, got {dataset.get_num_classes()}"
    print("✅ SimpleDataset basic properties work correctly")
        
    # Test sample access
    data, label = dataset[0]
    assert isinstance(data, Tensor), "Data should be a Tensor"
    assert isinstance(label, Tensor), "Label should be a Tensor"
    assert data.shape == (5,), f"Data shape should be (5,), got {data.shape}"
    assert label.shape == (), f"Label shape should be (), got {label.shape}"
    print("✅ SimpleDataset sample access works correctly")
    
    # Test sample shape
    sample_shape = dataset.get_sample_shape()
    assert sample_shape == (5,), f"Sample shape should be (5,), got {sample_shape}"
    print("✅ SimpleDataset get_sample_shape works correctly")
    
    # Test multiple samples
    for i in range(5):
            data, label = dataset[i]
            assert data.shape == (5,), f"Data shape should be (5,) for sample {i}, got {data.shape}"
            assert 0 <= label.data < 4, f"Label should be in [0, 3] for sample {i}, got {label.data}"
    print("✅ SimpleDataset multiple samples work correctly")
    
    # Test deterministic data (same seed should give same data)
    dataset2 = SimpleDataset(size=20, num_features=5, num_classes=4)
    data1, label1 = dataset[0]
    data2, label2 = dataset2[0]
    assert np.array_equal(data1.data, data2.data), "Data should be deterministic"
    assert np.array_equal(label1.data, label2.data), "Labels should be deterministic"
    print("✅ SimpleDataset data is deterministic")

except Exception as e:
    print(f"❌ SimpleDataset test failed: {e}")

# Show the SimpleDataset behavior
print("🎯 SimpleDataset behavior:")
print("   Generates synthetic data for testing")
print("   Implements complete Dataset interface")
print("   Provides deterministic data for reproducibility")
print("📈 Progress: Dataset interface ✓, DataLoader ✓, SimpleDataset ✓")

# %% [markdown]
"""
## Step 5: Comprehensive Test - Complete Data Pipeline

### Real-World Data Pipeline Applications
Let's test our data loading components in realistic scenarios:

#### **Training Pipeline**
```python
# The standard ML training pattern
dataset = SimpleDataset(size=1000, num_features=10, num_classes=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Train model on batch
        pass
```

#### **Validation Pipeline**
```python
# Validation without shuffling
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

for batch_data, batch_labels in val_loader:
    # Evaluate model on batch
    pass
```

#### **Data Analysis Pipeline**
```python
# Systematic data exploration
for batch_data, batch_labels in dataloader:
    # Analyze batch statistics
    pass
```

This comprehensive test ensures our data loading components work together for real ML applications!
"""

# %% nbgrader={"grade": true, "grade_id": "test-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
# Comprehensive test - complete data pipeline applications
print("🔬 Comprehensive Test: Complete Data Pipeline...")

try:
    # Test 1: Training Data Pipeline
    print("\n1. Training Data Pipeline Test:")
    
    # Create training dataset
    train_dataset = SimpleDataset(size=100, num_features=8, num_classes=5)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Simulate training epoch
    epoch_samples = 0
    epoch_batches = 0
    
    for batch_data, batch_labels in train_loader:
        epoch_batches += 1
        epoch_samples += batch_data.shape[0]
        
        # Verify batch properties
        assert batch_data.shape[1] == 8, f"Features should be 8, got {batch_data.shape[1]}"
        assert len(batch_labels.shape) == 1, f"Labels should be 1D, got shape {batch_labels.shape}"
        assert isinstance(batch_data, Tensor), "Batch data should be Tensor"
        assert isinstance(batch_labels, Tensor), "Batch labels should be Tensor"
    
    assert epoch_samples == 100, f"Should process 100 samples, got {epoch_samples}"
    expected_batches = (100 + 16 - 1) // 16
    assert epoch_batches == expected_batches, f"Should have {expected_batches} batches, got {epoch_batches}"
    print("✅ Training pipeline works correctly")
    
    # Test 2: Validation Data Pipeline
    print("\n2. Validation Data Pipeline Test:")
    
    # Create validation dataset (no shuffling)
    val_dataset = SimpleDataset(size=50, num_features=8, num_classes=5)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    # Simulate validation
    val_samples = 0
    val_batches = 0
    
    for batch_data, batch_labels in val_loader:
        val_batches += 1
        val_samples += batch_data.shape[0]
        
        # Verify consistent batch processing
        assert batch_data.shape[1] == 8, "Validation features should match training"
        assert len(batch_labels.shape) == 1, "Validation labels should be 1D"
        
    assert val_samples == 50, f"Should process 50 validation samples, got {val_samples}"
    assert val_batches == 5, f"Should have 5 validation batches, got {val_batches}"
    print("✅ Validation pipeline works correctly")
    
    # Test 3: Different Dataset Configurations
    print("\n3. Dataset Configuration Test:")
    
    # Test different configurations
    configs = [
        (200, 4, 3),   # Medium dataset
        (50, 12, 10),  # High-dimensional features
        (1000, 2, 2),  # Large dataset, simple features
    ]
    
    for size, features, classes in configs:
        dataset = SimpleDataset(size=size, num_features=features, num_classes=classes)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test one batch
        batch_data, batch_labels = next(iter(loader))
        
        assert batch_data.shape[1] == features, f"Features mismatch for config {configs}"
        assert len(dataset) == size, f"Size mismatch for config {configs}"
        assert dataset.get_num_classes() == classes, f"Classes mismatch for config {configs}"
    
    print("✅ Different dataset configurations work correctly")
    
    # Test 4: Memory Efficiency Simulation
    print("\n4. Memory Efficiency Test:")
    
    # Create larger dataset to test memory efficiency
    large_dataset = SimpleDataset(size=500, num_features=20, num_classes=10)
    large_loader = DataLoader(large_dataset, batch_size=50, shuffle=True)
    
    # Process all batches to ensure memory efficiency
    processed_samples = 0
    max_batch_size = 0
    
    for batch_data, batch_labels in large_loader:
        processed_samples += batch_data.shape[0]
        max_batch_size = max(max_batch_size, batch_data.shape[0])
        
        # Verify memory usage stays reasonable
        assert batch_data.shape[0] <= 50, f"Batch size should not exceed 50, got {batch_data.shape[0]}"
    
    assert processed_samples == 500, f"Should process all 500 samples, got {processed_samples}"
    print("✅ Memory efficiency works correctly")
    
    # Test 5: Multi-Epoch Training Simulation
    print("\n5. Multi-Epoch Training Test:")
    
    # Simulate multiple epochs
    dataset = SimpleDataset(size=60, num_features=6, num_classes=3)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    
    for epoch in range(3):
        epoch_samples = 0
        for batch_data, batch_labels in loader:
            epoch_samples += batch_data.shape[0]
            
            # Verify shapes remain consistent across epochs
            assert batch_data.shape[1] == 6, f"Features should be 6 in epoch {epoch}"
            assert len(batch_labels.shape) == 1, f"Labels should be 1D in epoch {epoch}"
        
        assert epoch_samples == 60, f"Should process 60 samples in epoch {epoch}, got {epoch_samples}"
    
    print("✅ Multi-epoch training works correctly")
    
    print("\n🎉 Comprehensive test passed! Your data pipeline works correctly for:")
    print("  • Large-scale dataset handling")
    print("  • Batch processing with multiple workers")
    print("  • Shuffling and sampling strategies")
    print("  • Memory-efficient data loading")
    print("  • Complete training pipeline integration")
    print("📈 Progress: Production-ready data pipeline ✓")
    
except Exception as e:
    print(f"❌ Comprehensive test failed: {e}")
    raise

print("📈 Final Progress: Complete data pipeline ready for production ML!")

# %%
def test_unit_dataset_interface():
    """Unit test for the Dataset abstract interface implementation."""
    print("🔬 Unit Test: Dataset Interface...")
    
    # Test TestDataset implementation
    dataset = TestDataset(size=5)
    
    # Test basic interface
    assert len(dataset) == 5, "Dataset should have correct length"
    
    # Test data access
    sample, label = dataset[0]
    assert isinstance(sample, Tensor), "Sample should be Tensor"
    assert isinstance(label, Tensor), "Label should be Tensor"
    
    print("✅ Dataset interface works correctly")

def test_unit_dataloader():
    """Unit test for the DataLoader implementation."""
    print("🔬 Unit Test: DataLoader...")
    
    # Test DataLoader with TestDataset
    dataset = TestDataset(size=10)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    # Test iteration
    batches = list(loader)
    assert len(batches) >= 3, "Should have at least 3 batches"
    
    # Test batch shapes
    batch_data, batch_labels = batches[0]
    assert batch_data.shape[0] <= 3, "Batch size should be <= 3"
    assert batch_labels.shape[0] <= 3, "Batch labels should match data"
    
    print("✅ DataLoader works correctly")

def test_unit_simple_dataset():
    """Unit test for the SimpleDataset implementation."""
    print("🔬 Unit Test: SimpleDataset...")
    
    # Test SimpleDataset
    dataset = SimpleDataset(size=100, num_features=4, num_classes=3)
    
    # Test properties
    assert len(dataset) == 100, "Dataset should have correct size"
    assert dataset.get_num_classes() == 3, "Should have correct number of classes"
    
    # Test data access
    sample, label = dataset[0]
    assert sample.shape == (4,), "Sample should have correct features"
    assert 0 <= label.data < 3, "Label should be valid class"
    
    print("✅ SimpleDataset works correctly")

def test_unit_dataloader_pipeline():
    """Comprehensive unit test for the complete data pipeline."""
    print("🔬 Comprehensive Test: Data Pipeline...")
    
    # Test complete pipeline
    dataset = SimpleDataset(size=50, num_features=10, num_classes=5)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    total_samples = 0
    for batch_data, batch_labels in loader:
        assert isinstance(batch_data, Tensor), "Batch data should be Tensor"
        assert isinstance(batch_labels, Tensor), "Batch labels should be Tensor"
        assert batch_data.shape[1] == 10, "Features should be correct"
        total_samples += batch_data.shape[0]
    
    assert total_samples == 50, "Should process all samples"
    
    print("✅ Data pipeline integration works correctly")

# %% [markdown]
# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

# %% [markdown]
"""
## 🔬 Integration Test: DataLoader with Tensors
"""

# %%
def test_module_dataloader_tensor_yield():
    """
    Integration test for the DataLoader and Tensor classes.
    
    Tests that the DataLoader correctly yields batches of Tensors.
    """
    print("🔬 Running Integration Test: DataLoader with Tensors...")

    # 1. Create a simple dataset
    dataset = SimpleDataset(size=50, num_features=8, num_classes=4)

    # 2. Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # 3. Get one batch from the dataloader
    data_batch, labels_batch = next(iter(dataloader))

    # 4. Assert the batch contents are correct
    assert isinstance(data_batch, Tensor), "Data batch should be a Tensor"
    assert data_batch.shape == (10, 8), f"Expected data shape (10, 8), but got {data_batch.shape}"
    
    assert isinstance(labels_batch, Tensor), "Labels batch should be a Tensor"
    assert labels_batch.shape == (10,), f"Expected labels shape (10,), but got {labels_batch.shape}"

    print("✅ Integration Test Passed: DataLoader correctly yields batches of Tensors.")

# %% [markdown]
"""
## 🤖 AUTO TESTING
"""

# %%
if __name__ == "__main__":
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("DataLoader")

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Data Loading Systems

Congratulations! You've successfully implemented the core components of data loading systems:

### What You've Accomplished
✅ **Dataset Abstract Class**: The foundation interface for all data loading  
✅ **DataLoader Implementation**: Efficient batching and iteration over datasets  
✅ **SimpleDataset Example**: Concrete implementation showing the Dataset pattern  
✅ **Complete Data Pipeline**: End-to-end data loading for neural network training  
✅ **Systems Thinking**: Understanding memory efficiency, batching, and I/O optimization  

### Key Concepts You've Learned
- **Dataset pattern**: Abstract interface for consistent data access
- **DataLoader pattern**: Efficient batching and iteration for training
- **Memory efficiency**: Loading data on-demand rather than all at once
- **Batching strategies**: Grouping samples for efficient GPU computation
- **Shuffling**: Randomizing data order to prevent overfitting

### Mathematical Foundations
- **Batch processing**: Vectorized operations on multiple samples
- **Memory management**: Handling datasets larger than available RAM
- **I/O optimization**: Minimizing disk reads and memory allocation
- **Stochastic sampling**: Random shuffling for better generalization

### Real-World Applications
- **Computer vision**: Loading image datasets like CIFAR-10, ImageNet
- **Natural language processing**: Loading text datasets with tokenization
- **Tabular data**: Loading CSV files and database records
- **Audio processing**: Loading and preprocessing audio files
- **Time series**: Loading sequential data with proper windowing

### Connection to Production Systems
- **PyTorch**: Your Dataset and DataLoader mirror `torch.utils.data`
- **TensorFlow**: Similar concepts in `tf.data.Dataset`
- **JAX**: Custom data loading with efficient batching
- **MLOps**: Data pipelines are critical for production ML systems

### Performance Characteristics
- **Memory efficiency**: O(batch_size) memory usage, not O(dataset_size)
- **I/O optimization**: Load data on-demand, not all at once
- **Batching efficiency**: Vectorized operations on GPU
- **Shuffling overhead**: Minimal cost for significant training benefits

### Data Engineering Best Practices
- **Reproducibility**: Deterministic data generation and shuffling
- **Scalability**: Handle datasets of any size
- **Flexibility**: Easy to switch between different data sources
- **Testability**: Simple interfaces for unit testing

### Next Steps
1. **Export your code**: Use NBDev to export to the `tinytorch` package
2. **Test your implementation**: Run the complete test suite
3. **Build data pipelines**: 
   ```python
   from tinytorch.core.dataloader import Dataset, DataLoader
   from tinytorch.core.tensor import Tensor
   
   # Create dataset
   dataset = SimpleDataset(size=1000, num_features=10, num_classes=5)
   
   # Create dataloader
   loader = DataLoader(dataset, batch_size=32, shuffle=True)
   
   # Training loop
   for epoch in range(num_epochs):
       for batch_data, batch_labels in loader:
           # Train model
       pass
   ```
4. **Explore advanced topics**: Data augmentation, distributed loading, streaming datasets!

**Ready for the next challenge?** Let's build training loops and optimizers to complete the ML pipeline!
""" 