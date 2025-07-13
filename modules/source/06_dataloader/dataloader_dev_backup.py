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
# Module 6: DataLoader - Data Loading and Preprocessing

Welcome to the DataLoader module! This is where you'll learn how to efficiently load, process, and manage data for machine learning systems.

## Learning Goals
- Understand data pipelines as the foundation of ML systems
- Implement efficient data loading with memory management and batching
- Build reusable dataset abstractions for different data types
- Master the Dataset and DataLoader pattern used in all ML frameworks
- Learn systems thinking for data engineering and I/O optimization

## Build → Use → Understand
1. **Build**: Create dataset classes and data loaders from scratch
2. **Use**: Load real datasets and feed them to neural networks
3. **Understand**: How data engineering affects system performance and scalability
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
## 🧠 The Mathematical Foundation of Data Engineering

### The Data Pipeline Equation
Every machine learning system follows this fundamental equation:

```
Model Performance = f(Data Quality × Data Quantity × Data Efficiency)
```

### Why Data Engineering is Critical
- **Data is the fuel**: Without proper data pipelines, nothing else works
- **I/O bottlenecks**: Data loading is often the biggest performance bottleneck
- **Memory management**: How you handle data affects everything else
- **Production reality**: Data pipelines are critical in real ML systems

### The Three Pillars of Data Engineering
1. **Abstraction**: Clean interfaces that hide complexity
2. **Efficiency**: Minimize I/O and memory overhead
3. **Scalability**: Handle datasets larger than memory

### Connection to Real ML Systems
Every framework uses the Dataset/DataLoader pattern:
- **PyTorch**: `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`
- **TensorFlow**: `tf.data.Dataset` with efficient data pipelines
- **JAX**: Custom data loading with `jax.numpy` integration
- **TinyTorch**: `tinytorch.core.dataloader.Dataset` and `DataLoader` (what we're building!)

### Performance Considerations
- **Memory efficiency**: Handle datasets larger than RAM
- **I/O optimization**: Read from disk efficiently with batching
- **Caching strategies**: When to cache vs recompute
- **Parallel processing**: Multi-threaded data loading
"""

# %% [markdown]
"""
## Step 1: Understanding Data Engineering

### What is Data Engineering?
**Data engineering** is the foundation of all machine learning systems. It involves loading, processing, and managing data efficiently so that models can learn from it.

### The Fundamental Insight
**Data engineering is about managing the flow of information through your system:**
```
Raw Data → Load → Preprocess → Batch → Feed to Model
```

### Real-World Examples
- **Image datasets**: CIFAR-10, ImageNet, MNIST
- **Text datasets**: Wikipedia, books, social media
- **Tabular data**: CSV files, databases, spreadsheets
- **Audio data**: Speech recordings, music files

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
        ### BEGIN SOLUTION
        # This is an abstract method - subclasses must implement it
        raise NotImplementedError("Subclasses must implement get_num_classes")
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Quick Test: Dataset Base Class

Let's understand the Dataset interface! While we can't test the abstract class directly, we'll create a simple test dataset.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dataset-interface-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test Dataset interface with a simple implementation
print("🔬 Testing Dataset interface...")

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
print("📈 Progress: Dataset interface ✓")

# %% [markdown]
"""
## Step 2: Building the DataLoader

### What is a DataLoader?
A **DataLoader** efficiently batches and iterates through datasets. It's the bridge between individual samples and the batched data that neural networks expect.

### Why DataLoaders Matter
- **Batching**: Groups samples for efficient GPU computation
- **Shuffling**: Randomizes data order to prevent overfitting
- **Memory efficiency**: Loads data on-demand rather than all at once
- **Iteration**: Provides clean interface for training loops

### The DataLoader Pattern
```
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
        ### BEGIN SOLUTION
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        ### END SOLUTION
    
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
        ### BEGIN SOLUTION
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
        ### END SOLUTION
    
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
        ### BEGIN SOLUTION
        # Calculate number of batches using ceiling division
        dataset_size = len(self.dataset)
        return (dataset_size + self.batch_size - 1) // self.batch_size
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Quick Test: DataLoader

Let's test your DataLoader implementation! This is the heart of efficient data loading for neural networks.
"""

# %% nbgrader={"grade": true, "grade_id": "test-dataloader-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
# Test DataLoader immediately after implementation
print("🔬 Testing DataLoader...")

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
## Step 3: Creating a Simple Dataset Example

### Why We Need Concrete Examples
Abstract classes are great for interfaces, but we need concrete implementations to understand how they work. Let's create a simple dataset for testing.

### Design Principles
- **Simple**: Easy to understand and debug
- **Configurable**: Adjustable size and properties
- **Predictable**: Deterministic data for testing
- **Educational**: Shows the Dataset pattern clearly
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
        ### BEGIN SOLUTION
        self.size = size
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Set seed for reproducible data
        np.random.seed(42)
        
        # Generate synthetic data
        self.data = np.random.randn(size, num_features).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=size)
        ### END SOLUTION
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample and label by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (data, label) tensors
            
        TODO: Return the sample and label at the given index.
        
        APPROACH:
        1. Get data at index from self.data
        2. Get label at index from self.labels
        3. Convert to tensors and return as tuple
        
        EXAMPLE:
        dataset[0] returns (Tensor([1.2, -0.5, 0.8, 0.1]), Tensor(2))
        
        HINTS:
        - Use self.data[index] and self.labels[index]
        - Convert to Tensor objects
        - Return as tuple (data, label)
        """
        ### BEGIN SOLUTION
        data = Tensor(self.data[index])
        label = Tensor(self.labels[index])
        return data, label
        ### END SOLUTION
    
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        TODO: Return the dataset size.
        
        HINTS:
        - Return self.size
        """
        ### BEGIN SOLUTION
        return self.size
        ### END SOLUTION
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.
        
        TODO: Return the number of classes.
        
        HINTS:
        - Return self.num_classes
        """
        ### BEGIN SOLUTION
        return self.num_classes
        ### END SOLUTION

# %% [markdown]
"""
## 🧪 Comprehensive DataLoader Testing Suite

Let's test all data loading components thoroughly with realistic ML data scenarios!
"""

# %% nbgrader={"grade": false, "grade_id": "test-dataloader-comprehensive", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_dataset_interface():
    """Test 1: Dataset interface comprehensive testing"""
    print("🔬 Testing Dataset Interface...")
    
    # Test 1.1: Abstract base class behavior
    try:
        # Test that we can't instantiate abstract Dataset
        try:
            base_dataset = Dataset()
            base_dataset[0]  # Should raise NotImplementedError
            assert False, "Should not be able to call abstract methods"
        except NotImplementedError:
            print("✅ Abstract Dataset correctly raises NotImplementedError")
    except Exception as e:
        print(f"❌ Abstract Dataset test failed: {e}")
        return False
    
    # Test 1.2: SimpleDataset implementation
    try:
        dataset = SimpleDataset(size=50, num_features=4, num_classes=3)
        
        # Test basic properties
        assert len(dataset) == 50, f"Dataset length should be 50, got {len(dataset)}"
        assert dataset.get_num_classes() == 3, f"Should have 3 classes, got {dataset.get_num_classes()}"
        
        # Test sample retrieval
        data, label = dataset[0]
        assert isinstance(data, Tensor), "Data should be a Tensor"
        assert isinstance(label, Tensor), "Label should be a Tensor"
        assert data.shape == (4,), f"Data shape should be (4,), got {data.shape}"
        
        # Test sample shape method
        sample_shape = dataset.get_sample_shape()
        assert sample_shape == (4,), f"Sample shape should be (4,), got {sample_shape}"
        
        print("✅ SimpleDataset implementation test passed")
    except Exception as e:
        print(f"❌ SimpleDataset implementation failed: {e}")
        return False
    
    # Test 1.3: Different dataset configurations
    try:
        # Small dataset
        small_dataset = SimpleDataset(size=5, num_features=2, num_classes=2)
        assert len(small_dataset) == 5, "Small dataset length wrong"
        assert small_dataset.get_num_classes() == 2, "Small dataset classes wrong"
        
        # Large dataset
        large_dataset = SimpleDataset(size=1000, num_features=10, num_classes=5)
        assert len(large_dataset) == 1000, "Large dataset length wrong"
        assert large_dataset.get_num_classes() == 5, "Large dataset classes wrong"
        
        # Test data consistency (seeded random)
        data1, _ = small_dataset[0]
        data2, _ = small_dataset[0]
        assert np.allclose(data1.data, data2.data), "Dataset should be deterministic"
        
        print("✅ Different dataset configurations test passed")
    except Exception as e:
        print(f"❌ Different dataset configurations failed: {e}")
        return False
    
    # Test 1.4: Edge cases and robustness
    try:
        # Test edge case: single sample
        single_dataset = SimpleDataset(size=1, num_features=1, num_classes=1)
        data, label = single_dataset[0]
        assert data.shape == (1,), "Single sample data shape wrong"
        assert isinstance(label.data, (int, np.integer)) or label.data.shape == (), "Single sample label wrong"
        
        # Test boundary indices
        dataset = SimpleDataset(size=10, num_features=3, num_classes=2)
        first_data, first_label = dataset[0]
        last_data, last_label = dataset[9]
        assert first_data.shape == (3,), "First sample shape wrong"
        assert last_data.shape == (3,), "Last sample shape wrong"
        
        print("✅ Edge cases and robustness test passed")
    except Exception as e:
        print(f"❌ Edge cases and robustness failed: {e}")
        return False
    
    print("🎯 Dataset interface: All tests passed!")
    return True

def test_dataloader_functionality():
    """Test 2: DataLoader functionality comprehensive testing"""
    print("🔬 Testing DataLoader Functionality...")
    
    # Test 2.1: Basic DataLoader operations
    try:
        dataset = SimpleDataset(size=32, num_features=4, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Test initialization
        assert dataloader.batch_size == 8, f"Batch size should be 8, got {dataloader.batch_size}"
        assert dataloader.shuffle == False, f"Shuffle should be False, got {dataloader.shuffle}"
        
        # Test length calculation
        expected_batches = (32 + 8 - 1) // 8  # Ceiling division: 4 batches
        assert len(dataloader) == expected_batches, f"Should have {expected_batches} batches, got {len(dataloader)}"
        
        print("✅ Basic DataLoader operations test passed")
    except Exception as e:
        print(f"❌ Basic DataLoader operations failed: {e}")
        return False
    
    # Test 2.2: Batch iteration and shapes
    try:
        dataset = SimpleDataset(size=25, num_features=3, num_classes=2)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        batch_count = 0
        total_samples = 0
        
        for batch_data, batch_labels in dataloader:
            batch_count += 1
            batch_size = batch_data.shape[0]
            total_samples += batch_size
            
            # Check batch shapes
            assert len(batch_data.shape) == 2, f"Batch data should be 2D, got {batch_data.shape}"
            assert batch_data.shape[1] == 3, f"Should have 3 features, got {batch_data.shape[1]}"
            assert batch_labels.shape[0] == batch_size, f"Labels should match batch size"
            
            # Check data types
            assert isinstance(batch_data, Tensor), "Batch data should be Tensor"
            assert isinstance(batch_labels, Tensor), "Batch labels should be Tensor"
        
        # Verify complete iteration
        assert total_samples == 25, f"Should process 25 samples, got {total_samples}"
        assert batch_count == 3, f"Should have 3 batches, got {batch_count}"  # 25/10 = 3 batches
        
        print("✅ Batch iteration and shapes test passed")
    except Exception as e:
        print(f"❌ Batch iteration and shapes failed: {e}")
        return False
    
    # Test 2.3: Different batch sizes
    try:
        dataset = SimpleDataset(size=100, num_features=5, num_classes=3)
        
        # Small batches
        small_loader = DataLoader(dataset, batch_size=7, shuffle=False)
        assert len(small_loader) == 15, f"Small loader should have 15 batches, got {len(small_loader)}"  # 100/7 = 15
        
        # Large batches
        large_loader = DataLoader(dataset, batch_size=30, shuffle=False)
        assert len(large_loader) == 4, f"Large loader should have 4 batches, got {len(large_loader)}"  # 100/30 = 4
        
        # Single sample batches
        single_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        assert len(single_loader) == 100, f"Single loader should have 100 batches, got {len(single_loader)}"
        
        print("✅ Different batch sizes test passed")
    except Exception as e:
        print(f"❌ Different batch sizes failed: {e}")
        return False
    
    # Test 2.4: Shuffling behavior
    try:
        dataset = SimpleDataset(size=20, num_features=2, num_classes=2)
        
        # Test with shuffling
        loader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
        loader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)
        
        # Get multiple batches to test shuffling
        shuffle_batches = list(loader_shuffle)
        no_shuffle_batches = list(loader_no_shuffle)
        
        assert len(shuffle_batches) == len(no_shuffle_batches), "Should have same number of batches"
        
        # Test that all original samples are present (just reordered)
        shuffle_all_data = np.concatenate([batch[0].data for batch in shuffle_batches])
        no_shuffle_all_data = np.concatenate([batch[0].data for batch in no_shuffle_batches])
        
        assert shuffle_all_data.shape == no_shuffle_all_data.shape, "Should have same total data shape"
        
        print("✅ Shuffling behavior test passed")
    except Exception as e:
        print(f"❌ Shuffling behavior failed: {e}")
        return False
    
    print("🎯 DataLoader functionality: All tests passed!")
    return True

def test_data_pipeline_scenarios():
    """Test 3: Real-world data pipeline scenarios"""
    print("🔬 Testing Data Pipeline Scenarios...")
    
    # Test 3.1: Image classification scenario
    try:
        # Simulate CIFAR-10 like dataset: 32x32 RGB images, 10 classes
        image_dataset = SimpleDataset(size=1000, num_features=32*32*3, num_classes=10)
        image_loader = DataLoader(image_dataset, batch_size=64, shuffle=True)
        
        # Test one epoch of training
        epoch_samples = 0
        for batch_data, batch_labels in image_loader:
            epoch_samples += batch_data.shape[0]
            
            # Verify image batch properties
            assert batch_data.shape[1] == 32*32*3, f"Should have 3072 features (32x32x3), got {batch_data.shape[1]}"
            assert batch_data.shape[0] <= 64, f"Batch size should be <= 64, got {batch_data.shape[0]}"
            
            # Simulate forward pass
            batch_size = batch_data.shape[0]
            assert batch_labels.shape[0] == batch_size, "Labels should match batch size"
        
        assert epoch_samples == 1000, f"Should process 1000 samples, got {epoch_samples}"
        print("✅ Image classification scenario test passed")
    except Exception as e:
        print(f"❌ Image classification scenario failed: {e}")
        return False
    
    # Test 3.2: Text classification scenario
    try:
        # Simulate text classification: 512 token embeddings, 5 sentiment classes
        text_dataset = SimpleDataset(size=500, num_features=512, num_classes=5)
        text_loader = DataLoader(text_dataset, batch_size=32, shuffle=True)
        
        # Test batch processing
        for batch_data, batch_labels in text_loader:
            # Verify text batch properties
            assert batch_data.shape[1] == 512, f"Should have 512 features, got {batch_data.shape[1]}"
            
            # Simulate text processing
            batch_size = batch_data.shape[0]
            assert batch_size <= 32, f"Batch size should be <= 32, got {batch_size}"
            break  # Just test first batch
        
        print("✅ Text classification scenario test passed")
    except Exception as e:
        print(f"❌ Text classification scenario failed: {e}")
        return False
    
    # Test 3.3: Tabular data scenario
    try:
        # Simulate tabular data: house prices with 20 features, 3 price ranges
        tabular_dataset = SimpleDataset(size=200, num_features=20, num_classes=3)
        tabular_loader = DataLoader(tabular_dataset, batch_size=16, shuffle=False)
        
        # Test systematic processing (no shuffling for tabular data)
        batch_count = 0
        for batch_data, batch_labels in tabular_loader:
            batch_count += 1
            
            # Verify tabular batch properties
            assert batch_data.shape[1] == 20, f"Should have 20 features, got {batch_data.shape[1]}"
            
            # Simulate tabular processing
            batch_size = batch_data.shape[0]
            assert batch_size <= 16, f"Batch size should be <= 16, got {batch_size}"
        
        expected_batches = (200 + 16 - 1) // 16  # 13 batches
        assert batch_count == expected_batches, f"Should have {expected_batches} batches, got {batch_count}"
        
        print("✅ Tabular data scenario test passed")
    except Exception as e:
        print(f"❌ Tabular data scenario failed: {e}")
        return False
    
    # Test 3.4: Small dataset scenario
    try:
        # Simulate small research dataset
        small_dataset = SimpleDataset(size=50, num_features=10, num_classes=2)
        small_loader = DataLoader(small_dataset, batch_size=8, shuffle=True)
        
        # Test multiple epochs
        for epoch in range(3):
            epoch_samples = 0
            for batch_data, batch_labels in small_loader:
                epoch_samples += batch_data.shape[0]
                
                # Verify small dataset properties
                assert batch_data.shape[1] == 10, f"Should have 10 features, got {batch_data.shape[1]}"
                
            assert epoch_samples == 50, f"Epoch {epoch}: should process 50 samples, got {epoch_samples}"
        
        print("✅ Small dataset scenario test passed")
    except Exception as e:
        print(f"❌ Small dataset scenario failed: {e}")
        return False
    
    print("🎯 Data pipeline scenarios: All tests passed!")
    return True

def test_integration_with_ml_workflow():
    """Test 4: Integration with ML workflow"""
    print("🔬 Testing Integration with ML Workflow...")
    
    # Test 4.1: Training loop integration
    try:
        # Create dataset for training
        train_dataset = SimpleDataset(size=100, num_features=8, num_classes=3)
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        
        # Simulate training loop
        for epoch in range(2):
            epoch_loss = 0
            batch_count = 0
            
            for batch_data, batch_labels in train_loader:
                batch_count += 1
                
                # Simulate forward pass
                batch_size = batch_data.shape[0]
                assert batch_data.shape == (batch_size, 8), f"Batch data shape wrong: {batch_data.shape}"
                assert batch_labels.shape[0] == batch_size, f"Batch labels shape wrong: {batch_labels.shape}"
                
                # Simulate loss computation
                mock_loss = np.random.random()
                epoch_loss += mock_loss
                
                # Verify we can iterate through all batches
                assert batch_count <= 5, f"Too many batches: {batch_count}"  # 100/20 = 5
            
            assert batch_count == 5, f"Should have 5 batches per epoch, got {batch_count}"
        
        print("✅ Training loop integration test passed")
    except Exception as e:
        print(f"❌ Training loop integration failed: {e}")
        return False
    
    # Test 4.2: Validation loop integration
    try:
        # Create dataset for validation
        val_dataset = SimpleDataset(size=50, num_features=8, num_classes=3)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)  # No shuffle for validation
        
        # Simulate validation loop
        total_correct = 0
        total_samples = 0
        
        for batch_data, batch_labels in val_loader:
            batch_size = batch_data.shape[0]
            total_samples += batch_size
            
            # Simulate prediction
            mock_predictions = np.random.randint(0, 3, size=batch_size)
            mock_correct = np.random.randint(0, batch_size + 1)
            total_correct += mock_correct
            
            # Verify batch properties
            assert batch_data.shape[1] == 8, f"Features should be 8, got {batch_data.shape[1]}"
            assert batch_labels.shape[0] == batch_size, f"Labels should match batch size"
        
        assert total_samples == 50, f"Should validate 50 samples, got {total_samples}"
        
        print("✅ Validation loop integration test passed")
    except Exception as e:
        print(f"❌ Validation loop integration failed: {e}")
        return False
    
    # Test 4.3: Model inference integration
    try:
        # Create dataset for inference
        test_dataset = SimpleDataset(size=30, num_features=5, num_classes=2)
        test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
        
        # Simulate inference
        all_predictions = []
        
        for batch_data, batch_labels in test_loader:
            batch_size = batch_data.shape[0]
            
            # Simulate model inference
            mock_predictions = np.random.random((batch_size, 2))  # 2 classes
            all_predictions.append(mock_predictions)
            
            # Verify inference batch properties
            assert batch_data.shape[1] == 5, f"Features should be 5, got {batch_data.shape[1]}"
            assert batch_size <= 5, f"Batch size should be <= 5, got {batch_size}"
        
        # Verify all predictions collected
        total_predictions = np.concatenate(all_predictions, axis=0)
        assert total_predictions.shape == (30, 2), f"Predictions shape should be (30, 2), got {total_predictions.shape}"
        
        print("✅ Model inference integration test passed")
    except Exception as e:
        print(f"❌ Model inference integration failed: {e}")
        return False
    
    # Test 4.4: Cross-validation scenario
    try:
        # Create dataset for cross-validation
        full_dataset = SimpleDataset(size=100, num_features=6, num_classes=4)
        
        # Simulate 5-fold cross-validation
        fold_size = 20
        
        for fold in range(5):
            # Create train/val split simulation
            train_size = 80  # 4 folds for training
            val_size = 20    # 1 fold for validation
            
            train_dataset = SimpleDataset(size=train_size, num_features=6, num_classes=4)
            val_dataset = SimpleDataset(size=val_size, num_features=6, num_classes=4)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Verify fold setup
            assert len(train_dataset) == train_size, f"Train size wrong for fold {fold}"
            assert len(val_dataset) == val_size, f"Val size wrong for fold {fold}"
            
            # Test one iteration of each
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            assert train_batch[0].shape[1] == 6, f"Train features wrong for fold {fold}"
            assert val_batch[0].shape[1] == 6, f"Val features wrong for fold {fold}"
        
        print("✅ Cross-validation scenario test passed")
    except Exception as e:
        print(f"❌ Cross-validation scenario failed: {e}")
        return False
    
    print("🎯 ML workflow integration: All tests passed!")
    return True

# Run all comprehensive tests
def run_comprehensive_dataloader_tests():
    """Run all comprehensive DataLoader tests"""
    print("🧪 Running Comprehensive DataLoader Test Suite...")
    print("=" * 60)
    
    test_results = []
    
    # Run all test functions
    test_results.append(test_dataset_interface())
    test_results.append(test_dataloader_functionality())
    test_results.append(test_data_pipeline_scenarios())
    test_results.append(test_integration_with_ml_workflow())
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Dataset Interface: {'PASSED' if test_results[0] else 'FAILED'}")
    print(f"✅ DataLoader Functionality: {'PASSED' if test_results[1] else 'FAILED'}")
    print(f"✅ Data Pipeline Scenarios: {'PASSED' if test_results[2] else 'FAILED'}")
    print(f"✅ ML Workflow Integration: {'PASSED' if test_results[3] else 'FAILED'}")
    
    all_passed = all(test_results)
    print(f"\n🎯 Overall Result: {'ALL TESTS PASSED! 🎉' if all_passed else 'SOME TESTS FAILED ❌'}")
    
    if all_passed:
        print("\n🚀 DataLoader Module Implementation Complete!")
        print("   ✓ Dataset interface working correctly")
        print("   ✓ DataLoader batching and iteration functional")
        print("   ✓ Real-world data pipeline scenarios tested")
        print("   ✓ ML workflow integration verified")
        print("\n🎓 Ready for production ML data pipelines!")
    
    return all_passed

# Run the comprehensive test suite
if __name__ == "__main__":
    run_comprehensive_dataloader_tests()

# %% [markdown]
"""
### 🧪 Test Your Data Loading Implementations

Once you implement the classes above, run these cells to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-dataset", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test Dataset abstract class
print("Testing Dataset abstract class...")

# Create a simple dataset
dataset = SimpleDataset(size=10, num_features=3, num_classes=2)

# Test basic functionality
assert len(dataset) == 10, f"Dataset length should be 10, got {len(dataset)}"
assert dataset.get_num_classes() == 2, f"Number of classes should be 2, got {dataset.get_num_classes()}"

# Test sample retrieval
data, label = dataset[0]
assert isinstance(data, Tensor), "Data should be a Tensor"
assert isinstance(label, Tensor), "Label should be a Tensor"
assert data.shape == (3,), f"Data shape should be (3,), got {data.shape}"
assert label.shape == (), f"Label shape should be (), got {label.shape}"

# Test sample shape
sample_shape = dataset.get_sample_shape()
assert sample_shape == (3,), f"Sample shape should be (3,), got {sample_shape}"

print("✅ Dataset tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-dataloader", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test DataLoader
print("Testing DataLoader...")

# Create dataset and dataloader
dataset = SimpleDataset(size=50, num_features=4, num_classes=3)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test dataloader length
expected_batches = (50 + 8 - 1) // 8  # Ceiling division
assert len(dataloader) == expected_batches, f"DataLoader length should be {expected_batches}, got {len(dataloader)}"

# Test batch iteration
batch_count = 0
total_samples = 0

for batch_data, batch_labels in dataloader:
    batch_count += 1
    batch_size = batch_data.shape[0]
    total_samples += batch_size
    
    # Check batch shapes
    assert batch_data.shape[1] == 4, f"Batch data should have 4 features, got {batch_data.shape[1]}"
    assert batch_labels.shape[0] == batch_size, f"Batch labels should match batch size, got {batch_labels.shape[0]}"
    
    # Check that we don't exceed expected batches
    assert batch_count <= expected_batches, f"Too many batches: {batch_count} > {expected_batches}"

# Verify we processed all samples
assert total_samples == 50, f"Should process 50 samples total, got {total_samples}"
assert batch_count == expected_batches, f"Should have {expected_batches} batches, got {batch_count}"

print("✅ DataLoader tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-dataloader-shuffle", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test DataLoader shuffling
print("Testing DataLoader shuffling...")

# Create dataset
dataset = SimpleDataset(size=20, num_features=2, num_classes=2)

# Test with shuffling
dataloader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
dataloader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)

# Get first batch from each
batch_shuffle = next(iter(dataloader_shuffle))
batch_no_shuffle = next(iter(dataloader_no_shuffle))

# With different random seeds, shuffled batches should be different
# (This is probabilistic, but very likely to be true)
shuffle_data = batch_shuffle[0].data
no_shuffle_data = batch_no_shuffle[0].data

# Check that shapes are correct
assert shuffle_data.shape == (5, 2), f"Shuffled batch shape should be (5, 2), got {shuffle_data.shape}"
assert no_shuffle_data.shape == (5, 2), f"No-shuffle batch shape should be (5, 2), got {no_shuffle_data.shape}"

print("✅ DataLoader shuffling tests passed!")

# %% nbgrader={"grade": true, "grade_id": "test-integration", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
# Test complete data pipeline integration
print("Testing complete data pipeline integration...")

# Create a larger dataset
dataset = SimpleDataset(size=100, num_features=8, num_classes=5)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Simulate training loop
epoch_samples = 0
epoch_batches = 0

for batch_data, batch_labels in dataloader:
    epoch_batches += 1
    epoch_samples += batch_data.shape[0]
    
    # Verify batch properties
    assert batch_data.shape[1] == 8, f"Features should be 8, got {batch_data.shape[1]}"
    assert len(batch_labels.shape) == 1, f"Labels should be 1D, got shape {batch_labels.shape}"
    
    # Verify data types
    assert isinstance(batch_data, Tensor), "Batch data should be Tensor"
    assert isinstance(batch_labels, Tensor), "Batch labels should be Tensor"

# Verify we processed all data
assert epoch_samples == 100, f"Should process 100 samples, got {epoch_samples}"
expected_batches = (100 + 16 - 1) // 16
assert epoch_batches == expected_batches, f"Should have {expected_batches} batches, got {epoch_batches}"

print("✅ Complete data pipeline integration tests passed!")

# %% [markdown]
"""
## 🎯 Module Summary

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

### Next Steps
1. **Export your code**: `tito package nbdev --export 06_dataloader`
2. **Test your implementation**: `tito module test 06_dataloader`
3. **Use your data loading**: 
   ```python
   from tinytorch.core.dataloader import Dataset, DataLoader, SimpleDataset
   
   # Create dataset and dataloader
   dataset = SimpleDataset(size=1000, num_features=10, num_classes=3)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
   # Training loop
   for batch_data, batch_labels in dataloader:
       # Train your network on batch_data, batch_labels
       pass
   ```
4. **Build real datasets**: Extend Dataset for your specific data types
5. **Optimize performance**: Add caching, parallel loading, and preprocessing

**Ready for the next challenge?** You now have all the core components to build complete machine learning systems: tensors, activations, layers, networks, and data loading. The next modules will focus on training (autograd, optimizers) and advanced topics!
""" 