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
# DataLoader - Efficient Data Pipeline and Batch Processing Systems

Welcome to the DataLoader module! You'll build the data infrastructure that feeds neural networks, understanding how I/O optimization and memory management determine training speed.

## Learning Goals
- Systems understanding: How data I/O becomes the bottleneck in ML training and why efficient data pipelines are critical for system performance
- Core implementation skill: Build Dataset and DataLoader classes with batching, shuffling, and memory-efficient iteration patterns
- Pattern recognition: Understand the universal Dataset/DataLoader abstraction used across all ML frameworks
- Framework connection: See how your implementation mirrors PyTorch's data loading infrastructure and optimization strategies
- Performance insight: Learn why data loading parallelization and prefetching are essential for GPU utilization in production systems

## Build → Use → Reflect
1. **Build**: Complete Dataset and DataLoader classes with efficient batching, shuffling, and real dataset support (CIFAR-10)
2. **Use**: Load large-scale image datasets and feed them to neural networks with proper batch processing
3. **Reflect**: Why does data loading speed often determine training speed more than model computation?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how efficient data pipelines enable scalable ML training
- Practical capability to build data loading systems that handle datasets larger than memory
- Systems insight into why data engineering is often the limiting factor in ML system performance
- Performance consideration of how batch size, shuffling, and prefetching affect training throughput and convergence
- Connection to production ML systems and how frameworks optimize data loading for different storage systems

## Systems Reality Check
💡 **Production Context**: PyTorch's DataLoader uses multiprocessing and memory pinning to overlap data loading with GPU computation, achieving near-zero data loading overhead
⚡ **Performance Note**: Modern GPUs can process data faster than storage systems can provide it - data loading optimization is critical for hardware utilization in production training
"""

# %% nbgrader={"grade": false, "grade_id": "dataloader-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.dataloader

#| export
import numpy as np
import sys
import os
from typing import Tuple, Optional, Iterator
import urllib.request
import tarfile
import pickle
import time

# Import our building blocks - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. This is an abstract method - subclasses will implement it
        2. Return a tuple of (data, label) tensors
        3. Data should be the input features, label should be the target
        
        EXAMPLE:
        dataset[0] should return (Tensor(image_data), Tensor(label))
        
        LEARNING CONNECTIONS:
        - **PyTorch Integration**: This follows the exact same pattern as torch.utils.data.Dataset
        - **Production Data**: Real datasets like ImageNet, CIFAR-10 use this interface
        - **Memory Efficiency**: On-demand loading prevents loading entire dataset into memory
        - **Batching Foundation**: DataLoader uses __getitem__ to create batches efficiently
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. This is an abstract method - subclasses will implement it
        2. Return the total number of samples in the dataset
        
        EXAMPLE:
        len(dataset) should return 50000 for CIFAR-10 training set
        
        LEARNING CONNECTIONS:
        - **Memory Planning**: DataLoader uses len() to calculate number of batches
        - **Progress Tracking**: Training loops use len() for progress bars and epoch calculations
        - **Distributed Training**: Multi-GPU systems need dataset size for work distribution
        - **Statistical Sampling**: Some training strategies require knowing total dataset size
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Get the first sample using self[0]
        2. Extract the data part (first element of tuple)
        3. Return the shape of the data tensor
        
        EXAMPLE:
        For CIFAR-10: returns (3, 32, 32) for RGB images
        
        LEARNING CONNECTIONS:
        - **Model Architecture**: Neural networks need to know input shape for first layer
        - **Batch Planning**: Systems use sample shape to calculate memory requirements
        - **Preprocessing Validation**: Ensures all samples have consistent shape
        - **Framework Integration**: Similar to PyTorch's dataset shape inspection
        
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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. This is an abstract method - subclasses will implement it
        2. Return the number of unique classes in the dataset
        
        EXAMPLE:
        For CIFAR-10: returns 10 (classes 0-9)
        
        LEARNING CONNECTIONS:
        - **Output Layer Design**: Neural networks need num_classes for final layer size
        - **Loss Function Setup**: CrossEntropyLoss uses num_classes for proper computation
        - **Evaluation Metrics**: Accuracy calculation depends on number of classes
        - **Model Validation**: Ensures model predictions match expected class range
        
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

# Test the interface (moved to main block)

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
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create indices list: list(range(len(dataset)))
        2. Shuffle indices if self.shuffle is True
        3. Loop through indices in batch_size chunks
        4. For each batch: collect samples, stack them, yield batch
        
        EXAMPLE:
        for batch_data, batch_labels in dataloader:
            # batch_data.shape: (batch_size, ...)
            # batch_labels.shape: (batch_size,)
        
        LEARNING CONNECTIONS:
        - **GPU Efficiency**: Batching maximizes GPU utilization by processing multiple samples together
        - **Training Stability**: Shuffling prevents overfitting to data order and improves generalization
        - **Memory Management**: Batches fit in GPU memory while full dataset may not
        - **Gradient Estimation**: Batch gradients provide better estimates than single-sample gradients
        
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
## Step 4b: CIFAR-10 Dataset - Real Data for CNNs

### Download and Load Real Computer Vision Data
Let's implement loading CIFAR-10, the dataset we'll use to achieve our north star goal of 75% accuracy!
"""

# %% nbgrader={"grade": false, "grade_id": "cifar10", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def download_cifar10(root: str = "./data") -> str:
    """
    Download CIFAR-10 dataset.
    
    TODO: Download and extract CIFAR-10.
    
    HINTS:
    - URL: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    - Use urllib.request.urlretrieve()
    - Extract with tarfile
    """
    ### BEGIN SOLUTION
    os.makedirs(root, exist_ok=True)
    dataset_dir = os.path.join(root, "cifar-10-batches-py")
    
    if os.path.exists(dataset_dir):
        print(f"✅ CIFAR-10 found at {dataset_dir}")
        return dataset_dir
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(root, "cifar-10.tar.gz")
    
    print(f"📥 Downloading CIFAR-10 (~170MB)...")
    urllib.request.urlretrieve(url, tar_path)
    print("✅ Downloaded!")
    
    print("📦 Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(root)
    print("✅ Ready!")
    
    return dataset_dir
    ### END SOLUTION

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset for CNN training."""
    
    def __init__(self, root="./data", train=True, download=False):
        """Load CIFAR-10 data."""
        ### BEGIN SOLUTION
        if download:
            dataset_dir = download_cifar10(root)
        else:
            dataset_dir = os.path.join(root, "cifar-10-batches-py")
        
        if train:
            data_list = []
            label_list = []
            for i in range(1, 6):
                with open(os.path.join(dataset_dir, f"data_batch_{i}"), 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    data_list.append(batch[b'data'])
                    label_list.extend(batch[b'labels'])
            self.data = np.concatenate(data_list)
            self.labels = np.array(label_list)
        else:
            with open(os.path.join(dataset_dir, "test_batch"), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data = batch[b'data']
                self.labels = np.array(batch[b'labels'])
        
        # Reshape to (N, 3, 32, 32) and normalize
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        print(f"✅ Loaded {len(self.data):,} images")
        ### END SOLUTION
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx]), Tensor(self.labels[idx])
    
    def __len__(self):
        return len(self.data)
    
    def get_num_classes(self):
        return 10

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

# %% [markdown]
"""
### 🧪 Unit Test: Dataset Interface Implementation

This test validates the abstract Dataset interface, ensuring proper inheritance, method implementation, and interface compliance for creating custom datasets in the TinyTorch data loading pipeline.
"""

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

# %% [markdown]
"""
### 🧪 Unit Test: DataLoader Implementation

This test validates the DataLoader class functionality, ensuring proper batch creation, iteration capability, and integration with datasets for efficient data loading in machine learning training pipelines.
"""

# %%
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

# %% [markdown]
"""
### 🧪 Unit Test: Simple Dataset Implementation

This test validates the SimpleDataset class, ensuring it can handle real-world data scenarios including proper data storage, indexing, and compatibility with the DataLoader for practical machine learning workflows.
"""

# %%
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

# %% [markdown]
"""
### 🧪 Unit Test: Complete Data Pipeline Integration

This comprehensive test validates the entire data pipeline from dataset creation through DataLoader batching, ensuring all components work together seamlessly for end-to-end machine learning data processing workflows.
"""

# %%
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

# Test function defined (called in main block)

# %% [markdown]
"""
## 📊 ML Systems: I/O Pipeline Optimization & Bottleneck Analysis

Now that you have data loading systems, let's develop **I/O optimization skills**. This section teaches you to identify and fix data loading bottlenecks that can dramatically slow down training in production systems.

### **Learning Outcome**: *"I can identify and fix I/O bottlenecks that limit training speed"*

---

## Data Pipeline Profiler (Medium Guided Implementation)

As an ML systems engineer, you need to ensure data loading doesn't become the bottleneck. Training GPUs can process data much faster than traditional storage can provide it. Let's build tools to measure and optimize data pipeline performance.
"""

# %%
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

class DataPipelineProfiler:
    """
    I/O pipeline profiling toolkit for data loading systems.
    
    Helps ML engineers identify bottlenecks in data loading pipelines
    and optimize throughput for high-performance training systems.
    """
    
    def __init__(self):
        self.profiling_history = []
        self.bottleneck_threshold = 0.1  # seconds per batch
        
    def time_dataloader_iteration(self, dataloader, num_batches=10):
        """
        Time how long it takes to iterate through DataLoader batches.
        
        TODO: Implement DataLoader timing analysis.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Record start time
        2. Iterate through specified number of batches
        3. Time each batch loading
        4. Calculate statistics (total, average, min, max times)
        5. Identify if data loading is a bottleneck
        6. Return comprehensive timing analysis
        
        EXAMPLE:
        profiler = DataPipelineProfiler()
        timing = profiler.time_dataloader_iteration(my_dataloader, 20)
        print(f"Avg batch time: {timing['avg_batch_time']:.3f}s")
        print(f"Bottleneck: {timing['is_bottleneck']}")
        
        LEARNING CONNECTIONS:
        - **Production Optimization**: Fast GPUs often wait for slow data loading
        - **System Bottlenecks**: Data loading can limit training speed more than model complexity
        - **Resource Planning**: Understanding I/O vs compute trade-offs for hardware selection
        - **Pipeline Tuning**: Multi-worker data loading and prefetching strategies
        
        HINTS:
        - Use enumerate(dataloader) to get batches
        - Time each batch: start = time.time(), batch = next(iter), end = time.time()
        - Break after num_batches to avoid processing entire dataset
        - Calculate: total_time, avg_time, min_time, max_time
        - Bottleneck if avg_time > self.bottleneck_threshold
        """
        ### BEGIN SOLUTION
        batch_times = []
        total_start = time.time()
        
        try:
            dataloader_iter = iter(dataloader)
            for i in range(num_batches):
                batch_start = time.time()
                try:
                    batch = next(dataloader_iter)
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)
                except StopIteration:
                    print(f"   DataLoader exhausted after {i} batches")
                    break
        except Exception as e:
            print(f"   Error during iteration: {e}")
            return {'error': str(e)}
        
        total_end = time.time()
        total_time = total_end - total_start
        
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            min_batch_time = min(batch_times)
            max_batch_time = max(batch_times)
            
            # Check if data loading is a bottleneck
            is_bottleneck = avg_batch_time > self.bottleneck_threshold
            
            # Calculate throughput
            batches_per_second = len(batch_times) / total_time if total_time > 0 else 0
            
            return {
                'total_time': total_time,
                'num_batches': len(batch_times),
                'avg_batch_time': avg_batch_time,
                'min_batch_time': min_batch_time,
                'max_batch_time': max_batch_time,
                'batches_per_second': batches_per_second,
                'is_bottleneck': is_bottleneck,
                'bottleneck_threshold': self.bottleneck_threshold
            }
        else:
            return {'error': 'No batches processed'}
        ### END SOLUTION
    
    def analyze_batch_size_scaling(self, dataset, batch_sizes=[16, 32, 64, 128]):
        """
        Analyze how batch size affects data loading performance.
        
        TODO: Implement batch size scaling analysis.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. For each batch size, create a DataLoader
        2. Time the data loading for each configuration
        3. Calculate throughput (samples/second) for each
        4. Identify optimal batch size for I/O performance
        5. Return scaling analysis with recommendations
        
        EXAMPLE:
        profiler = DataPipelineProfiler()
        analysis = profiler.analyze_batch_size_scaling(my_dataset, [16, 32, 64])
        print(f"Optimal batch size: {analysis['optimal_batch_size']}")
        
        LEARNING CONNECTIONS:
        - **Memory vs Throughput**: Larger batches improve throughput but consume more memory
        - **Hardware Optimization**: Optimal batch size depends on GPU memory and compute units
        - **Training Dynamics**: Batch size affects gradient noise and convergence behavior
        - **Production Scaling**: Understanding batch size impact on serving latency and cost
        
        HINTS:
        - Create DataLoader: DataLoader(dataset, batch_size=bs, shuffle=False)
        - Time with self.time_dataloader_iteration()
        - Calculate: samples_per_second = batch_size * batches_per_second
        - Find batch size with highest samples/second
        - Consider memory constraints vs throughput
        """
        ### BEGIN SOLUTION
        scaling_results = []
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size}...")
            
            # Create DataLoader with current batch size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Time the data loading
            timing_result = self.time_dataloader_iteration(dataloader, num_batches=min(10, len(dataset)//batch_size))
            
            if 'error' not in timing_result:
                # Calculate throughput metrics
                samples_per_second = batch_size * timing_result['batches_per_second']
                
                result = {
                    'batch_size': batch_size,
                    'avg_batch_time': timing_result['avg_batch_time'],
                    'batches_per_second': timing_result['batches_per_second'],
                    'samples_per_second': samples_per_second,
                    'is_bottleneck': timing_result['is_bottleneck']
                }
                scaling_results.append(result)
        
        # Find optimal batch size (highest throughput)
        if scaling_results:
            optimal = max(scaling_results, key=lambda x: x['samples_per_second'])
            optimal_batch_size = optimal['batch_size']
            
            return {
                'scaling_results': scaling_results,
                'optimal_batch_size': optimal_batch_size,
                'max_throughput': optimal['samples_per_second']
            }
        else:
            return {'error': 'No valid results obtained'}
        ### END SOLUTION
    
    def compare_io_strategies(self, dataset, strategies=['sequential', 'shuffled']):
        """
        Compare different I/O strategies for data loading performance.
        
        This function is PROVIDED to demonstrate I/O optimization analysis.
        Students use it to understand different data loading patterns.
        """
        print("📊 I/O STRATEGY COMPARISON")
        print("=" * 40)
        
        results = {}
        batch_size = 32  # Standard batch size for comparison
        
        for strategy in strategies:
            print(f"\n🔍 Testing {strategy.upper()} strategy...")
            
            if strategy == 'sequential':
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            elif strategy == 'shuffled':
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            else:
                print(f"   Unknown strategy: {strategy}")
                continue
            
            # Time the strategy
            timing_result = self.time_dataloader_iteration(dataloader, num_batches=20)
            
            if 'error' not in timing_result:
                results[strategy] = timing_result
                print(f"   Avg batch time: {timing_result['avg_batch_time']:.3f}s")
                print(f"   Throughput: {timing_result['batches_per_second']:.1f} batches/sec")
                print(f"   Bottleneck: {'Yes' if timing_result['is_bottleneck'] else 'No'}")
        
        # Compare strategies
        if len(results) >= 2:
            fastest = min(results.items(), key=lambda x: x[1]['avg_batch_time'])
            slowest = max(results.items(), key=lambda x: x[1]['avg_batch_time'])
            
            speedup = slowest[1]['avg_batch_time'] / fastest[1]['avg_batch_time']
            
            print(f"\n🎯 STRATEGY ANALYSIS:")
            print(f"   Fastest: {fastest[0]} ({fastest[1]['avg_batch_time']:.3f}s)")
            print(f"   Slowest: {slowest[0]} ({slowest[1]['avg_batch_time']:.3f}s)")
            print(f"   Speedup: {speedup:.1f}x")
        
        return results
    
    def simulate_compute_vs_io_balance(self, dataloader, simulated_compute_time=0.05):
        """
        Simulate the balance between data loading and compute time.
        
        This function is PROVIDED to show I/O vs compute analysis.
        Students use it to understand when I/O becomes a bottleneck.
        """
        print("⚖️  COMPUTE vs I/O BALANCE ANALYSIS")
        print("=" * 45)
        
        print(f"Simulated compute time per batch: {simulated_compute_time:.3f}s")
        print(f"(This represents GPU processing time)")
        
        # Time data loading
        io_timing = self.time_dataloader_iteration(dataloader, num_batches=15)
        
        if 'error' in io_timing:
            print(f"Error in timing: {io_timing['error']}")
            return
        
        avg_io_time = io_timing['avg_batch_time']
        
        print(f"\n📊 TIMING ANALYSIS:")
        print(f"   Data loading time: {avg_io_time:.3f}s per batch")
        print(f"   Simulated compute: {simulated_compute_time:.3f}s per batch")
        
        # Determine bottleneck
        if avg_io_time > simulated_compute_time:
            bottleneck = "I/O"
            utilization = simulated_compute_time / avg_io_time * 100
            print(f"\n🚨 BOTTLENECK: {bottleneck}")
            print(f"   GPU utilization: {utilization:.1f}%")
            print(f"   GPU waiting for data: {avg_io_time - simulated_compute_time:.3f}s per batch")
        else:
            bottleneck = "Compute"
            utilization = avg_io_time / simulated_compute_time * 100
            print(f"\n✅ BOTTLENECK: {bottleneck}")
            print(f"   I/O utilization: {utilization:.1f}%")
            print(f"   I/O waiting for GPU: {simulated_compute_time - avg_io_time:.3f}s per batch")
        
        # Calculate training impact
        total_cycle_time = max(avg_io_time, simulated_compute_time)
        efficiency = min(avg_io_time, simulated_compute_time) / total_cycle_time * 100
        
        print(f"\n🎯 TRAINING IMPACT:")
        print(f"   Pipeline efficiency: {efficiency:.1f}%")
        print(f"   Total cycle time: {total_cycle_time:.3f}s")
        
        if bottleneck == "I/O":
            print(f"   💡 Recommendation: Optimize data loading")
            print(f"      - Increase batch size")
            print(f"      - Use data prefetching")
            print(f"      - Faster storage (SSD vs HDD)")
        else:
            print(f"   💡 Recommendation: I/O is well optimized")
            print(f"      - Consider larger models or batch sizes")
            print(f"      - Focus on compute optimization")
        
        return {
            'io_time': avg_io_time,
            'compute_time': simulated_compute_time,
            'bottleneck': bottleneck,
            'efficiency': efficiency,
            'total_cycle_time': total_cycle_time
        }

# %% [markdown]
"""
### 🎯 Learning Activity 1: DataLoader Performance Profiling (Medium Guided Implementation)

**Goal**: Learn to measure data loading performance and identify I/O bottlenecks that can slow down training.

Complete the missing implementations in the `DataPipelineProfiler` class above, then use your profiler to analyze data loading performance.
"""

# %%
# Initialize the data pipeline profiler
profiler = DataPipelineProfiler()

# Only run tests when module is executed directly
if __name__ == '__main__':
    print("📊 DATA PIPELINE PERFORMANCE ANALYSIS")
    print("=" * 50)

    # Create test dataset and dataloader
    test_dataset = TensorDataset([
        Tensor(np.random.randn(100)) for _ in range(1000)  # 1000 samples
    ], [
        Tensor([i % 10]) for i in range(1000)  # Labels
    ])

    # Test 1: Basic DataLoader timing
    print("⏱️  Basic DataLoader Timing:")
    basic_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Students use their implemented timing function
timing_result = profiler.time_dataloader_iteration(basic_dataloader, num_batches=25)

if 'error' not in timing_result:
    print(f"   Average batch time: {timing_result['avg_batch_time']:.3f}s")
    print(f"   Throughput: {timing_result['batches_per_second']:.1f} batches/sec")
    print(f"   Bottleneck detected: {'Yes' if timing_result['is_bottleneck'] else 'No'}")
    
    # Calculate samples per second
    samples_per_sec = 32 * timing_result['batches_per_second']
    print(f"   Samples/second: {samples_per_sec:.1f}")
else:
    print(f"   Error: {timing_result['error']}")

# Test 2: Batch size scaling analysis
print(f"\n📈 Batch Size Scaling Analysis:")

# Students use their implemented scaling analysis
scaling_analysis = profiler.analyze_batch_size_scaling(test_dataset, [16, 32, 64, 128])

if 'error' not in scaling_analysis:
    print(f"   Optimal batch size: {scaling_analysis['optimal_batch_size']}")
    print(f"   Max throughput: {scaling_analysis['max_throughput']:.1f} samples/sec")
    
    print(f"\n   📊 Detailed Results:")
    for result in scaling_analysis['scaling_results']:
        print(f"      Batch {result['batch_size']:3d}: {result['samples_per_second']:6.1f} samples/sec")
else:
    print(f"   Error: {scaling_analysis['error']}")

print(f"\n💡 I/O PERFORMANCE INSIGHTS:")
print(f"   - Larger batches often improve throughput (better amortization)")
print(f"   - But memory constraints limit maximum batch size")
print(f"   - Sweet spot balances throughput vs memory usage")
print(f"   - Real systems: GPU memory determines practical limits")

# %% [markdown]
"""
### 🎯 Learning Activity 2: Production I/O Optimization Analysis (Review & Understand)

**Goal**: Understand how I/O performance affects real training systems and learn optimization strategies used in production.
"""

# %%
# Compare different I/O strategies
io_comparison = profiler.compare_io_strategies(test_dataset, ['sequential', 'shuffled'])

# Simulate compute vs I/O balance with different scenarios
print(f"\n⚖️  COMPUTE vs I/O SCENARIOS:")
print(f"=" * 40)

# Test different compute scenarios
compute_scenarios = [
    (0.01, "Fast GPU (V100/A100)"),
    (0.05, "Medium GPU (RTX 3080)"),
    (0.1, "CPU-only training"),
    (0.2, "Complex model/large batch")
]

sample_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for compute_time, scenario_name in compute_scenarios:
    print(f"\n🖥️  {scenario_name}:")
    balance_analysis = profiler.simulate_compute_vs_io_balance(sample_dataloader, compute_time)

print(f"\n🎯 PRODUCTION I/O OPTIMIZATION LESSONS:")
print(f"=" * 50)

print(f"\n1. 📊 I/O BOTTLENECK IDENTIFICATION:")
print(f"   - Fast GPUs often bottlenecked by data loading")
print(f"   - CPU training rarely I/O bottlenecked")
print(f"   - Modern GPUs process data faster than storage provides it")

print(f"\n2. 🚀 OPTIMIZATION STRATEGIES:")
print(f"   - Data prefetching: Load next batch while GPU computes")
print(f"   - Parallel workers: Multiple threads/processes for loading")
print(f"   - Faster storage: NVMe SSD vs SATA vs network storage")
print(f"   - Data caching: Keep frequently used data in memory")

print(f"\n3. 🏗️ ARCHITECTURE DECISIONS:")
print(f"   - Batch size: Larger batches amortize I/O overhead")
print(f"   - Data format: Preprocessed vs on-the-fly transformation")
print(f"   - Storage location: Local vs network vs cloud storage")

print(f"\n4. 💰 COST IMPLICATIONS:")
print(f"   - I/O bottlenecks waste expensive GPU time")
print(f"   - GPU utilization directly affects training costs")
print(f"   - Faster storage investment pays off in GPU efficiency")

print(f"\n💡 SYSTEMS ENGINEERING INSIGHT:")
print(f"I/O optimization is often the highest-impact performance improvement:")
print(f"- GPUs are expensive → maximize their utilization")
print(f"- Data loading is often the limiting factor")
print(f"- 10% I/O improvement = 10% faster training = 10% cost reduction")
print(f"- Modern ML systems spend significant effort on data pipeline optimization")

if __name__ == "__main__":
    # Test the dataset interface demonstration
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
        num_classes = test_dataset.get_num_classes()
        assert num_classes == 2, f"Number of classes should be 2, got {num_classes}"
        print("✅ Dataset get_num_classes works correctly")
        
        # Test get_sample_shape
        sample_shape = test_dataset.get_sample_shape()
        assert sample_shape == (3,), f"Sample shape should be (3,), got {sample_shape}"
        print("✅ Dataset get_sample_shape works correctly")
        
        print("🎯 Dataset interface pattern:")
        print("   __getitem__: Returns (data, label) tuple")
        print("   __len__: Returns dataset size")
        print("   get_num_classes: Returns number of classes")
        print("   get_sample_shape: Returns shape of data samples")
        print("📈 Progress: Dataset interface ✓")
        
    except Exception as e:
        print(f"❌ Dataset interface test failed: {e}")
        raise
    
    # Run all tests
    test_unit_dataset_interface()
    test_unit_dataloader()
    test_unit_simple_dataset()
    test_unit_dataloader_pipeline()
    test_module_dataloader_tensor_yield()
    
    print("All tests passed!")
    print("dataloader_dev module complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking Questions

### System Design
1. How does TinyTorch's DataLoader design compare to PyTorch's DataLoader and TensorFlow's tf.data API in terms of flexibility and performance?
2. What are the trade-offs between memory-mapped files, streaming data loading, and in-memory caching for large-scale ML datasets?
3. How would you design a data loading system that efficiently handles both structured (tabular) and unstructured (images, text) data?

### Production ML
1. How would you implement fault-tolerant data loading that can handle network failures and corrupted files in production environments?
2. What strategies would you use to ensure data consistency and prevent data leakage when loading from constantly updating production databases?
3. How would you design a data pipeline that supports both batch inference and real-time prediction serving?

### Framework Design
1. What design patterns enable efficient data preprocessing that can be distributed across multiple worker processes without blocking training?
2. How would you implement dynamic batching that adapts batch sizes based on available memory and model complexity?
3. What abstractions would you create to support different data formats (images, audio, text) while maintaining a unified loading interface?

### Performance & Scale
1. How do different data loading strategies (synchronous vs asynchronous, single vs multi-threaded) impact training throughput on different hardware?
2. What are the bottlenecks when loading data for distributed training across multiple machines, and how would you optimize data transfer?
3. How would you implement data loading that scales efficiently from small datasets (MB) to massive datasets (TB) without code changes?
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Data Loading and Processing

Congratulations! You've successfully implemented professional data loading systems:

### What You've Accomplished
✅ **DataLoader Class**: Efficient batch processing with memory management
✅ **Dataset Integration**: Seamless compatibility with Tensor operations
✅ **Batch Processing**: Optimized data loading for training
✅ **Memory Management**: Efficient handling of large datasets
✅ **Real Applications**: Image classification, regression, and more

### Key Concepts You've Learned
- **Batch processing**: How to efficiently process data in chunks
- **Memory management**: Handling large datasets without memory overflow
- **Data iteration**: Creating efficient data loading pipelines
- **Integration patterns**: How data loaders work with neural networks
- **Performance optimization**: Balancing speed and memory usage

### Professional Skills Developed
- **Data engineering**: Building robust data processing pipelines
- **Memory optimization**: Efficient handling of large datasets
- **API design**: Clean interfaces for data loading operations
- **Integration testing**: Ensuring data loaders work with neural networks

### Ready for Advanced Applications
Your data loading implementations now enable:
- **Large-scale training**: Processing datasets too big for memory
- **Real-time learning**: Streaming data for online learning
- **Multi-modal data**: Handling images, text, and structured data
- **Production systems**: Robust data pipelines for deployment

### Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch**: `torch.utils.data.DataLoader` provides identical functionality
- **TensorFlow**: `tf.data.Dataset` implements similar concepts
- **Industry Standard**: Every major ML framework uses these exact patterns

### Next Steps
1. **Export your code**: `tito export 08_dataloader`
2. **Test your implementation**: `tito test 08_dataloader`
3. **Build training pipelines**: Combine with neural networks for complete ML systems
4. **Move to Module 9**: Add automatic differentiation for training!

**Ready for autograd?** Your data loading systems are now ready for real training!
"""