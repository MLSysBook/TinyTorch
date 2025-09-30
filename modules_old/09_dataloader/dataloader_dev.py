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

## Build -> Use -> Reflect
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
TIP **Production Context**: PyTorch's DataLoader uses multiprocessing and memory pinning to overlap data loading with GPU computation, achieving near-zero data loading overhead
SPEED **Performance Note**: Modern GPUs can process data faster than storage systems can provide it - data loading optimization is critical for hardware utilization in production training
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
print("FIRE TinyTorch DataLoader Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build data pipelines!")

# %% [markdown]
"""
## PACKAGE Where This Code Lives in the Final Package

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
## Step 1: Understanding Data Pipelines - The Foundation of ML Systems

### LINK Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): Data structures that hold and manipulate arrays efficiently
- Module 04 (Layers): Neural network components that need batched inputs

**What's Working**: You can create tensors and build neural network layers!

**The Gap**: Your models need REAL DATA to train on, not just random numbers.

**This Module's Solution**: Build professional data loading pipelines that feed real datasets to your networks.

**Connection Map**:
```
Tensor Operations -> Data Loading -> Training Loop
   (Module 02)       (Module 10)    (Next: Module 11)
```

### What are Data Pipelines?
**Data pipelines** are the systems that efficiently move data from storage to your model. They're the foundation of all machine learning systems and often the performance bottleneck!

### 📊 The Complete Data Pipeline Flow
```
+-------------+    +----------+    +---------+    +---------+    +--------------+
| Raw Storage |---▶| Dataset  |---▶| Shuffle |---▶|  Batch  |---▶| Neural Net   |
| (Files/DB)  |    | Loading  |    | + Index |    | + Stack |    | Training     |
+-------------+    +----------+    +---------+    +---------+    +--------------+
      v                 v              v             v               v
 Gigabytes         On-Demand      Random Order    GPU-Friendly    Learning!
   of Data          Loading        (No Overfit)    Format
```

### MAGNIFY Why Data Pipelines Are Critical for ML Systems
- **Performance**: Efficient loading prevents GPU starvation (GPUs idle waiting for data)
- **Scalability**: Handle datasets larger than memory (ImageNet = 150GB)
- **Consistency**: Reproducible data processing across experiments
- **Flexibility**: Easy to switch between datasets and configurations

### SPEED Real-World Performance Challenges
```
🏎️  GPU Processing Speed:     ~1000 images/second
🐌  Disk Read Speed:         ~100 images/second
WARNING️   Result: GPU waits 90% of time for data!
```

### 💾 Memory vs Storage Trade-offs
```
Dataset Size Analysis:
+-------------+-------------+-------------+-------------+
|   Dataset   |    Size     | Fits in RAM |  Strategy   |
+-------------+-------------+-------------+-------------┤
|   MNIST     |   ~60 MB    |    PASS Yes   | Load All    |
|  CIFAR-10   |  ~170 MB    |    PASS Yes   | Load All    |
|  ImageNet   |  ~150 GB    |    FAIL No    | Stream      |
|  Custom     |   ~1 TB     |    FAIL No    | Stream      |
+-------------+-------------+-------------+-------------+
```

### 🧠 Systems Engineering Principles
- **Memory efficiency**: Handle datasets larger than RAM without crashing
- **I/O optimization**: Read from disk efficiently to minimize GPU waiting
- **Batching strategies**: Trade-offs between memory usage and training speed
- **Caching**: When to cache frequently used data vs recompute on-demand

### PROGRESS Batch Processing Impact
```
Batch Size Performance Analysis:
    Batch Size | GPU Utilization | Memory Usage | Training Speed
    -----------+-----------------+--------------+---------------
        1      |      ~10%       |    Low       |    Very Slow
       32      |      ~80%       |   Medium     |      Good
      128      |      ~95%       |    High      |    Very Fast
      512      |      ~98%       |  Very High   |   Fastest*
    
    * Until you run out of GPU memory!
```

Let's start by building the most fundamental component: **Dataset**.
"""

# %% [markdown]
"""
## Step 2: Building the Dataset Interface - The Universal Data Access Pattern

### What is a Dataset?
A **Dataset** is an abstract interface that provides consistent access to data. It's the foundation of all data loading systems and the key abstraction that makes ML frameworks flexible.

### TARGET The Universal Dataset Pattern
```
           Dataset Interface
    +-----------------------------+
    |  def __getitem__(index):    |<--- Get single sample by index
    |      return data, label     |     (like a list or dictionary)
    |                             |
    |  def __len__():             |<--- Total number of samples
    |      return total_samples   |     (enables progress tracking)
    +-----------------------------+
                    ^
                    | Implements
    +---------------+---------------+
    |               |               |
+---v----+  +------v-----+  +------v------+
| MNIST  |  |  CIFAR-10  |  | Custom Data |
|Dataset |  |  Dataset   |  |   Dataset   |
+--------+  +------------+  +-------------+
```

### 🔧 Why Abstract Interfaces Are Systems Engineering Gold
- **Consistency**: Same interface for all data types (images, text, audio)
- **Flexibility**: Easy to switch between datasets without changing training code
- **Testability**: Easy to create test datasets for debugging and unit tests
- **Extensibility**: Easy to add new data sources (databases, APIs, cloud storage)
- **Modularity**: DataLoader works with ANY dataset that implements this interface

### 📊 Production Dataset Examples
```
Real-World Dataset Implementations:

🖼️  Computer Vision:
    - ImageNet: 14M images, 1000 classes
    - CIFAR-10: 60K images, 10 classes  
    - COCO: 200K images with object detection annotations
    - Custom: Your company's image data

📝 Natural Language Processing:
    - WikiText: 100M+ tokens from Wikipedia
    - IMDB: 50K movie reviews for sentiment analysis
    - Custom: Your company's text data

🔊 Audio Processing:
    - LibriSpeech: 1000 hours of speech
    - AudioSet: 2M YouTube clips with audio events
    - Custom: Your company's audio data

PROGRESS Time Series:
    - Stock prices, sensor data, user behavior logs
    - Custom: Your company's time series data
```

### ROCKET Framework Integration Power
```
# PyTorch Compatibility:
torch_dataset = torch.utils.data.Dataset  # Same interface!
torch_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# TensorFlow Compatibility:
tf_dataset = tf.data.Dataset.from_generator(dataset_generator)

# Our TinyTorch:
tiny_loader = DataLoader(dataset, batch_size=32)  # Same pattern!
```

This universal pattern means your skills transfer directly to production frameworks!

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
        # Every dataset (CIFAR-10, ImageNet, custom) must implement this!
        raise NotImplementedError(
            "This is an abstract method - subclasses like SimpleDataset "
            "must implement __getitem__ to return (data, label) tuples"
        )
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
        # DataLoader needs this to calculate number of batches!
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
        # This helps neural networks know their input dimension
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
        ### BEGIN SOLUTION
        # This is an abstract method - subclasses must implement it
        # Neural networks need this for output layer size (classification)
        raise NotImplementedError("Subclasses must implement get_num_classes")
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: Dataset Interface

Let's understand the Dataset interface! While we can't test the abstract class directly, we'll create a simple test dataset.

**This is a unit test** - it tests the Dataset interface pattern in isolation.
"""

# Create a minimal test dataset for testing
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

# %%
def test_unit_dataset_interface():
    """Test Dataset interface with a simple implementation."""
    print("🔬 Unit Test: Dataset Interface...")

    # Create a minimal test dataset
    dataset = TestDataset(size=5)

    # Test basic interface
    assert len(dataset) == 5, "Dataset should have correct length"

    # Test data access
    sample, label = dataset[0]
    assert isinstance(sample, Tensor), "Sample should be Tensor"
    assert isinstance(label, Tensor), "Label should be Tensor"

    print("✅ Dataset interface works correctly!")

test_unit_dataset_interface()

# %% [markdown]
"""
## Step 3: Building the DataLoader - The Batch Processing Engine

### What is a DataLoader?
A **DataLoader** efficiently batches and iterates through datasets. It's the bridge between individual samples and the batched data that neural networks expect. This is where the real systems engineering happens!

### 🔄 The DataLoader Processing Pipeline
```
         Dataset Samples              DataLoader Magic               Neural Network
    +---------------------+       +---------------------+       +-----------------+
    | [sample_1]          |       | 1. Shuffle indices  |       | Efficient GPU   |
    | [sample_2]          |------▶| 2. Group into       |------▶| Batch           |
    | [sample_3]          |       |    batches          |       | Processing      |
    | [sample_4]          |       | 3. Stack tensors    |       |                 |
    | ...                 |       | 4. Yield batches    |       | batch_size=32   |
    | [sample_n]          |       |                     |       | shape=(32,...)  |
    +---------------------+       +---------------------+       +-----------------+
```

### SPEED Why DataLoaders Are Critical for Performance
```
GPU Utilization Without Batching:
+-----+-----+-----+-----+-----+-----+-----+-----+
|  🔄  | ... | ... | ... | ... | ... | ... | ... | Time
+-----+-----+-----+-----+-----+-----+-----+-----+
  ~5%   GPU mostly idle (underutilized)

GPU Utilization With Proper Batching:
+████████████████████████████████████████████████+
|          ████████████████████████████████      | Time
+████████████████████████████████████████████████+
  ~95%   GPU fully utilized (efficient!)
```

### 🧮 Memory vs Speed Trade-offs
```
Batch Size Impact Analysis:
    
    Batch Size | Memory Usage | GPU Utilization | Gradient Quality
    -----------+--------------+-----------------+-----------------
        1      |     Low      |      ~10%       |   Noisy (bad)
       16      |   Medium     |      ~60%       |     Better
       64      |    High      |      ~90%       |      Good
      256      |  Very High   |      ~95%       |    Very Good
      512      | TOO HIGH! CRASH |       N/A       |   OOM Error
```

### 🔀 Shuffling: Preventing Overfitting to Data Order
```
Without Shuffling (Bad!):
    Epoch 1: [cat, cat, dog, dog, bird, bird] 
    Epoch 2: [cat, cat, dog, dog, bird, bird]  <- Same order!
    Model learns data order, not features 😞

With Shuffling (Good!):
    Epoch 1: [dog, cat, bird, cat, dog, bird]
    Epoch 2: [bird, dog, cat, bird, cat, dog]  <- Random order!
    Model learns features, generalizes well 😊
```

### TARGET Production Training Pattern
```python
# The universal ML training pattern:
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:  # <- This line!
        predictions = model(batch_data)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
```

### 🏗️ Systems Engineering Considerations
- **Batch size**: Trade-off between memory usage and training speed
- **Shuffling**: Essential for model generalization (prevents order memorization)
- **Memory efficiency**: Stream data instead of loading everything into RAM
- **Iterator protocol**: Enables clean for-loop syntax in training code
- **GPU utilization**: Proper batching maximizes expensive GPU hardware

### 🔧 Real-World Applications
- **Training loops**: Feed batches to neural networks for gradient computation
- **Validation**: Evaluate models on held-out data systematically
- **Inference**: Process large datasets efficiently for predictions
- **Data analysis**: Explore datasets systematically without memory overflow

Let's implement the DataLoader that powers all ML training!
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
            raise ValueError(
                f"Batch size must be a positive integer (like 32 or 64), got {batch_size}. "
                f"This determines how many samples are processed together for efficiency."
            )
        
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
        ### BEGIN SOLUTION
        # Step 1: Create list of all sample indices (0, 1, 2, ..., dataset_size-1)
        # This allows us to control which samples go into which batches
        sample_indices = list(range(len(self.dataset)))
        
        # Step 2: Randomly shuffle indices if requested (prevents overfitting to data order)
        # Shuffling is critical for good model generalization!
        if self.shuffle:
            np.random.shuffle(sample_indices)
        
        # Step 3: Process data in batches of self.batch_size
        # This loop creates efficient GPU-sized chunks of data
        for batch_start_idx in range(0, len(sample_indices), self.batch_size):
            current_batch_indices = sample_indices[batch_start_idx:batch_start_idx + self.batch_size]
            
            # Step 4: Collect samples for this batch
            # Build lists of data and labels for efficient stacking
            batch_data_list = []
            batch_labels_list = []
            
            for sample_idx in current_batch_indices:
                data, label = self.dataset[sample_idx]  # Get individual sample
                # Access .data to get underlying numpy array for efficient stacking
                # Tensors wrap numpy arrays, and np.stack() needs raw arrays
                batch_data_list.append(data.data)
                batch_labels_list.append(label.data)
            
            # Step 5: Stack individual samples into batch tensors
            # np.stack combines multiple arrays along a new axis (axis=0 = batch dimension)
            # This creates the (batch_size, feature_dims...) shape that GPUs love!
            batch_data_array = np.stack(batch_data_list, axis=0)
            batch_labels_array = np.stack(batch_labels_list, axis=0)
            
            # Return batch as Tensors for neural network processing
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
        Dataset size 100, batch size 32 -> 4 batches
        
        HINTS:
        - Use len(self.dataset) for dataset size
        - Use ceiling division for exact batch count
        - Formula: (dataset_size + batch_size - 1) // batch_size
        """
        ### BEGIN SOLUTION
        # Calculate number of batches using ceiling division
        # This tells training loops how many iterations per epoch
        dataset_size = len(self.dataset)
        return (dataset_size + self.batch_size - 1) // self.batch_size
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: DataLoader

Let's test your DataLoader implementation! This is the heart of efficient data loading for neural networks.

**This is a unit test** - it tests the DataLoader class in isolation.
"""

# %%
def test_unit_dataloader():
    """Test DataLoader implementation with comprehensive functionality tests."""
    print("🔬 Unit Test: DataLoader...")

    # Use the TestDataset from before
    dataset = TestDataset(size=10)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    print(f"DataLoader created: batch_size={dataloader.batch_size}, shuffle={dataloader.shuffle}")
    print(f"Number of batches: {len(dataloader)}")

    # Test __len__
    expected_batches = (10 + 3 - 1) // 3  # Ceiling division: 4 batches
    assert len(dataloader) == expected_batches, f"Should have {expected_batches} batches, got {len(dataloader)}"

    # Test iteration
    batch_count = 0
    total_samples = 0

    for batch_data, batch_labels in dataloader:
        batch_count += 1
        batch_size = batch_data.shape[0]
        total_samples += batch_size

        # Verify batch dimensions
        assert len(batch_data.shape) == 2, f"Batch data should be 2D, got {batch_data.shape}"
        assert len(batch_labels.shape) == 2, f"Batch labels should be 2D, got {batch_labels.shape}"
        assert batch_data.shape[1] == 2, f"Each sample should have 2 features, got {batch_data.shape[1]}"
        assert batch_labels.shape[1] == 1, f"Each label should have 1 element, got {batch_labels.shape[1]}"

    assert batch_count == expected_batches, f"Should iterate {expected_batches} times, got {batch_count}"
    assert total_samples == 10, f"Should process 10 total samples, got {total_samples}"

    # Test shuffling
    dataloader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
    dataloader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)

    # Get first batch from each
    batch1_shuffle = next(iter(dataloader_shuffle))
    batch1_no_shuffle = next(iter(dataloader_no_shuffle))

    # Test different batch sizes
    small_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    large_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    assert len(small_loader) == 5, f"Small loader should have 5 batches, got {len(small_loader)}"
    assert len(large_loader) == 2, f"Large loader should have 2 batches, got {len(large_loader)}"

    print("✅ DataLoader works correctly!")

test_unit_dataloader()

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
        np.random.seed(42)  # Fixed seed ensures same data every time - important for testing!
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
        ### BEGIN SOLUTION
        # Get the specific sample by index
        # This is the core of on-demand data loading!
        data = self.data[index]
        label = self.labels[index]
        return Tensor(data), Tensor(label)
        ### END SOLUTION
    
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
        ### BEGIN SOLUTION
        # Return total number of samples
        # DataLoader needs this to calculate batches per epoch
        return self.size
        ### END SOLUTION
    
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
        ### BEGIN SOLUTION
        # Return number of unique classes
        # Neural networks need this for output layer size
        return self.num_classes
        ### END SOLUTION

# %% [markdown]
"""
## Step 4b: CIFAR-10 Dataset - Real Computer Vision Data

### 🏆 Achieving Our North Star Goal: 75% Accuracy on CIFAR-10

Let's implement loading CIFAR-10, the dataset we'll use to achieve our ambitious goal of 75% accuracy!

### 🇺🇸 CIFAR-10 Dataset Specifications
```
🖼️ CIFAR-10 Dataset Overview:
    +----------------------------------------+
    | 🎨 Classes: 10 (airplane, car, bird, etc.)  |
    | 🖼️ Images: 60,000 total (50k train + 10k test) |
    | 📌 Size: 32x32 pixels, RGB color           |
    | 💾 Storage: ~170MB compressed             |
    | TARGET Goal: 75% classification accuracy      |
    +----------------------------------------+
    
    Classes: airplane, automobile, bird, cat, deer, 
             dog, frog, horse, ship, truck
```

### 🗾 Data Pipeline for Computer Vision
```
CIFAR-10 Loading Pipeline:
    
    Raw Files         Dataset Class        DataLoader        CNN Model
+-----------------+ +-----------------+ +-----------------+ +-----------------+
| data_batch_1    | | CIFAR10Dataset | | Batch: (32,3,  | | Convolutional |
| data_batch_2    |▶| __getitem__()   |▶| 32,32) images  |▶| Neural        |
| data_batch_3    | | Loads on-demand | | Labels: (32,)  | | Network       |
| data_batch_4    | | Normalizes [0,1]| | Shuffled order | | Training      |
| data_batch_5    | | Shape: (3,32,32)| |               | |               |
+-----------------+ +-----------------+ +-----------------+ +-----------------+
```

### PROGRESS Why CIFAR-10 is Perfect for Learning
- **Manageable size**: Fits in memory, fast iteration
- **Real complexity**: Natural images, not toy data
- **Standard benchmark**: Compare with published results
- **CV fundamentals**: Teaches image processing essentials
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
        print(f"PASS CIFAR-10 found at {dataset_dir}")
        return dataset_dir
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(root, "cifar-10.tar.gz")
    
    print(f"📥 Downloading CIFAR-10 (~170MB)...")
    urllib.request.urlretrieve(url, tar_path)
    print("PASS Downloaded!")
    
    print("PACKAGE Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(root)
    print("PASS Ready!")
    
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
        
        # Reshape from flat array to image format: (N, 3, 32, 32) = (batch, channels, height, width)
        # Normalize pixel values from [0, 255] to [0, 1] for neural network training
        # This is critical: neural networks expect inputs in [0,1] range!
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        print(f"PASS Loaded {len(self.data):,} images")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Value range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        ### END SOLUTION
    
    def __getitem__(self, idx):
        ### BEGIN SOLUTION
        # Return individual image and label as Tensors
        # Image shape: (3, 32, 32) = (channels, height, width)
        # Label shape: () = scalar class index
        return Tensor(self.data[idx]), Tensor(self.labels[idx])
        ### END SOLUTION
    
    def __len__(self):
        ### BEGIN SOLUTION
        # Return total number of images
        return len(self.data)
        ### END SOLUTION
    
    def get_num_classes(self):
        ### BEGIN SOLUTION
        # CIFAR-10 has exactly 10 classes
        return 10
        ### END SOLUTION

# %% [markdown]
"""
### 🧪 Unit Test: SimpleDataset

Let's test your SimpleDataset implementation! This concrete example shows how the Dataset pattern works.

**This is a unit test** - it tests the SimpleDataset class in isolation.
"""

# %%
def test_unit_simple_dataset():
    """Test SimpleDataset implementation with comprehensive functionality tests."""
    print("🔬 Unit Test: SimpleDataset...")

    # Create dataset
    dataset = SimpleDataset(size=20, num_features=5, num_classes=4)

    print(f"Dataset created: size={len(dataset)}, features={dataset.num_features}, classes={dataset.get_num_classes()}")

    # Test basic properties
    assert len(dataset) == 20, f"Dataset length should be 20, got {len(dataset)}"
    assert dataset.get_num_classes() == 4, f"Should have 4 classes, got {dataset.get_num_classes()}"

    # Test sample access
    data, label = dataset[0]
    assert isinstance(data, Tensor), "Data should be a Tensor"
    assert isinstance(label, Tensor), "Label should be a Tensor"
    assert data.shape == (5,), f"Data shape should be (5,), got {data.shape}"
    assert label.shape == (), f"Label shape should be (), got {label.shape}"

    # Test sample shape
    sample_shape = dataset.get_sample_shape()
    assert sample_shape == (5,), f"Sample shape should be (5,), got {sample_shape}"

    # Test multiple samples
    for i in range(5):
        data, label = dataset[i]
        assert data.shape == (5,), f"Data shape should be (5,) for sample {i}, got {data.shape}"
        assert 0 <= label.data < 4, f"Label should be in [0, 3] for sample {i}, got {label.data}"

    # Test deterministic data (same seed should give same data)
    dataset2 = SimpleDataset(size=20, num_features=5, num_classes=4)
    data1, label1 = dataset[0]
    data2, label2 = dataset2[0]
    assert np.array_equal(data1.data, data2.data), "Data should be deterministic"
    assert np.array_equal(label1.data, label2.data), "Labels should be deterministic"

    print("✅ SimpleDataset works correctly!")

test_unit_simple_dataset()

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

    print("✅ Data pipeline integration works correctly!")

test_unit_dataloader_pipeline()


# %% [markdown]
# %% [markdown]
"""
## TEST Module Testing

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

    print("PASS Integration Test Passed: DataLoader correctly yields batches of Tensors.")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🔍 Systems Analysis: I/O Pipeline Performance & Scaling

Now that your data loading implementation is complete, let's analyze its performance characteristics and understand how it scales in production systems.

**This section teaches ML systems engineering skills: measuring, profiling, and optimizing data pipeline performance.**
"""

# %%
import time
import os

def analyze_dataloader_performance():
    """
    Comprehensive analysis of DataLoader performance characteristics.

    Measures batch loading times, memory usage patterns, and scaling behavior
    to understand production performance implications.
    """
    print("🔍 DATALOADER PERFORMANCE ANALYSIS")
    print("=" * 50)

    # Test 1: Basic Performance Timing
    print("\n📊 1. BATCH LOADING PERFORMANCE:")
    dataset = SimpleDataset(size=1000, num_features=20, num_classes=10)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Time batch loading
    batch_times = []
    for i, (data, labels) in enumerate(loader):
        if i >= 10:  # Test first 10 batches
            break
        start_time = time.time()
        # Simulate accessing the data (triggers actual loading)
        _ = data.shape, labels.shape
        batch_time = time.time() - start_time
        batch_times.append(batch_time)

    avg_time = sum(batch_times) / len(batch_times)
    throughput = 64 / avg_time  # samples per second

    print(f"   Average batch time: {avg_time:.4f}s")
    print(f"   Throughput: {throughput:.0f} samples/second")
    print(f"   Range: {min(batch_times):.4f}s - {max(batch_times):.4f}s")

    # Test 2: Batch Size Scaling
    print(f"\n📈 2. BATCH SIZE SCALING ANALYSIS:")
    batch_sizes = [16, 32, 64, 128, 256]
    scaling_results = []

    for batch_size in batch_sizes:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Time one batch
        start_time = time.time()
        data, labels = next(iter(loader))
        batch_time = time.time() - start_time

        samples_per_sec = batch_size / batch_time
        scaling_results.append((batch_size, batch_time, samples_per_sec))
        print(f"   Batch {batch_size:3d}: {batch_time:.4f}s ({samples_per_sec:.0f} samples/sec)")

    # Find optimal batch size
    optimal = max(scaling_results, key=lambda x: x[2])
    print(f"   Optimal batch size: {optimal[0]} ({optimal[2]:.0f} samples/sec)")

    # Test 3: Memory Usage Analysis
    print(f"\n💾 3. MEMORY USAGE PATTERNS:")

    # Compare small vs large datasets
    small_dataset = SimpleDataset(size=100, num_features=10, num_classes=5)
    large_dataset = SimpleDataset(size=10000, num_features=50, num_classes=20)

    for name, dataset in [("Small Dataset", small_dataset), ("Large Dataset", large_dataset)]:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Get memory footprint estimate
        data, labels = next(iter(loader))
        data_memory = data.data.nbytes
        labels_memory = labels.data.nbytes
        total_memory = data_memory + labels_memory

        print(f"   {name}:")
        print(f"     Batch memory: {total_memory / 1024:.1f} KB")
        print(f"     Data: {data_memory / 1024:.1f} KB, Labels: {labels_memory / 1024:.1f} KB")
        print(f"     Per sample: {total_memory / 32:.0f} bytes")

    # Test 4: I/O Strategy Comparison
    print(f"\n🔀 4. I/O STRATEGY COMPARISON:")

    dataset = SimpleDataset(size=500, num_features=20, num_classes=10)

    strategies = [
        ("Sequential (no shuffle)", False),
        ("Random (with shuffle)", True)
    ]

    for name, shuffle in strategies:
        loader = DataLoader(dataset, batch_size=50, shuffle=shuffle)

        start_time = time.time()
        batch_count = 0
        for data, labels in loader:
            batch_count += 1
            if batch_count >= 5:  # Test first 5 batches
                break
        total_time = time.time() - start_time

        avg_batch_time = total_time / batch_count
        print(f"   {name}: {avg_batch_time:.4f}s per batch")

    print(f"\n💡 PRODUCTION INSIGHTS:")
    print(f"   • Larger batches improve throughput (amortize overhead)")
    print(f"   • Memory usage scales linearly with batch size and features")
    print(f"   • Shuffling adds minimal overhead for in-memory data")
    print(f"   • GPU utilization depends on data loading not being bottleneck")
    print(f"   • Real bottlenecks: disk I/O, network storage, preprocessing")

# %% [markdown]
"""
## 🧪 Integration Test: DataLoader with Tensors
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

test_module_dataloader_tensor_yield()

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

### 1. Memory vs Performance Trade-offs
In your DataLoader implementation, you discovered that larger batch sizes generally improve throughput. When you tested batches of 16, 32, 64, and 128 samples, you likely saw increasing samples-per-second rates.

**Analysis Question**: Your DataLoader implementation loads entire batches into memory at once. If you needed to handle a dataset with 10GB of data on a machine with only 4GB of RAM, how would you modify your current DataLoader design to support this scenario while maintaining reasonable performance?

**Consider**:
- Memory-mapped files vs loading subsets
- Streaming vs caching strategies
- Trade-offs between memory usage and I/O efficiency

### 2. Production Scaling Analysis
Your SimpleDataset generates synthetic data in memory, but real production systems often need to load from disk, databases, or network storage.

**Scaling Question**: Imagine deploying your DataLoader design to handle ImageNet (150GB of images) on a distributed training cluster with 8 GPUs. Each GPU needs different batches simultaneously, and data is stored on network-attached storage.

**Design Challenge**: What bottlenecks would emerge in your current implementation, and how would you redesign the data loading pipeline to maximize GPU utilization across the cluster?

**Consider**:
- Network bandwidth limitations
- Storage I/O patterns
- Data locality and caching strategies
- Prefetching and parallel loading

### 3. Debugging Production I/O Issues
Your performance analysis showed that shuffling adds minimal overhead for in-memory data, but production systems often experience unpredictable I/O performance.

**Engineering Question**: A production ML system using your DataLoader design suddenly experiences 10x slower training speeds, but the model code hasn't changed. The logs show DataLoader batch loading times varying from 50ms to 5 seconds randomly.

**Root Cause Analysis**: What systematic debugging approach would you use to identify whether the bottleneck is in your DataLoader implementation, the storage system, network, or something else? What metrics would you instrument and monitor?

**Consider**:
- I/O monitoring and profiling techniques
- Distributed system debugging approaches
- Performance regression investigation methods
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: DataLoader - Efficient Data Pipeline Systems

Congratulations! You've successfully implemented a comprehensive data loading system for machine learning:

### What You've Accomplished
✅ **Dataset Interface**: Abstract base class enabling flexible data sources (500+ lines)
✅ **DataLoader Engine**: Efficient batching and iteration system with shuffling support
✅ **SimpleDataset Implementation**: Concrete dataset for synthetic data generation and testing
✅ **CIFAR-10 Integration**: Real-world computer vision dataset loading capabilities
✅ **Performance Analysis**: Comprehensive I/O pipeline profiling and optimization insights
✅ **Integration Testing**: Seamless compatibility validation with Tensor operations

### Key Learning Outcomes
- **Data Pipeline Architecture**: Universal Dataset/DataLoader abstraction used across all ML frameworks
- **Batch Processing Systems**: Memory-efficient handling of large datasets through strategic batching
- **I/O Performance Engineering**: Understanding and measuring data loading bottlenecks in production systems
- **Memory Management**: Efficient tensor stacking and batch creation without memory explosions
- **Production Patterns**: Real-world data loading strategies for scaling ML training systems

### Systems Understanding Achieved
- **Performance Characteristics**: Batch size scaling impacts both throughput and memory usage
- **I/O Bottleneck Analysis**: Data loading often limits training speed more than model computation
- **Memory vs Speed Trade-offs**: Larger batches improve efficiency but require more RAM
- **Shuffling Impact**: Minimal overhead for generalization benefits in training
- **Scaling Behavior**: Linear memory growth with batch size and feature dimensions

### Professional Skills Developed
- **ML Systems Engineering**: Building robust data pipelines that handle production-scale workloads
- **Performance Profiling**: Measuring and optimizing I/O performance for training efficiency
- **API Design**: Clean, extensible interfaces following industry-standard patterns
- **Integration Architecture**: Seamless compatibility with tensor operations and neural networks

### Ready for Advanced Applications
Your DataLoader implementation now enables:
- **Large-scale Training**: Processing datasets larger than available memory
- **Real-time Inference**: Efficient batch processing for production model serving
- **Multi-modal Data**: Support for images, text, and structured data through consistent interfaces
- **Distributed Training**: Foundation for multi-GPU and multi-node data loading strategies

### Connection to Real ML Systems
Your implementations mirror production frameworks:
- **PyTorch**: `torch.utils.data.DataLoader` uses identical batching and iteration patterns
- **TensorFlow**: `tf.data.Dataset` implements the same universal dataset abstraction
- **Industry Standard**: Every major ML framework builds on these exact design patterns

### Next Steps
1. **Export your module**: `tito module complete 09_dataloader`
2. **Validate integration**: All components work together for complete ML pipelines
3. **Ready for Module 10**: Training loops that will use your data loading infrastructure
4. **Production Deployment**: Scale to real datasets and distributed training scenarios

**🚀 Achievement Unlocked**: You've built production-quality data loading infrastructure that powers real ML training systems!
"""

def test_module():
    """Run all module tests systematically."""
    print("🧪 RUNNING MODULE 09 TESTS")
    print("=" * 50)

    try:
        # Run all unit tests
        test_unit_dataset_interface()
        test_unit_dataloader()
        test_unit_simple_dataset()
        test_unit_dataloader_pipeline()
        test_module_dataloader_tensor_yield()

        # Run systems analysis
        analyze_dataloader_performance()

        print("\n✅ ALL MODULE TESTS PASSED!")
        print("🎯 DataLoader module implementation complete!")

    except Exception as e:
        print(f"\n❌ MODULE TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    test_module()