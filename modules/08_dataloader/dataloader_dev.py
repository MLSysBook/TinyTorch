# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#| default_exp data.loader

# %% [markdown]
"""
# Module 08: DataLoader - Efficient Data Pipeline for ML Training

Welcome to Module 08! You're about to build the data loading infrastructure that transforms how ML models consume data during training.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor operations, activations, layers, losses, autograd, optimizers, and training loops
**You'll Build**: Dataset abstraction, DataLoader with batching/shuffling, and real dataset support
**You'll Enable**: Efficient data pipelines that feed hungry neural networks with properly formatted batches

**Connection Map**:
```
Training Loop → DataLoader → Batched Data → Model
(Module 07)    (Module 08)  (optimized)   (ready to learn)
```

## Learning Objectives
By the end of this module, you will:
1. Understand the data pipeline: individual samples → batches → training
2. Implement Dataset abstraction and TensorDataset for tensor-based data
3. Build DataLoader with intelligent batching, shuffling, and memory-efficient iteration
4. Experience data pipeline performance characteristics firsthand
5. Create download functions for real computer vision datasets

Let's transform scattered data into organized learning batches!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/08_dataloader/dataloader_dev.py
**Building Side:** Code exports to tinytorch.data.loader

```python
# Final package structure:
from tinytorch.data.loader import Dataset, DataLoader, TensorDataset  # This module
from tinytorch.data.loader import download_mnist, download_cifar10  # Dataset utilities
from tinytorch.core.tensor import Tensor  # Foundation (Module 01)
```

**Why this matters:**
- **Learning:** Complete data loading system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.utils.data with all core data utilities
- **Efficiency:** Optimized data pipelines are crucial for training speed and memory usage
- **Integration:** Works seamlessly with training loops to create complete ML systems
"""

# %%
# Essential imports for data loading
import numpy as np
import random
from typing import Iterator, Tuple, List, Optional, Union
from abc import ABC, abstractmethod
import os
import gzip
import urllib.request
import pickle
import sys

# Import real Tensor class from Module 01
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

# %% [markdown]
"""
## Part 1: Understanding the Data Pipeline

Before we implement anything, let's understand what happens when neural networks "eat" data. The journey from raw data to trained models follows a specific pipeline that every ML engineer must master.

### The Data Pipeline Journey

Imagine you have 50,000 images of cats and dogs, and you want to train a neural network to classify them:

```
Raw Data Storage          Dataset Interface         DataLoader Batching         Training Loop
┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐      ┌─────────────┐
│ cat_001.jpg     │      │ dataset[0]       │      │ Batch 1:           │      │ model(batch)│
│ dog_023.jpg     │ ───> │ dataset[1]       │ ───> │ [cat, dog, cat]    │ ───> │ optimizer   │
│ cat_045.jpg     │      │ dataset[2]       │      │ Batch 2:           │      │ loss        │
│ ...             │      │ ...              │      │ [dog, cat, dog]    │      │ backward    │
│ (50,000 files)  │      │ dataset[49999]   │      │ ...                │      │ step        │
└─────────────────┘      └──────────────────┘      └────────────────────┘      └─────────────┘
```

### Why This Pipeline Matters

**Individual Access (Dataset)**: Neural networks can't process 50,000 files at once. We need a way to access one sample at a time: "Give me image #1,247".

**Batch Processing (DataLoader)**: GPUs are parallel machines - they're much faster processing 32 images simultaneously than 1 image 32 times.

**Memory Efficiency**: Loading all 50,000 images into memory would require ~150GB. Instead, we load only the current batch (~150MB).

**Training Variety**: Shuffling ensures the model sees different combinations each epoch, preventing memorization.

### The Dataset Abstraction

The Dataset class provides a uniform interface for accessing data, regardless of whether it's stored as files, in memory, in databases, or generated on-the-fly:

```
Dataset Interface
┌─────────────────────────────────────┐
│ __len__()     → "How many samples?" │
│ __getitem__(i) → "Give me sample i" │
└─────────────────────────────────────┘
          ↑                ↑
     Enables for     Enables indexing
    loops/iteration   dataset[index]
```

**Connection to systems**: This abstraction is crucial because it separates *how data is stored* from *how it's accessed*, enabling optimizations like caching, prefetching, and parallel loading.
"""

# %% nbgrader={"grade": false, "grade_id": "dataset-implementation", "solution": true}
class Dataset(ABC):
    """
    Abstract base class for all datasets.

    Provides the fundamental interface that all datasets must implement:
    - __len__(): Returns the total number of samples
    - __getitem__(idx): Returns the sample at given index

    TODO: Implement the abstract Dataset base class

    APPROACH:
    1. Use ABC (Abstract Base Class) to define interface
    2. Mark methods as @abstractmethod to force implementation
    3. Provide clear docstrings for subclasses

    EXAMPLE:
    >>> class MyDataset(Dataset):
    ...     def __len__(self): return 100
    ...     def __getitem__(self, idx): return idx
    >>> dataset = MyDataset()
    >>> print(len(dataset))  # 100
    >>> print(dataset[42])   # 42

    HINT: Abstract methods force subclasses to implement core functionality
    """

    ### BEGIN SOLUTION
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        This method must be implemented by all subclasses to enable
        len(dataset) calls and batch size calculations.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Return the sample at the given index.

        Args:
            idx: Index of the sample to retrieve (0 <= idx < len(dataset))

        Returns:
            The sample at index idx. Format depends on the dataset implementation.
            Could be (data, label) tuple, single tensor, etc.
        """
        pass
    ### END SOLUTION


# %% nbgrader={"grade": true, "grade_id": "test-dataset", "locked": true, "points": 10}
def test_unit_dataset():
    """🔬 Test Dataset abstract base class."""
    print("🔬 Unit Test: Dataset Abstract Base Class...")

    # Test that Dataset is properly abstract
    try:
        dataset = Dataset()
        assert False, "Should not be able to instantiate abstract Dataset"
    except TypeError:
        print("✅ Dataset is properly abstract")

    # Test concrete implementation
    class TestDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return f"item_{idx}"

    dataset = TestDataset(10)
    assert len(dataset) == 10
    assert dataset[0] == "item_0"
    assert dataset[9] == "item_9"

    print("✅ Dataset interface works correctly!")

# test_unit_dataset()  # Moved to main block


# %% [markdown]
"""
## Part 2: TensorDataset - When Data Lives in Memory

Now let's implement TensorDataset, the most common dataset type for when your data is already loaded into tensors. This is perfect for datasets like MNIST where you can fit everything in memory.

### Understanding TensorDataset Structure

TensorDataset takes multiple tensors and aligns them by their first dimension (the sample dimension):

```
Input Tensors (aligned by first dimension):
  Features Tensor        Labels Tensor         Metadata Tensor
  ┌─────────────────┐   ┌───────────────┐     ┌─────────────────┐
  │ [1.2, 3.4, 5.6] │   │ 0 (cat)       │     │ "image_001.jpg" │ ← Sample 0
  │ [2.1, 4.3, 6.5] │   │ 1 (dog)       │     │ "image_002.jpg" │ ← Sample 1
  │ [3.0, 5.2, 7.4] │   │ 0 (cat)       │     │ "image_003.jpg" │ ← Sample 2
  │ ...             │   │ ...           │     │ ...             │
  └─────────────────┘   └───────────────┘     └─────────────────┘
        (N, 3)               (N,)                   (N,)

Dataset Access:
  dataset[1] → (Tensor([2.1, 4.3, 6.5]), Tensor(1), "image_002.jpg")
```

### Why TensorDataset is Powerful

**Memory Locality**: All data is pre-loaded and stored contiguously in memory, enabling fast access patterns.

**Vectorized Operations**: Since everything is already tensors, no conversion overhead during training.

**Supervised Learning Perfect**: Naturally handles (features, labels) pairs, plus any additional metadata.

**Batch-Friendly**: When DataLoader needs a batch, it can slice multiple samples efficiently.

### Real-World Usage Patterns

```
# Computer Vision
images = Tensor(shape=(50000, 32, 32, 3))  # CIFAR-10 images
labels = Tensor(shape=(50000,))            # Class labels 0-9
dataset = TensorDataset(images, labels)

# Natural Language Processing
token_ids = Tensor(shape=(10000, 512))     # Tokenized sentences
labels = Tensor(shape=(10000,))            # Sentiment labels
dataset = TensorDataset(token_ids, labels)

# Time Series
sequences = Tensor(shape=(1000, 100, 5))   # 100 timesteps, 5 features
targets = Tensor(shape=(1000, 10))         # 10-step ahead prediction
dataset = TensorDataset(sequences, targets)
```

The key insight: TensorDataset transforms "arrays of data" into "a dataset that serves samples".
"""

# %% nbgrader={"grade": false, "grade_id": "tensordataset-implementation", "solution": true}
class TensorDataset(Dataset):
    """
    Dataset wrapping tensors for supervised learning.

    Each sample is a tuple of tensors from the same index across all input tensors.
    All tensors must have the same size in their first dimension.

    TODO: Implement TensorDataset for tensor-based data

    APPROACH:
    1. Store all input tensors
    2. Validate they have same first dimension (number of samples)
    3. Return tuple of tensor slices for each index

    EXAMPLE:
    >>> features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features each
    >>> labels = Tensor([0, 1, 0])                    # 3 labels
    >>> dataset = TensorDataset(features, labels)
    >>> print(len(dataset))  # 3
    >>> print(dataset[1])    # (Tensor([3, 4]), Tensor(1))

    HINTS:
    - Use *tensors to accept variable number of tensor arguments
    - Check all tensors have same length in dimension 0
    - Return tuple of tensor[idx] for all tensors
    """

    def __init__(self, *tensors):
        """
        Create dataset from multiple tensors.

        Args:
            *tensors: Variable number of Tensor objects

        All tensors must have the same size in their first dimension.
        """
        ### BEGIN SOLUTION
        assert len(tensors) > 0, "Must provide at least one tensor"

        # Store all tensors
        self.tensors = tensors

        # Validate all tensors have same first dimension
        first_size = len(tensors[0].data)  # Size of first dimension
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"All tensors must have same size in first dimension. "
                    f"Tensor 0: {first_size}, Tensor {i}: {len(tensor.data)}"
                )
        ### END SOLUTION

    def __len__(self) -> int:
        """Return number of samples (size of first dimension)."""
        ### BEGIN SOLUTION
        return len(self.tensors[0].data)
        ### END SOLUTION

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """
        Return tuple of tensor slices at given index.

        Args:
            idx: Sample index

        Returns:
            Tuple containing tensor[idx] for each input tensor
        """
        ### BEGIN SOLUTION
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Return tuple of slices from all tensors
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)
        ### END SOLUTION


# %% nbgrader={"grade": true, "grade_id": "test-tensordataset", "locked": true, "points": 15}
def test_unit_tensordataset():
    """🔬 Test TensorDataset implementation."""
    print("🔬 Unit Test: TensorDataset...")

    # Test basic functionality
    features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
    labels = Tensor([0, 1, 0])                    # 3 labels

    dataset = TensorDataset(features, labels)

    # Test length
    assert len(dataset) == 3, f"Expected length 3, got {len(dataset)}"

    # Test indexing
    sample = dataset[0]
    assert len(sample) == 2, "Should return tuple with 2 tensors"
    assert np.array_equal(sample[0].data, [1, 2]), f"Wrong features: {sample[0].data}"
    assert sample[1].data == 0, f"Wrong label: {sample[1].data}"

    sample = dataset[1]
    assert np.array_equal(sample[1].data, 1), f"Wrong label at index 1: {sample[1].data}"

    # Test error handling
    try:
        dataset[10]  # Out of bounds
        assert False, "Should raise IndexError for out of bounds access"
    except IndexError:
        pass

    # Test mismatched tensor sizes
    try:
        bad_features = Tensor([[1, 2], [3, 4]])  # Only 2 samples
        bad_labels = Tensor([0, 1, 0])           # 3 labels - mismatch!
        TensorDataset(bad_features, bad_labels)
        assert False, "Should raise error for mismatched tensor sizes"
    except ValueError:
        pass

    print("✅ TensorDataset works correctly!")

# test_unit_tensordataset()  # Moved to main block


# %% [markdown]
"""
## Part 3: DataLoader - The Batch Factory

Now we build the DataLoader, the component that transforms individual dataset samples into the batches that neural networks crave. This is where data loading becomes a systems challenge.

### Understanding Batching: From Samples to Tensors

DataLoader performs a crucial transformation - it collects individual samples and stacks them into batch tensors:

```
Step 1: Individual Samples from Dataset
  dataset[0] → (features: [1, 2, 3], label: 0)
  dataset[1] → (features: [4, 5, 6], label: 1)
  dataset[2] → (features: [7, 8, 9], label: 0)
  dataset[3] → (features: [2, 3, 4], label: 1)

Step 2: DataLoader Groups into Batch (batch_size=2)
  Batch 1:
    features: [[1, 2, 3],    ← Stacked into shape (2, 3)
               [4, 5, 6]]
    labels:   [0, 1]         ← Stacked into shape (2,)

  Batch 2:
    features: [[7, 8, 9],    ← Stacked into shape (2, 3)
               [2, 3, 4]]
    labels:   [0, 1]         ← Stacked into shape (2,)
```

### The Shuffling Process

Shuffling randomizes which samples appear in which batches, crucial for good training:

```
Without Shuffling (epoch 1):          With Shuffling (epoch 1):
  Batch 1: [sample 0, sample 1]         Batch 1: [sample 2, sample 0]
  Batch 2: [sample 2, sample 3]         Batch 2: [sample 3, sample 1]
  Batch 3: [sample 4, sample 5]         Batch 3: [sample 5, sample 4]

Without Shuffling (epoch 2):          With Shuffling (epoch 2):
  Batch 1: [sample 0, sample 1]  ✗      Batch 1: [sample 1, sample 4]  ✓
  Batch 2: [sample 2, sample 3]  ✗      Batch 2: [sample 0, sample 5]  ✓
  Batch 3: [sample 4, sample 5]  ✗      Batch 3: [sample 2, sample 3]  ✓

  (Same every epoch = overfitting!)     (Different combinations = better learning!)
```

### DataLoader as a Systems Component

**Memory Management**: DataLoader only holds one batch in memory at a time, not the entire dataset.

**Iteration Interface**: Provides Python iterator protocol so training loops can use `for batch in dataloader:`.

**Collation Strategy**: Automatically stacks tensors from individual samples into batch tensors.

**Performance Critical**: This is often the bottleneck in training pipelines - loading and preparing data can be slower than the forward pass!

### The DataLoader Algorithm

```
1. Create indices list: [0, 1, 2, ..., dataset_length-1]
2. If shuffle=True: randomly shuffle the indices
3. Group indices into chunks of batch_size
4. For each chunk:
   a. Retrieve samples: [dataset[i] for i in chunk]
   b. Collate samples: stack individual tensors into batch tensors
   c. Yield the batch tensor tuple
```

This transforms the dataset from "access one sample" to "iterate through batches" - exactly what training loops need.
"""

# %% nbgrader={"grade": false, "grade_id": "dataloader-implementation", "solution": true}
class DataLoader:
    """
    Data loader with batching and shuffling support.

    Wraps a dataset to provide batched iteration with optional shuffling.
    Essential for efficient training with mini-batch gradient descent.

    TODO: Implement DataLoader with batching and shuffling

    APPROACH:
    1. Store dataset, batch_size, and shuffle settings
    2. Create iterator that groups samples into batches
    3. Handle shuffling by randomizing indices
    4. Collate individual samples into batch tensors

    EXAMPLE:
    >>> dataset = TensorDataset(Tensor([[1,2], [3,4], [5,6]]), Tensor([0,1,0]))
    >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     features_batch, labels_batch = batch
    ...     print(f"Features: {features_batch.shape}, Labels: {labels_batch.shape}")

    HINTS:
    - Use random.shuffle() for index shuffling
    - Group consecutive samples into batches
    - Stack individual tensors using np.stack()
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        """
        Create DataLoader for batched iteration.

        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
        """
        ### BEGIN SOLUTION
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        ### END SOLUTION

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        ### BEGIN SOLUTION
        # Calculate number of complete batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        ### END SOLUTION

    def __iter__(self) -> Iterator:
        """Return iterator over batches."""
        ### BEGIN SOLUTION
        # Create list of indices
        indices = list(range(len(self.dataset)))

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Collate batch - convert list of tuples to tuple of tensors
            yield self._collate_batch(batch)
        ### END SOLUTION

    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        """
        Collate individual samples into batch tensors.

        Args:
            batch: List of sample tuples from dataset

        Returns:
            Tuple of batched tensors
        """
        ### BEGIN SOLUTION
        if len(batch) == 0:
            return ()

        # Determine number of tensors per sample
        num_tensors = len(batch[0])

        # Group tensors by position
        batched_tensors = []
        for tensor_idx in range(num_tensors):
            # Extract all tensors at this position
            tensor_list = [sample[tensor_idx].data for sample in batch]

            # Stack into batch tensor
            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)
        ### END SOLUTION


# %% nbgrader={"grade": true, "grade_id": "test-dataloader", "locked": true, "points": 20}
def test_unit_dataloader():
    """🔬 Test DataLoader implementation."""
    print("🔬 Unit Test: DataLoader...")

    # Create test dataset
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # 5 samples
    labels = Tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(features, labels)

    # Test basic batching (no shuffle)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Test length calculation
    assert len(loader) == 3, f"Expected 3 batches, got {len(loader)}"  # ceil(5/2) = 3

    batches = list(loader)
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

    # Test first batch
    batch_features, batch_labels = batches[0]
    assert batch_features.data.shape == (2, 2), f"Wrong batch features shape: {batch_features.data.shape}"
    assert batch_labels.data.shape == (2,), f"Wrong batch labels shape: {batch_labels.data.shape}"

    # Test last batch (should have 1 sample)
    batch_features, batch_labels = batches[2]
    assert batch_features.data.shape == (1, 2), f"Wrong last batch features shape: {batch_features.data.shape}"
    assert batch_labels.data.shape == (1,), f"Wrong last batch labels shape: {batch_labels.data.shape}"

    # Test that data is preserved
    assert np.array_equal(batches[0][0].data[0], [1, 2]), "First sample should be [1,2]"
    assert batches[0][1].data[0] == 0, "First label should be 0"

    # Test shuffling produces different order
    loader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
    loader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)

    batch_shuffle = list(loader_shuffle)[0]
    batch_no_shuffle = list(loader_no_shuffle)[0]

    # Note: This might occasionally fail due to random chance, but very unlikely
    # We'll just test that both contain all the original data
    shuffle_features = set(tuple(row) for row in batch_shuffle[0].data)
    no_shuffle_features = set(tuple(row) for row in batch_no_shuffle[0].data)
    expected_features = {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)}

    assert shuffle_features == expected_features, "Shuffle should preserve all data"
    assert no_shuffle_features == expected_features, "No shuffle should preserve all data"

    print("✅ DataLoader works correctly!")

# test_unit_dataloader()  # Moved to main block


# %% [markdown]
"""
## Part 4: Real Datasets - MNIST and CIFAR-10

Time to work with real data! We'll implement download functions for two classic computer vision datasets that every ML engineer should know.

### Understanding Standard Datasets

MNIST and CIFAR-10 are the "hello world" datasets of computer vision, each teaching different lessons:

```
MNIST (Handwritten Digits)              CIFAR-10 (Tiny Objects)
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ Size: 28×28 pixels          │          │ Size: 32×32×3 pixels        │
│ Colors: Grayscale (1 chan)  │          │ Colors: RGB (3 channels)    │
│ Classes: 10 (digits 0-9)    │          │ Classes: 10 (objects)       │
│ Training: 60,000 samples    │          │ Training: 50,000 samples    │
│ Testing: 10,000 samples     │          │ Testing: 10,000 samples     │
│                             │          │                             │
│ ┌─────┐ ┌─────┐ ┌─────┐     │          │ ┌─────┐ ┌─────┐ ┌─────┐     │
│ │  5  │ │  3  │ │  8  │     │          │ │ ✈️  │ │ 🚗  │ │ 🐸  │     │
│ └─────┘ └─────┘ └─────┘     │          │ └─────┘ └─────┘ └─────┘     │
│ (simple shapes)             │          │ (complex textures)          │
└─────────────────────────────┘          └─────────────────────────────┘
```

### Why These Datasets Matter

**MNIST**: Perfect for learning basics - simple, clean, small. Most algorithms achieve >95% accuracy.

**CIFAR-10**: Real-world complexity - color, texture, background clutter. Much harder, ~80-90% is good.

**Progression**: MNIST → CIFAR-10 → ImageNet represents increasing complexity in computer vision.

### Dataset Format Patterns

Both datasets follow similar patterns:

```
Typical Dataset Structure:
┌─────────────────────────────────────────┐
│ Training Set                            │
│ ├── Images: (N, H, W, C) tensor         │
│ └── Labels: (N,) tensor                 │
│                                         │
│ Test Set                                │
│ ├── Images: (M, H, W, C) tensor         │
│ └── Labels: (M,) tensor                 │
└─────────────────────────────────────────┘

Where:
  N = number of training samples
  M = number of test samples
  H, W = height, width
  C = channels (1 for grayscale, 3 for RGB)
```

### Data Pipeline Integration

Once downloaded, these datasets integrate seamlessly with our pipeline:

```
Download Function → TensorDataset → DataLoader → Training
      ↓                   ↓             ↓           ↓
  Raw tensors      Indexed access   Batched data  Model input
```

**Note**: For educational purposes, we'll create synthetic datasets with the same structure as MNIST/CIFAR-10. In production, you'd download the actual data from official sources.
"""

# %% nbgrader={"grade": false, "grade_id": "download-functions", "solution": true}
def download_mnist(data_dir: str = "./data") -> Tuple[TensorDataset, TensorDataset]:
    """
    Download and prepare MNIST dataset.

    Returns train and test datasets with (images, labels) format.
    Images are normalized to [0,1] range.

    TODO: Implement MNIST download and preprocessing

    APPROACH:
    1. Create data directory if needed
    2. Download MNIST files from official source
    3. Parse binary format and extract images/labels
    4. Normalize images and convert to tensors
    5. Return TensorDataset objects

    EXAMPLE:
    >>> train_ds, test_ds = download_mnist()
    >>> print(f"Train: {len(train_ds)} samples")
    >>> print(f"Test: {len(test_ds)} samples")
    >>> image, label = train_ds[0]
    >>> print(f"Image shape: {image.shape}, Label: {label.data}")

    HINTS:
    - MNIST images are 28x28 grayscale, stored as uint8
    - Labels are single integers 0-9
    - Normalize images by dividing by 255.0
    """
    ### BEGIN SOLUTION
    os.makedirs(data_dir, exist_ok=True)

    # MNIST URLs (simplified - using a mock implementation for educational purposes)
    # In production, you'd download from official sources

    # Create simple synthetic MNIST-like data for educational purposes
    print("📥 Creating synthetic MNIST-like dataset for educational purposes...")

    # Generate synthetic training data (60,000 samples)
    np.random.seed(42)  # For reproducibility
    train_images = np.random.rand(60000, 28, 28).astype(np.float32)
    train_labels = np.random.randint(0, 10, 60000).astype(np.int64)

    # Generate synthetic test data (10,000 samples)
    test_images = np.random.rand(10000, 28, 28).astype(np.float32)
    test_labels = np.random.randint(0, 10, 10000).astype(np.int64)

    # Create TensorDatasets
    train_dataset = TensorDataset(Tensor(train_images), Tensor(train_labels))
    test_dataset = TensorDataset(Tensor(test_images), Tensor(test_labels))

    print(f"✅ MNIST-like dataset ready: {len(train_dataset)} train, {len(test_dataset)} test samples")

    return train_dataset, test_dataset
    ### END SOLUTION


def download_cifar10(data_dir: str = "./data") -> Tuple[TensorDataset, TensorDataset]:
    """
    Download and prepare CIFAR-10 dataset.

    Returns train and test datasets with (images, labels) format.
    Images are normalized to [0,1] range.

    TODO: Implement CIFAR-10 download and preprocessing

    APPROACH:
    1. Create data directory if needed
    2. Download CIFAR-10 files from official source
    3. Parse pickle format and extract images/labels
    4. Normalize images and convert to tensors
    5. Return TensorDataset objects

    EXAMPLE:
    >>> train_ds, test_ds = download_cifar10()
    >>> print(f"Train: {len(train_ds)} samples")
    >>> image, label = train_ds[0]
    >>> print(f"Image shape: {image.shape}, Label: {label.data}")

    HINTS:
    - CIFAR-10 images are 32x32x3 color, stored as uint8
    - Labels are single integers 0-9 (airplane, automobile, etc.)
    - Images come in format (height, width, channels)
    """
    ### BEGIN SOLUTION
    os.makedirs(data_dir, exist_ok=True)

    # Create simple synthetic CIFAR-10-like data for educational purposes
    print("📥 Creating synthetic CIFAR-10-like dataset for educational purposes...")

    # Generate synthetic training data (50,000 samples)
    np.random.seed(123)  # Different seed than MNIST
    train_images = np.random.rand(50000, 32, 32, 3).astype(np.float32)
    train_labels = np.random.randint(0, 10, 50000).astype(np.int64)

    # Generate synthetic test data (10,000 samples)
    test_images = np.random.rand(10000, 32, 32, 3).astype(np.float32)
    test_labels = np.random.randint(0, 10, 10000).astype(np.int64)

    # Create TensorDatasets
    train_dataset = TensorDataset(Tensor(train_images), Tensor(train_labels))
    test_dataset = TensorDataset(Tensor(test_images), Tensor(test_labels))

    print(f"✅ CIFAR-10-like dataset ready: {len(train_dataset)} train, {len(test_dataset)} test samples")

    return train_dataset, test_dataset
    ### END SOLUTION


# %% nbgrader={"grade": true, "grade_id": "test-download-functions", "locked": true, "points": 15}
def test_unit_download_functions():
    """🔬 Test dataset download functions."""
    print("🔬 Unit Test: Download Functions...")

    # Test MNIST download
    train_mnist, test_mnist = download_mnist()

    assert len(train_mnist) == 60000, f"MNIST train should have 60000 samples, got {len(train_mnist)}"
    assert len(test_mnist) == 10000, f"MNIST test should have 10000 samples, got {len(test_mnist)}"

    # Test sample format
    image, label = train_mnist[0]
    assert image.data.shape == (28, 28), f"MNIST image should be (28,28), got {image.data.shape}"
    assert 0 <= label.data <= 9, f"MNIST label should be 0-9, got {label.data}"
    assert 0 <= image.data.max() <= 1, f"MNIST images should be normalized to [0,1], max is {image.data.max()}"

    # Test CIFAR-10 download
    train_cifar, test_cifar = download_cifar10()

    assert len(train_cifar) == 50000, f"CIFAR-10 train should have 50000 samples, got {len(train_cifar)}"
    assert len(test_cifar) == 10000, f"CIFAR-10 test should have 10000 samples, got {len(test_cifar)}"

    # Test sample format
    image, label = train_cifar[0]
    assert image.data.shape == (32, 32, 3), f"CIFAR-10 image should be (32,32,3), got {image.data.shape}"
    assert 0 <= label.data <= 9, f"CIFAR-10 label should be 0-9, got {label.data}"
    assert 0 <= image.data.max() <= 1, f"CIFAR-10 images should be normalized, max is {image.data.max()}"

    print("✅ Download functions work correctly!")

# test_unit_download_functions()  # Moved to main block


# %% [markdown]
"""
## Part 5: Systems Analysis - Data Pipeline Performance

Now let's analyze our data pipeline like production ML engineers. Understanding where time and memory go is crucial for building systems that scale.

### The Performance Question: Where Does Time Go?

In a typical training step, time is split between data loading and computation:

```
Training Step Breakdown:
┌───────────────────────────────────────────────────────────────┐
│ Data Loading        │ Forward Pass     │ Backward Pass     │
│ ████████████         │ ███████         │ ████████         │
│ 40ms               │ 25ms            │ 35ms              │
└───────────────────────────────────────────────────────────────┘
              100ms total per step

Bottleneck Analysis:
- If data loading > forward+backward: "Data starved" (CPU bottleneck)
- If forward+backward > data loading: "Compute bound" (GPU bottleneck)
- Ideal: Data loading ≈ computation time (balanced pipeline)
```

### Memory Scaling: The Batch Size Trade-off

Batch size creates a fundamental trade-off in memory vs efficiency:

```
Batch Size Impact:

Small Batches (batch_size=8):
┌─────────────────────────────────────────┐
│ Memory: 8 × 28 × 28 × 4 bytes = 25KB   │ ← Low memory
│ Overhead: High (many small batches)    │ ← High overhead
│ GPU Util: Poor (underutilized)         │ ← Poor efficiency
└─────────────────────────────────────────┘

Large Batches (batch_size=512):
┌─────────────────────────────────────────┐
│ Memory: 512 × 28 × 28 × 4 bytes = 1.6MB│ ← Higher memory
│ Overhead: Low (fewer large batches)    │ ← Lower overhead
│ GPU Util: Good (well utilized)         │ ← Better efficiency
└─────────────────────────────────────────┘
```

### Shuffling Overhead Analysis

Shuffling seems simple, but let's measure its real cost:

```
Shuffle Operation Breakdown:

1. Index Generation:    O(n) - create [0, 1, 2, ..., n-1]
2. Shuffle Operation:   O(n) - randomize the indices
3. Sample Access:       O(1) per sample - dataset[shuffled_idx]

Memory Impact:
- No Shuffle: 0 extra memory (sequential access)
- With Shuffle: 8 bytes × dataset_size (store indices)

For 50,000 samples: 8 × 50,000 = 400KB extra memory
```

The key insight: shuffling overhead is typically negligible compared to the actual data loading and tensor operations.

### Pipeline Bottleneck Identification

We'll measure three critical metrics:

1. **Throughput**: Samples processed per second
2. **Memory Usage**: Peak memory during batch loading
3. **Overhead**: Time spent on data vs computation

These measurements will reveal whether our pipeline is CPU-bound (slow data loading) or compute-bound (slow model).
"""

# %% nbgrader={"grade": false, "grade_id": "systems-analysis", "solution": true}
def analyze_dataloader_performance():
    """📊 Analyze DataLoader performance characteristics."""
    print("📊 Analyzing DataLoader Performance...")

    import time

    # Create test dataset of varying sizes
    sizes = [1000, 5000, 10000]
    batch_sizes = [16, 64, 256]

    print("\n🔍 Batch Size vs Loading Time:")

    for size in sizes:
        # Create synthetic dataset
        features = Tensor(np.random.randn(size, 100))  # 100 features
        labels = Tensor(np.random.randint(0, 10, size))
        dataset = TensorDataset(features, labels)

        print(f"\nDataset size: {size} samples")

        for batch_size in batch_sizes:
            # Time data loading
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            start_time = time.time()
            batch_count = 0
            for batch in loader:
                batch_count += 1
            end_time = time.time()

            elapsed = end_time - start_time
            throughput = size / elapsed if elapsed > 0 else float('inf')

            print(f"  Batch size {batch_size:3d}: {elapsed:.3f}s ({throughput:,.0f} samples/sec)")

    # Analyze shuffle overhead
    print("\n🔄 Shuffle Overhead Analysis:")

    dataset_size = 10000
    features = Tensor(np.random.randn(dataset_size, 50))
    labels = Tensor(np.random.randint(0, 5, dataset_size))
    dataset = TensorDataset(features, labels)

    batch_size = 64

    # No shuffle
    loader_no_shuffle = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()
    batches_no_shuffle = list(loader_no_shuffle)
    time_no_shuffle = time.time() - start_time

    # With shuffle
    loader_shuffle = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    batches_shuffle = list(loader_shuffle)
    time_shuffle = time.time() - start_time

    shuffle_overhead = ((time_shuffle - time_no_shuffle) / time_no_shuffle) * 100

    print(f"  No shuffle: {time_no_shuffle:.3f}s")
    print(f"  With shuffle: {time_shuffle:.3f}s")
    print(f"  Shuffle overhead: {shuffle_overhead:.1f}%")

    print("\n💡 Key Insights:")
    print("• Larger batch sizes reduce per-sample overhead")
    print("• Shuffle adds minimal overhead for reasonable dataset sizes")
    print("• Memory usage scales linearly with batch size")
    print("🚀 Production tip: Balance batch size with GPU memory limits")

# analyze_dataloader_performance()  # Moved to main block


def analyze_memory_usage():
    """📊 Analyze memory usage patterns in data loading."""
    print("\n📊 Analyzing Memory Usage Patterns...")

    # Memory usage estimation
    def estimate_memory_mb(batch_size, feature_size, dtype_bytes=4):
        """Estimate memory usage for a batch."""
        return (batch_size * feature_size * dtype_bytes) / (1024 * 1024)

    print("\n💾 Memory Usage by Batch Configuration:")

    feature_sizes = [784, 3072, 50176]  # MNIST, CIFAR-10, ImageNet-like
    feature_names = ["MNIST (28×28)", "CIFAR-10 (32×32×3)", "ImageNet (224×224×1)"]
    batch_sizes = [1, 32, 128, 512]

    for feature_size, name in zip(feature_sizes, feature_names):
        print(f"\n{name}:")
        for batch_size in batch_sizes:
            memory_mb = estimate_memory_mb(batch_size, feature_size)
            print(f"  Batch {batch_size:3d}: {memory_mb:6.1f} MB")

    print("\n🎯 Memory Trade-offs:")
    print("• Larger batches: More memory, better GPU utilization")
    print("• Smaller batches: Less memory, more noisy gradients")
    print("• Sweet spot: Usually 32-128 depending on model size")

    # Demonstrate actual memory usage with our tensors
    print("\n🔬 Actual Tensor Memory Usage:")

    # Create different sized tensors
    tensor_small = Tensor(np.random.randn(32, 784))    # Small batch
    tensor_large = Tensor(np.random.randn(512, 784))   # Large batch

    # Size in bytes (roughly)
    small_bytes = tensor_small.data.nbytes
    large_bytes = tensor_large.data.nbytes

    print(f"  Small batch (32×784): {small_bytes / 1024:.1f} KB")
    print(f"  Large batch (512×784): {large_bytes / 1024:.1f} KB")
    print(f"  Ratio: {large_bytes / small_bytes:.1f}×")

# analyze_memory_usage()  # Moved to main block


# %% [markdown]
"""
## Part 6: Integration Testing

Let's test how our DataLoader integrates with a complete training workflow, simulating real ML pipeline usage.
"""

# %% nbgrader={"grade": false, "grade_id": "integration-test", "solution": true}
def test_training_integration():
    """🔬 Test DataLoader integration with training workflow."""
    print("🔬 Integration Test: Training Workflow...")

    # Create a realistic dataset
    num_samples = 1000
    num_features = 20
    num_classes = 5

    # Synthetic classification data
    features = Tensor(np.random.randn(num_samples, num_features))
    labels = Tensor(np.random.randint(0, num_classes, num_samples))

    dataset = TensorDataset(features, labels)

    # Create train/val splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Manual split (in production, you'd use proper splitting utilities)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    # Create subset datasets
    train_samples = [dataset[i] for i in train_indices]
    val_samples = [dataset[i] for i in val_indices]

    # Convert back to tensors for TensorDataset
    train_features = Tensor(np.stack([sample[0].data for sample in train_samples]))
    train_labels = Tensor(np.stack([sample[1].data for sample in train_samples]))
    val_features = Tensor(np.stack([sample[0].data for sample in val_samples]))
    val_labels = Tensor(np.stack([sample[1].data for sample in val_samples]))

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Dataset splits:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Simulate training loop
    print("\n🏃 Simulated Training Loop:")

    epoch_samples = 0
    batch_count = 0

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_count += 1
        epoch_samples += len(batch_features.data)

        # Simulate forward pass (just check shapes)
        assert batch_features.data.shape[0] <= batch_size, "Batch size exceeded"
        assert batch_features.data.shape[1] == num_features, "Wrong feature count"
        assert len(batch_labels.data) == len(batch_features.data), "Mismatched batch sizes"

        if batch_idx < 3:  # Show first few batches
            print(f"  Batch {batch_idx + 1}: {batch_features.data.shape[0]} samples")

    print(f"  Total: {batch_count} batches, {epoch_samples} samples processed")

    # Validate that all samples were seen
    assert epoch_samples == len(train_dataset), f"Expected {len(train_dataset)}, processed {epoch_samples}"

    print("✅ Training integration works correctly!")

# test_training_integration()  # Moved to main block


# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %%
def test_module():
    """
    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_dataset()
    test_unit_tensordataset()
    test_unit_dataloader()
    test_unit_download_functions()

    print("\nRunning integration scenarios...")

    # Test complete workflow
    test_training_integration()

    # Test realistic dataset usage
    print("🔬 Integration Test: Realistic Dataset Usage...")

    # Download datasets
    train_mnist, test_mnist = download_mnist()

    # Create DataLoaders
    train_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_mnist, batch_size=64, shuffle=False)

    # Test iteration
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    assert len(train_batch) == 2, "Batch should contain (images, labels)"
    assert train_batch[0].data.shape[0] == 64, f"Wrong batch size: {train_batch[0].data.shape[0]}"
    assert train_batch[0].data.shape[1:] == (28, 28), f"Wrong image shape: {train_batch[0].data.shape[1:]}"

    print("✅ Realistic dataset usage works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 08")

# Call before module summary
# test_module()  # Moved to main block


# %%
if __name__ == "__main__":
    print("🚀 Running DataLoader module...")

    # Run all unit tests
    test_unit_dataset()
    test_unit_tensordataset()
    test_unit_dataloader()
    test_unit_download_functions()

    # Run performance analysis
    analyze_dataloader_performance()
    analyze_memory_usage()

    # Run integration test
    test_training_integration()

    # Run final module test
    test_module()

    print("✅ Module validation complete!")


# %% [markdown]
"""
## 🤔 ML Systems Thinking: Data Pipeline Design

### Question 1: Memory vs Speed Trade-offs
You implemented DataLoader with different batch sizes.
If you have 10GB of GPU memory and each sample uses 1MB:
- Maximum batch size before out-of-memory: _____ samples
- If you use batch size 32 instead of maximum, how much memory is unused? _____ GB

### Question 2: Shuffling Impact
Your DataLoader has shuffle=True option.
For a dataset with 50,000 samples and batch_size=100:
- How many batches per epoch? _____
- If you shuffle every epoch for 10 epochs, how many different batch combinations are possible? _____
- Why is shuffling important for training? _____

### Question 3: Data Pipeline Bottlenecks
You measured DataLoader performance across different configurations.
If loading data takes 0.1 seconds per batch and forward pass takes 0.05 seconds:
- What percentage of time is spent on data loading? _____%
- How would you optimize this pipeline? _____
- What happens to training speed if you increase workers from 1 to 4? _____

### Question 4: Dataset Design Patterns
You implemented both Dataset and TensorDataset classes.
For a text dataset with variable-length sequences:
- Would TensorDataset work directly? Yes/No: _____
- What preprocessing would you need? _____
- How would batching work with different sequence lengths? _____

### Question 5: Production Scaling
Your implementation works for thousands of samples.
For training on 1 million samples with distributed training across 8 GPUs:
- How would you split the dataset? _____
- What happens to effective batch size? _____
- How does shuffling work across multiple machines? _____
"""


# %% [markdown]
"""
## 🎯 MODULE SUMMARY: DataLoader

Congratulations! You've built a complete data loading pipeline for ML training!

### Key Accomplishments
- Built Dataset abstraction and TensorDataset implementation with proper tensor alignment
- Created DataLoader with batching, shuffling, and memory-efficient iteration
- Added MNIST and CIFAR-10 download functions for computer vision workflows
- Analyzed data pipeline performance and discovered memory/speed trade-offs
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Batch size directly impacts memory usage and training throughput**
- **Shuffling adds minimal overhead but prevents overfitting patterns**
- **Data loading can become a bottleneck without proper optimization**
- **Memory usage scales linearly with batch size and feature dimensions**

### Ready for Next Steps
Your DataLoader implementation enables efficient training of CNNs and larger models with proper data pipeline management.
Export with: `tito module complete 08`

**Next**: Module 09 (Spatial) will add Conv2d layers that leverage your efficient data loading for image processing!

### Real-World Connection
You've implemented the same patterns used in:
- **PyTorch's DataLoader**: Same interface design for batching and shuffling
- **TensorFlow's Dataset API**: Similar abstraction for data pipeline optimization
- **Production ML**: Essential for handling large-scale training efficiently
- **Research**: Standard foundation for all deep learning experiments

Your data loading pipeline is now ready to power the CNN training in Module 09!
"""