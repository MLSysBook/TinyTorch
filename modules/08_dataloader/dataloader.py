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
#| export

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

**Learning Side:** You work in `modules/08_dataloader/dataloader_dev.py`  
**Building Side:** Code exports to `tinytorch.data.loader`

```python
# How to use this module:
from tinytorch.data.loader import Dataset, DataLoader, TensorDataset
from tinytorch.data.loader import download_mnist, download_cifar10
```

**Why this matters:**
- **Learning:** Complete data loading system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.utils.data with all core data utilities
- **Efficiency:** Optimized data pipelines are crucial for training speed and memory usage
- **Integration:** Works seamlessly with training loops to create complete ML systems
"""

# %%
#| export
# Essential imports for data loading
import numpy as np
import random
import time
import sys
from typing import Iterator, Tuple, List, Optional, Union
from abc import ABC, abstractmethod

# Import real Tensor class from tinytorch package
from tinytorch.core.tensor import Tensor

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
#| export
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

if __name__ == "__main__":
    test_unit_dataset()


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
#| export
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
    labels = Tensor([0, 1, 0])                   # 3 labels

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

if __name__ == "__main__":
    test_unit_tensordataset()


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
#| export
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

if __name__ == "__main__":
    test_unit_dataloader()


# %% nbgrader={"grade": true, "grade_id": "test-dataloader-deterministic", "locked": true, "points": 5}
def test_unit_dataloader_deterministic():
    """🔬 Test DataLoader deterministic shuffling with fixed seed."""
    print("🔬 Unit Test: DataLoader Deterministic Shuffling...")

    # Create test dataset
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    labels = Tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)

    # Test that same seed produces same shuffle
    random.seed(42)
    loader1 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches1 = list(loader1)

    random.seed(42)
    loader2 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches2 = list(loader2)

    # Should produce identical batches with same seed
    for i, (batch1, batch2) in enumerate(zip(batches1, batches2)):
        assert np.array_equal(batch1[0].data, batch2[0].data), \
            f"Batch {i} features should be identical with same seed"
        assert np.array_equal(batch1[1].data, batch2[1].data), \
            f"Batch {i} labels should be identical with same seed"

    # Test that different seeds produce different shuffles
    random.seed(42)
    loader3 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches3 = list(loader3)

    random.seed(123)  # Different seed
    loader4 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches4 = list(loader4)

    # Should produce different batches with different seeds (very likely)
    different = False
    for batch3, batch4 in zip(batches3, batches4):
        if not np.array_equal(batch3[0].data, batch4[0].data):
            different = True
            break

    assert different, "Different seeds should produce different shuffles"

    print("✅ Deterministic shuffling works correctly!")

if __name__ == "__main__":
    test_unit_dataloader_deterministic()


# %% [markdown]
"""
## Part 4: Working with Real Datasets

Now that you've built the DataLoader abstraction, you're ready to use it with real data!

### Using Real Datasets: The TinyTorch Approach

TinyTorch separates **mechanics** (this module) from **application** (examples/milestones):

```
Module 08 (DataLoader)          Examples & Milestones
┌──────────────────────┐       ┌────────────────────────┐
│ Dataset abstraction  │       │ Real MNIST digits      │
│ TensorDataset impl   │  ───> │ CIFAR-10 images        │
│ DataLoader batching  │       │ Custom datasets        │
│ Shuffle & iteration  │       │ Download utilities     │
└──────────────────────┘       └────────────────────────┘
   (Learn mechanics)              (Apply to real data)
```

### Understanding Image Data

**What does image data actually look like?**

Images are just 2D arrays of numbers (pixels). Here are actual 8×8 handwritten digits:

```
Digit "5" (8×8):        Digit "3" (8×8):        Digit "8" (8×8):
 0  0 12 13  5  0  0  0   0  0 11 12  0  0  0  0   0  0 10 14  8  1  0  0
 0  0 13 15 10  0  0  0   0  2 16 16 16  7  0  0   0  0 16 15 15  9  0  0
 0  3 15 13 16  7  0  0   0  0  8 16  8  0  0  0   0  0 15  5  5 13  0  0
 0  8 13  6 15  4  0  0   0  0  0 12 13  0  0  0   0  1 16  5  5 13  0  0
 0  0  0  6 16  5  0  0   0  0  1 16 15  9  0  0   0  6 16 16 16 16  1  0
 0  0  5 15 16  9  0  0   0  0 14 16 16 16  7  0   1 16  3  1  1 15  1  0
 0  0  9 16  9  0  0  0   0  5 16  8  8 16  0  0   0  9 16 16 16 15  0  0
 0  0  0  0  0  0  0  0   0  3 16 16 16 12  0  0   0  0  0  0  0  0  0  0

Visual representation:           
░█████░          ░█████░          ░█████░
░█░░░█░          ░░░░░█░          █░░░░█░
░░░░█░░          ░░███░░          ░█████░
░░░█░░░          ░░░░█░░          █░░░░█░
░░█░░░░          ░█████░          ░█████░
```

**Shape transformations in DataLoader:**

```
Individual Sample (from Dataset):
  image: (8, 8)      ← Single 8×8 image
  label: scalar      ← Single digit (0-9)

After DataLoader batching (batch_size=32):
  images: (32, 8, 8)  ← Stack of 32 images
  labels: (32,)       ← Array of 32 labels
  
This is what your model sees during training!
```

### Quick Start with Real Data

**Tiny Datasets (ships with TinyTorch):**
```python
# 8×8 handwritten digits - instant, no downloads!
import numpy as np
data = np.load('datasets/tiny/digits_8x8.npz')
images = Tensor(data['images'])  # (1797, 8, 8)
labels = Tensor(data['labels'])  # (1797,)

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Each batch contains real digit images!
for batch_images, batch_labels in loader:
    # batch_images: (32, 8, 8) - 32 digit images
    # batch_labels: (32,) - their labels (0-9)
    break
```

**Full Datasets (for serious training):**
```python
# See milestones/03_mlp_revival_1986/ for MNIST download (28×28 images)
# See milestones/04_cnn_revolution_1998/ for CIFAR-10 download (32×32×3 images)
```

### What You've Accomplished

You've built the **data loading infrastructure** that powers all modern ML:
- ✅ Dataset abstraction (universal interface)
- ✅ TensorDataset (in-memory efficiency)
- ✅ DataLoader (batching, shuffling, iteration)

**Next steps:** Apply your DataLoader to real datasets in the milestones!

**Real-world connection:** You've implemented the same patterns as:
- PyTorch's `torch.utils.data.DataLoader`
- TensorFlow's `tf.data.Dataset`
- Production ML pipelines everywhere
"""


# %% [markdown]
"""
## Part 5: Systems Analysis - Data Pipeline Performance

**Note:** This section provides performance analysis tools for understanding DataLoader behavior. The analysis functions are defined below but not run automatically. To explore performance characteristics, uncomment and run `analyze_dataloader_performance()` or `analyze_memory_usage()` manually.

Now let's understand data pipeline performance like production ML engineers. Understanding where time and memory go is crucial for building systems that scale.

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

# analyze_dataloader_performance()  # Optional: Run manually for performance insights


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

    # Measure actual memory (data array + object overhead)
    small_bytes = tensor_small.data.nbytes
    large_bytes = tensor_large.data.nbytes

    # Also measure Python object overhead
    small_total = sys.getsizeof(tensor_small.data) + sys.getsizeof(tensor_small)
    large_total = sys.getsizeof(tensor_large.data) + sys.getsizeof(tensor_large)

    print(f"  Small batch (32×784):")
    print(f"    - Data only: {small_bytes / 1024:.1f} KB")
    print(f"    - With object overhead: {small_total / 1024:.1f} KB")
    print(f"  Large batch (512×784):")
    print(f"    - Data only: {large_bytes / 1024:.1f} KB")
    print(f"    - With object overhead: {large_total / 1024:.1f} KB")
    print(f"  Ratio: {large_bytes / small_bytes:.1f}× (data scales linearly)")

    print("\n🎯 Memory Optimization Tips:")
    print("• Object overhead becomes negligible with larger batches")
    print("• Use float32 instead of float64 to halve memory usage")
    print("• Consider gradient accumulation for effective larger batches")

# analyze_memory_usage()  # Optional: Run manually for memory insights


def analyze_collation_overhead():
    """📊 Analyze the cost of collating samples into batches."""
    print("\n📊 Analyzing Collation Overhead...")

    # Test different batch sizes to see collation cost
    dataset_size = 1000
    feature_size = 100
    features = Tensor(np.random.randn(dataset_size, feature_size))
    labels = Tensor(np.random.randint(0, 10, dataset_size))
    dataset = TensorDataset(features, labels)

    print("\n⚡ Collation Time by Batch Size:")

    for batch_size in [8, 32, 128, 512]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        start_time = time.time()
        for batch in loader:
            pass  # Just iterate, measuring collation overhead
        total_time = time.time() - start_time

        batches = len(loader)
        time_per_batch = (total_time / batches) * 1000  # Convert to ms

        print(f"  Batch size {batch_size:3d}: {time_per_batch:.2f}ms per batch ({batches} batches total)")

    print("\n💡 Collation Insights:")
    print("• Larger batches take longer to collate (more np.stack operations)")
    print("• But fewer large batches are more efficient than many small ones")
    print("• Optimal: Balance between batch size and iteration overhead")

# analyze_collation_overhead()  # Optional: Run manually for collation insights


# %% [markdown]
"""
## Part 6: Common Pitfalls and Best Practices

Before we move to integration testing, let's cover common mistakes students and practitioners make with data loading:

### ⚠️ Common Mistakes to Avoid

**1. Forgetting to Shuffle Training Data**
```python
# ❌ WRONG - No shuffling means same batches every epoch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# ✅ CORRECT - Shuffle for training, but not for validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```
**Why it matters:** Without shuffling, your model sees the same batch combinations every epoch, leading to overfitting to batch-specific patterns rather than general patterns.

**2. Batch Size Too Large (Out of Memory)**
```python
# ❌ WRONG - Batch size might exceed GPU memory
loader = DataLoader(dataset, batch_size=1024)  # Might cause OOM!

# ✅ CORRECT - Start small and increase gradually
loader = DataLoader(dataset, batch_size=32)    # Safe starting point
# Monitor GPU memory, then try 64, 128, etc.
```
**Why it matters:** Batch size directly determines peak memory usage. Too large = crash. Too small = slow training.

**3. Improper Train/Validation Split**
```python
# ❌ WRONG - Validation data leaking into training
all_data = dataset
train_loader = DataLoader(all_data, shuffle=True)  # No split!

# ✅ CORRECT - Separate train and validation
train_size = int(0.8 * len(dataset))
train_data = dataset[:train_size]
val_data = dataset[train_size:]
train_loader = DataLoader(train_data, shuffle=True)
val_loader = DataLoader(val_data, shuffle=False)
```
**Why it matters:** Using the same data for training and validation gives falsely optimistic performance metrics.

**4. Not Handling Uneven Batches**
```python
# Dataset with 1000 samples, batch_size=128
# Creates: [128, 128, 128, 128, 128, 128, 128, 104] samples per batch
# Your model must handle variable batch sizes!

# Example: Don't assume batch_size in forward pass
def forward(self, x):
    batch_size = x.shape[0]  # ✅ Get actual batch size
    # Don't hardcode: batch_size = 128  # ❌ Breaks on last batch
```

### 🚀 Best Practices for Production

**1. Batch Size Selection Strategy**
```
Start with: 32 (almost always works)
↓
Monitor GPU memory usage
↓
If memory < 80%: double to 64
If memory > 90%: keep at 32
↓
Repeat until you find the sweet spot (usually 32-256)
```

**2. Data Augmentation Placement**
- **Option A:** In Dataset's `__getitem__` (random crop, flip, etc.)
- **Option B:** After DataLoader in training loop (batch-level operations)
- **Rule:** Image-level augmentation in Dataset, batch-level in loop

**3. Shuffling Strategy**
- **Training:** Always shuffle (`shuffle=True`)
- **Validation:** Never shuffle (`shuffle=False`)
- **Testing:** Never shuffle (`shuffle=False`)
- **Reason:** Validation/test need reproducible metrics

**4. Memory-Constrained Training**
```python
# Technique: Gradient Accumulation (effective larger batch)
effective_batch_size = 128
actual_batch_size = 32
accumulation_steps = effective_batch_size // actual_batch_size  # = 4

loader = DataLoader(dataset, batch_size=32)  # Fits in memory
# In training loop: accumulate 4 batches before optimizer step
# Result: Same as batch_size=128 but uses less memory!
```

These patterns will save you hours of debugging and help you build robust training pipelines!
"""

# %% [markdown]
"""
## Part 7: Integration Testing

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
    test_unit_dataloader_deterministic()

    print("\nRunning integration scenarios...")

    # Test complete workflow
    test_training_integration()

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 08")

# %%
# Run comprehensive module test
if __name__ == "__main__":
    test_module()




# %% [markdown]
"""
## 🎯 MODULE SUMMARY: DataLoader

Congratulations! You've built a complete data loading pipeline for ML training!

### Key Accomplishments
- Built Dataset abstraction and TensorDataset implementation with proper tensor alignment
- Created DataLoader with batching, shuffling, and memory-efficient iteration
- Analyzed data pipeline performance and discovered memory/speed trade-offs
- Learned how to apply DataLoader to real datasets (see examples/milestones)
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Batch size directly impacts memory usage and training throughput**
- **Shuffling adds minimal overhead but prevents overfitting patterns**
- **Data loading can become a bottleneck without proper optimization**
- **Memory usage scales linearly with batch size and feature dimensions**

### Ready for Next Steps
Your DataLoader implementation enables efficient training of CNNs and larger models with proper data pipeline management.
Export with: `tito export 08_dataloader`

**Apply your knowledge:**
- Milestone 03: Train MLP on real MNIST digits
- Milestone 04: Train CNN on CIFAR-10 images

**Then continue with:** Module 09 (Spatial) for Conv2d layers!

### Real-World Connection
You've implemented the same patterns used in:
- **PyTorch's DataLoader**: Same interface design for batching and shuffling
- **TensorFlow's Dataset API**: Similar abstraction for data pipeline optimization
- **Production ML**: Essential for handling large-scale training efficiently
- **Research**: Standard foundation for all deep learning experiments

Your data loading pipeline is now ready to power the CNN training in Module 09!
"""