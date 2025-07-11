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
# Module 4: Data - Data Loading and Preprocessing

Welcome to the Data module! This is where you'll learn how to efficiently load, process, and manage data for machine learning systems.

## Learning Goals
- Understand data pipelines as the foundation of ML systems
- Implement efficient data loading with memory management
- Build reusable dataset abstractions for different data types
- Master batching strategies and I/O optimization
- Learn systems thinking for data engineering

## Build → Use → Understand
1. **Build**: Create dataset classes and data loaders
2. **Use**: Load real datasets and train models
3. **Understand**: How data engineering affects system performance

## Module Dependencies
This module builds on previous modules:
- **tensor** → **activations** → **layers** → **networks** → **data**
- Data feeds into training: data → autograd → training
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/data/data_dev.py`  
**Building Side:** Code exports to `tinytorch.core.data`

```python
# Final package structure:
from tinytorch.core.data import Dataset, DataLoader, CIFAR10Dataset
from tinytorch.core.tensor import Tensor
from tinytorch.core.networks import Sequential
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.utils.data`
- **Consistency:** All data loading utilities live together in `core.data`
"""

# %%
#| default_exp core.data

# Setup and imports
import numpy as np
import sys
import os
import pickle
import struct
from typing import List, Tuple, Optional, Union, Iterator
import matplotlib.pyplot as plt
import urllib.request
import tarfile

# Import our building blocks
from tinytorch.core.tensor import Tensor

print("🔥 TinyTorch Data Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build data pipelines!")

# %%
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

# Import our building blocks
from tinytorch.core.tensor import Tensor

# %%
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    return 'pytest' not in sys.modules and 'test' not in sys.argv

# %% [markdown]
"""
## Step 1: What is Data Engineering?

### Definition
**Data engineering** is the foundation of all machine learning systems. It involves loading, processing, and managing data efficiently so that models can learn from it.

### Why Data Engineering Matters
- **Data is the fuel**: Without proper data pipelines, nothing else works
- **I/O bottlenecks**: Data loading is often the biggest performance bottleneck
- **Memory management**: How you handle data affects everything else
- **Production reality**: Data pipelines are critical in real ML systems

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

# %%
#| export
class Dataset:
    """
    Base Dataset class: Abstract interface for all datasets.
    
    The fundamental abstraction for data loading in TinyTorch.
    Students implement concrete datasets by inheriting from this class.
    
    TODO: Implement the base Dataset class with required methods.
    
    APPROACH:
    1. Define the interface that all datasets must implement
    2. Include methods for getting individual samples and dataset size
    3. Make it easy to extend for different data types
    
    EXAMPLE:
    dataset = CIFAR10Dataset("data/cifar10/")
    sample, label = dataset[0]  # Get first sample
    size = len(dataset)  # Get dataset size
    
    HINTS:
    - Use abstract methods that subclasses must implement
    - Include __getitem__ for indexing and __len__ for size
    - Add helper methods for getting sample shapes and number of classes
    """
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample and label by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (data, label) tensors
            
        TODO: Implement abstract method for getting samples.
        
        STEP-BY-STEP:
        1. This is an abstract method - subclasses will implement it
        2. Return a tuple of (data, label) tensors
        3. Data should be the input features, label should be the target
        
        EXAMPLE:
        dataset[0] should return (Tensor(image_data), Tensor(label))
        """
        raise NotImplementedError("Student implementation required")
    
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        TODO: Implement abstract method for getting dataset size.
        
        STEP-BY-STEP:
        1. This is an abstract method - subclasses will implement it
        2. Return the total number of samples in the dataset
        
        EXAMPLE:
        len(dataset) should return 50000 for CIFAR-10 training set
        """
        raise NotImplementedError("Student implementation required")
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of a single data sample.
        
        TODO: Implement method to get sample shape.
        
        STEP-BY-STEP:
        1. Get the first sample using self[0]
        2. Extract the data part (first element of tuple)
        3. Return the shape of the data tensor
        
        EXAMPLE:
        For CIFAR-10: returns (3, 32, 32) for RGB images
        """
        raise NotImplementedError("Student implementation required")
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.
        
        TODO: Implement abstract method for getting number of classes.
        
        STEP-BY-STEP:
        1. This is an abstract method - subclasses will implement it
        2. Return the total number of classes in the dataset
        
        EXAMPLE:
        For CIFAR-10: returns 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
        """
        raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
class Dataset:
    """Base Dataset class: Abstract interface for all datasets."""
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get a single sample and label by index."""
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__")
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single data sample."""
        sample, _ = self[0]
        return sample.shape
    
    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        raise NotImplementedError("Subclasses must implement get_num_classes")

# %% [markdown]
"""
### 🧪 Test Your Base Dataset
"""

# %%
# Test the base Dataset class
print("Testing base Dataset class...")

try:
    # Create a simple test dataset
    class TestDataset(Dataset):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples
            self.data = [Tensor(np.random.randn(3, 32, 32)) for _ in range(num_samples)]
            self.labels = [Tensor(np.array(i % 3)) for i in range(num_samples)]
        
        def __getitem__(self, index):
            return self.data[index], self.labels[index]
        
        def __len__(self):
            return self.num_samples
        
        def get_num_classes(self):
            return 3
    
    # Test the dataset
    dataset = TestDataset(5)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test indexing
    sample, label = dataset[0]
    print(f"✅ Sample shape: {sample.shape}")
    print(f"✅ Label: {label}")
    
    # Test helper methods
    print(f"✅ Sample shape: {dataset.get_sample_shape()}")
    print(f"✅ Number of classes: {dataset.get_num_classes()}")
    
    print("🎉 Base Dataset class works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the base Dataset class above!")

# %% [markdown]
"""
## Step 2: Understanding CIFAR-10 Dataset

Now let's build a real dataset! We'll focus on **CIFAR-10** - the perfect dataset for learning data loading.

### Why CIFAR-10?
- **Perfect size**: 170MB - large enough for optimization, small enough to manage
- **Real data**: 32x32 color images, 10 classes
- **Classic dataset**: Every ML student should know it
- **Good complexity**: Requires proper data loading techniques

### The CIFAR-10 Format
```
File structure:
- data_batch_1: 10,000 images + labels
- data_batch_2: 10,000 images + labels
- ...
- test_batch: 10,000 test images
- batches.meta: Class names and metadata

Binary format:
- Each image: 3073 bytes (3072 for RGB + 1 for label)
- Images stored as: [label, R, G, B, R, G, B, ...]
- 32x32x3 = 3072 bytes per image
```

### Data Loading Challenges
- **Binary file parsing**: CIFAR-10 uses a custom binary format
- **Memory management**: 60,000 images need efficient handling
- **Batching**: Grouping samples for efficient processing
- **Preprocessing**: Normalization, augmentation, etc.

Let's implement CIFAR-10 loading step by step!
"""

# %%
#| export
class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 Dataset: Load and manage CIFAR-10 image data.
    
    CIFAR-10 contains 60,000 32x32 color images in 10 classes.
    Perfect for learning data loading and image processing.
    
    Args:
        root_dir: Directory containing CIFAR-10 files
        train: If True, load training data. If False, load test data.
        download: If True, download dataset if not present
        
    TODO: Implement CIFAR-10 dataset loading.
    
    APPROACH:
    1. Handle dataset download if needed (with progress bar!)
    2. Parse binary files to extract images and labels
    3. Store data efficiently in memory
    4. Implement indexing and size methods
    
    EXAMPLE:
    dataset = CIFAR10Dataset("data/cifar10/", train=True)
    image, label = dataset[0]  # Get first image
    print(f"Image shape: {image.shape}")  # (3, 32, 32)
    print(f"Label: {label}")  # Tensor with class index
    
    HINTS:
    - Use pickle to load binary files
    - Each batch file contains 'data' and 'labels' keys
    - Reshape data to (3, 32, 32) format
    - Store images and labels as separate lists
    - Add progress bar with urllib.request.urlretrieve(url, filename, reporthook=progress_function)
    - Progress function receives (block_num, block_size, total_size) parameters
    """
    
    def __init__(self, root_dir: str, train: bool = True, download: bool = True):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            root_dir: Directory to store/load dataset
            train: If True, load training data. If False, load test data.
            download: If True, download dataset if not present
            
        TODO: Implement CIFAR-10 initialization.
        
        STEP-BY-STEP:
        1. Create root directory if it doesn't exist
        2. Download dataset if needed and not present (with progress bar!)
        3. Load binary files and parse data
        4. Store images and labels in memory
        5. Set up class names
        
        EXAMPLE:
        CIFAR10Dataset("data/cifar10/", train=True)
        creates a dataset with 50,000 training images
        
        PROGRESS BAR HINT:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded * 100) // total_size
            print(f"\\rDownloading: {percent}%", end='', flush=True)
        """
        raise NotImplementedError("Student implementation required")
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single image and label by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (image, label) tensors
            
        TODO: Implement sample retrieval.
        
        STEP-BY-STEP:
        1. Get image from self.images[index]
        2. Get label from self.labels[index]
        3. Return (Tensor(image), Tensor(label))
        
        EXAMPLE:
        image, label = dataset[0]
        image.shape should be (3, 32, 32)
        label should be integer 0-9
        """
        raise NotImplementedError("Student implementation required")
    
    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        TODO: Return the length of the dataset.
        
        STEP-BY-STEP:
        1. Return len(self.images)
        
        EXAMPLE:
        Training set: 50,000 samples
        Test set: 10,000 samples
        """
        raise NotImplementedError("Student implementation required")
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes in CIFAR-10.
        
        TODO: Return the number of classes.
        
        STEP-BY-STEP:
        1. CIFAR-10 has 10 classes
        2. Return 10
        
        EXAMPLE:
        Returns 10 for CIFAR-10
        """
        raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset: Load and manage CIFAR-10 image data."""
    
    def __init__(self, root_dir: str, train: bool = True, download: bool = True):
        self.root_dir = root_dir
        self.train = train
        self.class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create directory if it doesn't exist
        os.makedirs(root_dir, exist_ok=True)
        
        # Download if needed
        if download:
            self._download_if_needed()
        
        # Load data
        self._load_data()
    
    def _download_if_needed(self):
        """Download CIFAR-10 if not present."""
        cifar_path = os.path.join(self.root_dir, "cifar-10-batches-py")
        if not os.path.exists(cifar_path):
            print("🔄 Downloading CIFAR-10 dataset...")
            print("📦 Size: ~170MB (this may take a few minutes)")
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            filename = os.path.join(self.root_dir, "cifar-10-python.tar.gz")
            
            try:
                # Download with progress bar
                def show_progress(block_num, block_size, total_size):
                    """Show download progress bar."""
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, (downloaded * 100) // total_size)
                        bar_length = 50
                        filled_length = (percent * bar_length) // 100
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        
                        # Convert bytes to MB
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        
                        print(f"\r📥 [{bar}] {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
                    else:
                        # Fallback if total size unknown
                        downloaded_mb = downloaded / (1024 * 1024)
                        print(f"\r📥 Downloaded: {downloaded_mb:.1f} MB", end='', flush=True)
                
                urllib.request.urlretrieve(url, filename, reporthook=show_progress)
                print()  # New line after progress bar
                
                # Extract
                print("📂 Extracting CIFAR-10 files...")
                with tarfile.open(filename, 'r:gz') as tar:
                    tar.extractall(self.root_dir, filter='data')
                
                # Clean up
                os.remove(filename)
                print("✅ CIFAR-10 downloaded and extracted successfully!")
                
            except Exception as e:
                print(f"\n❌ Download failed: {e}")
                print("Please download CIFAR-10 manually from https://www.cs.toronto.edu/~kriz/cifar.html")
    
    def _load_data(self):
        """Load CIFAR-10 data from binary files."""
        cifar_path = os.path.join(self.root_dir, "cifar-10-batches-py")
        
        self.images = []
        self.labels = []
        
        if self.train:
            # Load training batches
            for i in range(1, 6):
                batch_file = os.path.join(cifar_path, f"data_batch_{i}")
                if os.path.exists(batch_file):
                    with open(batch_file, 'rb') as f:
                        batch = pickle.load(f, encoding='bytes')
                        # Convert bytes keys to strings
                        batch = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in batch.items()}
                        
                        # Extract images and labels
                        images = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
                        labels = batch['labels']
                        
                        self.images.extend(images)
                        self.labels.extend(labels)
        else:
            # Load test batch
            test_file = os.path.join(cifar_path, "test_batch")
            if os.path.exists(test_file):
                with open(test_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    # Convert bytes keys to strings
                    batch = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in batch.items()}
                    
                    # Extract images and labels
                    self.images = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
                    self.labels = batch['labels']
        
        print(f"✅ Loaded {len(self.images)} {'training' if self.train else 'test'} samples")
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get a single image and label by index."""
        image = Tensor(self.images[index])
        label = Tensor(np.array(self.labels[index]))
        return image, label
    
    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.images)
    
    def get_num_classes(self) -> int:
        """Get the number of classes in CIFAR-10."""
        return 10

# %% [markdown]
"""
### 🧪 Test Your CIFAR-10 Dataset
"""

# %%
# Test CIFAR-10 dataset (skip download for now)
print("Testing CIFAR-10 dataset...")

try:
    # Create a mock dataset for testing without download
    class MockCIFAR10Dataset(Dataset):
        def __init__(self, size, train=True):
            self.size = size
            self.train = train
            self.data = [np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(size)]
            self.labels = [np.random.randint(0, 10) for _ in range(size)]
        
        def __getitem__(self, index):
            return Tensor(self.data[index].astype(np.float32)), Tensor(np.array(self.labels[index]))
        
        def __len__(self):
            return self.size
        
        def get_num_classes(self):
            return 10
    
    # Test the dataset
    dataset = MockCIFAR10Dataset(50)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test indexing
    image, label = dataset[0]
    print(f"✅ Image shape: {image.shape}")
    print(f"✅ Label: {label}")
    print(f"✅ Number of classes: {dataset.get_num_classes()}")
    
    # Test multiple samples
    for i in range(3):
        img, lbl = dataset[i]
        print(f"✅ Sample {i}: {img.shape}, class {int(lbl.data)}")
    
    print("🎉 CIFAR-10 dataset structure works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the CIFAR-10 dataset above!")

# %% [markdown]
"""
### 👁️ Visual Feedback: See Your Data!

Let's add a visualization function to actually **see** the CIFAR-10 images we're loading. 
This provides immediate visual feedback and builds intuition about the data.
"""

# %%
def show_cifar10_samples(dataset, num_samples=8, title="CIFAR-10 Samples"):
    """
    Display a grid of CIFAR-10 images with their labels.
    
    Args:
        dataset: CIFAR-10 dataset
        num_samples: Number of samples to display
        title: Title for the plot
        
    TODO: Implement visualization function.
    
    APPROACH:
    1. Create a matplotlib subplot grid
    2. Get random samples from dataset
    3. Display each image with its class label
    4. Handle the image format (CHW -> HWC, normalize to 0-1)
    
    EXAMPLE:
    show_cifar10_samples(dataset, num_samples=8)
    # Shows 8 CIFAR-10 images in a 2x4 grid
    
    HINTS:
    - Use plt.subplots() to create grid
    - Convert image from (C, H, W) to (H, W, C) for display
    - Normalize pixel values to [0, 1] range
    - Use dataset.class_names for labels
    
    NOTE: This is a development/learning tool, not part of the core package.
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
def show_cifar10_samples(dataset, num_samples=8, title="CIFAR-10 Samples"):
    """Display a grid of CIFAR-10 images with their labels."""
    if not _should_show_plots():
        return
    
    # Create subplot grid
    rows = 2
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        
        # Get image and label
        image, label = dataset[idx]
        
        # Convert from (C, H, W) to (H, W, C) and normalize to [0, 1]
        if hasattr(image, 'data'):
            img_data = image.data
        else:
            img_data = image
        
        # Handle different tensor formats
        if img_data.shape[0] == 3:  # (C, H, W)
            img_display = np.transpose(img_data, (1, 2, 0))
        else:
            img_display = img_data
        
        # Normalize to [0, 1] range
        img_display = img_display.astype(np.float32)
        if img_display.max() > 1.0:
            img_display = img_display / 255.0
        
        # Ensure values are in [0, 1]
        img_display = np.clip(img_display, 0, 1)
        
        # Display image
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        ax.imshow(img_display)
        ax.axis('off')
        
        # Add label
        if hasattr(label, 'data'):
            label_idx = int(label.data)
        else:
            label_idx = int(label)
        
        if hasattr(dataset, 'class_names'):
            class_name = dataset.class_names[label_idx]
            ax.set_title(f'{class_name} ({label_idx})', fontsize=10)
        else:
            ax.set_title(f'Class {label_idx}', fontsize=10)
    
    plt.tight_layout()
    if _should_show_plots():
        plt.show()

# %%
# Test visual feedback with real CIFAR-10 data
print("🎨 Testing visual feedback with real CIFAR-10...")

try:
    # Create real CIFAR-10 dataset for visualization
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load real CIFAR-10 dataset
        cifar_dataset = CIFAR10Dataset(temp_dir, train=True, download=True)
        
        print(f"✅ Loaded {len(cifar_dataset)} real CIFAR-10 samples")
        print(f"✅ Class names: {cifar_dataset.class_names}")
        
        # Show sample images
        if _should_show_plots():
            print("🖼️ Displaying sample images...")
            show_cifar10_samples(cifar_dataset, num_samples=8, title="Real CIFAR-10 Training Samples")
        
        # Show some statistics
        sample_images = [cifar_dataset[i][0] for i in range(100)]
        pixel_values = [img.data for img in sample_images]
        all_pixels = np.concatenate([pixels.flatten() for pixels in pixel_values])
        
        print(f"✅ Pixel value range: [{all_pixels.min():.1f}, {all_pixels.max():.1f}]")
        print(f"✅ Mean pixel value: {all_pixels.mean():.1f}")
        print(f"✅ Std pixel value: {all_pixels.std():.1f}")
        
        print("🎉 Visual feedback works! You can see your data!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure CIFAR-10 dataset is implemented correctly!")

# %% [markdown]
"""
## Step 3: Understanding Data Loading

Now let's build a **DataLoader** to efficiently batch and iterate through our dataset.

### Why DataLoaders Matter
- **Batching**: Process multiple samples at once (GPU efficiency)
- **Shuffling**: Randomize order for better training
- **Memory management**: Handle large datasets efficiently
- **I/O optimization**: Load data in parallel with training

### The DataLoader Pattern
```
Dataset: [sample1, sample2, sample3, ...]
DataLoader: [[batch1], [batch2], [batch3], ...]
```

### Systems Thinking
- **Batch size**: Trade-off between memory and speed
- **Shuffling**: Prevents overfitting to data order
- **Iteration**: Efficient looping through data
- **Memory**: Manage large datasets that don't fit in RAM

Let's implement a DataLoader!
"""

# %%
#| export
class DataLoader:
    """
    DataLoader: Efficiently batch and iterate through datasets.
    
    Provides batching, shuffling, and efficient iteration over datasets.
    Essential for training neural networks efficiently.
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data each epoch
        
    TODO: Implement DataLoader with batching and shuffling.
    
    APPROACH:
    1. Store dataset and configuration
    2. Implement __iter__ to yield batches
    3. Handle shuffling and batching logic
    4. Stack individual samples into batches
    
    EXAMPLE:
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_images, batch_labels in dataloader:
        print(f"Batch shape: {batch_images.shape}")  # (32, 3, 32, 32)
    
    HINTS:
    - Use np.random.permutation for shuffling
    - Stack samples using np.stack
    - Yield batches as (batch_data, batch_labels)
    - Handle last batch that might be smaller
    """
    
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            
        TODO: Store configuration and dataset.
        
        STEP-BY-STEP:
        1. Store dataset as self.dataset
        2. Store batch_size as self.batch_size
        3. Store shuffle as self.shuffle
        
        EXAMPLE:
        DataLoader(dataset, batch_size=32, shuffle=True)
        """
        raise NotImplementedError("Student implementation required")
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Iterate through dataset in batches.
        
        Returns:
            Iterator yielding (batch_data, batch_labels) tuples
            
        TODO: Implement batching and shuffling logic.
        
        STEP-BY-STEP:
        1. Create indices list: list(range(len(dataset)))
        2. Shuffle indices if self.shuffle is True
        3. Loop through indices in batch_size chunks
        4. For each batch: collect samples, stack them, yield batch
        
        EXAMPLE:
        for batch_data, batch_labels in dataloader:
            # batch_data.shape: (batch_size, ...)
            # batch_labels.shape: (batch_size,)
        """
        raise NotImplementedError("Student implementation required")
    
    def __len__(self) -> int:
        """
        Get the number of batches per epoch.
        
        TODO: Calculate number of batches.
        
        STEP-BY-STEP:
        1. Get dataset size: len(self.dataset)
        2. Calculate: (dataset_size + batch_size - 1) // batch_size
        3. This handles the last partial batch correctly
        
        EXAMPLE:
        Dataset size: 100, batch_size: 32
        Number of batches: 4 (32, 32, 32, 4)
        """
        raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
class DataLoader:
    """DataLoader: Efficiently batch and iterate through datasets."""
    
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Iterate through dataset in batches."""
        # Create indices
        indices = list(range(len(self.dataset)))
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Collect samples for this batch
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                data, label = self.dataset[idx]
                batch_data.append(data.data)
                batch_labels.append(label.data)
            
            # Stack into batches
            batch_data = np.stack(batch_data, axis=0)
            batch_labels = np.stack(batch_labels, axis=0)
            
            yield Tensor(batch_data), Tensor(batch_labels)
    
    def __len__(self) -> int:
        """Get the number of batches per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# %% [markdown]
"""
### 🧪 Test Your DataLoader
"""

# %%
# Test DataLoader
print("Testing DataLoader...")

try:
    # Create a test dataset
    class SimpleDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.data = [np.random.randn(3, 32, 32) for _ in range(size)]
            self.labels = [i % 10 for i in range(size)]
        
        def __getitem__(self, index):
            return Tensor(self.data[index]), Tensor(np.array(self.labels[index]))
        
        def __len__(self):
            return self.size
        
        def get_num_classes(self):
            return 10
    
    # Test DataLoader
    dataset = SimpleDataset(100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"✅ Dataset size: {len(dataset)}")
    print(f"✅ Number of batches: {len(dataloader)}")
    
    # Test iteration
    batch_count = 0
    for batch_data, batch_labels in dataloader:
        batch_count += 1
        print(f"✅ Batch {batch_count}: data shape {batch_data.shape}, labels shape {batch_labels.shape}")
        if batch_count >= 3:  # Only show first 3 batches
            break
    
    print("🎉 DataLoader works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the DataLoader above!")

# %%
# Test DataLoader with visual feedback using real CIFAR-10
print("🎨 Testing DataLoader with visual feedback...")

try:
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load real CIFAR-10 dataset
        cifar_dataset = CIFAR10Dataset(temp_dir, train=True, download=True)
        
        # Create DataLoader
        dataloader = DataLoader(cifar_dataset, batch_size=16, shuffle=True)
        
        print(f"✅ Created DataLoader with {len(dataloader)} batches")
        
        # Get first batch
        batch_data, batch_labels = next(iter(dataloader))
        print(f"✅ First batch shape: {batch_data.shape}")
        print(f"✅ First batch labels: {batch_labels.shape}")
        
        # Show first few images from the batch
        print("🖼️ Displaying first batch images...")
        
        # Create a temporary dataset-like object for visualization
        class BatchDataset:
            def __init__(self, batch_data, batch_labels, class_names):
                self.batch_data = batch_data
                self.batch_labels = batch_labels
                self.class_names = class_names
            
            def __getitem__(self, index):
                return Tensor(self.batch_data.data[index]), Tensor(self.batch_labels.data[index])
            
            def __len__(self):
                return self.batch_data.shape[0]
        
        batch_dataset = BatchDataset(batch_data, batch_labels, cifar_dataset.class_names)
        show_cifar10_samples(batch_dataset, num_samples=8, title="DataLoader Batch - Real CIFAR-10")
        
        print("🎉 DataLoader visual feedback works!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure DataLoader and visualization are implemented correctly!")

# %% [markdown]
"""
## Step 4: Understanding Data Preprocessing

Finally, let's build a **Normalizer** to preprocess our data for better training.

### Why Normalization Matters
- **Gradient stability**: Prevents exploding/vanishing gradients
- **Training speed**: Faster convergence
- **Numerical stability**: Prevents overflow/underflow
- **Consistent scales**: All features have similar ranges

### Common Normalization Techniques
- **Min-Max**: Scale to [0, 1] range
- **Z-score**: Zero mean, unit variance
- **ImageNet**: Specific mean/std for pretrained models

### The Normalization Process
```
Raw Data: [0, 255] pixel values
Normalized: [-1, 1] or [0, 1] range
```

Let's implement a flexible normalizer!
"""

# %%
#| export
class Normalizer:
    """
    Data Normalizer: Standardize data for better training.
    
    Computes mean and standard deviation from training data,
    then applies normalization to new data.
    
    TODO: Implement data normalization.
    
    APPROACH:
    1. Fit: Compute mean and std from training data
    2. Transform: Apply normalization using computed stats
    3. Handle both single tensors and batches
    
    EXAMPLE:
    normalizer = Normalizer()
    normalizer.fit(training_data)  # Compute stats
    normalized = normalizer.transform(new_data)  # Apply normalization
    
    HINTS:
    - Store mean and std as instance variables
    - Use np.mean and np.std for statistics
    - Apply: (data - mean) / std
    - Handle division by zero (add small epsilon)
    """
    
    def __init__(self):
        """
        Initialize normalizer.
        
        TODO: Initialize mean and std to None.
        
        STEP-BY-STEP:
        1. Set self.mean = None
        2. Set self.std = None
        3. Set self.epsilon = 1e-8 (for numerical stability)
        
        EXAMPLE:
        normalizer = Normalizer()
        """
        raise NotImplementedError("Student implementation required")
    
    def fit(self, data: List[Tensor]):
        """
        Compute normalization statistics from training data.
        
        Args:
            data: List of tensors to compute statistics from
            
        TODO: Compute mean and standard deviation.
        
        STEP-BY-STEP:
        1. Stack all tensors: np.stack([t.data for t in data])
        2. Compute mean: np.mean(stacked_data)
        3. Compute std: np.std(stacked_data)
        4. Store as self.mean and self.std
        
        EXAMPLE:
        normalizer.fit([tensor1, tensor2, tensor3])
        """
        raise NotImplementedError("Student implementation required")
    
    def transform(self, data: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        """
        Apply normalization to data.
        
        Args:
            data: Tensor or list of tensors to normalize
            
        Returns:
            Normalized tensor(s)
            
        TODO: Apply normalization using computed statistics.
        
        STEP-BY-STEP:
        1. Check if mean and std are computed (not None)
        2. If single tensor: apply (data - mean) / (std + epsilon)
        3. If list: apply to each tensor in the list
        4. Return normalized data
        
        EXAMPLE:
        normalized = normalizer.transform(tensor)
        """
        raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
class Normalizer:
    """Data Normalizer: Standardize data for better training."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.epsilon = 1e-8
    
    def fit(self, data: List[Tensor]):
        """Compute normalization statistics from training data."""
        # Stack all data
        all_data = np.stack([t.data for t in data])
        
        # Compute statistics
        self.mean = np.mean(all_data)
        self.std = np.std(all_data)
        
        print(f"✅ Computed normalization stats: mean={self.mean:.4f}, std={self.std:.4f}")
    
    def transform(self, data: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        """Apply normalization to data."""
        if self.mean is None or self.std is None:
            raise ValueError("Must call fit() before transform()")
        
        if isinstance(data, list):
            # Transform list of tensors
            return [Tensor((t.data - self.mean) / (self.std + self.epsilon)) for t in data]
        else:
            # Transform single tensor
            return Tensor((data.data - self.mean) / (self.std + self.epsilon))

# %% [markdown]
"""
### 🧪 Test Your Normalizer
"""

# %%
# Test Normalizer
print("Testing Normalizer...")

try:
    # Create test data
    data = [
        Tensor(np.random.randn(3, 32, 32) * 50 + 100),  # Mean ~100, std ~50
        Tensor(np.random.randn(3, 32, 32) * 50 + 100),
        Tensor(np.random.randn(3, 32, 32) * 50 + 100)
    ]
    
    # Test normalizer
    normalizer = Normalizer()
    
    # Fit to data
    normalizer.fit(data)
    
    # Transform data
    normalized = normalizer.transform(data)
    
    # Check results
    print(f"✅ Original data mean: {np.mean([t.data for t in data]):.4f}")
    print(f"✅ Original data std: {np.std([t.data for t in data]):.4f}")
    print(f"✅ Normalized data mean: {np.mean([t.data for t in normalized]):.4f}")
    print(f"✅ Normalized data std: {np.std([t.data for t in normalized]):.4f}")
    
    # Test single tensor
    single_tensor = Tensor(np.random.randn(3, 32, 32) * 50 + 100)
    normalized_single = normalizer.transform(single_tensor)
    print(f"✅ Single tensor normalized: {normalized_single.shape}")
    
    print("🎉 Normalizer works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Normalizer above!")

# %% [markdown]
"""
## Step 5: Building a Complete Data Pipeline

Now let's put everything together into a complete data pipeline!

### The Complete Pipeline
```
Raw Data → Dataset → DataLoader → Normalizer → Model
```

This is the foundation of every machine learning system!
"""

# %%
#| export
def create_data_pipeline(dataset_path: str = "data/cifar10/", 
                        batch_size: int = 32, 
                        normalize: bool = True,
                        shuffle: bool = True):
    """
    Create a complete data pipeline for training.
    
    Args:
        dataset_path: Path to dataset
        batch_size: Batch size for training
        normalize: Whether to normalize data
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (train_loader, test_loader)
        
    TODO: Implement complete data pipeline.
    
    APPROACH:
    1. Create train and test datasets
    2. Create data loaders
    3. Fit normalizer on training data
    4. Return all components
    
    EXAMPLE:
    train_loader, test_loader = create_data_pipeline()
    for batch_data, batch_labels in train_loader:
        # Ready for training!
    """
    raise NotImplementedError("Student implementation required")

# %%
#| hide
#| export
def create_data_pipeline(dataset_path: str = "data/cifar10/", 
                        batch_size: int = 32, 
                        normalize: bool = True,
                        shuffle: bool = True):
    """Create a complete data pipeline for training."""
    
    print("🔧 Creating data pipeline...")
    
    # Create datasets with real CIFAR-10 data
    train_dataset = CIFAR10Dataset(dataset_path, train=True, download=True)
    test_dataset = CIFAR10Dataset(dataset_path, train=False, download=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create normalizer
    normalizer = None
    if normalize:
        normalizer = Normalizer()
        # Fit on a subset of training data for efficiency
        sample_data = [train_dataset[i][0] for i in range(min(1000, len(train_dataset)))]
        normalizer.fit(sample_data)
        print(f"✅ Computed normalization stats: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    
    print(f"✅ Pipeline created:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Normalization: {normalize}")
    
    return train_loader, test_loader

# %% [markdown]
"""
### 🧪 Test Your Complete Data Pipeline
"""

# %%
# Test complete data pipeline
print("Testing complete data pipeline...")

try:
    # Create pipeline
    train_loader, test_loader = create_data_pipeline(
        batch_size=16, normalize=True, shuffle=True
    )
    
    # Test training loop
    print("\n🔥 Testing training loop:")
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"   Batch {i+1}: data {batch_data.shape}, labels {batch_labels.shape}")
        
        # Note: Data is already normalized in the pipeline if normalize=True
        
        if i >= 2:  # Only show first 3 batches
            break
    
    # Test test loop
    print("\n🧪 Testing test loop:")
    for i, (batch_data, batch_labels) in enumerate(test_loader):
        print(f"   Test batch {i+1}: data {batch_data.shape}, labels {batch_labels.shape}")
        if i >= 1:  # Only show first 2 batches
            break
    
    print("\n🎉 Complete data pipeline works!")
    print("Ready for training neural networks!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the data pipeline above!")

# %%
# Test complete pipeline with visual feedback
print("🎨 Testing complete pipeline with visual feedback...")

try:
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create complete pipeline
        train_loader, test_loader = create_data_pipeline(
            dataset_path=temp_dir,
            batch_size=16, 
            normalize=True, 
            shuffle=True
        )
        
        # Get a batch from training data
        train_batch_data, train_batch_labels = next(iter(train_loader))
        print(f"✅ Training batch shape: {train_batch_data.shape}")
        
        # Get a batch from test data
        test_batch_data, test_batch_labels = next(iter(test_loader))
        print(f"✅ Test batch shape: {test_batch_data.shape}")
        
        # Show training batch images
        print("🖼️ Displaying training batch...")
        class PipelineBatchDataset:
            def __init__(self, batch_data, batch_labels):
                self.batch_data = batch_data
                self.batch_labels = batch_labels
                self.class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
                                   'dog', 'frog', 'horse', 'ship', 'truck']
            
            def __getitem__(self, index):
                return Tensor(self.batch_data.data[index]), Tensor(self.batch_labels.data[index])
            
            def __len__(self):
                return self.batch_data.shape[0]
        
        train_batch_dataset = PipelineBatchDataset(train_batch_data, train_batch_labels)
        show_cifar10_samples(train_batch_dataset, num_samples=8, title="Complete Pipeline - Training Batch")
        
        # Show test batch images
        print("🖼️ Displaying test batch...")
        test_batch_dataset = PipelineBatchDataset(test_batch_data, test_batch_labels)
        show_cifar10_samples(test_batch_dataset, num_samples=8, title="Complete Pipeline - Test Batch")
        
        # Show data statistics
        print(f"✅ Training data range: [{train_batch_data.data.min():.3f}, {train_batch_data.data.max():.3f}]")
        print(f"✅ Training data mean: {train_batch_data.data.mean():.3f}")
        print(f"✅ Training data std: {train_batch_data.data.std():.3f}")
        
        print("🎉 Complete pipeline visual feedback works!")
        print("🚀 You can see your entire data pipeline in action!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure complete pipeline and visualization work correctly!")

# %% [markdown]
"""
## 🎯 Summary

Congratulations! You've built a complete data loading system:

### What You Built
1. **Dataset**: Abstract interface for data loading
2. **CIFAR10Dataset**: Real dataset implementation
3. **DataLoader**: Efficient batching and iteration
4. **Normalizer**: Data preprocessing for better training
5. **Data Pipeline**: Complete system integration

### Key Concepts Learned
- **Data engineering**: The foundation of ML systems
- **Batching**: Efficient processing of multiple samples
- **Normalization**: Preprocessing for stable training
- **Systems thinking**: Memory, I/O, and performance considerations

### Next Steps
- **Autograd**: Automatic differentiation for training
- **Training**: Optimization loops and loss functions
- **Advanced data**: Augmentation, distributed loading, etc.

### Real-World Impact
This data loading system is the foundation of every ML pipeline:
- **Production systems**: Handle millions of samples efficiently
- **Research**: Enable experimentation with different datasets
- **MLOps**: Integrate with training and deployment pipelines

You now understand how data flows through ML systems! 🚀
""" 