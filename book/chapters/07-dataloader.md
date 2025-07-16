---
title: "DataLoader"
description: "Dataset interfaces and data loading pipelines"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: ['✅ Understand data engineering as the foundation of ML systems', '✅ Implement reusable dataset abstractions and interfaces', '✅ Build efficient data loaders with batching and shuffling', '✅ Create data preprocessing pipelines for normalization', '✅ Apply systems thinking to data I/O and memory management', '✅ Have a complete data pipeline ready for neural network training']
---

# DataLoader
---
**Course Navigation:** [Home](../intro.html) → [Module 7: 07 Dataloader](#)

---


<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ ⭐⭐⭐ | <strong>Time:</strong> 4-5 hours</p>
</div>



## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 5-7 hours
- **Prerequisites**: Tensor, Layers modules
- **Next Steps**: Training, Networks modules

Build the data pipeline foundation of TinyTorch! This module implements efficient data loading, preprocessing, and batching systems - the critical infrastructure that feeds neural networks during training.

## 🎯 Learning Objectives

By the end of this module, you will:
- ✅ Understand data engineering as the foundation of ML systems
- ✅ Implement reusable dataset abstractions and interfaces
- ✅ Build efficient data loaders with batching and shuffling
- ✅ Create data preprocessing pipelines for normalization
- ✅ Apply systems thinking to data I/O and memory management
- ✅ Have a complete data pipeline ready for neural network training

## 📋 Module Structure

```
modules/dataloader/
├── README.md                 # 📖 This file - Module overview
├── dataloader_dev.py         # 🔧 Main development file  
├── dataloader_dev.ipynb      # 📓 Generated notebook (auto-created)
├── tests/
│   └── test_dataloader.py    # 🧪 Automated tests
└── check_dataloader.py       # ✅ Manual verification (coming soon)
```

## 🚀 Getting Started

### Step 1: Complete Prerequisites
Make sure you've completed the foundational modules:
```bash
tito test --module setup    # Should pass
tito test --module tensor   # Should pass
tito test --module layers   # Should pass
```

### Step 2: Open the Data Development File
```bash
# Start from the dataloader module directory
cd modules/dataloader/

# Convert to notebook if needed
tito notebooks --module dataloader

# Open the development notebook
jupyter lab dataloader_dev.ipynb
```

### Step 3: Work Through the Implementation
The development file guides you through building:
1. **Dataset base class** - Abstract interface for all datasets
2. **CIFAR-10 implementation** - Real dataset with binary file parsing
3. **DataLoader** - Efficient batching and shuffling system
4. **Normalizer** - Data preprocessing for stable training
5. **Complete pipeline** - Integration of all components

### Step 4: Export and Test
```bash
# Export your dataloader implementation
tito sync --module dataloader

# Test your implementation
tito test --module dataloader
```

## 📚 What You'll Implement

### Core Data Infrastructure
You'll build a complete data loading system that supports:

#### 1. Dataset Abstraction
```python
# Abstract base class for all datasets
class Dataset:
    def __getitem__(self, index):
        # Get single sample and label
        pass
    
    def __len__(self):
        # Get total number of samples
        pass
    
    def get_num_classes(self):
        # Get number of classes
        pass

# Concrete implementation
dataset = CIFAR10Dataset("data/cifar10/", train=True)
image, label = dataset[0]  # Get first sample
```

#### 2. Real Dataset Loading
```python
# CIFAR-10 dataset with download and parsing
dataset = CIFAR10Dataset("data/cifar10/", train=True, download=True)
print(f"Dataset size: {len(dataset)}")           # 50,000 training samples
print(f"Sample shape: {dataset.get_sample_shape()}")  # (3, 32, 32)
print(f"Classes: {dataset.get_num_classes()}")        # 10 classes
```

#### 3. Efficient Data Loading
```python
# DataLoader with batching and shuffling
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_images, batch_labels in dataloader:
    print(f"Batch shape: {batch_images.shape}")  # (32, 3, 32, 32)
    print(f"Labels shape: {batch_labels.shape}")  # (32,)
    # Ready for neural network training!
```

#### 4. Data Preprocessing
```python
# Normalizer for stable training
normalizer = Normalizer()
normalizer.fit(training_data)  # Compute statistics
normalized_data = normalizer.transform(test_data)  # Apply normalization
```

#### 5. Complete Pipeline
```python
# One-function pipeline creation
train_loader, test_loader, normalizer = create_data_pipeline(
    dataset_path="data/cifar10/",
    batch_size=32,
    normalize=True,
    shuffle=True
)
```

### Technical Requirements
Your data system must:
- Handle multiple dataset types through common interface
- Efficiently load and parse binary data files
- Support batching with configurable batch sizes
- Implement shuffling for training randomization
- Provide data normalization for stable training
- Export to `tinytorch.core.dataloader`

## 🧪 Testing Your Implementation

### Progressive Testing with Real Data

The tests follow the **"Build → Use → Understand"** pattern with real CIFAR-10 data:

```bash
# Run all tests (downloads real CIFAR-10 data)
tito test --module dataloader

# Run specific test categories
python -m pytest tests/test_dataloader.py::TestDatasetInterface -v      # Test abstract interface
python -m pytest tests/test_dataloader.py::TestCIFAR10Dataset -v        # Test real data loading
python -m pytest tests/test_dataloader.py::TestDataLoader -v            # Test batching real data
python -m pytest tests/test_dataloader.py::TestNormalizer -v            # Test normalizing real data
python -m pytest tests/test_dataloader.py::TestDataPipeline -v          # Test complete pipeline
```

### Real Data Testing Flow

Each test builds on the previous component using actual CIFAR-10 data:

1. **Build Dataset** → **Test**: Download and load real CIFAR-10 images (50,000 training, 10,000 test)
2. **Build DataLoader** → **Test**: Batch real images with proper shuffling and iteration
3. **Build Normalizer** → **Test**: Normalize real pixel values (0-255 range → standardized)
4. **Build Pipeline** → **Test**: Complete pipeline with real data flow and preprocessing

### Why Real Data Testing Matters

- **Real-world validation**: Tests work with actual data students will use in training
- **Immediate feedback**: See your pipeline working with real images, not fake data
- **Systems thinking**: Understand I/O, memory, and performance with real data distributions
- **Debugging**: Catch issues that only appear with real data (file formats, edge cases)

**Note**: First test run downloads ~170MB CIFAR-10 dataset with progress bar. Subsequent runs use cached data.

### Interactive Testing with Visual Feedback
```python
# Test in the notebook or Python REPL
from tinytorch.core.dataloader import Dataset, DataLoader, CIFAR10Dataset

# Create and test datasets with real data
dataset = CIFAR10Dataset("data/cifar10/", train=True, download=True)
print(f"Loaded {len(dataset)} real CIFAR-10 samples")

# Test data loading
dataloader = DataLoader(dataset, batch_size=16)
for batch_data, batch_labels in dataloader:
    print(f"Real batch shape: {batch_data.shape}")  # (16, 3, 32, 32)
    print(f"Real labels: {batch_labels}")  # Actual CIFAR-10 classes
    break
```

### 🎨 Development Visual Feedback

The development notebook (`dataloader_dev.py`) includes **visual feedback** for learning:

```python
# 👁️ SEE your data - Available in development notebook only
show_cifar10_samples(dataset, num_samples=8, title="My CIFAR-10 Data")
```

### 🎨 Visual Feedback Features (Development Only)

The development notebook includes **visual feedback** for learning and debugging:

- **Download progress bar**: Visual progress indicator during CIFAR-10 download (~170MB)
- **`show_cifar10_samples()`**: Display a grid of CIFAR-10 images with class labels
- **Real image visualization**: See actual airplanes, cars, birds, cats, etc.
- **Batch visualization**: View what your DataLoader is producing
- **Pipeline visualization**: See the complete data flow in action

**Why Visual Feedback Matters:**
- **Build confidence**: See that your data pipeline is working correctly
- **Debug issues**: Spot problems like incorrect normalization or corrupted images
- **Understand data**: Build intuition about what your model will be learning from
- **Immediate feedback**: Visual confirmation follows the "Build → Use → Understand" pattern

**Note**: Visual feedback is available in the development notebook (`data_dev.py`) for learning purposes. The core package exports only the essential data loading components.

## 🎯 Success Criteria

Your data module is complete when:

1. **All tests pass**: `tito test --module dataloader`
2. **Data classes import correctly**: `from tinytorch.core.dataloader import Dataset, DataLoader`
3. **Dataset loading works**: Can create datasets and access samples
4. **Batching works**: DataLoader produces correct batch shapes
5. **Preprocessing works**: Normalizer computes and applies statistics
6. **Pipeline works**: Complete pipeline creates train/test loaders

## 💡 Implementation Tips

### Start with the Interface
1. **Dataset base class** - Define the abstract interface
2. **Simple test dataset** - Create mock data for testing
3. **Basic DataLoader** - Implement batching without shuffling
4. **Add shuffling** - Randomize sample order
5. **Test frequently** - Verify each component works

### Design Patterns
```python
class Dataset:
    def __getitem__(self, index):
        # Return (data, label) tuple
        return data_tensor, label_tensor
    
    def __len__(self):
        # Return total number of samples
        return self.num_samples

class DataLoader:
    def __iter__(self):
        # Yield batches of (batch_data, batch_labels)
        for batch in self._create_batches():
            yield batch_data, batch_labels
```

### Systems Thinking
- **Memory management**: Don't load entire dataset into RAM
- **I/O efficiency**: Batch file operations when possible
- **Preprocessing**: Compute statistics once, apply many times
- **Interface design**: Make components easily swappable

### Common Challenges
- **Binary file parsing** - CIFAR-10 uses custom format
- **Batch size handling** - Last batch may be smaller
- **Data type consistency** - Convert to consistent types
- **Error handling** - Provide helpful debugging messages

## 🔧 Advanced Features (Optional)

If you finish early, try implementing:
- **Data augmentation** - Random transformations for training
- **Multi-worker loading** - Parallel data loading
- **Caching** - Store processed data for faster access
- **Different datasets** - MNIST, Fashion-MNIST, etc.

## 🚀 Next Steps

Once you complete the data module:

1. **Move to Autograd**: `cd modules/autograd/`
2. **Build automatic differentiation**: Enable gradient computation
3. **Combine with data**: Train models on real datasets
4. **Prepare for training**: Ready for the training module

## 🔗 Why Data Engineering Matters

Data engineering is the foundation of all ML systems:
- **Training loops** need efficient data loading
- **Model performance** depends on data quality
- **Production systems** require scalable data pipelines
- **Research** needs flexible data interfaces

Your data implementation will power all TinyTorch training!

## 📊 Real-World Connection

The patterns you'll implement are used in:
- **PyTorch DataLoader** - Same interface and concepts
- **TensorFlow tf.data** - Similar pipeline architecture
- **Production ML** - Scalable data processing systems
- **Research** - Flexible experimentation frameworks

## 🎉 Ready to Build?

The data module is where TinyTorch becomes a real ML system. You're about to create the infrastructure that will feed neural networks, enable training loops, and power production ML pipelines.

Focus on clean interfaces, efficient implementation, and systems thinking! 🔥 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/07_dataloader/dataloader_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/07_dataloader/dataloader_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/07_dataloader/dataloader_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/06_cnn.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/08_autograd.html" title="next page">Next Module →</a>
</div>
