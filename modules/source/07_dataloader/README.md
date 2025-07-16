# 🔥 Module: DataLoader

## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 5-7 hours
- **Prerequisites**: Tensor, Layers modules
- **Next Steps**: Training, Networks modules

Build the data pipeline foundation of TinyTorch! This module implements efficient data loading, preprocessing, and batching systems—the critical infrastructure that feeds neural networks during training and powers real-world ML systems.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Design data pipeline architectures**: Understand data engineering as the foundation of scalable ML systems
- **Implement reusable dataset abstractions**: Build flexible interfaces that support multiple data sources and formats
- **Create efficient data loaders**: Develop batching, shuffling, and streaming systems for optimal training performance
- **Build preprocessing pipelines**: Implement normalization, augmentation, and transformation systems
- **Apply systems engineering principles**: Handle memory management, I/O optimization, and error recovery in data pipelines

## 🧠 Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Implement dataset abstractions, data loaders, and preprocessing pipelines from engineering principles
2. **Use**: Apply your data system to real CIFAR-10 dataset with complete train/test workflows
3. **Optimize**: Analyze performance characteristics, memory usage, and system bottlenecks for production readiness

## 📚 What You'll Build

### Complete Data Pipeline System
```python
# End-to-end data pipeline creation
train_loader, test_loader, normalizer = create_data_pipeline(
    dataset_path="data/cifar10/",
    batch_size=32,
    normalize=True,
    shuffle=True
)

# Ready for neural network training
for batch_images, batch_labels in train_loader:
    # batch_images.shape: (32, 3, 32, 32) - normalized pixel values
    # batch_labels.shape: (32,) - class indices
    predictions = model(batch_images)
    loss = compute_loss(predictions, batch_labels)
    # Continue training loop...
```

### Dataset Abstraction System
```python
# Flexible interface supporting multiple datasets
class Dataset:
    def __getitem__(self, index): 
        # Return (data, label) for any dataset type
        pass
    def __len__(self): 
        # Enable len() and iteration
        pass

# Concrete implementation with real data
dataset = CIFAR10Dataset("data/cifar10/", train=True, download=True)
print(f"Loaded {len(dataset)} real samples")  # 50,000 training images
image, label = dataset[0]  # Access individual samples
print(f"Sample shape: {image.shape}, Label: {label}")
```

### Efficient Data Loading System
```python
# High-performance batching with memory optimization
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,          # Configurable batch size
    shuffle=True,           # Training randomization
    drop_last=False         # Handle incomplete batches
)

# Pythonic iteration interface
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: {batch_data.shape}")
    # Automatic batching handles all the complexity
```

### Data Preprocessing Pipeline
```python
# Production-ready normalization system
normalizer = Normalizer()

# Fit on training data (compute statistics once)
normalizer.fit(training_images)
print(f"Mean: {normalizer.mean}, Std: {normalizer.std}")

# Apply to any dataset (training, validation, test)
normalized_images = normalizer.transform(test_images)
# Ensures consistent preprocessing across data splits
```

## 🚀 Getting Started

### Prerequisites
Ensure you have the foundational tensor operations:

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Verify prerequisite modules
tito test --module tensor
tito test --module layers
```

### Development Workflow
1. **Open the development file**: `modules/source/07_dataloader/dataloader_dev.py`
2. **Implement Dataset abstraction**: Create the base interface for all data sources
3. **Build CIFAR-10 dataset**: Implement real dataset loading with binary file parsing
4. **Create DataLoader system**: Add batching, shuffling, and iteration functionality
5. **Add preprocessing tools**: Implement normalizer and transformation pipeline
6. **Export and verify**: `tito export --module dataloader && tito test --module dataloader`

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify data engineering functionality:

```bash
# TinyTorch CLI (recommended)
tito test --module dataloader

# Direct pytest execution
python -m pytest tests/ -k dataloader -v
```

### Test Coverage Areas
- ✅ **Dataset Interface**: Verify abstract base class and concrete implementations
- ✅ **Real Data Loading**: Test with actual CIFAR-10 dataset (downloads ~170MB)
- ✅ **Batching System**: Ensure correct batch shapes and memory efficiency
- ✅ **Data Preprocessing**: Verify normalization statistics and transformations
- ✅ **Pipeline Integration**: Test complete train/test workflow with real data

### Inline Testing & Real Data Validation
The module includes comprehensive feedback using real CIFAR-10 data:
```python
# Example inline test output
🔬 Unit Test: CIFAR-10 dataset loading...
📥 Downloading CIFAR-10 dataset (170MB)...
✅ Successfully loaded 50,000 training samples
✅ Sample shapes correct: (3, 32, 32)
✅ Labels in valid range: [0, 9]
📈 Progress: CIFAR-10 Dataset ✓

# DataLoader testing with real data
🔬 Unit Test: DataLoader batching...
✅ Batch shapes correct: (32, 3, 32, 32)
✅ Shuffling produces different orders
✅ Iteration covers all samples exactly once
📈 Progress: DataLoader ✓
```

### Manual Testing Examples
```python
from tinytorch.core.tensor import Tensor
from dataloader_dev import CIFAR10Dataset, DataLoader, Normalizer

# Test dataset loading with real data
dataset = CIFAR10Dataset("data/cifar10/", train=True, download=True)
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.get_num_classes()}")

# Test data loading pipeline
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
for batch_images, batch_labels in dataloader:
    print(f"Batch shape: {batch_images.shape}")
    print(f"Label range: {batch_labels.min()} to {batch_labels.max()}")
    break  # Just test first batch

# Test preprocessing pipeline
normalizer = Normalizer()
sample_batch, _ = next(iter(dataloader))
normalizer.fit(sample_batch)
normalized = normalizer.transform(sample_batch)
print(f"Original range: [{sample_batch.min():.2f}, {sample_batch.max():.2f}]")
print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
```

## 🎯 Key Concepts

### Real-World Applications
- **Production ML Systems**: Companies like Netflix, Spotify use similar data pipelines for recommendation training
- **Computer Vision**: ImageNet, COCO dataset loaders power research and production vision systems
- **Natural Language Processing**: Text preprocessing pipelines enable language model training
- **Autonomous Systems**: Real-time data streams from sensors require efficient pipeline architectures

### Data Engineering Principles
- **Interface Design**: Abstract Dataset class enables switching between data sources seamlessly
- **Memory Efficiency**: Streaming data loading prevents memory overflow with large datasets
- **I/O Optimization**: Batching reduces system calls and improves throughput
- **Preprocessing Consistency**: Fit-transform pattern ensures identical preprocessing across data splits

### Systems Performance Considerations
- **Batch Size Trade-offs**: Larger batches improve GPU utilization but increase memory usage
- **Shuffling Strategy**: Random access patterns for training vs sequential for inference
- **Caching and Storage**: Balance between memory usage and I/O performance
- **Error Handling**: Robust handling of corrupted data, network failures, disk issues

### Production ML Pipeline Patterns
- **ETL Design**: Extract (load files), Transform (preprocess), Load (batch) pattern
- **Data Versioning**: Reproducible datasets with consistent preprocessing
- **Pipeline Monitoring**: Track data quality, distribution shifts, processing times
- **Scalability Planning**: Design for growing datasets and distributed processing

## 🎉 Ready to Build?

You're about to build the data engineering foundation that powers every successful ML system! From startup prototypes to billion-dollar recommendation engines, they all depend on robust data pipelines like the one you're building.

This module teaches you the systems thinking that separates hobby projects from production ML systems. You'll work with real data, handle real performance constraints, and build infrastructure that scales. Take your time, think about edge cases, and enjoy building the backbone of machine learning!

```{grid} 3
:gutter: 3
:margin: 2

{grid-item-card} 🚀 Launch Builder
:link: https://mybinder.org/v2/gh/VJProductions/TinyTorch/main?filepath=modules/source/07_dataloader/dataloader_dev.py
:class-title: text-center
:class-body: text-center

Interactive development environment

{grid-item-card} 📓 Open in Colab  
:link: https://colab.research.google.com/github/VJProductions/TinyTorch/blob/main/modules/source/07_dataloader/dataloader_dev.ipynb
:class-title: text-center
:class-body: text-center

Google Colab notebook

{grid-item-card} 👀 View Source
:link: https://github.com/VJProductions/TinyTorch/blob/main/modules/source/07_dataloader/dataloader_dev.py  
:class-title: text-center
:class-body: text-center

Browse the code on GitHub
``` 