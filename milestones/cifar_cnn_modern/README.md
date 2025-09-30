# 🖼️ CIFAR-10 CNN Example

## What This Demonstrates
A modern CNN architecture for natural image classification using YOUR TinyTorch implementations!

## Prerequisites
Complete these TinyTorch modules first:
- Module 02 (Tensor) - Data structures
- Module 03 (Activations) - ReLU 
- Module 04 (Layers) - Linear layers
- Module 07 (Optimizers) - Adam
- Module 09 (Spatial) - Conv2d, MaxPool2D
- Module 10 (DataLoader) - Dataset, DataLoader

## 🚀 Quick Start

```bash
# Test architecture only (no data download)
python train_cnn.py --test-only

# Train with real CIFAR-10 data (~170MB download)
python train_cnn.py

# Quick test with subset of data
python train_cnn.py --quick-test
```

## 📊 Dataset Information

### CIFAR-10 Details
- **Size**: 60,000 32×32 color images (50K train, 10K test)
- **Classes**: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
- **Download**: ~170MB from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- **Storage**: Cached in `examples/datasets/cifar-10/` after first download

### Data Flow
1. **First Run**: Downloads CIFAR-10 from the web (shows progress)
2. **Subsequent Runs**: Uses cached data (no re-download)
3. **Offline Mode**: Falls back to synthetic data if download fails

### Dataset Handling
```python
# The example uses DataManager for downloading
data_manager = DatasetManager()  # Handles download/caching
(train_data, train_labels), (test_data, test_labels) = data_manager.get_cifar10()

# Then wraps in YOUR Dataset interface
train_dataset = CIFARDataset(train_data, train_labels)  # YOUR Dataset

# Finally uses YOUR DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32)  # YOUR DataLoader
```

## 🏗️ Architecture
```
Input (32×32×3) → Conv2d (3→32) → ReLU → MaxPool (2×2) → 
Conv2d (32→64) → ReLU → MaxPool (2×2) → Flatten → 
Linear (2304→256) → ReLU → Linear (256→10) → Output
```

## 📈 Expected Results
- **Training Time**: 3-5 minutes for demo (3 epochs, 100 batches/epoch)
- **Accuracy**: 65%+ on test set (with simple architecture)
- **Parameters**: ~600K weights

## 🔧 Command Line Options
- `--epochs N`: Number of training epochs (default: 3)
- `--batch-size N`: Batch size (default: 32)
- `--test-only`: Test architecture without training
- `--quick-test`: Use subset of data for quick testing
- `--no-visualize`: Skip visualization

## 💡 What You Learn
- How CNNs extract hierarchical features from images
- Why spatial structure matters for vision
- How YOUR Conv2d, MaxPool2D, and DataLoader work together
- Complete end-to-end training pipeline with real data

## 🐛 Troubleshooting

### Download Issues
If CIFAR-10 download fails:
- Check internet connection
- The example will automatically use synthetic data
- You can manually download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

### Memory Issues
If you run out of memory:
- Use smaller batch size: `--batch-size 16`
- Use quick test mode: `--quick-test`
- Reduce number of epochs: `--epochs 1`

## 📚 Educational Notes
This example shows how YOUR implementations handle:
- **Spatial feature extraction** through convolutions
- **Efficient data loading** with batching and shuffling  
- **Real-world datasets** with proper train/test splits
- **Complete training loops** with YOUR optimizer and autograd