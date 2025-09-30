# 🔢 MNIST MLP (1986) - Backpropagation Revolution

## What This Demonstrates
Multi-layer network solving real vision! Backpropagation enables training deep networks on actual handwritten digits.

## Prerequisites
Complete these TinyTorch modules first:
- Module 02 (Tensor) - Data structures
- Module 03 (Activations) - ReLU, Softmax
- Module 04 (Layers) - Linear layers
- Module 06 (Autograd) - Backpropagation
- Module 07 (Optimizers) - SGD optimizer  
- Module 08 (Training) - Training loops

Note: Runs BEFORE Module 10 (DataLoader), so uses manual batching.

## 🚀 Quick Start

```bash
# Train on MNIST digits
python train_mlp.py

# Test architecture only
python train_mlp.py --test-only

# Quick training (fewer epochs)
python train_mlp.py --epochs 3
```

## 📊 Dataset Information

### MNIST Handwritten Digits
- **Size**: 70,000 grayscale 28×28 images (60K train, 10K test)
- **Classes**: Digits 0-9
- **Download**: ~10MB from http://yann.lecun.com/exdb/mnist/
- **Storage**: Cached in `examples/datasets/mnist/` after first download

### Sample Digits
```
  "7"          "2"          "1"
░░░████░░    █████████    ░░░██░░░
░░░░░██░░    ░░░░░░██░    ░░███░░░
░░░░██░░░    ░░░░░██░░    ░░░██░░░
░░░██░░░░    ░░░██░░░░    ░░░██░░░
░░██░░░░░    ░░██░░░░░    ░░░██░░░
░░██░░░░░    ██████████   ░░░██░░░
```

### Data Flow
1. **Download**: Automatic from LeCun's website
2. **Format**: Flatten 28×28 → 784 features
3. **Batching**: Manual (DataLoader not available yet)

## 🏗️ Architecture
```
Input (784) → Linear (784→128) → ReLU → Linear (128→64) → ReLU → Linear (64→10) → Output
                ↑                          ↑                         ↑
           Hidden Layer 1            Hidden Layer 2            10 Classes
```

## 📈 Expected Results
- **Training Time**: 2-3 minutes (5 epochs)
- **Accuracy**: 95%+ on test set
- **Parameters**: ~100K weights

## 💡 Historical Significance
- **1986**: Backprop paper enables deep learning
- **Innovation**: Automatic gradient computation
- **Impact**: Proved neural networks could solve real problems
- **YOUR Version**: Same architecture, YOUR implementation!

## 🔧 Command Line Options
- `--test-only`: Test architecture without training
- `--epochs N`: Training epochs (default: 5)

## 📚 What You Learn
- How to handle real vision datasets
- Multi-layer networks for complex patterns
- Manual batching before DataLoader
- YOUR complete training pipeline works!

## 🐛 Troubleshooting

### Download Issues
If MNIST download fails:
- Check internet connection
- Falls back to synthetic data automatically
- Manual download: http://yann.lecun.com/exdb/mnist/

### Memory Issues
- Reduce batch size in the code (default: 32)
- Train for fewer epochs: `--epochs 2`