# 🧠 Perceptron (1957) - Rosenblatt

## What This Demonstrates
The first trainable neural network in history! Using YOUR TinyTorch implementations to recreate Rosenblatt's pioneering perceptron.

## Prerequisites
Complete these TinyTorch modules first:
- Module 02 (Tensor) - Data structures with gradients
- Module 03 (Activations) - Sigmoid activation
- Module 04 (Layers) - Linear layer

## 🚀 Quick Start

```bash
# Run the perceptron training
python rosenblatt_perceptron.py

# Test architecture only
python rosenblatt_perceptron.py --test-only

# Custom epochs
python rosenblatt_perceptron.py --epochs 200
```

## 📊 Dataset Information

### Synthetic Linearly Separable Data
- **Generated**: 1,000 points in 2D space
- **Classes**: Binary (0 or 1)
- **Property**: Linearly separable by design
- **No Download Required**: Data generated on-the-fly

### Why Synthetic Data?
The perceptron can only solve linearly separable problems. We generate data that's guaranteed to be separable to demonstrate the algorithm works when its assumptions are met.

## 🏗️ Architecture
```
Input (x1, x2) → Linear (2→1) → Sigmoid → Binary Output
```

Simple but revolutionary - this proved machines could learn!

## 📈 Expected Results
- **Training Time**: ~30 seconds
- **Accuracy**: 95%+ (problem is linearly separable)
- **Parameters**: Just 3 (2 weights + 1 bias)

## 💡 Historical Significance
- **1957**: First demonstration of machine learning
- **Innovation**: Weights that adjust based on errors
- **Limitation**: Can't solve XOR (see xor_1969 example)
- **Legacy**: Foundation for all modern neural networks

## 🔧 Command Line Options
- `--test-only`: Test architecture without training
- `--epochs N`: Number of training epochs (default: 100)

## 📚 What You Learn
- How the first neural network worked
- Why gradients enable learning
- YOUR Linear layer performs the same math as 1957
- Limitations that led to multi-layer networks