# Module 05: Loss Functions - Learning Objectives for Neural Networks

**Essential loss functions that define learning objectives and enable neural networks to learn from data through gradient-based optimization.**

## 🎯 Learning Objectives

By the end of this module, you will understand:

- **Mathematical Foundation**: How loss functions translate learning problems into optimization objectives
- **Numerical Stability**: Why proper implementation prevents catastrophic training failures in production
- **Problem Matching**: When to use each loss function based on problem structure and data characteristics
- **Production Integration**: How loss functions integrate with neural network training pipelines

## 🏗️ What You'll Build

### Core Loss Functions
- **MeanSquaredError**: Regression loss for continuous value prediction
- **CrossEntropyLoss**: Multi-class classification with numerically stable softmax
- **BinaryCrossEntropyLoss**: Optimized binary classification loss

### Key Features
- ✅ Numerically stable implementations that handle edge cases
- ✅ Efficient batch processing for scalable training  
- ✅ Clean interfaces that integrate with neural networks
- ✅ Comprehensive testing with real-world scenarios

## 🚀 Quick Start

```python
from tinytorch.core.losses import MeanSquaredError, CrossEntropyLoss, BinaryCrossEntropyLoss

# Regression: Predicting house prices
mse = MeanSquaredError()
regression_loss = mse(predicted_prices, actual_prices)

# Multi-class classification: Image recognition
ce_loss = CrossEntropyLoss() 
classification_loss = ce_loss(model_logits, class_indices)

# Binary classification: Spam detection
bce_loss = BinaryCrossEntropyLoss()
binary_loss = bce_loss(spam_logits, spam_labels)
```

## 📚 Usage Examples

### When to Use Each Loss Function

**Mean Squared Error (MSE)**
- **Best for**: Regression problems (house prices, temperatures, ages)
- **Output**: Any real number
- **Activation**: Linear (no activation)

**Cross-Entropy Loss**  
- **Best for**: Multi-class classification (image classification, text categorization)
- **Output**: Class probabilities (sum to 1)
- **Activation**: Softmax

**Binary Cross-Entropy Loss**
- **Best for**: Binary classification (spam detection, medical diagnosis)
- **Output**: Single probability (0 to 1) 
- **Activation**: Sigmoid

## 🧪 Testing Your Implementation

Run the module to test all loss functions:

```bash
# Test implementations
python modules/05_losses/losses_dev.py

# Export to package
tito module complete 05_losses
```

Expected output:
```
🧪 Testing Mean Squared Error Loss...
✅ Perfect predictions test passed
✅ All MSE loss tests passed!

🧪 Testing Cross-Entropy Loss... 
✅ Perfect predictions test passed
✅ All Cross-Entropy loss tests passed!

🎉 Complete loss function foundation ready!
```

## 🔗 Integration Examples

### Training Loop Integration
```python
from tinytorch.core.layers import Sequential, Linear
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.losses import CrossEntropyLoss

# Build classifier
model = Sequential([
    Linear(784, 128), ReLU(),
    Linear(128, 10), Softmax()
])

# Set up training
loss_fn = CrossEntropyLoss()

# Training step
predictions = model(batch_inputs)
loss = loss_fn(predictions, batch_targets)
# loss.backward()  # Triggers gradient computation (with autograd)
```

## 🎯 Module Structure

```
05_losses/
├── losses_dev.py          # Main implementation
├── README.md              # This file
└── module.yaml           # Module configuration
```

## 🔬 Key Implementation Details

### Numerical Stability Features
- **Cross-Entropy**: Uses log-sum-exp trick and probability clipping
- **Binary Cross-Entropy**: Stable logits formulation prevents overflow
- **All Losses**: Robust handling of edge cases and extreme values

### Performance Optimizations
- Efficient batch processing across multiple samples
- Vectorized operations using NumPy
- Memory-efficient computation for large datasets

## 🚀 What's Next

With loss functions implemented, you're ready for:
- **Training Loops**: Complete end-to-end neural network training
- **Optimizers**: Gradient-based parameter updates  
- **Advanced Training**: Monitoring, checkpointing, and convergence analysis

## 💡 Key Insights

1. **Loss functions are the interface between business objectives and mathematical optimization**
2. **Numerical stability is critical for reliable production training**
3. **Different problem types require different loss functions for optimal performance**
4. **Proper batch processing enables scalable training on large datasets**

---

**Next Module**: Training Infrastructure - Build complete training loops that bring all components together!