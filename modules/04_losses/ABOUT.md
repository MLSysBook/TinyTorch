---
title: "Loss Functions"
description: "Implement MSE and CrossEntropy loss functions for training neural networks"
difficulty: 2
time_estimate: "3-4 hours"
prerequisites: ["Tensor", "Activations", "Layers"]
next_steps: ["Autograd"]
learning_objectives:
  - "Implement MSE loss for regression tasks with proper numerical stability"
  - "Build CrossEntropy loss for classification with log-sum-exp trick"
  - "Understand mathematical properties of loss functions and their gradients"
  - "Recognize how loss functions connect model outputs to optimization objectives"
  - "Apply appropriate loss functions for different machine learning tasks"
---

# 04. Losses

**🏗️ FOUNDATION TIER** | Difficulty: ⭐⭐ (2/4) | Time: 3-4 hours

## Overview

Implement the mathematical functions that measure how wrong your model's predictions are. Loss functions are the bridge between model outputs and the optimization process—they define what "better" means and drive the entire learning process.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement MSE loss** for regression tasks with numerically stable computation
2. **Build CrossEntropy loss** for classification using the log-sum-exp trick for numerical stability
3. **Understand mathematical properties** of loss landscapes and their impact on optimization
4. **Recognize the role** of loss functions in connecting predictions to training objectives
5. **Apply appropriate losses** for regression, binary classification, and multi-class classification

## Why This Matters

### Production Context

Loss functions are fundamental to all machine learning systems:

- **Recommendation Systems** use MSE and ranking losses to learn user preferences
- **Image Classification** relies on CrossEntropy loss for category prediction (ImageNet, CIFAR-10)
- **Language Models** use CrossEntropy to predict next tokens in GPT, Claude, and all LLMs
- **Autonomous Driving** combines multiple losses for perception, planning, and control

### Historical Context

Loss functions evolved with machine learning itself:

- **Least Squares (1805)**: Gauss invented MSE for astronomical orbit predictions
- **Maximum Likelihood (1912)**: Fisher formalized statistical foundations of loss functions
- **CrossEntropy (1950s)**: Information theory brought entropy-based losses to ML
- **Modern Deep Learning (2012+)**: Careful loss design enables training billion-parameter models

## Build → Use → Understand

This module follows the classic pedagogy for foundational concepts:

1. **Build**: Implement MSE and CrossEntropy loss functions from mathematical definitions
2. **Use**: Apply losses to regression and classification tasks, seeing how they drive learning
3. **Understand**: Analyze loss landscapes, gradients, and numerical stability considerations

## Implementation Guide

### Step 1: MSE (Mean Squared Error) Loss

Implement L2 loss for regression:

```python
class MSELoss:
    """Mean Squared Error loss for regression."""
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute MSE: (1/n) * Σ(predictions - targets)²
        
        Args:
            predictions: Model outputs
            targets: Ground truth values
        Returns:
            Scalar loss value
        """
        diff = predictions - targets
        squared = diff * diff
        return squared.mean()
```

### Step 2: CrossEntropy Loss

Implement log-likelihood loss for classification:

```python
class CrossEntropyLoss:
    """CrossEntropy loss for multi-class classification."""
    
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute CrossEntropy with log-sum-exp trick for numerical stability.
        
        Args:
            logits: Raw model outputs (before softmax)
            targets: Class indices or one-hot vectors
        Returns:
            Scalar loss value
        """
        # Log-sum-exp trick for numerical stability
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = (logits - max_logits).exp()
        log_probs = logits - max_logits - exp_logits.sum(axis=1, keepdims=True).log()
        
        # Negative log-likelihood
        return -log_probs.mean()
```

### Step 3: Loss Function Properties

Understand key mathematical properties:

- **Convexity**: MSE is convex; CrossEntropy is convex in logits
- **Gradients**: Smooth gradients enable effective optimization
- **Scale**: Loss magnitude affects learning rate tuning
- **Numerical Stability**: Requires careful implementation (log-sum-exp trick)

## Testing

### Inline Tests

The module includes immediate feedback:

```python
# Example inline test output
🔬 Unit Test: MSE Loss...
✅ MSE computes squared error correctly
✅ MSE gradient flows properly
✅ MSE handles batch dimensions correctly
📈 Progress: MSE Loss ✓

🔬 Unit Test: CrossEntropy Loss...
✅ CrossEntropy numerically stable
✅ CrossEntropy matches PyTorch implementation
✅ CrossEntropy handles multi-class problems
📈 Progress: CrossEntropy Loss ✓
```

### Export and Validate

```bash
# Export to package
tito export --module 04_losses

# Run test suite
tito test --module 04_losses
```

## Where This Code Lives

```
tinytorch/
├── nn/
│   └── losses.py          # MSELoss, CrossEntropyLoss
└── core/
    └── tensor.py          # Underlying tensor operations
```

After export, use as:
```python
from tinytorch.nn import MSELoss, CrossEntropyLoss

# For regression
mse = MSELoss()
loss = mse(predictions, targets)

# For classification
ce = CrossEntropyLoss()
loss = ce(logits, labels)
```

## Systems Thinking Questions

1. **Why does CrossEntropy require the log-sum-exp trick?** What numerical instability occurs without it?

2. **How does loss scale affect learning?** If you multiply your loss by 100, what happens to gradients and learning?

3. **Why do we use MSE for regression but CrossEntropy for classification?** What makes each appropriate for its task?

4. **How do loss functions connect to probability theory?** What is the relationship between CrossEntropy and maximum likelihood?

5. **What happens if you use the wrong loss function?** Try MSE for classification or CrossEntropy for regression—what breaks?

## Real-World Connections

### Industry Applications

- **Computer Vision**: CrossEntropy trains all classification models (ResNet, EfficientNet, Vision Transformers)
- **NLP**: CrossEntropy is the foundation of all language models (GPT, BERT, T5)
- **Recommendation**: MSE and ranking losses optimize Netflix, Spotify, YouTube recommendations
- **Robotics**: MSE trains continuous control policies for manipulation and navigation

### Production Considerations

- **Numerical Stability**: Log-sum-exp trick prevents overflow/underflow in production systems
- **Loss Scaling**: Careful scaling enables mixed-precision training (FP16/BF16)
- **Weighted Losses**: Class weights handle imbalanced datasets in production
- **Custom Losses**: Production systems often combine multiple loss terms

## What's Next?

Now that you can measure prediction quality, you're ready for **Module 05: Autograd** where you'll learn how to automatically compute gradients of these loss functions, enabling the optimization that drives all of machine learning.

**Preview**: Autograd will automatically compute ∂Loss/∂weights for any loss function you build, making training possible without manual gradient derivations!

---

**Need Help?**
- Check the inline tests in `modules/04_losses/losses_dev.py`
- Review mathematical derivations in the module comments
- Compare your implementation against PyTorch's losses

