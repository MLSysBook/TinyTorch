---
title: "Loss Functions"
description: "Neural network loss functions (MSE, Cross-Entropy, Binary Cross-Entropy)"
module_number: 4
tier: "foundation"
difficulty: "intermediate"
time_estimate: "3-4 hours"
prerequisites: ["01. Tensor", "02. Activations", "03. Layers"]
next_module: "05. Autograd"
learning_objectives:
  - "Understand loss functions as objectives that measure model performance"
  - "Implement MSE for regression and Cross-Entropy for classification"
  - "Recognize numerical stability requirements in loss computation"
  - "Connect loss function choice to problem type and output distributions"
  - "Analyze gradient behavior and its impact on learning dynamics"
---

# 04. Losses

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 0.5rem; display: inline-block; margin-bottom: 2rem; font-weight: 600;">
Foundation Tier | Module 04 of 20
</div>

**Implement objective functions that measure model performance.**

Difficulty: Intermediate | Time: 3-4 hours | Prerequisites: Modules 01-03

---

## What You'll Build

Loss functions quantify how wrong a model's predictions are. They're the objective we minimize during training—the compass that guides learning.

By the end of this module, you'll have implemented three essential loss functions:

- **MSE (Mean Squared Error)** - For regression problems
- **Cross-Entropy Loss** - For multi-class classification
- **Binary Cross-Entropy** - For binary classification

### Example Usage

```python
from tinytorch.core.losses import MSE, CrossEntropyLoss, BCELoss
from tinytorch.core.tensor import Tensor

# MSE for regression
mse = MSE()
predictions = Tensor([2.5, 3.1, 4.2])
targets = Tensor([2.0, 3.0, 4.0])
loss = mse.forward(predictions, targets)  # Scalar

# Cross-Entropy for classification
ce = CrossEntropyLoss()
logits = Tensor([[2.0, 1.0, 0.1]])  # Unnormalized scores
targets = Tensor([0])  # Class index
loss = ce.forward(logits, targets)

# Binary Cross-Entropy
bce = BCELoss()
predictions = Tensor([0.7, 0.3, 0.9])  # Probabilities [0,1]
targets = Tensor([1.0, 0.0, 1.0])
loss = bce.forward(predictions, targets)
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement three loss functions with numerical stability handling for exponentials and logarithms.

### 2. Use
Apply losses to real prediction scenarios, understanding when each loss is appropriate for different problem types.

### 3. Understand
Grasp how loss function choice affects gradient behavior and learning dynamics, and why numerical stability is critical.

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize loss functions as differentiable objectives that bridge model predictions and training dynamics

2. **Core Implementation**: Build numerically stable implementations of MSE, Cross-Entropy, and Binary Cross-Entropy

3. **Pattern Recognition**: Understand the connection between problem type (regression vs classification) and appropriate loss function

4. **Framework Connection**: See how your losses mirror PyTorch's `nn.MSELoss()`, `nn.CrossEntropyLoss()`, and `nn.BCELoss()`

5. **Performance Trade-offs**: Analyze gradient magnitude and how loss function choice affects training speed and stability

---

## Why This Matters

### Production Context

Loss functions are fundamental to every ML application:

- **Computer Vision**: Cross-Entropy for image classification (ImageNet, COCO)
- **NLP**: Cross-Entropy for language modeling (GPT, BERT use it for next-token prediction)
- **Regression**: MSE for price prediction, demand forecasting, time series
- **Ranking**: Binary Cross-Entropy for click prediction, recommendation systems

The loss function defines what "good" means for your model. Choose wrong, train poorly.

### Systems Reality Check

**Performance Note**: Loss computation is O(n) for n predictions, but gradient computation through loss can be expensive. Cross-Entropy with Softmax has numerical stability tricks that matter.

**Memory Note**: Loss is a scalar, but its gradient must flow back through all predictions. This gradient tensor has the same shape as predictions.

---

## Implementation Guide

### Prerequisites Check

```bash
tito test 01 02 03
```

### Development Workflow

```bash
cd modules/source/04_losses/
jupyter lab losses_dev.py
```

### Step-by-Step Build

#### Step 1: Mean Squared Error (MSE)

For regression tasks:

```python
class MSE:
    def forward(self, predictions, targets):
        """
        MSE Loss: mean((y_pred - y_true)^2)
        
        Args:
            predictions: Tensor of shape (N,) or (N, D)
            targets: Tensor of same shape as predictions
        
        Returns:
            Scalar loss
        """
        diff = predictions - targets
        squared = diff ** 2
        return squared.mean()
```

**Why squared error**: Penalizes large errors more than small ones. Differentiable everywhere (unlike absolute error).

**Use case**: Predicting continuous values (house prices, temperatures, stock prices).

#### Step 2: Cross-Entropy Loss

For multi-class classification:

```python
class CrossEntropyLoss:
    def forward(self, logits, targets):
        """
        Cross-Entropy: -log(softmax(logits)[targets])
        
        Args:
            logits: Tensor of shape (N, C) - unnormalized scores
            targets: Tensor of shape (N,) - class indices
        
        Returns:
            Scalar loss
        """
        # Numerical stability: subtract max before exp
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        
        # Compute log softmax
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits.data), axis=1, keepdims=True))
        log_softmax = shifted_logits.data - log_sum_exp
        
        # Select log probabilities of correct classes
        batch_size = logits.shape[0]
        log_probs = log_softmax[np.arange(batch_size), targets.data.astype(int)]
        
        # Negative log likelihood
        return Tensor(-np.mean(log_probs))
```

**Critical stability trick**: Subtract max before exponentiating prevents overflow. Log-sum-exp trick is standard in production frameworks.

**Use case**: Multi-class classification (ImageNet 1000 classes, CIFAR-10, MNIST digits).

#### Step 3: Binary Cross-Entropy (BCE)

For binary classification:

```python
class BCELoss:
    def forward(self, predictions, targets):
        """
        Binary Cross-Entropy: -[t*log(p) + (1-t)*log(1-p)]
        
        Args:
            predictions: Tensor of shape (N,) - probabilities in [0,1]
            targets: Tensor of shape (N,) - binary labels {0,1}
        
        Returns:
            Scalar loss
        """
        # Clip to prevent log(0)
        eps = 1e-7
        p = np.clip(predictions.data, eps, 1 - eps)
        
        # Binary cross-entropy
        bce = -(targets.data * np.log(p) + (1 - targets.data) * np.log(1 - p))
        
        return Tensor(np.mean(bce))
```

**Numerical stability**: Clipping prevents `log(0) = -inf`. Essential for stability.

**Use case**: Binary decisions (spam detection, click prediction, medical diagnosis).

---

## Testing Your Implementation

### Inline Tests

```python
# Test MSE
mse = MSE()
preds = Tensor([1.0, 2.0, 3.0])
targets = Tensor([1.0, 2.0, 3.0])
loss = mse.forward(preds, targets)
assert abs(loss.data) < 1e-6  # Perfect prediction = 0 loss
print("✓ MSE working")

# Test Cross-Entropy
ce = CrossEntropyLoss()
logits = Tensor([[10.0, 0.0, 0.0]])  # High confidence class 0
targets = Tensor([0])
loss = ce.forward(logits, targets)
assert loss.data < 0.1  # Low loss for correct prediction
print("✓ Cross-Entropy working")

# Test BCE
bce = BCELoss()
preds = Tensor([0.9, 0.1])  # High confidence
targets = Tensor([1.0, 0.0])
loss = bce.forward(preds, targets)
assert loss.data < 0.2
print("✓ BCE working")
```

### Module Export & Validation

```bash
tito export 04
tito test 04
```

**Expected output**:
```
✓ All tests passed! [15/15]
✓ Module 04 complete!
```

---

## Where This Code Lives

Your losses drive training:

```python
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.layers import Sequential, Linear
from tinytorch.core.activations import ReLU

# Training setup
model = Sequential([Linear(784, 128), ReLU(), Linear(128, 10)])
loss_fn = CrossEntropyLoss()

# Training loop
predictions = model.forward(batch_x)
loss = loss_fn.forward(predictions, batch_y)
# Later: loss.backward() to compute gradients
```

**Package structure**:
```
tinytorch/
├── core/
│   ├── losses.py  ← YOUR implementations
│   ├── layers.py
│   ├── autograd.py  (uses losses)
```

---

## Systems Thinking Questions

1. **Gradient Behavior**: MSE gives gradients proportional to error magnitude. Cross-Entropy gives gradients proportional to prediction confidence. How does this affect learning dynamics?

2. **Numerical Stability**: Why must Cross-Entropy subtract max before exponentiation? What happens if you don't? Try it.

3. **Loss Scaling**: A batch with 32 samples computes mean loss over 32 examples. How does batch size affect gradient magnitude? Should learning rate scale with batch size?

4. **Problem Mismatch**: What happens if you use MSE for classification? Or Cross-Entropy for regression? Why do these fail?

5. **Framework Comparison**: PyTorch's `CrossEntropyLoss` combines Softmax + log + NLL in one operation. Why is this more efficient than computing separately?

---

## Real-World Connections

### Industry Applications

- **ImageNet Classification**: Cross-Entropy loss to classify 1000 object categories
- **GPT Language Modeling**: Cross-Entropy to predict next token from 50K vocabulary
- **Click Prediction**: Binary Cross-Entropy for ad click probability
- **Demand Forecasting**: MSE to predict sales quantities

### Training Dynamics

- **Loss Plateaus**: Cross-Entropy can saturate (near-zero gradients) when model is very confident
- **Loss Spikes**: Numerical instability in exponentials causes training to diverge
- **Gradient Clipping**: Large loss values can cause exploding gradients

---

## What's Next?

**Excellent work!** You can now measure model performance. Next, you'll implement automatic differentiation—the engine that computes gradients for these losses.

**Module 05: Autograd** - Build computational graphs and automatic differentiation for efficient gradient computation

[Continue to Module 05: Autograd →](05-autograd.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Losses API Reference](../appendices/api-reference.html#losses)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)
