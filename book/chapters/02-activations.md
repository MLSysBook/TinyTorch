---
title: "Activation Functions"
description: "Neural network activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU)"
module_number: 2
tier: "foundation"
difficulty: "beginner"
time_estimate: "3-4 hours"
prerequisites: ["01. Tensor"]
next_module: "03. Layers"
learning_objectives:
  - "Understand how activation functions enable neural networks to learn nonlinear patterns"
  - "Implement core activations (ReLU, Sigmoid, Tanh, Softmax, GELU) with numerical stability"
  - "Recognize the properties and appropriate use cases for each activation function"
  - "Connect activation design to production frameworks like PyTorch and TensorFlow"
  - "Analyze computational complexity and numerical stability trade-offs"
---

# 02. Activations

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1.5rem; border-radius: 0.5rem; display: inline-block; margin-bottom: 2rem; font-weight: 600;">
Foundation Tier | Module 02 of 20
</div>

**Implement nonlinear functions that enable neural networks to learn complex patterns.**

Difficulty: Beginner | Time: 3-4 hours | Prerequisites: Module 01 (Tensor)

---

## What You'll Build

Activation functions are the key to neural network expressiveness. Without them, networks would be limited to linear transformations—unable to learn complex patterns like image features or language semantics.

By the end of this module, you'll have implemented five essential activation functions:

- **ReLU** (Rectified Linear Unit) - The workhorse of modern deep learning
- **Sigmoid** - Probabilistic outputs for binary classification
- **Tanh** (Hyperbolic Tangent) - Zero-centered activation for hidden layers
- **Softmax** - Multi-class probability distributions
- **GELU** (Gaussian Error Linear Unit) - Used in transformers like GPT

### Example Usage

```python
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax, GELU
from tinytorch.core.tensor import Tensor

# ReLU: Simple and effective
relu = ReLU()
output = relu.forward(Tensor([-2, -1, 0, 1, 2]))  # [0, 0, 0, 1, 2]

# Sigmoid: Probabilistic outputs
sigmoid = Sigmoid()
output = sigmoid.forward(Tensor([0, 1, -1]))  # [0.5, 0.73, 0.27]

# Tanh: Zero-centered
tanh = Tanh()
output = tanh.forward(Tensor([0, 1, -1]))  # [0, 0.76, -0.76]

# Softmax: Multi-class probabilities
softmax = Softmax()
logits = Tensor([2.0, 1.0, 0.1])
probs = softmax.forward(logits)  # Sums to 1.0

# GELU: Smooth activation for transformers
gelu = GELU()
output = gelu.forward(Tensor([0, 1, -1]))  # Smooth, probabilistic gating
```

---

## Learning Pattern: Build → Use → Understand

### 1. Build
Implement five activation functions with proper numerical stability handling. Learn to prevent overflow/underflow in exponential operations.

### 2. Use
Apply activations in test scenarios, visualize their behavior, and understand their mathematical properties through experimentation.

### 3. Understand
Grasp when to use each activation (ReLU for hidden layers, Sigmoid for binary outputs, Softmax for multi-class, GELU for transformers) and understand the trade-offs.

---

## Learning Objectives

By completing this module, you will:

1. **Systems Understanding**: Recognize activation functions as the nonlinearity that separates neural networks from linear regression, enabling universal function approximation

2. **Core Implementation**: Build numerically stable implementations of five activation functions, handling edge cases like overflow in exponentials

3. **Pattern Recognition**: Understand activation function properties (range, smoothness, sparsity) and how they affect training dynamics

4. **Framework Connection**: See how your implementations mirror PyTorch's `nn.ReLU()`, `nn.Sigmoid()`, etc., understanding the design decisions frameworks make

5. **Performance Trade-offs**: Analyze computational cost (ReLU is cheap, Softmax is expensive) and numerical stability requirements

---

## Why This Matters

### Production Context

Activation functions are fundamental to every neural network:

- **Computer Vision**: CNNs use ReLU in convolutional layers (AlexNet, ResNet, YOLO)
- **NLP**: Transformers use GELU in feedforward layers (BERT, GPT, T5)
- **Classification**: Softmax converts logits to probabilities in output layers
- **Reinforcement Learning**: Sigmoid/Tanh used in policy networks and actor-critic methods

Choosing the wrong activation can cripple training - ReLU's dead neurons, Sigmoid's vanishing gradients, or numerically unstable Softmax can all break models.

### Systems Reality Check

**Performance Note**: ReLU is O(n) and trivially parallelizable - just a comparison with zero. Softmax is O(n) but requires two passes (max finding, then exponentiation/normalization), making it ~10x slower than ReLU.

**Memory Note**: Most activations are computed in-place during inference (replacing input with output), reducing memory by 2x. PyTorch's `inplace=True` parameter enables this optimization.

---

## Implementation Guide

### Prerequisites Check

Verify Module 01 is complete:

```bash
tito test 01
```

### Development Workflow

```bash
cd modules/source/02_activations/
jupyter lab activations_dev.py
```

### Step-by-Step Build

#### Step 1: ReLU (Rectified Linear Unit)

The simplest and most widely used activation:

```python
class ReLU:
    def forward(self, x):
        """Apply ReLU: f(x) = max(0, x)"""
        return Tensor(np.maximum(0, x.data))
```

**Why this matters**: ReLU's simplicity makes it fast and its sparsity (many zeros) can improve generalization. However, "dead neurons" (always outputting zero) can be problematic.

**Production insight**: ResNet uses ReLU throughout its 152 layers. The function's unbounded positive range prevents vanishing gradients in deep networks.

#### Step 2: Sigmoid

Squashes values to (0, 1) for probability interpretation:

```python
class Sigmoid:
    def forward(self, x):
        """Apply Sigmoid: f(x) = 1 / (1 + e^(-x))"""
        # Numerical stability: avoid overflow in exp(-x)
        return Tensor(1 / (1 + np.exp(-np.clip(x.data, -500, 500))))
```

**Numerical stability**: For large negative `x`, `exp(-x)` can overflow. Clipping prevents this.

**Use case**: Binary classification output layers, attention mechanisms (scaled attention scores).

#### Step 3: Tanh (Hyperbolic Tangent)

Zero-centered activation squashing to (-1, 1):

```python
class Tanh:
    def forward(self, x):
        """Apply Tanh: f(x) = tanh(x)"""
        return Tensor(np.tanh(x.data))
```

**Why zero-centered matters**: Tanh outputs have mean near 0, which can improve training dynamics compared to Sigmoid's (0,1) range.

**Use case**: Hidden layers in RNNs/LSTMs, when you want symmetric outputs.

#### Step 4: Softmax

Converts logits to probability distributions:

```python
class Softmax:
    def forward(self, x):
        """Apply Softmax: exp(x) / sum(exp(x))"""
        # Numerical stability: subtract max before exp
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        return Tensor(exp_x / np.sum(exp_x, axis=-1, keepdims=True))
```

**Critical stability trick**: Subtracting the max before exponentiation prevents overflow while preserving the result (softmax is shift-invariant).

**Use case**: Multi-class classification, attention mechanisms.

#### Step 5: GELU (Gaussian Error Linear Unit)

Smooth, probabilistic activation used in transformers:

```python
class GELU:
    def forward(self, x):
        """Apply GELU: x * Φ(x) where Φ is Gaussian CDF"""
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return Tensor(0.5 * x.data * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3)
        )))
```

**Why transformers use GELU**: Smoother than ReLU, with probabilistic interpretation. BERT and GPT-2/3 use GELU in feedforward layers.

---

## Testing Your Implementation

### Inline Tests

```python
# Test ReLU
relu = ReLU()
x = Tensor([-2, -1, 0, 1, 2])
output = relu.forward(x)
assert output.data.tolist() == [0, 0, 0, 1, 2]
print("✓ ReLU working")

# Test Sigmoid range
sigmoid = Sigmoid()
x = Tensor([0])
output = sigmoid.forward(x)
assert 0.49 < output.data[0] < 0.51  # Should be ~0.5
print("✓ Sigmoid working")

# Test Softmax sums to 1
softmax = Softmax()
x = Tensor([1, 2, 3])
output = softmax.forward(x)
assert abs(np.sum(output.data) - 1.0) < 1e-6
print("✓ Softmax working")
```

### Module Export & Validation

```bash
tito export 02
tito test 02
```

**Expected output**:
```
✓ All tests passed! [20/20]
✓ Module 02 complete!
```

---

## Where This Code Lives

After export, your implementations are used throughout TinyTorch:

```python
# Your activations power layers and networks:
from tinytorch.core.activations import ReLU, Sigmoid, Softmax
from tinytorch.core.layers import Linear

# Example: Linear layer with ReLU activation
layer = Linear(784, 128)
activation = ReLU()

output = activation.forward(layer.forward(x))  # Nonlinear transformation!
```

**Package structure**:
```
tinytorch/
├── core/
│   ├── activations.py  ← YOUR implementations
│   ├── layers.py       (uses activations)
│   ├── networks.py     (chains activations)
```

---

## Systems Thinking Questions

1. **Complexity Analysis**: ReLU is O(n) with just a comparison. Softmax requires finding max, exponentiation, and normalization. How does this affect training time for large models?

2. **Dead Neurons**: ReLU neurons can "die" (always output 0) if they receive only negative inputs. How would you detect this during training? What alternatives exist (Leaky ReLU, ELU)?

3. **Vanishing Gradients**: Sigmoid and Tanh have gradients near 0 for large inputs. Why does this make training deep networks difficult? How does ReLU avoid this?

4. **Numerical Stability**: Why must Softmax subtract the max before exponentiation? What would happen without this trick?

5. **Framework Comparison**: PyTorch offers `nn.ReLU(inplace=True)` for memory efficiency. What does this mean? When is it safe to use?

---

## Real-World Connections

### Industry Applications

- **ResNet (2015)**: Uses ReLU throughout 152 layers for image classification
- **BERT/GPT (2018-2020)**: Uses GELU in transformer feedforward layers
- **AlphaGo (2016)**: Uses ReLU in policy/value networks
- **YOLO (2015-present)**: Uses Leaky ReLU for object detection

### Research Insights

- **Dying ReLU Problem**: ~40% of ReLU neurons can become permanently dead in large networks
- **GELU vs ReLU**: GELU improves perplexity by ~0.5 points in language models
- **Swish (2017)**: Google's discovered activation performs slightly better than ReLU on ImageNet

---

## What's Next?

**Excellent work!** You've implemented the nonlinear functions that make neural networks powerful. Now you'll combine tensors and activations to build composable network layers.

**Module 03: Layers** - Implement Linear layers and Sequential containers to build complete neural network architectures

[Continue to Module 03: Layers →](03-layers.html)

---

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Activations API Reference](../appendices/api-reference.html#activations)
- [Report Issues](https://github.com/mlsysbook/TinyTorch/issues)
