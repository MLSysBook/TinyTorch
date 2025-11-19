# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Module 03: Layers - Building Blocks of Neural Networks

Welcome to Module 03! You're about to build the fundamental building blocks that make neural networks possible.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor class (Module 01) with all operations and activations (Module 02)
**You'll Build**: Linear layers and Dropout regularization
**You'll Enable**: Multi-layer neural networks, trainable parameters, and forward passes

**Connection Map**:
```
Tensor → Activations → Layers → Networks
(data)   (intelligence) (building blocks) (architectures)
```

## Learning Objectives
By the end of this module, you will:
1. Implement Linear layers with proper weight initialization
2. Add Dropout for regularization during training
3. Understand parameter management and counting
4. Test individual layer components

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/03_layers/layers_dev.py
**Building Side:** Code exports to tinytorch.core.layers

```python
# Final package structure:
from tinytorch.core.layers import Linear, Dropout  # This module
from tinytorch.core.tensor import Tensor  # Module 01 - foundation
from tinytorch.core.activations import ReLU, Sigmoid  # Module 02 - intelligence
```

**Why this matters:**
- **Learning:** Complete layer system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn with all layer building blocks together
- **Consistency:** All layer operations and parameter management in core.layers
- **Integration:** Works seamlessly with tensors and activations for complete neural networks
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "solution": true}
#| default_exp core.layers
#| export

import numpy as np

# Import dependencies from tinytorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid

# %% [markdown]
"""
## 1. Introduction: What are Neural Network Layers?

Neural network layers are the fundamental building blocks that transform data as it flows through a network. Each layer performs a specific computation:

- **Linear layers** apply learned transformations: `y = xW + b`
- **Dropout layers** randomly zero elements for regularization

Think of layers as processing stations in a factory:
```
Input Data → Layer 1 → Layer 2 → Layer 3 → Output
    ↓          ↓         ↓         ↓         ↓
  Features   Hidden   Hidden   Hidden   Predictions
```

Each layer learns its own piece of the puzzle. Linear layers learn which features matter, while dropout prevents overfitting by forcing robustness.
"""

# %% [markdown]
"""
## 2. Foundations: Mathematical Background

### Linear Layer Mathematics
A linear layer implements: **y = xW + b**

```
Input x (batch_size, in_features)  @  Weight W (in_features, out_features)  +  Bias b (out_features)
                                   =  Output y (batch_size, out_features)
```

### Weight Initialization
Random initialization is crucial for breaking symmetry:
- **Xavier/Glorot**: Scale by sqrt(1/fan_in) for stable gradients
- **He**: Scale by sqrt(2/fan_in) for ReLU activation
- **Too small**: Gradients vanish, learning is slow
- **Too large**: Gradients explode, training unstable

### Parameter Counting
```
Linear(784, 256): 784 × 256 + 256 = 200,960 parameters

Manual composition:
    layer1 = Linear(784, 256)  # 200,960 params
    activation = ReLU()        # 0 params
    layer2 = Linear(256, 10)   # 2,570 params
                               # Total: 203,530 params
```

Memory usage: 4 bytes/param × 203,530 = ~814KB for weights alone
"""

# %% [markdown]
"""
## 3. Implementation: Building Layer Foundation

Let's build our layer system step by step. We'll implement two essential layer types:

1. **Linear Layer** - The workhorse of neural networks
2. **Dropout Layer** - Prevents overfitting

### Key Design Principles:
- All methods defined INSIDE classes (no monkey-patching)
- Parameter tensors have requires_grad=True (ready for Module 05)
- Forward methods return new tensors, preserving immutability
- parameters() method enables optimizer integration
"""

# %% [markdown]
"""
### 🏗️ Linear Layer - The Foundation of Neural Networks

Linear layers (also called Dense or Fully Connected layers) are the fundamental building blocks of neural networks. They implement the mathematical operation:

**y = xW + b**

Where:
- **x**: Input features (what we know)
- **W**: Weight matrix (what we learn)
- **b**: Bias vector (adjusts the output)
- **y**: Output features (what we predict)

### Why Linear Layers Matter

Linear layers learn **feature combinations**. Each output neuron asks: "What combination of input features is most useful for my task?" The network discovers these combinations through training.

### Data Flow Visualization
```
Input Features     Weight Matrix        Bias Vector      Output Features
[batch, in_feat] @ [in_feat, out_feat] + [out_feat]  =  [batch, out_feat]

Example: MNIST Digit Recognition
[32, 784]       @  [784, 10]          + [10]        =  [32, 10]
  ↑                   ↑                    ↑             ↑
32 images         784 pixels          10 classes    10 probabilities
                  to 10 classes       adjustments   per image
```

### Memory Layout
```
Linear(784, 256) Parameters:
┌─────────────────────────────┐
│ Weight Matrix W             │  784 × 256 = 200,704 params
│ [784, 256] float32          │  × 4 bytes = 802.8 KB
├─────────────────────────────┤
│ Bias Vector b               │  256 params
│ [256] float32               │  × 4 bytes = 1.0 KB
└─────────────────────────────┘
                Total: 803.8 KB for one layer
```
"""

# %% nbgrader={"grade": false, "grade_id": "linear-layer", "solution": true}
#| export
class Linear:
    """
    Linear (fully connected) layer: y = xW + b

    This is the fundamental building block of neural networks.
    Applies a linear transformation to incoming data.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize linear layer with proper weight initialization.

        TODO: Initialize weights and bias with Xavier initialization

        APPROACH:
        1. Create weight matrix (in_features, out_features) with Xavier scaling
        2. Create bias vector (out_features,) initialized to zeros if bias=True
        3. Set requires_grad=True for parameters (ready for Module 05)

        EXAMPLE:
        >>> layer = Linear(784, 10)  # MNIST classifier final layer
        >>> print(layer.weight.shape)
        (784, 10)
        >>> print(layer.bias.shape)
        (10,)

        HINTS:
        - Xavier init: scale = sqrt(1/in_features)
        - Use np.random.randn() for normal distribution
        - bias=None when bias=False
        """
        ### BEGIN SOLUTION
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot initialization for stable gradients
        scale = np.sqrt(1.0 / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data, requires_grad=True)

        # Initialize bias to zeros or None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through linear layer.

        TODO: Implement y = xW + b

        APPROACH:
        1. Matrix multiply input with weights: xW
        2. Add bias if it exists
        3. Return result as new Tensor

        EXAMPLE:
        >>> layer = Linear(3, 2)
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features
        >>> y = layer.forward(x)
        >>> print(y.shape)
        (2, 2)  # 2 samples, 2 outputs

        HINTS:
        - Use tensor.matmul() for matrix multiplication
        - Handle bias=None case
        - Broadcasting automatically handles bias addition
        """
        ### BEGIN SOLUTION
        # Linear transformation: y = xW
        output = x.matmul(self.weight)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
        ### END SOLUTION

    def __call__(self, x):
        """Allows the layer to be called like a function."""
        return self.forward(x)

    def parameters(self):
        """
        Return list of trainable parameters.

        TODO: Return all tensors that need gradients

        APPROACH:
        1. Start with weight (always present)
        2. Add bias if it exists
        3. Return as list for optimizer
        """
        ### BEGIN SOLUTION
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
        ### END SOLUTION

    def __repr__(self):
        """String representation for debugging."""
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"

# %% [markdown]
"""
### 🔬 Unit Test: Linear Layer
This test validates our Linear layer implementation works correctly.
**What we're testing**: Weight initialization, forward pass, parameter management
**Why it matters**: Foundation for all neural network architectures
**Expected**: Proper shapes, Xavier scaling, parameter counting
"""

# %% nbgrader={"grade": true, "grade_id": "test-linear", "locked": true, "points": 15}
def test_unit_linear_layer():
    """🔬 Test Linear layer implementation."""
    print("🔬 Unit Test: Linear Layer...")

    # Test layer creation
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)
    assert layer.weight.requires_grad == True
    assert layer.bias.requires_grad == True

    # Test Xavier initialization (weights should be reasonably scaled)
    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(1.0 / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, f"Weight std {weight_std} not close to Xavier {expected_std}"

    # Test bias initialization (should be zeros)
    assert np.allclose(layer.bias.data, 0), "Bias should be initialized to zeros"

    # Test forward pass
    x = Tensor(np.random.randn(32, 784))  # Batch of 32 samples
    y = layer.forward(x)
    assert y.shape == (32, 256), f"Expected shape (32, 256), got {y.shape}"

    # Test no bias option
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1  # Only weight, no bias

    # Test parameters method
    params = layer.parameters()
    assert len(params) == 2  # Weight and bias
    assert params[0] is layer.weight
    assert params[1] is layer.bias

    print("✅ Linear layer works correctly!")

if __name__ == "__main__":
    test_unit_linear_layer()





# %% [markdown]
"""
### 🎲 Dropout Layer - Preventing Overfitting

Dropout is a regularization technique that randomly "turns off" neurons during training. This forces the network to not rely too heavily on any single neuron, making it more robust and generalizable.

### Why Dropout Matters

**The Problem**: Neural networks can memorize training data instead of learning generalizable patterns. This leads to poor performance on new, unseen data.

**The Solution**: Dropout randomly zeros out neurons, forcing the network to learn multiple independent ways to solve the problem.

### Dropout in Action
```
Training Mode (p=0.5 dropout):
Input:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
         ↓ Random mask with 50% survival rate
Mask:   [1,   0,   1,   0,   1,   1,   0,   1  ]
         ↓ Apply mask and scale by 1/(1-p) = 2.0
Output: [2.0, 0.0, 6.0, 0.0, 10.0, 12.0, 0.0, 16.0]

Inference Mode (no dropout):
Input:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
         ↓ Pass through unchanged
Output: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

### Training vs Inference Behavior
```
                Training Mode              Inference Mode
               ┌─────────────────┐        ┌─────────────────┐
Input Features │ [×] [ ] [×] [×] │        │ [×] [×] [×] [×] │
               │ Active Dropped  │   →    │   All Active    │
               │ Active Active   │        │                 │
               └─────────────────┘        └─────────────────┘
                      ↓                           ↓
                "Learn robustly"            "Use all knowledge"
```

### Memory and Performance
```
Dropout Memory Usage:
┌─────────────────────────────┐
│ Input Tensor: X MB          │
├─────────────────────────────┤
│ Random Mask: X/4 MB         │  (boolean mask, 1 byte/element)
├─────────────────────────────┤
│ Output Tensor: X MB         │
└─────────────────────────────┘
        Total: ~2.25X MB peak memory

Computational Overhead: Minimal (element-wise operations)
```
"""

# %% nbgrader={"grade": false, "grade_id": "dropout-layer", "solution": true}
#| export
class Dropout:
    """
    Dropout layer for regularization.

    During training: randomly zeros elements with probability p
    During inference: scales outputs by (1-p) to maintain expected value

    This prevents overfitting by forcing the network to not rely on specific neurons.
    """

    def __init__(self, p=0.5):
        """
        Initialize dropout layer.

        TODO: Store dropout probability

        Args:
            p: Probability of zeroing each element (0.0 = no dropout, 1.0 = zero everything)

        EXAMPLE:
        >>> dropout = Dropout(0.5)  # Zero 50% of elements during training
        """
        ### BEGIN SOLUTION
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p
        ### END SOLUTION

    def forward(self, x, training=True):
        """
        Forward pass through dropout layer.

        TODO: Apply dropout during training, pass through during inference

        APPROACH:
        1. If not training, return input unchanged
        2. If training, create random mask with probability (1-p)
        3. Multiply input by mask and scale by 1/(1-p)
        4. Return result as new Tensor

        EXAMPLE:
        >>> dropout = Dropout(0.5)
        >>> x = Tensor([1, 2, 3, 4])
        >>> y_train = dropout.forward(x, training=True)   # Some elements zeroed
        >>> y_eval = dropout.forward(x, training=False)   # All elements preserved

        HINTS:
        - Use np.random.random() < keep_prob for mask
        - Scale by 1/(1-p) to maintain expected value
        - training=False should return input unchanged
        """
        ### BEGIN SOLUTION
        if not training or self.p == 0.0:
            # During inference or no dropout, pass through unchanged
            return x

        if self.p == 1.0:
            # Drop everything (preserve requires_grad for gradient flow)
            return Tensor(np.zeros_like(x.data), requires_grad=x.requires_grad if hasattr(x, 'requires_grad') else False)

        # During training, apply dropout
        keep_prob = 1.0 - self.p

        # Create random mask: True where we keep elements
        mask = np.random.random(x.data.shape) < keep_prob

        # Apply mask and scale using Tensor operations to preserve gradients!
        mask_tensor = Tensor(mask.astype(np.float32), requires_grad=False)  # Mask doesn't need gradients
        scale = Tensor(np.array(1.0 / keep_prob), requires_grad=False)
        
        # Use Tensor operations: x * mask * scale
        output = x * mask_tensor * scale
        return output
        ### END SOLUTION

    def __call__(self, x, training=True):
        """Allows the layer to be called like a function."""
        return self.forward(x, training)

    def parameters(self):
        """Dropout has no parameters."""
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"

# %% [markdown]
"""
### 🔬 Unit Test: Dropout Layer
This test validates our Dropout layer implementation works correctly.
**What we're testing**: Training vs inference behavior, probability scaling, randomness
**Why it matters**: Essential for preventing overfitting in neural networks
**Expected**: Correct masking during training, passthrough during inference
"""

# %% nbgrader={"grade": true, "grade_id": "test-dropout", "locked": true, "points": 10}
def test_unit_dropout_layer():
    """🔬 Test Dropout layer implementation."""
    print("🔬 Unit Test: Dropout Layer...")

    # Test dropout creation
    dropout = Dropout(0.5)
    assert dropout.p == 0.5

    # Test inference mode (should pass through unchanged)
    x = Tensor([1, 2, 3, 4])
    y_inference = dropout.forward(x, training=False)
    assert np.array_equal(x.data, y_inference.data), "Inference should pass through unchanged"

    # Test training mode with zero dropout (should pass through unchanged)
    dropout_zero = Dropout(0.0)
    y_zero = dropout_zero.forward(x, training=True)
    assert np.array_equal(x.data, y_zero.data), "Zero dropout should pass through unchanged"

    # Test training mode with full dropout (should zero everything)
    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x, training=True)
    assert np.allclose(y_full.data, 0), "Full dropout should zero everything"

    # Test training mode with partial dropout
    # Note: This is probabilistic, so we test statistical properties
    np.random.seed(42)  # For reproducible test
    x_large = Tensor(np.ones((1000,)))  # Large tensor for statistical significance
    y_train = dropout.forward(x_large, training=True)

    # Count non-zero elements (approximately 50% should survive)
    non_zero_count = np.count_nonzero(y_train.data)
    expected_survival = 1000 * 0.5
    # Allow 10% tolerance for randomness
    assert 0.4 * 1000 < non_zero_count < 0.6 * 1000, f"Expected ~500 survivors, got {non_zero_count}"

    # Test scaling (surviving elements should be scaled by 1/(1-p) = 2.0)
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0  # 1.0 / (1 - 0.5)
    assert np.allclose(surviving_values, expected_value), f"Surviving values should be {expected_value}"

    # Test no parameters
    params = dropout.parameters()
    assert len(params) == 0, "Dropout should have no parameters"

    # Test invalid probability
    try:
        Dropout(-0.1)
        assert False, "Should raise ValueError for negative probability"
    except ValueError:
        pass

    try:
        Dropout(1.1)
        assert False, "Should raise ValueError for probability > 1"
    except ValueError:
        pass

    print("✅ Dropout layer works correctly!")

if __name__ == "__main__":
    test_unit_dropout_layer()

# %% [markdown]
"""
## 4. Integration: Bringing It Together

Now that we've built both layer types, let's see how they work together to create a complete neural network architecture. We'll manually compose a realistic 3-layer MLP for MNIST digit classification.

### Network Architecture Visualization
```
MNIST Classification Network (3-Layer MLP):

    Input Layer          Hidden Layer 1        Hidden Layer 2        Output Layer
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     784         │    │      256        │    │      128        │    │       10        │
│   Pixels        │───▶│   Features      │───▶│   Features      │───▶│    Classes      │
│  (28×28 image)  │    │   + ReLU        │    │   + ReLU        │    │  (0-9 digits)   │
│                 │    │   + Dropout     │    │   + Dropout     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        ↓                       ↓                       ↓                       ↓
   "Raw pixels"          "Edge detectors"        "Shape detectors"        "Digit classifier"

Data Flow:
[32, 784] → Linear(784,256) → ReLU → Dropout(0.5) → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,10) → [32, 10]
```

### Parameter Count Analysis
```
Parameter Breakdown (Manual Layer Composition):
┌─────────────────────────────────────────────────────────────┐
│ layer1 = Linear(784 → 256)                               │
│   Weights: 784 × 256 = 200,704 params                      │
│   Bias:    256 params                                       │
│   Subtotal: 200,960 params                                  │
├─────────────────────────────────────────────────────────────┤
│ activation1 = ReLU(), dropout1 = Dropout(0.5)              │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ layer2 = Linear(256 → 128)                               │
│   Weights: 256 × 128 = 32,768 params                       │
│   Bias:    128 params                                       │
│   Subtotal: 32,896 params                                   │
├─────────────────────────────────────────────────────────────┤
│ activation2 = ReLU(), dropout2 = Dropout(0.3)              │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ layer3 = Linear(128 → 10)                                │
│   Weights: 128 × 10 = 1,280 params                         │
│   Bias:    10 params                                        │
│   Subtotal: 1,290 params                                    │
└─────────────────────────────────────────────────────────────┘
                    TOTAL: 235,146 parameters
                    Memory: ~940 KB (float32)
```
"""


# %% [markdown]
"""
## 5. Systems Analysis: Memory and Performance

Now let's analyze the systems characteristics of our layer implementations. Understanding memory usage and computational complexity helps us build efficient neural networks.

### Memory Analysis Overview
```
Layer Memory Components:
┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER MEMORY                         │
├─────────────────────────────────────────────────────────────┤
│ • Weights: Persistent, shared across batches               │
│ • Biases: Small but necessary for output shifting          │
│ • Total: Grows with network width and depth                │
├─────────────────────────────────────────────────────────────┤
│                   ACTIVATION MEMORY                         │
├─────────────────────────────────────────────────────────────┤
│ • Input tensors: batch_size × features × 4 bytes           │
│ • Output tensors: batch_size × features × 4 bytes          │
│ • Intermediate results during forward pass                  │
│ • Total: Grows with batch size and layer width             │
├─────────────────────────────────────────────────────────────┤
│                   TEMPORARY MEMORY                          │
├─────────────────────────────────────────────────────────────┤
│ • Dropout masks: batch_size × features × 1 byte            │
│ • Computation buffers for matrix operations                 │
│ • Total: Peak during forward/backward passes               │
└─────────────────────────────────────────────────────────────┘
```

### Computational Complexity Overview
```
Layer Operation Complexity:
┌─────────────────────────────────────────────────────────────┐
│ Linear Layer Forward Pass:                                  │
│   Matrix Multiply: O(batch × in_features × out_features)    │
│   Bias Addition: O(batch × out_features)                    │
│   Dominant: Matrix multiplication                           │
├─────────────────────────────────────────────────────────────┤
│ Multi-layer Forward Pass:                                   │
│   Sum of all layer complexities                             │
│   Memory: Peak of all intermediate activations              │
├─────────────────────────────────────────────────────────────┤
│ Dropout Forward Pass:                                        │
│   Mask Generation: O(elements)                              │
│   Element-wise Multiply: O(elements)                        │
│   Overhead: Minimal compared to linear layers               │
└─────────────────────────────────────────────────────────────┘
```
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-layer-memory", "solution": true}
def analyze_layer_memory():
    """📊 Analyze memory usage patterns in layer operations."""
    print("📊 Analyzing Layer Memory Usage...")

    # Test different layer sizes
    layer_configs = [
        (784, 256),   # MNIST → hidden
        (256, 256),   # Hidden → hidden
        (256, 10),    # Hidden → output
        (2048, 2048), # Large hidden
    ]

    print("\nLinear Layer Memory Analysis:")
    print("Configuration → Weight Memory → Bias Memory → Total Memory")

    for in_feat, out_feat in layer_configs:
        # Calculate memory usage
        weight_memory = in_feat * out_feat * 4  # 4 bytes per float32
        bias_memory = out_feat * 4
        total_memory = weight_memory + bias_memory

        print(f"({in_feat:4d}, {out_feat:4d}) → {weight_memory/1024:7.1f} KB → {bias_memory/1024:6.1f} KB → {total_memory/1024:7.1f} KB")

    # Analyze multi-layer memory scaling
    print("\n💡 Multi-layer Model Memory Scaling:")
    hidden_sizes = [128, 256, 512, 1024, 2048]

    for hidden_size in hidden_sizes:
        # 3-layer MLP: 784 → hidden → hidden/2 → 10
        layer1_params = 784 * hidden_size + hidden_size
        layer2_params = hidden_size * (hidden_size // 2) + (hidden_size // 2)
        layer3_params = (hidden_size // 2) * 10 + 10

        total_params = layer1_params + layer2_params + layer3_params
        memory_mb = total_params * 4 / (1024 * 1024)

        print(f"Hidden={hidden_size:4d}: {total_params:7,} params = {memory_mb:5.1f} MB")

# Analysis will be run in main block

# %% nbgrader={"grade": false, "grade_id": "analyze-layer-performance", "solution": true}
def analyze_layer_performance():
    """📊 Analyze computational complexity of layer operations."""
    print("📊 Analyzing Layer Computational Complexity...")

    # Test forward pass FLOPs
    batch_sizes = [1, 32, 128, 512]
    layer = Linear(784, 256)

    print("\nLinear Layer FLOPs Analysis:")
    print("Batch Size → Matrix Multiply FLOPs → Bias Add FLOPs → Total FLOPs")

    for batch_size in batch_sizes:
        # Matrix multiplication: (batch, in) @ (in, out) = batch * in * out FLOPs
        matmul_flops = batch_size * 784 * 256
        # Bias addition: batch * out FLOPs
        bias_flops = batch_size * 256
        total_flops = matmul_flops + bias_flops

        print(f"{batch_size:10d} → {matmul_flops:15,} → {bias_flops:13,} → {total_flops:11,}")

    print("\n💡 Key Insights:")
    print("🚀 Linear layer complexity: O(batch_size × in_features × out_features)")
    print("🚀 Memory grows linearly with batch size, quadratically with layer width")
    print("🚀 Dropout adds minimal computational overhead (element-wise operations)")

# Analysis will be run in main block

# %% [markdown]
# """
# # 🧪 Module Integration Test
#
# Final validation that everything works together correctly.
# """
#
# def import_previous_module(module_name: str, component_name: str):
#     import sys
#     import os
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..', module_name))
#     module = __import__(f"{module_name.split('_')[1]}_dev")
#     return getattr(module, component_name)

# %% nbgrader={"grade": true, "grade_id": "module-integration", "locked": true, "points": 20}
def test_module():
    """
    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_linear_layer()
    test_unit_dropout_layer()

    print("\nRunning integration scenarios...")

    # Test realistic neural network construction with manual composition
    print("🔬 Integration Test: Multi-layer Network...")

    # Import ReLU from Module 02 (already imported at top of file)
    # ReLU is available from: from tinytorch.core.activations import ReLU

    # Build individual layers for manual composition
    layer1 = Linear(784, 128)
    activation1 = ReLU()
    dropout1 = Dropout(0.5)
    layer2 = Linear(128, 64)
    activation2 = ReLU()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64, 10)

    # Test end-to-end forward pass with manual composition
    batch_size = 16
    x = Tensor(np.random.randn(batch_size, 784))

    # Manual forward pass
    x = layer1.forward(x)
    x = activation1.forward(x)
    x = dropout1.forward(x)
    x = layer2.forward(x)
    x = activation2.forward(x)
    x = dropout2.forward(x)
    output = layer3.forward(x)

    assert output.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {output.shape}"

    # Test parameter counting from individual layers
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6  # 3 weights + 3 biases from 3 Linear layers
    assert len(all_params) == expected_params, f"Expected {expected_params} parameters, got {len(all_params)}"

    # Test all parameters have requires_grad=True
    for param in all_params:
        assert param.requires_grad == True, "All parameters should have requires_grad=True"

    # Test individual layer functionality
    test_x = Tensor(np.random.randn(4, 784))
    # Test dropout in training vs inference
    dropout_test = Dropout(0.5)
    train_output = dropout_test.forward(test_x, training=True)
    infer_output = dropout_test.forward(test_x, training=False)
    assert np.array_equal(test_x.data, infer_output.data), "Inference mode should pass through unchanged"

    print("✅ Multi-layer network integration works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 03_layers")

# Run comprehensive module test
if __name__ == "__main__":
    test_module()


# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Layers

Congratulations! You've built the fundamental building blocks that make neural networks possible!

### Key Accomplishments
- Built Linear layers with proper Xavier initialization and parameter management
- Created Dropout layers for regularization with training/inference mode handling
- Demonstrated manual layer composition for building neural networks
- Analyzed memory scaling and computational complexity of layer operations
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your layer implementation enables building complete neural networks! The Linear layer provides learnable transformations, manual composition chains them together, and Dropout prevents overfitting.

Export with: `tito module complete 03_layers`

**Next**: Module 04 will add loss functions (CrossEntropyLoss, MSELoss) that measure how wrong your model is - the foundation for learning!
"""
