# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
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
**You'll Build**: Linear layers, Sequential composition, and Dropout regularization
**You'll Enable**: Multi-layer neural networks, trainable parameters, and forward passes

**Connection Map**:
```
Tensor → Activations → Layers → Networks
(data)   (intelligence) (building blocks) (architectures)
```

## Learning Objectives
By the end of this module, you will:
1. Implement Linear layers with proper weight initialization
2. Build Sequential containers for chaining operations
3. Add Dropout for regularization during training
4. Understand parameter management and counting
5. Test layer composition and shape preservation

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/03_layers/layers_dev.py
**Building Side:** Code exports to tinytorch.core.layers

```python
# Final package structure:
from tinytorch.core.layers import Linear, Sequential, Dropout  # This module
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

import numpy as np
import sys
import os

# Import the proper Tensor class from Module 01
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

# %% [markdown]
"""
## 1. Introduction: What are Neural Network Layers?

Neural network layers are the fundamental building blocks that transform data as it flows through a network. Each layer performs a specific computation:

- **Linear layers** apply learned transformations: `y = xW + b`
- **Sequential containers** chain multiple operations together
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
Sequential([
    Linear(784, 256),  # 200,960 params
    ReLU(),            # 0 params
    Linear(256, 10)    # 2,570 params
])                     # Total: 203,530 params
```

Memory usage: 4 bytes/param × 203,530 = ~814KB for weights alone
"""

# %% [markdown]
"""
## 3. Implementation: Building Layer Foundation

Let's build our layer system step by step. We'll implement three essential layer types:

1. **Linear Layer** - The workhorse of neural networks
2. **Sequential Container** - Chains layers together
3. **Dropout Layer** - Prevents overfitting

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

# Test will be run in main block

# %% [markdown]
"""
### 🔗 Sequential Container - Chaining Operations Together

The Sequential container is like a assembly line for data processing. It takes multiple layers and applies them one after another, passing the output of each layer as input to the next.

### Why Sequential Matters

Most neural networks are **sequential compositions** of simpler operations. Instead of manually calling each layer, Sequential automates the process and manages the data flow.

### Architecture Visualization
```
Sequential Network Flow:

Input Data          Layer 1            Layer 2            Layer 3         Output
[32, 784]    →    Linear(784,256)  →  ReLU()  →      Linear(256,10)   →  [32, 10]
  MNIST            Feature           Non-linear        Classification      Class
  Images          Extraction         Activation         Layer             Scores
    ↓                 ↓                 ↓                 ↓                 ↓
"What do I see?" → "Extract edges" → "Activate patterns" → "Classify" → "It's a 7!"
```

### Sequential vs Manual Chaining
```
# Manual approach (tedious and error-prone):
def forward(x):
    x = layer1.forward(x)
    x = layer2.forward(x)
    x = layer3.forward(x)
    return x

# Sequential approach (clean and automatic):
model = Sequential(layer1, layer2, layer3)
output = model.forward(x)
```

### Parameter Management
```
Sequential Parameter Collection:
┌─────────────────┐
│ Layer 1: Linear │ → params: [weight1, bias1]
├─────────────────┤
│ Layer 2: ReLU   │ → params: [] (no learnable params)
├─────────────────┤
│ Layer 3: Linear │ → params: [weight3, bias3]
└─────────────────┘
         ↓
  model.parameters() = [weight1, bias1, weight3, bias3]
```
"""

# %% nbgrader={"grade": false, "grade_id": "sequential-container", "solution": true}
class Sequential:
    """
    Sequential container for chaining multiple layers.

    Applies layers in order: output = layer_n(...layer_2(layer_1(input)))
    This is the most common way to build neural networks.
    """

    def __init__(self, *layers):
        """
        Initialize sequential container with list of layers.

        TODO: Store layers for sequential application

        EXAMPLE:
        >>> model = Sequential(
        ...     Linear(784, 128),
        ...     ReLU(),  # Would be from Module 02
        ...     Linear(128, 10)
        ... )
        """
        ### BEGIN SOLUTION
        self.layers = list(layers)
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through all layers in sequence.

        TODO: Apply each layer to the output of the previous layer

        APPROACH:
        1. Start with input x
        2. Apply each layer in order
        3. Return final output

        EXAMPLE:
        >>> x = Tensor(np.random.randn(32, 784))
        >>> output = model.forward(x)  # Goes through Linear -> ReLU -> Linear
        """
        ### BEGIN SOLUTION
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
        ### END SOLUTION

    def parameters(self):
        """
        Return all parameters from all layers.

        TODO: Collect parameters from all layers that have them

        APPROACH:
        1. Iterate through layers
        2. Check if layer has parameters() method
        3. Collect all parameters into single list
        """
        ### BEGIN SOLUTION
        all_params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                all_params.extend(layer.parameters())
        return all_params
        ### END SOLUTION

    def __len__(self):
        """Return number of layers."""
        return len(self.layers)

    def __getitem__(self, idx):
        """Access layer by index."""
        return self.layers[idx]

    def __repr__(self):
        """String representation showing all layers."""
        layer_strs = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return f"Sequential(\n" + "\n".join(layer_strs) + "\n)"

# %% [markdown]
"""
### 🔬 Unit Test: Sequential Container
This test validates our Sequential container works correctly.
**What we're testing**: Layer chaining, parameter collection, forward pass
**Why it matters**: Enables building multi-layer neural networks
**Expected**: Correct data flow, parameter aggregation, shape preservation
"""

# %% nbgrader={"grade": true, "grade_id": "test-sequential", "locked": true, "points": 15}
def test_unit_sequential_container():
    """🔬 Test Sequential container implementation."""
    print("🔬 Unit Test: Sequential Container...")

    # Create simple mock activation for testing
    class MockReLU:
        def forward(self, x):
            return Tensor(np.maximum(0, x.data))
        def __repr__(self):
            return "ReLU()"

    # Test sequential creation
    model = Sequential(
        Linear(784, 128),
        MockReLU(),
        Linear(128, 10)
    )

    assert len(model) == 3
    assert isinstance(model[0], Linear)
    assert isinstance(model[1], MockReLU)
    assert isinstance(model[2], Linear)

    # Test forward pass
    x = Tensor(np.random.randn(32, 784))
    output = model.forward(x)
    assert output.shape == (32, 10), f"Expected shape (32, 10), got {output.shape}"

    # Test parameter collection (should have params from Linear layers only)
    params = model.parameters()
    expected_params = 4  # 2 weights + 2 biases from 2 Linear layers
    assert len(params) == expected_params, f"Expected {expected_params} parameters, got {len(params)}"

    # Verify parameters are from correct layers
    layer1_params = model[0].parameters()
    layer3_params = model[2].parameters()
    expected_param_count = len(layer1_params) + len(layer3_params)
    assert len(params) == expected_param_count

    print("✅ Sequential container works correctly!")

# Test will be run in main block

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
            # Drop everything
            return Tensor(np.zeros_like(x.data))

        # During training, apply dropout
        keep_prob = 1.0 - self.p

        # Create random mask: True where we keep elements
        mask = np.random.random(x.data.shape) < keep_prob

        # Apply mask and scale to maintain expected value
        output_data = (x.data * mask) / keep_prob
        return Tensor(output_data)
        ### END SOLUTION

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

# Test will be run in main block

# %% [markdown]
"""
## 4. Integration: Bringing It Together

Now that we've built all three layer types, let's see how they work together to create a complete neural network architecture. We'll build a realistic 3-layer MLP for MNIST digit classification.

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
Parameter Breakdown:
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Linear(784 → 256)                                 │
│   Weights: 784 × 256 = 200,704 params                      │
│   Bias:    256 params                                       │
│   Subtotal: 200,960 params                                  │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: ReLU + Dropout                                     │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Linear(256 → 128)                                 │
│   Weights: 256 × 128 = 32,768 params                       │
│   Bias:    128 params                                       │
│   Subtotal: 32,896 params                                   │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: ReLU + Dropout                                     │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Linear(128 → 10)                                  │
│   Weights: 128 × 10 = 1,280 params                         │
│   Bias:    10 params                                        │
│   Subtotal: 1,290 params                                    │
└─────────────────────────────────────────────────────────────┘
                    TOTAL: 235,146 parameters
                    Memory: ~940 KB (float32)
```
"""

# %% nbgrader={"grade": false, "grade_id": "integration-demo", "solution": true}
def demonstrate_layer_integration():
    """
    Demonstrate layers working together in a realistic neural network.

    This simulates a 3-layer MLP for MNIST classification:
    784 → 256 → 128 → 10
    """
    print("🔗 Integration Demo: 3-Layer MLP")
    print("Architecture: 784 → 256 → 128 → 10 (MNIST classifier)")

    # Create mock activation for demonstration
    class MockReLU:
        def forward(self, x):
            return Tensor(np.maximum(0, x.data))
        def parameters(self):
            return []
        def __repr__(self):
            return "ReLU()"

    # Build the network
    model = Sequential(
        Linear(784, 256),   # Input layer
        MockReLU(),         # Activation
        Dropout(0.5),       # Regularization
        Linear(256, 128),   # Hidden layer
        MockReLU(),         # Activation
        Dropout(0.3),       # Less aggressive dropout
        Linear(128, 10)     # Output layer
    )

    print(f"\nModel architecture:")
    print(model)

    # Test forward pass with MNIST-like data
    batch_size = 32
    x = Tensor(np.random.randn(batch_size, 784))
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model.forward(x)
    print(f"Output shape: {output.shape}")

    # Count parameters
    params = model.parameters()
    total_params = sum(p.size for p in params)
    print(f"\nTotal parameters: {total_params:,}")

    # Break down by layer
    print("\nParameter breakdown:")
    layer1_params = sum(p.size for p in model[0].parameters())  # Linear(784, 256)
    layer2_params = sum(p.size for p in model[3].parameters())  # Linear(256, 128)
    layer3_params = sum(p.size for p in model[6].parameters())  # Linear(128, 10)

    print(f"  Layer 1 (784→256): {layer1_params:,} params")
    print(f"  Layer 2 (256→128): {layer2_params:,} params")
    print(f"  Layer 3 (128→10):  {layer3_params:,} params")

    # Memory estimate
    memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"\nMemory usage: ~{memory_mb:.1f} MB (weights only)")

    return model, output

# Integration demo will be run in main block

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
│ Sequential Forward Pass:                                     │
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

    # Analyze Sequential memory scaling
    print("\n💡 Sequential Model Memory Scaling:")
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
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

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
    test_unit_sequential_container()
    test_unit_dropout_layer()

    print("\nRunning integration scenarios...")

    # Test realistic neural network construction
    print("🔬 Integration Test: Multi-layer Network...")

    # Create mock activation for integration test
    class MockActivation:
        def forward(self, x):
            return Tensor(np.maximum(0, x.data))  # ReLU-like
        def parameters(self):
            return []
        def __repr__(self):
            return "MockActivation()"

    # Build a complete 3-layer network
    network = Sequential(
        Linear(784, 128),
        MockActivation(),
        Dropout(0.5),
        Linear(128, 64),
        MockActivation(),
        Dropout(0.3),
        Linear(64, 10)
    )

    # Test end-to-end forward pass
    batch_size = 16
    x = Tensor(np.random.randn(batch_size, 784))

    # Forward pass
    output = network.forward(x)
    assert output.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {output.shape}"

    # Test parameter counting
    params = network.parameters()
    expected_layers_with_params = 3  # Three Linear layers
    linear_layers = [layer for layer in network.layers if isinstance(layer, Linear)]
    total_expected_params = sum(len(layer.parameters()) for layer in linear_layers)
    assert len(params) == total_expected_params, f"Expected {total_expected_params} parameters, got {len(params)}"

    # Test all parameters have requires_grad=True
    for param in params:
        assert param.requires_grad == True, "All parameters should have requires_grad=True"

    # Test dropout in inference mode
    output_train = network.forward(x)  # Default training=True in our simplified version
    # Note: In full implementation, we'd test training vs inference mode

    print("✅ Multi-layer network integration works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 03_layers")

# Module test will be run in main block

# %%
if __name__ == "__main__":
    print("🚀 Running Layers module...")

    # Run all unit tests
    test_unit_linear_layer()
    test_unit_sequential_container()
    test_unit_dropout_layer()

    # Run integration demo
    model, output = demonstrate_layer_integration()

    # Run systems analysis
    analyze_layer_memory()
    analyze_layer_performance()

    # Run final module test
    test_module()

    print("✅ Module validation complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Layer Architecture

Now that you've built a complete layer system, let's reflect on the systems implications of your implementation.
"""

# %% nbgrader={"grade": false, "grade_id": "systems-q1", "solution": true}
# %% [markdown]
"""
### Question 1: Parameter Memory Scaling
You implemented Linear layers with weight matrices that scale as in_features × out_features.

**a) Memory Growth**: For a 4-layer MLP with architecture [784, 512, 256, 128, 10]:
- Layer 1: 784 × 512 + 512 = _____ parameters
- Layer 2: 512 × 256 + 256 = _____ parameters
- Layer 3: 256 × 128 + 128 = _____ parameters
- Layer 4: 128 × 10 + 10 = _____ parameters
- Total memory at 4 bytes/param: _____ MB

**b) Width vs Depth Trade-off**: Compare memory usage:
- Wide: [784, 1024, 10] vs Deep: [784, 256, 256, 256, 10]
- Which uses more memory? Why might you choose one over the other?

*Think about: representational capacity, gradient flow, overfitting risk*
"""

# %% nbgrader={"grade": false, "grade_id": "systems-q2", "solution": true}
# %% [markdown]
"""
### Question 2: Dropout Implementation Choices
Your Dropout layer uses per-element random masks during training.

**a) Memory Pattern**: When applying dropout to a (1000, 512) tensor:
- Original tensor: 1000 × 512 × 4 bytes = _____ MB
- Dropout mask: 1000 × 512 × 1 byte = _____ KB
- Output tensor: 1000 × 512 × 4 bytes = _____ MB
- Peak memory during forward pass: _____ MB

**b) Alternative Implementations**: What are the trade-offs of:
- In-place dropout: `x.data *= mask` (modify original)
- Structured dropout: Drop entire neurons instead of elements
- Deterministic dropout: Use fixed patterns instead of random

*Consider: memory usage, randomness benefits, gradient flow*
"""

# %% nbgrader={"grade": false, "grade_id": "systems-q3", "solution": true}
# %% [markdown]
"""
### Question 3: Sequential Container Design
Your Sequential container applies layers one after another in a simple loop.

**a) Memory Efficiency**: In your implementation, when computing Sequential([Layer1, Layer2, Layer3]).forward(x):
- How many intermediate tensors exist simultaneously in memory?
- What's the peak memory usage for a 4-layer network?
- How could you reduce memory usage? What would you sacrifice?

**b) Computational Graph**: Each layer creates new Tensor objects. For gradient computation:
- How does this affect the computation graph in Module 05?
- What's the memory cost of storing all intermediate activations?
- When might you want to trade computation for memory?

*Think about: activation checkpointing, in-place operations, gradient accumulation*
"""

# %% nbgrader={"grade": false, "grade_id": "systems-q4", "solution": true}
# %% [markdown]
"""
### Question 4: Xavier Initialization Impact
Your Linear layer uses Xavier initialization with scale = sqrt(1/in_features).

**a) Scaling Behavior**: For layers with different input sizes:
- Linear(784, 256): scale = sqrt(1/784) ≈ _____
- Linear(64, 256): scale = sqrt(1/64) ≈ _____
- Which layer has larger initial weights? Why does this matter for training?

**b) Alternative Schemes**: Compare initialization strategies:
- Xavier: sqrt(1/in_features) - good for Sigmoid/Tanh
- He: sqrt(2/in_features) - good for ReLU
- LeCun: sqrt(1/in_features) - good for SELU
- Why do different activations need different initialization?

*Think about: gradient magnitudes, activation ranges, vanishing/exploding gradients*
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Layers

Congratulations! You've built the fundamental building blocks that make neural networks possible!

### Key Accomplishments
- Built Linear layers with proper Xavier initialization and parameter management
- Implemented Sequential containers for chaining operations with automatic parameter collection
- Created Dropout layers for regularization with training/inference mode handling
- Analyzed memory scaling and computational complexity of layer operations
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your layer implementation enables building complete neural networks! The Linear layer provides learnable transformations, Sequential chains them together, and Dropout prevents overfitting.

Export with: `tito module complete 03_layers`

**Next**: Module 04 will add loss functions (CrossEntropyLoss, MSELoss) that measure how wrong your model is - the foundation for learning!
"""