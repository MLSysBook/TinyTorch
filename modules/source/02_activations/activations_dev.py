# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Module 2: Activations - Nonlinearity in Neural Networks

Welcome to the Activations module! This is where neural networks get their power through nonlinearity.

## Learning Goals
- Understand why activation functions are essential for neural networks
- Implement the four most important activation functions: ReLU, Sigmoid, Tanh, and Softmax
- Visualize how activations transform data and enable complex learning
- See how activations work with layers to build powerful networks
- Master the NBGrader workflow with comprehensive testing

## Build → Use → Understand
1. **Build**: Activation functions that add nonlinearity
2. **Use**: Transform tensors and see immediate results
3. **Understand**: How nonlinearity enables complex pattern learning
"""

# %% nbgrader={"grade": false, "grade_id": "activations-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.activations

#| export
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Union, List

# Import our Tensor class - try from package first, then from local module
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local tensor module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# %% nbgrader={"grade": false, "grade_id": "activations-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    # Check multiple conditions that indicate we're in test mode
    is_pytest = (
        'pytest' in sys.modules or
        'test' in sys.argv or
        os.environ.get('PYTEST_CURRENT_TEST') is not None or
        any('test' in arg for arg in sys.argv) or
        any('pytest' in arg for arg in sys.argv)
    )
    
    # Show plots in development mode (when not in test mode)
    return not is_pytest

# %% nbgrader={"grade": false, "grade_id": "activations-visualization", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
#| export
def visualize_activation_function(activation_fn, name: str, x_range: tuple = (-5, 5), num_points: int = 100):
    """Visualize an activation function's behavior"""
    if not _should_show_plots():
        return
        
    try:
        
        # Generate input values
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        
        # Apply activation function
        y_vals = []
        for x in x_vals:
            input_tensor = Tensor([[x]])
            output = activation_fn(input_tensor)
            y_vals.append(output.data.item())
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'{name} Activation')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel(f'{name}(x)')
        plt.title(f'{name} Activation Function')
        plt.legend()
        plt.show()
        
    except ImportError:
        print("   📊 Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")

def visualize_activation_on_data(activation_fn, name: str, data: Tensor):
    """Show activation function applied to sample data"""
    if not _should_show_plots():
        return
        
    try:
        output = activation_fn(data)
        print(f"   📊 {name} Example:")
        print(f"      Input:  {data.data.flatten()}")
        print(f"      Output: {output.data.flatten()}")
        print(f"      Range:  [{output.data.min():.3f}, {output.data.max():.3f}]")
        
    except Exception as e:
        print(f"   ⚠️  Data visualization error: {e}")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/02_activations/activations_dev.py`  
**Building Side:** Code exports to `tinytorch.core.activations`

```python
# Final package structure:
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax  # All activations together!
from tinytorch.core.tensor import Tensor  # The foundation
from tinytorch.core.layers import Dense, Conv2D  # Coming next!
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.functional`
- **Consistency:** All activation functions live together in `core.activations`
- **Integration:** Works seamlessly with tensors and layers
"""

# %% [markdown]
"""
## 🧠 The Mathematical Foundation of Nonlinearity

### The Universal Approximation Theorem
**Key Insight:** Neural networks with nonlinear activation functions can approximate any continuous function!

```
Without activation: f(x) = W₃(W₂(W₁x + b₁) + b₂) + b₃ = Wx + b (still linear!)
With activation: f(x) = W₃σ(W₂σ(W₁x + b₁) + b₂) + b₃ (nonlinear!)
```

### Why Nonlinearity is Critical
- **Linear Limitations**: Without activations, any deep network collapses to a single linear transformation
- **Feature Learning**: Nonlinear functions create complex decision boundaries
- **Representation Power**: Each layer can learn different levels of abstraction
- **Biological Inspiration**: Neurons fire (activate) only above certain thresholds

### Mathematical Properties We Care About
- **Differentiability**: For gradient-based optimization
- **Computational Efficiency**: Fast forward and backward passes
- **Numerical Stability**: Avoiding vanishing/exploding gradients
- **Sparsity**: Some activations (like ReLU) produce sparse representations

### Connection to Real ML Systems
Every major framework has these same activations:
- **PyTorch**: `torch.nn.ReLU()`, `torch.nn.Sigmoid()`, etc.
- **TensorFlow**: `tf.nn.relu()`, `tf.nn.sigmoid()`, etc.
- **JAX**: `jax.nn.relu()`, `jax.nn.sigmoid()`, etc.
- **TinyTorch**: `tinytorch.core.activations.ReLU()` (what we're building!)
"""

# %% [markdown]
"""
## Step 1: What is an Activation Function?

### Definition
An **activation function** is a mathematical function that adds nonlinearity to neural networks. It transforms the output of a layer before passing it to the next layer.

### Why Activation Functions Matter
**Without activation functions, neural networks are just linear transformations!**

```
Linear → Linear → Linear = Still Linear
```

No matter how many layers you stack, without activation functions, you can only learn linear relationships. Activation functions introduce the nonlinearity that allows neural networks to:
- Learn complex patterns
- Approximate any continuous function
- Solve non-linear problems

### Visual Analogy
Think of activation functions as **decision makers** at each neuron:
- **ReLU**: "If positive, pass it through; if negative, block it"
- **Sigmoid**: "Squash everything between 0 and 1"
- **Tanh**: "Squash everything between -1 and 1"
- **Softmax**: "Convert to probabilities that sum to 1"

### Connection to Previous Modules
In Module 1 (Tensor), we learned how to store and manipulate data. Now we add the nonlinear functions that make neural networks powerful.
"""

# %% [markdown]
"""
## Step 2: ReLU - The Workhorse of Deep Learning

### What is ReLU?
**ReLU (Rectified Linear Unit)** is the most popular activation function in deep learning.

**Mathematical Definition:**
```
f(x) = max(0, x)
```

**In Plain English:**
- If input is positive → pass it through unchanged
- If input is negative → output zero

### Why ReLU is Popular
1. **Simple**: Easy to compute and understand
2. **Fast**: No expensive operations (no exponentials)
3. **Sparse**: Outputs many zeros, creating sparse representations
4. **Gradient-friendly**: Gradient is either 0 or 1 (no vanishing gradient for positive inputs)

### Real-World Analogy
ReLU is like a **one-way valve** - it only lets positive "pressure" through, blocking negative values completely.

### When to Use ReLU
- **Hidden layers** in most neural networks (90% of cases)
- **Convolutional layers** in image processing (CNNs)
- **When you want sparse activations** (many zeros)
- **Deep networks** (doesn't suffer from vanishing gradients)

### Real-World Applications
- **Image Classification**: ResNet, VGG, AlexNet all use ReLU
- **Object Detection**: YOLO, R-CNN use ReLU in backbone networks
- **Natural Language Processing**: Transformer models use ReLU in feedforward layers
- **Recommendation Systems**: Deep collaborative filtering with ReLU

### Mathematical Properties
- **Derivative**: f'(x) = 1 if x > 0, else 0
- **Range**: [0, ∞)
- **Sparsity**: Outputs exactly 0 for negative inputs
- **Computational Cost**: O(1) - just a max operation
"""

# %% nbgrader={"grade": false, "grade_id": "relu-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class ReLU:
    """
    ReLU Activation Function: f(x) = max(0, x)
    
    The most popular activation function in deep learning.
    Simple, fast, and effective for most applications.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU activation: f(x) = max(0, x)
        
        TODO: Implement ReLU activation
        
        APPROACH:
        1. For each element in the input tensor, apply max(0, element)
        2. Return a new Tensor with the results
        
        EXAMPLE:
        Input: Tensor([[-1, 0, 1, 2, -3]])
        Expected: Tensor([[0, 0, 1, 2, 0]])
        
        HINTS:
        - Use np.maximum(0, x.data) for element-wise max
        - Remember to return a new Tensor object
        - The shape should remain the same as input
        """
        ### BEGIN SOLUTION
        result = np.maximum(0, x.data)
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable: relu(x) instead of relu.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: ReLU Activation

Let's test your ReLU implementation right away! This gives you immediate feedback on whether your activation function works correctly.

**This is a unit test** - it tests one specific activation function (ReLU) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test ReLU activation immediately after implementation
print("🔬 Unit Test: ReLU Activation...")

# Create ReLU instance
relu = ReLU()

# Test with mixed positive/negative values
try:
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = relu(test_input)
    expected = np.array([[0, 0, 0, 1, 2]])
    
    assert np.array_equal(result.data, expected), f"ReLU failed: expected {expected}, got {result.data}"
    print(f"✅ ReLU test: input {test_input.data} → output {result.data}")
    
    # Test that negative values become zero
    assert np.all(result.data >= 0), "ReLU should make all negative values zero"
    print("✅ ReLU correctly zeros negative values")
    
    # Test that positive values remain unchanged
    positive_input = Tensor([[1, 2, 3, 4, 5]])
    positive_result = relu(positive_input)
    assert np.array_equal(positive_result.data, positive_input.data), "ReLU should preserve positive values"
    print("✅ ReLU preserves positive values")
    
except Exception as e:
    print(f"❌ ReLU test failed: {e}")
    raise

# Show visual example
print("🎯 ReLU behavior:")
print("   Negative → 0 (blocked)")
print("   Zero → 0 (blocked)")  
print("   Positive → unchanged (passed through)")
print("📈 Progress: ReLU ✓")

# %% [markdown]
"""
## Step 3: Sigmoid - The Smooth Squasher

### What is Sigmoid?
**Sigmoid** is a smooth S-shaped function that squashes inputs to the range (0, 1).

**Mathematical Definition:**
```
f(x) = 1 / (1 + e^(-x))
```

**Properties:**
- **Range**: (0, 1) - never exactly 0 or 1
- **Smooth**: Differentiable everywhere
- **Monotonic**: Always increasing
- **Centered**: Around 0.5

### Why Sigmoid is Useful
1. **Probabilistic**: Output can be interpreted as probabilities
2. **Bounded**: Output is always between 0 and 1
3. **Smooth**: Good for gradient-based optimization
4. **Historical**: Was the standard before ReLU

### Real-World Analogy
Sigmoid is like a **soft switch** - it gradually turns on as input increases, unlike ReLU's hard cutoff.

### Real-World Applications
- **Binary Classification**: Final layer for yes/no decisions (spam detection, medical diagnosis)
- **Logistic Regression**: The classic ML algorithm uses sigmoid
- **Attention Mechanisms**: Gating mechanisms in LSTM/GRU
- **Probability Estimation**: When you need outputs between 0 and 1

### Mathematical Properties
- **Derivative**: f'(x) = f(x)(1 - f(x)) - elegant and efficient!
- **Range**: (0, 1) - never exactly 0 or 1
- **Symmetry**: Sigmoid(0) = 0.5 (centered)
- **Saturation**: Gradients approach 0 for large |x| (vanishing gradient problem)

### When to Use Sigmoid
- **Binary classification** (output layer)
- **Gates** in LSTM/GRU networks
- **When you need probabilistic outputs**
- **Avoid in deep networks** (vanishing gradients)
"""

# %% nbgrader={"grade": false, "grade_id": "sigmoid-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Sigmoid:
    """
    Sigmoid Activation Function: f(x) = 1 / (1 + e^(-x))
    
    Smooth S-shaped function that squashes inputs to (0, 1).
    Useful for binary classification and probabilistic outputs.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Sigmoid activation: f(x) = 1 / (1 + e^(-x))
        
        TODO: Implement Sigmoid activation with numerical stability
        
        APPROACH:
        1. Clip input values to prevent overflow (e.g., between -500 and 500)
        2. Apply the sigmoid formula: 1 / (1 + exp(-x))
        3. Return a new Tensor with the results
        
        EXAMPLE:
        Input: Tensor([[-2, 0, 2]])
        Expected: Tensor([[0.119, 0.5, 0.881]]) (approximately)
        
        HINTS:
        - Use np.clip(x.data, -500, 500) for numerical stability
        - Use np.exp() for the exponential function
        - Be careful with very large/small inputs to avoid overflow
        """
        ### BEGIN SOLUTION
        # Clip for numerical stability
        clipped = np.clip(x.data, -500, 500)
        result = 1 / (1 + np.exp(-clipped))
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable: sigmoid(x) instead of sigmoid.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Sigmoid Activation

Let's test your Sigmoid implementation! This should squash all values to the range (0, 1).

**This is a unit test** - it tests one specific activation function (Sigmoid) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-sigmoid-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test Sigmoid activation immediately after implementation
print("🔬 Unit Test: Sigmoid Activation...")

# Create Sigmoid instance
sigmoid = Sigmoid()

# Test with various inputs
try:
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = sigmoid(test_input)
    
    # Check that all outputs are between 0 and 1
    assert np.all(result.data > 0), "Sigmoid outputs should be > 0"
    assert np.all(result.data < 1), "Sigmoid outputs should be < 1"
    print(f"✅ Sigmoid test: input {test_input.data} → output {result.data}")
    
    # Test specific values
    zero_input = Tensor([[0]])
    zero_result = sigmoid(zero_input)
    assert np.allclose(zero_result.data, 0.5, atol=1e-6), f"Sigmoid(0) should be 0.5, got {zero_result.data}"
    print("✅ Sigmoid(0) = 0.5 (correct)")
    
    # Test that it's monotonic (larger inputs give larger outputs)
    small_input = Tensor([[-1]])
    large_input = Tensor([[1]])
    small_result = sigmoid(small_input)
    large_result = sigmoid(large_input)
    assert small_result.data < large_result.data, "Sigmoid should be monotonic"
    print("✅ Sigmoid is monotonic (increasing)")
    
except Exception as e:
    print(f"❌ Sigmoid test failed: {e}")
    raise

# Show visual example
print("🎯 Sigmoid behavior:")
print("   Large negative → approaches 0")
print("   Zero → 0.5")
print("   Large positive → approaches 1")
print("📈 Progress: ReLU ✓, Sigmoid ✓")

# %% [markdown]
"""
## Step 4: Tanh - The Zero-Centered Squasher

### What is Tanh?
**Tanh (Hyperbolic Tangent)** is similar to Sigmoid but centered around zero.

**Mathematical Definition:**
```
f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Properties:**
- **Range**: (-1, 1) - symmetric around zero
- **Zero-centered**: Output averages to zero
- **Smooth**: Differentiable everywhere
- **Stronger gradients**: Than sigmoid in some regions

### Why Tanh is Useful
1. **Zero-centered**: Better for training (gradients don't all have same sign)
2. **Symmetric**: Treats positive and negative inputs equally
3. **Stronger gradients**: Can help with training dynamics
4. **Bounded**: Output is always between -1 and 1

### Real-World Analogy
Tanh is like a **balanced scale** - it can tip positive or negative, with zero as the neutral point.

### When to Use Tanh
- **Hidden layers** (alternative to ReLU)
- **RNNs** (traditional choice)
- **When you need zero-centered outputs**
"""

# %% nbgrader={"grade": false, "grade_id": "tanh-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Tanh:
    """
    Tanh Activation Function: f(x) = tanh(x)
    
    Zero-centered S-shaped function that squashes inputs to (-1, 1).
    Better than sigmoid for hidden layers due to zero-centered outputs.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh activation: f(x) = tanh(x)
        
        TODO: Implement Tanh activation
        
        APPROACH:
        1. Use NumPy's tanh function for numerical stability
        2. Apply to the tensor data
        3. Return a new Tensor with the results
        
        EXAMPLE:
        Input: Tensor([[-2, 0, 2]])
        Expected: Tensor([[-0.964, 0.0, 0.964]]) (approximately)
        
        HINTS:
        - Use np.tanh(x.data) - NumPy handles the math
        - Much simpler than implementing the formula manually
        - NumPy's tanh is numerically stable
        """
        ### BEGIN SOLUTION
        result = np.tanh(x.data)
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable: tanh(x) instead of tanh.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Tanh Activation

Let's test your Tanh implementation! This should squash all values to the range (-1, 1) and be zero-centered.

**This is a unit test** - it tests one specific activation function (Tanh) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-tanh-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test Tanh activation immediately after implementation
print("🔬 Unit Test: Tanh Activation...")

# Create Tanh instance
tanh = Tanh()

# Test with various inputs
try:
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = tanh(test_input)
    
    # Check that all outputs are between -1 and 1
    assert np.all(result.data > -1), "Tanh outputs should be > -1"
    assert np.all(result.data < 1), "Tanh outputs should be < 1"
    print(f"✅ Tanh test: input {test_input.data} → output {result.data}")
    
    # Test specific values
    zero_input = Tensor([[0]])
    zero_result = tanh(zero_input)
    assert np.allclose(zero_result.data, 0.0, atol=1e-6), f"Tanh(0) should be 0.0, got {zero_result.data}"
    print("✅ Tanh(0) = 0.0 (zero-centered)")
    
    # Test symmetry: tanh(-x) = -tanh(x)
    pos_input = Tensor([[1]])
    neg_input = Tensor([[-1]])
    pos_result = tanh(pos_input)
    neg_result = tanh(neg_input)
    assert np.allclose(pos_result.data, -neg_result.data, atol=1e-6), "Tanh should be symmetric"
    print("✅ Tanh is symmetric: tanh(-x) = -tanh(x)")
    
except Exception as e:
    print(f"❌ Tanh test failed: {e}")
    raise

# Show visual example
print("🎯 Tanh behavior:")
print("   Large negative → approaches -1")
print("   Zero → 0.0 (zero-centered)")
print("   Large positive → approaches 1")
print("📈 Progress: ReLU ✓, Sigmoid ✓, Tanh ✓")

# %% [markdown]
"""
## Step 5: Softmax - The Probability Converter

### What is Softmax?
**Softmax** converts a vector of numbers into a probability distribution.

**Mathematical Definition:**
```
f(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
```

**Properties:**
- **Probabilities**: All outputs sum to 1
- **Non-negative**: All outputs are ≥ 0
- **Differentiable**: Smooth everywhere
- **Competitive**: Amplifies differences between inputs

### Why Softmax is Essential
1. **Multi-class classification**: Converts logits to probabilities
2. **Attention mechanisms**: Focuses on important elements
3. **Interpretable**: Output can be understood as confidence
4. **Competitive**: Emphasizes the largest input

### Real-World Analogy
Softmax is like **dividing a pie** - it takes any set of numbers and converts them into slices that sum to 100%.

### When to Use Softmax
- **Multi-class classification** (output layer)
- **Attention mechanisms** in transformers
- **When you need probability distributions**
"""

# %% nbgrader={"grade": false, "grade_id": "softmax-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Softmax:
    """
    Softmax Activation Function: f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Converts a vector of numbers into a probability distribution.
    Essential for multi-class classification and attention mechanisms.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        TODO: Implement Softmax activation with numerical stability
        
        APPROACH:
        1. Subtract max value from inputs for numerical stability
        2. Compute exponentials: e^(x_i - max)
        3. Divide by sum of exponentials
        4. Return a new Tensor with the results
        
        EXAMPLE:
        Input: Tensor([[1, 2, 3]])
        Expected: Tensor([[0.09, 0.24, 0.67]]) (approximately, sums to 1)
        
        HINTS:
        - Use np.max(x.data, axis=-1, keepdims=True) for stability
        - Use np.exp() for exponentials
        - Use np.sum() for the denominator
        - Make sure the result sums to 1 along the last axis
        """
        ### BEGIN SOLUTION
        # Subtract max for numerical stability
        x_max = np.max(x.data, axis=-1, keepdims=True)
        x_shifted = x.data - x_max
        
        # Compute softmax
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        result = exp_x / sum_exp
        
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable: softmax(x) instead of softmax.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Softmax Activation

Let's test your Softmax implementation! This should convert any vector into a probability distribution that sums to 1.

**This is a unit test** - it tests one specific activation function (Softmax) in isolation.
"""

# %% nbgrader={"grade": true, "grade_id": "test-softmax-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# Test Softmax activation immediately after implementation
print("🔬 Unit Test: Softmax Activation...")

# Create Softmax instance
softmax = Softmax()

# Test with various inputs
try:
    test_input = Tensor([[1, 2, 3]])
    result = softmax(test_input)
    
    # Check that all outputs are non-negative
    assert np.all(result.data >= 0), "Softmax outputs should be non-negative"
    print(f"✅ Softmax test: input {test_input.data} → output {result.data}")
    
    # Check that outputs sum to 1
    sum_result = np.sum(result.data)
    assert np.allclose(sum_result, 1.0, atol=1e-6), f"Softmax should sum to 1, got {sum_result}"
    print(f"✅ Softmax sums to 1: {sum_result:.6f}")
    
    # Test that larger inputs get higher probabilities
    large_input = Tensor([[1, 2, 5]])  # 5 should get the highest probability
    large_result = softmax(large_input)
    max_idx = np.argmax(large_result.data)
    assert max_idx == 2, f"Largest input should get highest probability, got max at index {max_idx}"
    print("✅ Softmax gives highest probability to largest input")
    
    # Test numerical stability with large numbers
    stable_input = Tensor([[1000, 1001, 1002]])
    stable_result = softmax(stable_input)
    assert not np.any(np.isnan(stable_result.data)), "Softmax should be numerically stable"
    assert np.allclose(np.sum(stable_result.data), 1.0, atol=1e-6), "Softmax should still sum to 1 with large inputs"
    print("✅ Softmax is numerically stable with large inputs")
    
except Exception as e:
    print(f"❌ Softmax test failed: {e}")
    raise

# Show visual example
print("🎯 Softmax behavior:")
print("   Converts any vector → probability distribution")
print("   All outputs ≥ 0, sum = 1")
print("   Larger inputs → higher probabilities")
print("📈 Progress: ReLU ✓, Sigmoid ✓, Tanh ✓, Softmax ✓")
print("🚀 All activation functions ready!")

# %% [markdown]
"""
### 🧪 Test Your Activation Functions

Once you implement the activation functions above, run these cells to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
# Test ReLU activation
print("Testing ReLU activation...")

relu = ReLU()

# Test basic functionality
input_tensor = Tensor([[-2, -1, 0, 1, 2]])
output = relu(input_tensor)
expected = np.array([[0, 0, 0, 1, 2]])
assert np.array_equal(output.data, expected), f"ReLU failed: expected {expected}, got {output.data}"

# Test with matrix
matrix_input = Tensor([[-1, 2], [3, -4]])
matrix_output = relu(matrix_input)
expected_matrix = np.array([[0, 2], [3, 0]])
assert np.array_equal(matrix_output.data, expected_matrix), f"ReLU matrix failed: expected {expected_matrix}, got {matrix_output.data}"

# Test shape preservation
assert output.shape == input_tensor.shape, f"ReLU should preserve shape: input {input_tensor.shape}, output {output.shape}"

print("✅ ReLU tests passed!")
print(f"✅ ReLU({input_tensor.data.flatten()}) = {output.data.flatten()}")

# %% nbgrader={"grade": true, "grade_id": "test-sigmoid", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
# Test Sigmoid activation
print("Testing Sigmoid activation...")

sigmoid = Sigmoid()

# Test basic functionality
input_tensor = Tensor([[0]])
output = sigmoid(input_tensor)
expected_value = 0.5
assert abs(output.data.item() - expected_value) < 1e-6, f"Sigmoid(0) should be 0.5, got {output.data.item()}"

# Test range bounds (allowing for floating-point precision at extremes)
large_input = Tensor([[100]])
large_output = sigmoid(large_input)
assert 0 < large_output.data.item() <= 1, f"Sigmoid output should be in (0,1], got {large_output.data.item()}"

small_input = Tensor([[-100]])
small_output = sigmoid(small_input)
assert 0 <= small_output.data.item() < 1, f"Sigmoid output should be in [0,1), got {small_output.data.item()}"

# Test with multiple values
multi_input = Tensor([[-2, 0, 2]])
multi_output = sigmoid(multi_input)
assert multi_output.shape == multi_input.shape, "Sigmoid should preserve shape"
assert np.all((multi_output.data > 0) & (multi_output.data < 1)), "All sigmoid outputs should be in (0,1)"

print("✅ Sigmoid tests passed!")
print(f"✅ Sigmoid({multi_input.data.flatten()}) = {multi_output.data.flatten()}")

# %% nbgrader={"grade": true, "grade_id": "test-tanh", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
# Test Tanh activation
print("Testing Tanh activation...")

tanh = Tanh()

# Test basic functionality
input_tensor = Tensor([[0]])
output = tanh(input_tensor)
expected_value = 0.0
assert abs(output.data.item() - expected_value) < 1e-6, f"Tanh(0) should be 0.0, got {output.data.item()}"

# Test range bounds (allowing for floating-point precision at extremes)
large_input = Tensor([[100]])
large_output = tanh(large_input)
assert -1 <= large_output.data.item() <= 1, f"Tanh output should be in [-1,1], got {large_output.data.item()}"

small_input = Tensor([[-100]])
small_output = tanh(small_input)
assert -1 <= small_output.data.item() <= 1, f"Tanh output should be in [-1,1], got {small_output.data.item()}"

# Test symmetry: tanh(-x) = -tanh(x)
test_input = Tensor([[2]])
pos_output = tanh(test_input)
neg_input = Tensor([[-2]])
neg_output = tanh(neg_input)
assert abs(pos_output.data.item() + neg_output.data.item()) < 1e-6, "Tanh should be symmetric: tanh(-x) = -tanh(x)"

print("✅ Tanh tests passed!")
print(f"✅ Tanh(±2) = ±{abs(pos_output.data.item()):.3f}")

# %% nbgrader={"grade": true, "grade_id": "test-softmax", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
# Test Softmax activation
print("Testing Softmax activation...")

softmax = Softmax()

# Test basic functionality
input_tensor = Tensor([[1, 2, 3]])
output = softmax(input_tensor)

# Check that outputs sum to 1
sum_output = np.sum(output.data)
assert abs(sum_output - 1.0) < 1e-6, f"Softmax outputs should sum to 1, got {sum_output}"

# Check that all outputs are positive
assert np.all(output.data > 0), "All softmax outputs should be positive"

# Check that larger inputs give larger outputs
assert output.data[0, 2] > output.data[0, 1] > output.data[0, 0], "Softmax should preserve order"

# Test with matrix (multiple rows)
matrix_input = Tensor([[1, 2], [3, 4]])
matrix_output = softmax(matrix_input)
row_sums = np.sum(matrix_output.data, axis=1)
assert np.allclose(row_sums, 1.0), f"Each row should sum to 1, got {row_sums}"

print("✅ Softmax tests passed!")
print(f"✅ Softmax({input_tensor.data.flatten()}) = {output.data.flatten()}")
print(f"✅ Sum = {np.sum(output.data):.6f}")

# %% nbgrader={"grade": true, "grade_id": "test-activation-integration", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
# Test activation function integration
print("Testing activation function integration...")

# Create test data
test_data = Tensor([[-2, -1, 0, 1, 2]])

# Test all activations
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()
softmax = Softmax()

# Apply all activations
relu_out = relu(test_data)
sigmoid_out = sigmoid(test_data)
tanh_out = tanh(test_data)
softmax_out = softmax(test_data)

# Check shapes are preserved
assert relu_out.shape == test_data.shape, "ReLU should preserve shape"
assert sigmoid_out.shape == test_data.shape, "Sigmoid should preserve shape"
assert tanh_out.shape == test_data.shape, "Tanh should preserve shape"
assert softmax_out.shape == test_data.shape, "Softmax should preserve shape"

# Check ranges (allowing for floating-point precision at extremes)
assert np.all(relu_out.data >= 0), "ReLU outputs should be non-negative"
assert np.all((sigmoid_out.data >= 0) & (sigmoid_out.data <= 1)), "Sigmoid outputs should be in [0,1]"
assert np.all((tanh_out.data >= -1) & (tanh_out.data <= 1)), "Tanh outputs should be in [-1,1]"
assert np.all(softmax_out.data > 0), "Softmax outputs should be positive"

# Test chaining (composition)
chained = relu(sigmoid(test_data))
assert chained.shape == test_data.shape, "Chained activations should preserve shape"

print("✅ Activation integration tests passed!")
print(f"✅ All activation functions work correctly")
print(f"✅ Input:   {test_data.data.flatten()}")
print(f"✅ ReLU:    {relu_out.data.flatten()}")
print(f"✅ Sigmoid: {sigmoid_out.data.flatten()}")
print(f"✅ Tanh:    {tanh_out.data.flatten()}")
print(f"✅ Softmax: {softmax_out.data.flatten()}")

# %% [markdown]
"""
## 🧪 Comprehensive Testing: All Activation Functions

Let's thoroughly test all your activation functions to make sure they work correctly in all scenarios.
This comprehensive testing ensures your implementations are robust and ready for real ML applications.
"""

# %% nbgrader={"grade": true, "grade_id": "test-activations-comprehensive", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_activations_comprehensive():
    """Comprehensive test of all activation functions."""
    print("🔬 Testing all activation functions comprehensively...")
    
    tests_passed = 0
    total_tests = 12
    
    # Test 1: ReLU Basic Functionality
    try:
        relu = ReLU()
        test_input = Tensor([[-2, -1, 0, 1, 2]])
        result = relu(test_input)
        expected = np.array([[0, 0, 0, 1, 2]])
        
        assert np.array_equal(result.data, expected), f"ReLU failed: expected {expected}, got {result.data}"
        assert result.shape == test_input.shape, "ReLU should preserve shape"
        assert np.all(result.data >= 0), "ReLU outputs should be non-negative"
        
        print(f"✅ ReLU basic: {test_input.data.flatten()} → {result.data.flatten()}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ ReLU basic test failed: {e}")
    
    # Test 2: ReLU Edge Cases
    try:
        relu = ReLU()
        
        # Test with zeros
        zero_input = Tensor([[0, 0, 0]])
        zero_result = relu(zero_input)
        assert np.array_equal(zero_result.data, np.array([[0, 0, 0]])), "ReLU(0) should be 0"
        
        # Test with large values
        large_input = Tensor([[1000, -1000]])
        large_result = relu(large_input)
        expected_large = np.array([[1000, 0]])
        assert np.array_equal(large_result.data, expected_large), "ReLU should handle large values"
        
        # Test with matrix
        matrix_input = Tensor([[-1, 2], [3, -4]])
        matrix_result = relu(matrix_input)
        expected_matrix = np.array([[0, 2], [3, 0]])
        assert np.array_equal(matrix_result.data, expected_matrix), "ReLU should work with matrices"
        
        print("✅ ReLU edge cases: zeros, large values, matrices")
        tests_passed += 1
    except Exception as e:
        print(f"❌ ReLU edge cases failed: {e}")
    
    # Test 3: Sigmoid Basic Functionality
    try:
        sigmoid = Sigmoid()
        
        # Test sigmoid(0) = 0.5
        zero_input = Tensor([[0]])
        zero_result = sigmoid(zero_input)
        assert abs(zero_result.data.item() - 0.5) < 1e-6, f"Sigmoid(0) should be 0.5, got {zero_result.data.item()}"
        
        # Test range bounds
        test_input = Tensor([[-10, -1, 0, 1, 10]])
        result = sigmoid(test_input)
        assert np.all((result.data > 0) & (result.data < 1)), "Sigmoid outputs should be in (0,1)"
        assert result.shape == test_input.shape, "Sigmoid should preserve shape"
        
        print(f"✅ Sigmoid basic: range (0,1), sigmoid(0)=0.5")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Sigmoid basic test failed: {e}")
    
    # Test 4: Sigmoid Properties
    try:
        sigmoid = Sigmoid()
        
        # Test monotonicity
        inputs = Tensor([[-2, -1, 0, 1, 2]])
        outputs = sigmoid(inputs)
        output_values = outputs.data.flatten()
        
        # Check that outputs are increasing
        for i in range(len(output_values) - 1):
            assert output_values[i] < output_values[i + 1], "Sigmoid should be monotonic increasing"
        
        # Test numerical stability with extreme values
        extreme_input = Tensor([[-1000, 1000]])
        extreme_result = sigmoid(extreme_input)
        assert not np.any(np.isnan(extreme_result.data)), "Sigmoid should handle extreme values without NaN"
        assert not np.any(np.isinf(extreme_result.data)), "Sigmoid should handle extreme values without Inf"
        
        print("✅ Sigmoid properties: monotonic, numerically stable")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Sigmoid properties failed: {e}")
    
    # Test 5: Tanh Basic Functionality
    try:
        tanh = Tanh()
        
        # Test tanh(0) = 0
        zero_input = Tensor([[0]])
        zero_result = tanh(zero_input)
        assert abs(zero_result.data.item() - 0.0) < 1e-6, f"Tanh(0) should be 0.0, got {zero_result.data.item()}"
        
        # Test range bounds
        test_input = Tensor([[-10, -1, 0, 1, 10]])
        result = tanh(test_input)
        assert np.all((result.data >= -1) & (result.data <= 1)), "Tanh outputs should be in [-1,1]"
        assert result.shape == test_input.shape, "Tanh should preserve shape"
        
        print(f"✅ Tanh basic: range [-1,1], tanh(0)=0")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Tanh basic test failed: {e}")
    
    # Test 6: Tanh Symmetry
    try:
        tanh = Tanh()
        
        # Test symmetry: tanh(-x) = -tanh(x)
        test_values = [1, 2, 3, 5]
        for val in test_values:
            pos_input = Tensor([[val]])
            neg_input = Tensor([[-val]])
            pos_result = tanh(pos_input)
            neg_result = tanh(neg_input)
            
            assert abs(pos_result.data.item() + neg_result.data.item()) < 1e-6, f"Tanh should be symmetric: tanh(-{val}) ≠ -tanh({val})"
        
        # Test numerical stability
        extreme_input = Tensor([[-1000, 1000]])
        extreme_result = tanh(extreme_input)
        assert not np.any(np.isnan(extreme_result.data)), "Tanh should handle extreme values without NaN"
        
        print("✅ Tanh symmetry: tanh(-x) = -tanh(x), numerically stable")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Tanh symmetry failed: {e}")
    
    # Test 7: Softmax Basic Functionality
    try:
        softmax = Softmax()
        
        # Test that outputs sum to 1
        test_input = Tensor([[1, 2, 3]])
        result = softmax(test_input)
        sum_result = np.sum(result.data)
        assert abs(sum_result - 1.0) < 1e-6, f"Softmax outputs should sum to 1, got {sum_result}"
        
        # Test that all outputs are positive
        assert np.all(result.data > 0), "All softmax outputs should be positive"
        
        # Test that larger inputs give larger outputs
        assert result.data[0, 2] > result.data[0, 1] > result.data[0, 0], "Softmax should preserve order"
        
        print(f"✅ Softmax basic: sums to 1, all positive, preserves order")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Softmax basic test failed: {e}")
    
    # Test 8: Softmax with Multiple Rows
    try:
        softmax = Softmax()
        
        # Test with matrix (multiple rows)
        matrix_input = Tensor([[1, 2, 3], [4, 5, 6]])
        matrix_result = softmax(matrix_input)
        
        # Each row should sum to 1
        row_sums = np.sum(matrix_result.data, axis=1)
        assert np.allclose(row_sums, 1.0), f"Each row should sum to 1, got {row_sums}"
        
        # All values should be positive
        assert np.all(matrix_result.data > 0), "All softmax outputs should be positive"
        
        # Test numerical stability with extreme values
        extreme_input = Tensor([[1000, 1001, 1002]])
        extreme_result = softmax(extreme_input)
        assert not np.any(np.isnan(extreme_result.data)), "Softmax should handle extreme values without NaN"
        assert abs(np.sum(extreme_result.data) - 1.0) < 1e-6, "Softmax should still sum to 1 with extreme values"
        
        print("✅ Softmax matrices: each row sums to 1, numerically stable")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Softmax matrices failed: {e}")
    
    # Test 9: Shape Preservation
    try:
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        softmax = Softmax()
        
        # Test different shapes
        test_shapes = [
            Tensor([[1]]),                    # 1x1
            Tensor([[1, 2, 3]]),             # 1x3
            Tensor([[1], [2], [3]]),         # 3x1
            Tensor([[1, 2], [3, 4]]),        # 2x2
            Tensor([[1, 2], [3, 4]]),        # 2x2
        ]
        
        for i, test_tensor in enumerate(test_shapes):
            original_shape = test_tensor.shape
            
            relu_result = relu(test_tensor)
            sigmoid_result = sigmoid(test_tensor)
            tanh_result = tanh(test_tensor)
            softmax_result = softmax(test_tensor)
            
            assert relu_result.shape == original_shape, f"ReLU shape mismatch for test {i}"
            assert sigmoid_result.shape == original_shape, f"Sigmoid shape mismatch for test {i}"
            assert tanh_result.shape == original_shape, f"Tanh shape mismatch for test {i}"
            assert softmax_result.shape == original_shape, f"Softmax shape mismatch for test {i}"
        
        print("✅ Shape preservation: all activations preserve input shapes")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Shape preservation failed: {e}")
    
    # Test 10: Function Composition
    try:
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        # Test chaining activations
        test_input = Tensor([[-2, -1, 0, 1, 2]])
        
        # Chain: input → tanh → relu
        tanh_result = tanh(test_input)
        relu_tanh_result = relu(tanh_result)
        
        # Chain: input → sigmoid → tanh
        sigmoid_result = sigmoid(test_input)
        tanh_sigmoid_result = tanh(sigmoid_result)
        
        # All should preserve shape
        assert relu_tanh_result.shape == test_input.shape, "Chained activations should preserve shape"
        assert tanh_sigmoid_result.shape == test_input.shape, "Chained activations should preserve shape"
        
        # Results should be valid
        assert np.all(relu_tanh_result.data >= 0), "ReLU after Tanh should be non-negative"
        assert np.all((tanh_sigmoid_result.data >= -1) & (tanh_sigmoid_result.data <= 1)), "Tanh after Sigmoid should be in [-1,1]"
        
        print("✅ Function composition: activations can be chained together")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Function composition failed: {e}")
    
    # Test 11: Real ML Scenario
    try:
        # Simulate a neural network layer output
        logits = Tensor([[2.0, 1.0, 0.1]])  # Raw network outputs
        
        # Apply softmax for classification
        softmax = Softmax()
        probabilities = softmax(logits)
        
        # Check that we get valid probabilities
        assert abs(np.sum(probabilities.data) - 1.0) < 1e-6, "Probabilities should sum to 1"
        assert np.all(probabilities.data > 0), "All probabilities should be positive"
        
        # The highest logit should give the highest probability
        max_logit_idx = np.argmax(logits.data)
        max_prob_idx = np.argmax(probabilities.data)
        assert max_logit_idx == max_prob_idx, "Highest logit should give highest probability"
        
        # Apply ReLU to hidden layer
        hidden_activations = Tensor([[-0.5, 0.8, -1.2, 2.1]])
        relu = ReLU()
        relu_output = relu(hidden_activations)
        
        # Should zero out negative values
        expected_relu = np.array([[0.0, 0.8, 0.0, 2.1]])
        assert np.array_equal(relu_output.data, expected_relu), "ReLU should zero negative values"
        
        print("✅ Real ML scenario: classification probabilities, hidden layer activation")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Real ML scenario failed: {e}")
    
    # Test 12: Performance and Stability
    try:
        # Test with large tensors
        large_input = Tensor(np.random.randn(100, 50))
        
        relu = ReLU()
        sigmoid = Sigmoid()
        tanh = Tanh()
        softmax = Softmax()
        
        # All should handle large tensors
        relu_large = relu(large_input)
        sigmoid_large = sigmoid(large_input)
        tanh_large = tanh(large_input)
        softmax_large = softmax(large_input)
        
        # Check for NaN or Inf
        assert not np.any(np.isnan(relu_large.data)), "ReLU should not produce NaN"
        assert not np.any(np.isnan(sigmoid_large.data)), "Sigmoid should not produce NaN"
        assert not np.any(np.isnan(tanh_large.data)), "Tanh should not produce NaN"
        assert not np.any(np.isnan(softmax_large.data)), "Softmax should not produce NaN"
        
        assert not np.any(np.isinf(relu_large.data)), "ReLU should not produce Inf"
        assert not np.any(np.isinf(sigmoid_large.data)), "Sigmoid should not produce Inf"
        assert not np.any(np.isinf(tanh_large.data)), "Tanh should not produce Inf"
        assert not np.any(np.isinf(softmax_large.data)), "Softmax should not produce Inf"
        
        print("✅ Performance and stability: large tensors handled without NaN/Inf")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Performance and stability failed: {e}")
    
    # Results summary
    print(f"\n📊 Activation Functions Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All activation function tests passed! Your implementations support:")
        print("  • ReLU: Fast, sparse activation for hidden layers")
        print("  • Sigmoid: Smooth probabilistic outputs (0,1)")
        print("  • Tanh: Zero-centered activation (-1,1)")
        print("  • Softmax: Probability distributions for classification")
        print("  • All functions preserve shapes and handle edge cases")
        print("  • Numerical stability with extreme values")
        print("  • Function composition for complex networks")
        print("📈 Progress: All Activation Functions ✓")
        return True
    else:
        print("⚠️  Some activation tests failed. Common issues:")
        print("  • Check mathematical formulas (especially sigmoid and tanh)")
        print("  • Verify numerical stability (clip extreme values)")
        print("  • Ensure proper shape preservation")
        print("  • Test with edge cases (zeros, large values)")
        print("  • Verify softmax sums to 1 for each row")
        return False

# Run the comprehensive test
success = test_activations_comprehensive()

# %% [markdown]
"""
### 🧪 Integration Test: Activation Functions in Neural Networks

Let's test how your activation functions work in a realistic neural network scenario.
"""

# %% nbgrader={"grade": true, "grade_id": "test-activations-integration", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_activations_integration():
    """Integration test with realistic neural network scenario."""
    print("🔬 Testing activation functions in neural network scenario...")
    
    try:
        print("🧠 Simulating a 3-layer neural network...")
        
        # Layer 1: Input data (batch of 3 samples, 4 features each)
        input_data = Tensor([[1.0, -2.0, 3.0, -1.0],
                           [2.0, 1.0, -1.0, 0.5],
                           [-1.0, 3.0, 2.0, -0.5]])
        print(f"📊 Input data shape: {input_data.shape}")
        
        # Layer 2: Hidden layer with ReLU activation
        # Simulate some linear transformation results
        hidden_raw = Tensor([[2.1, -1.5, 0.8],
                           [1.2, 3.4, -0.3],
                           [-0.7, 2.8, 1.9]])
        
        relu = ReLU()
        hidden_activated = relu(hidden_raw)
        print(f"✅ Hidden layer (ReLU): {hidden_raw.data.flatten()[:3]} → {hidden_activated.data.flatten()[:3]}")
        
        # Verify ReLU worked correctly
        assert np.all(hidden_activated.data >= 0), "Hidden layer should have non-negative activations"
        
        # Layer 3: Output layer for binary classification (sigmoid)
        output_raw = Tensor([[0.8], [2.1], [-0.5]])
        
        sigmoid = Sigmoid()
        output_probs = sigmoid(output_raw)
        print(f"✅ Output layer (Sigmoid): {output_raw.data.flatten()} → {output_probs.data.flatten()}")
        
        # Verify sigmoid outputs are valid probabilities
        assert np.all((output_probs.data > 0) & (output_probs.data < 1)), "Output should be valid probabilities"
        
        # Alternative: Multi-class classification with softmax
        multiclass_raw = Tensor([[1.0, 2.0, 0.5],
                               [0.1, 0.8, 2.1],
                               [1.5, 0.3, 1.2]])
        
        softmax = Softmax()
        class_probs = softmax(multiclass_raw)
        print(f"✅ Multi-class output (Softmax): each row sums to {np.sum(class_probs.data, axis=1)}")
        
        # Verify softmax outputs
        row_sums = np.sum(class_probs.data, axis=1)
        assert np.allclose(row_sums, 1.0), "Each sample should have probabilities summing to 1"
        
        # Test activation function chaining
        print("\n🔗 Testing activation function chaining...")
        
        # Chain: Tanh → ReLU (unusual but valid)
        tanh = Tanh()
        test_input = Tensor([[-2, -1, 0, 1, 2]])
        
        tanh_result = tanh(test_input)
        relu_tanh_result = relu(tanh_result)
        
        print(f"✅ Tanh → ReLU: {test_input.data.flatten()} → {tanh_result.data.flatten()} → {relu_tanh_result.data.flatten()}")
        
        # Verify chaining worked
        assert relu_tanh_result.shape == test_input.shape, "Chained activations should preserve shape"
        assert np.all(relu_tanh_result.data >= 0), "Final result should be non-negative (ReLU effect)"
        
        # Test different activation choices
        print("\n🎯 Testing activation function choices...")
        
        # Compare different activations on same input
        comparison_input = Tensor([[0.5, -0.5, 1.0, -1.0]])
        
        relu_comp = relu(comparison_input)
        sigmoid_comp = sigmoid(comparison_input)
        tanh_comp = tanh(comparison_input)
        
        print(f"Input:   {comparison_input.data.flatten()}")
        print(f"ReLU:    {relu_comp.data.flatten()}")
        print(f"Sigmoid: {sigmoid_comp.data.flatten()}")
        print(f"Tanh:    {tanh_comp.data.flatten()}")
        
        # Show how different activations affect the same input
        print("\n📈 Activation function characteristics:")
        print("• ReLU: Sparse (many zeros), unbounded positive")
        print("• Sigmoid: Smooth, bounded (0,1), good for probabilities")
        print("• Tanh: Zero-centered (-1,1), symmetric")
        print("• Softmax: Probability distribution, sums to 1")
        
        print("\n🎉 Integration test passed! Your activation functions work correctly in:")
        print("  • Multi-layer neural networks")
        print("  • Binary and multi-class classification")
        print("  • Function composition and chaining")
        print("  • Different architectural choices")
        print("📈 Progress: All activation functions ready for neural networks!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("\n💡 This suggests an issue with:")
        print("  • Basic activation function implementation")
        print("  • Shape handling in neural network context")
        print("  • Mathematical correctness of the functions")
        print("  • Check your activation function implementations")
        return False

# Run the integration test
success = test_activations_integration() and success

# Print final summary
print(f"\n{'='*60}")
print("🎯 ACTIVATION FUNCTIONS MODULE TESTING COMPLETE")
print(f"{'='*60}")

if success:
    print("🎉 CONGRATULATIONS! All activation function tests passed!")
    print("\n✅ Your activation functions successfully implement:")
    print("  • ReLU: max(0, x) for sparse hidden layer activation")
    print("  • Sigmoid: 1/(1+e^(-x)) for binary classification")
    print("  • Tanh: tanh(x) for zero-centered activation")
    print("  • Softmax: probability distributions for multi-class classification")
    print("  • Numerical stability with extreme values")
    print("  • Shape preservation and function composition")
    print("  • Real neural network integration")
    print("\n🚀 You're ready to build neural network layers!")
    print("📈 Final Progress: Activation Functions Module ✓ COMPLETE")
else:
    print("⚠️  Some tests failed. Please review the error messages above.")
    print("\n🔧 To fix issues:")
    print("  1. Check the specific activation function that failed")
    print("  2. Review the mathematical formulas")
    print("  3. Verify numerical stability (especially for sigmoid/tanh)")
    print("  4. Test with edge cases (zeros, large values)")
    print("  5. Ensure softmax sums to 1")
    print("\n💪 Keep going! These functions are the key to neural network power.")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented the core activation functions for TinyTorch:

### What You've Accomplished
✅ **ReLU**: The workhorse activation for hidden layers  
✅ **Sigmoid**: Smooth probabilistic outputs for binary classification  
✅ **Tanh**: Zero-centered activation for better training dynamics  
✅ **Softmax**: Probability distributions for multi-class classification  
✅ **Integration**: All functions work together and preserve tensor shapes  

### Key Concepts You've Learned
- **Nonlinearity** is essential for neural networks to learn complex patterns
- **ReLU** is simple, fast, and effective for most hidden layers
- **Sigmoid** squashes outputs to (0,1) for probabilistic interpretation
- **Tanh** is zero-centered and often better than sigmoid for hidden layers
- **Softmax** converts logits to probability distributions
- **Numerical stability** is crucial for functions with exponentials

### Next Steps
1. **Export your code**: `tito package nbdev --export 02_activations`
2. **Test your implementation**: `tito module test 02_activations`
3. **Use your activations**: 
   ```python
   from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
   from tinytorch.core.tensor import Tensor
   
   relu = ReLU()
   x = Tensor([[-1, 0, 1, 2]])
   y = relu(x)  # Your activation in action!
   ```
4. **Move to Module 3**: Start building neural network layers!

**Ready for the next challenge?** Let's combine tensors and activations to build the fundamental building blocks of neural networks!
""" 