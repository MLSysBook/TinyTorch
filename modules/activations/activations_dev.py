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
# Module 3: Activation Functions - The Spark of Intelligence

**Learning Goals:**
- Understand why activation functions are essential for neural networks
- Implement four fundamental activation functions from scratch
- Learn the mathematical properties and use cases of each activation
- Visualize activation function behavior and understand their impact

**Why This Matters:**
Without activation functions, neural networks would just be linear transformations - no matter how many layers you stack, you'd only get linear relationships. Activation functions introduce the nonlinearity that allows neural networks to learn complex patterns and approximate any function.

**Real-World Context:**
Every neural network you've heard of - from image recognition to language models - relies on activation functions. Understanding them deeply is crucial for designing effective architectures and debugging training issues.
"""

# %%
#| default_exp core.activations

# %%
#| export
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Union, List

# Import our Tensor class from the main package (rock solid foundation)
from tinytorch.core.tensor import Tensor

# %%
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

# %%
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
In Module 2 (Layers), we learned how to transform data through linear operations (matrix multiplication + bias). Now we add the nonlinear activation functions that make neural networks powerful.
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
- **Hidden layers** in most neural networks
- **Convolutional layers** in image processing
- **When you want sparse activations**
"""

# %%
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
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Allow calling the activation like a function: relu(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class ReLU:
    """ReLU Activation: f(x) = max(0, x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0, x.data)
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your ReLU Implementation

Let's test your ReLU implementation right away to make sure it's working correctly:
"""

# %%
try:
    # Create ReLU activation
    relu = ReLU()
    
    # Test 1: Basic functionality
    print("🔧 Testing ReLU Implementation")
    print("=" * 40)
    
    # Test with mixed positive/negative values
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    expected = Tensor([[0, 0, 0, 1, 2]])
    
    result = relu(test_input)
    print(f"Input:    {test_input.data.flatten()}")
    print(f"Output:   {result.data.flatten()}")
    print(f"Expected: {expected.data.flatten()}")
    
    # Verify correctness
    if np.allclose(result.data, expected.data):
        print("✅ Basic ReLU test passed!")
    else:
        print("❌ Basic ReLU test failed!")
        print("   Check your max(0, x) implementation")
    
    # Test 2: Edge cases
    edge_cases = Tensor([[-100, -0.1, 0, 0.1, 100]])
    edge_result = relu(edge_cases)
    expected_edge = np.array([[0, 0, 0, 0.1, 100]])
    
    print(f"\nEdge cases: {edge_cases.data.flatten()}")
    print(f"Output:     {edge_result.data.flatten()}")
    
    if np.allclose(edge_result.data, expected_edge):
        print("✅ Edge case test passed!")
    else:
        print("❌ Edge case test failed!")
    
    # Test 3: Shape preservation
    multi_dim = Tensor([[1, -1], [2, -2], [0, 3]])
    multi_result = relu(multi_dim)
    
    if multi_result.data.shape == multi_dim.data.shape:
        print("✅ Shape preservation test passed!")
    else:
        print("❌ Shape preservation test failed!")
        print(f"   Expected shape: {multi_dim.data.shape}, got: {multi_result.data.shape}")
    
    print("✅ ReLU tests complete!")
    
except NotImplementedError:
    print("⚠️  ReLU not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in ReLU: {e}")
    print("   Check your implementation in the forward method")

print()  # Add spacing

# %%
# 🎨 ReLU Visualization (development only - not exported)
if _should_show_plots():
    try:
        relu = ReLU()
        print("🎨 Visualizing ReLU behavior...")
        visualize_activation_function(relu, "ReLU", x_range=(-3, 3))
        
        # Show ReLU with real data
        sample_data = Tensor([[-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5]])
        visualize_activation_on_data(relu, "ReLU", sample_data)
    except:
        pass  # Skip if ReLU not implemented

# %% [markdown]
"""
## Step 3: Sigmoid - The Smooth Classifier

### What is Sigmoid?
**Sigmoid** is a smooth, S-shaped activation function that squashes inputs to the range (0, 1).

**Mathematical Definition:**
```
f(x) = 1 / (1 + e^(-x))
```

**Key Properties:**
- **Range**: (0, 1) - never exactly 0 or 1
- **Smooth**: Differentiable everywhere
- **Monotonic**: Always increasing
- **Symmetric**: Around the point (0, 0.5)

### Why Sigmoid is Useful
1. **Probability interpretation**: Output can be interpreted as probability
2. **Smooth gradients**: Nice for optimization
3. **Bounded output**: Prevents extreme values

### Real-World Analogy
Sigmoid is like a **smooth dimmer switch** - it gradually transitions from "off" (near 0) to "on" (near 1), unlike ReLU's sharp cutoff.

### When to Use Sigmoid
- **Binary classification** (output layer)
- **Gate mechanisms** (in LSTMs)
- **When you need probabilities**

### Numerical Stability Note
For very large positive or negative inputs, sigmoid can cause numerical issues. We'll handle this with clipping.
"""

# %%
#| export
class Sigmoid:
    """
    Sigmoid Activation Function: f(x) = 1 / (1 + e^(-x))
    
    Squashes inputs to the range (0, 1), useful for binary classification
    and probability interpretation.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Sigmoid activation: f(x) = 1 / (1 + e^(-x))
        
        TODO: Implement Sigmoid activation
        
        APPROACH:
        1. For numerical stability, clip x to reasonable range (e.g., -500 to 500)
        2. Compute 1 / (1 + exp(-x)) for each element
        3. Return a new Tensor with the results
        
        EXAMPLE:
        Input: Tensor([[-2, -1, 0, 1, 2]])
        Expected: Tensor([[0.119, 0.269, 0.5, 0.731, 0.881]]) (approximately)
        
        HINTS:
        - Use np.clip(x.data, -500, 500) for numerical stability
        - Use np.exp(-clipped_x) for the exponential
        - Formula: 1 / (1 + np.exp(-clipped_x))
        - Remember to return a new Tensor object
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Allow calling the activation like a function: sigmoid(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Sigmoid:
    """Sigmoid Activation: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        # Clip for numerical stability
        clipped = np.clip(x.data, -500, 500)
        result = 1 / (1 + np.exp(-clipped))
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Sigmoid Implementation

Let's test your Sigmoid implementation to ensure it's working correctly:
"""

# %%
try:
    # Create Sigmoid activation
    sigmoid = Sigmoid()
    
    print("🔧 Testing Sigmoid Implementation")
    print("=" * 40)
    
    # Test 1: Basic functionality
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = sigmoid(test_input)
    
    print(f"Input:  {test_input.data.flatten()}")
    print(f"Output: {result.data.flatten()}")
    
    # Check properties
    # 1. All outputs should be between 0 and 1
    if np.all(result.data >= 0) and np.all(result.data <= 1):
        print("✅ Range test passed: all outputs in (0, 1)")
    else:
        print("❌ Range test failed: outputs should be in (0, 1)")
    
    # 2. Sigmoid(0) should be 0.5
    zero_input = Tensor([[0]])
    zero_result = sigmoid(zero_input)
    if abs(zero_result.data.item() - 0.5) < 1e-6:
        print("✅ Sigmoid(0) = 0.5 test passed!")
    else:
        print(f"❌ Sigmoid(0) should be 0.5, got {zero_result.data.item()}")
    
    # 3. Test symmetry: sigmoid(-x) = 1 - sigmoid(x)
    x_val = 2.0
    pos_result = sigmoid(Tensor([[x_val]])).data.item()
    neg_result = sigmoid(Tensor([[-x_val]])).data.item()
    
    if abs(pos_result + neg_result - 1.0) < 1e-6:
        print("✅ Symmetry test passed!")
    else:
        print(f"❌ Symmetry test failed: sigmoid({x_val}) + sigmoid({-x_val}) should equal 1")
    
    # 4. Test numerical stability with extreme values
    extreme_input = Tensor([[-1000, 1000]])
    extreme_result = sigmoid(extreme_input)
    
    # Should not produce NaN or inf
    if not np.any(np.isnan(extreme_result.data)) and not np.any(np.isinf(extreme_result.data)):
        print("✅ Numerical stability test passed!")
    else:
        print("❌ Numerical stability test failed: extreme values produced NaN/inf")
    
    print("✅ Sigmoid tests complete!")
    
    # 🎨 Visualize Sigmoid behavior (development only)
    if _should_show_plots():
        print("\n🎨 Visualizing Sigmoid behavior...")
        visualize_activation_function(sigmoid, "Sigmoid", x_range=(-5, 5))
        
        # Show Sigmoid with real data
        sample_data = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
        visualize_activation_on_data(sigmoid, "Sigmoid", sample_data)
    
except NotImplementedError:
    print("⚠️  Sigmoid not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in Sigmoid: {e}")
    print("   Check your implementation in the forward method")

print()  # Add spacing

# %% [markdown]
"""
## Step 4: Tanh - The Centered Alternative

### What is Tanh?
**Tanh (Hyperbolic Tangent)** is similar to Sigmoid but centered around zero, with range (-1, 1).

**Mathematical Definition:**
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Alternative form:**
```
f(x) = 2 * sigmoid(2x) - 1
```

**Key Properties:**
- **Range**: (-1, 1) - symmetric around zero
- **Zero-centered**: Output has mean closer to zero
- **Smooth**: Differentiable everywhere
- **Stronger gradients**: Steeper than sigmoid

### Why Tanh is Better Than Sigmoid
1. **Zero-centered**: Helps with gradient flow in deep networks
2. **Stronger gradients**: Faster convergence in some cases
3. **Symmetric**: Better for certain applications

### Real-World Analogy
Tanh is like a **balanced scale** - it can tip strongly in either direction (-1 to +1) but defaults to neutral (0).

### When to Use Tanh
- **Hidden layers** (alternative to ReLU)
- **Recurrent networks** (RNNs, LSTMs)
- **When you need zero-centered outputs**
"""

# %%
#| export
class Tanh:
    """
    Tanh Activation Function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Zero-centered activation function with range (-1, 1).
    Often preferred over Sigmoid for hidden layers.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        TODO: Implement Tanh activation
        
        APPROACH:
        1. Use numpy's built-in tanh function: np.tanh(x.data)
        2. Return a new Tensor with the results
        
        ALTERNATIVE APPROACH:
        1. Compute e^x and e^(-x)
        2. Use formula: (e^x - e^(-x)) / (e^x + e^(-x))
        
        EXAMPLE:
        Input: Tensor([[-2, -1, 0, 1, 2]])
        Expected: Tensor([[-0.964, -0.762, 0.0, 0.762, 0.964]]) (approximately)
        
        HINTS:
        - np.tanh() is the simplest approach
        - Output range is (-1, 1)
        - tanh(0) = 0 (zero-centered)
        - Remember to return a new Tensor object
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Allow calling the activation like a function: tanh(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Tanh:
    """Tanh Activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Tanh Implementation

Let's test your Tanh implementation to ensure it's working correctly:
"""

# %%
try:
    # Create Tanh activation
    tanh = Tanh()
    
    print("🔧 Testing Tanh Implementation")
    print("=" * 40)
    
    # Test 1: Basic functionality
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = tanh(test_input)
    
    print(f"Input:  {test_input.data.flatten()}")
    print(f"Output: {result.data.flatten()}")
    
    # Check properties
    # 1. All outputs should be between -1 and 1
    if np.all(result.data >= -1) and np.all(result.data <= 1):
        print("✅ Range test passed: all outputs in (-1, 1)")
    else:
        print("❌ Range test failed: outputs should be in (-1, 1)")
    
    # 2. Tanh(0) should be 0
    zero_input = Tensor([[0]])
    zero_result = tanh(zero_input)
    if abs(zero_result.data.item()) < 1e-6:
        print("✅ Tanh(0) = 0 test passed!")
    else:
        print(f"❌ Tanh(0) should be 0, got {zero_result.data.item()}")
    
    # 3. Test antisymmetry: tanh(-x) = -tanh(x)
    x_val = 1.5
    pos_result = tanh(Tensor([[x_val]])).data.item()
    neg_result = tanh(Tensor([[-x_val]])).data.item()
    
    if abs(pos_result + neg_result) < 1e-6:
        print("✅ Antisymmetry test passed!")
    else:
        print(f"❌ Antisymmetry test failed: tanh({x_val}) + tanh({-x_val}) should equal 0")
    
    # 4. Test that tanh is stronger than sigmoid
    # For the same input, |tanh(x)| should be > |sigmoid(x) - 0.5|
    test_val = 1.0
    tanh_result = abs(tanh(Tensor([[test_val]])).data.item())
    sigmoid_result = abs(sigmoid(Tensor([[test_val]])).data.item() - 0.5)
    
    if tanh_result > sigmoid_result:
        print("✅ Stronger gradient test passed!")
    else:
        print("❌ Tanh should have stronger gradients than sigmoid")
    
    print("✅ Tanh tests complete!")
    
    # 🎨 Visualize Tanh behavior (development only)
    if _should_show_plots():
        print("\n🎨 Visualizing Tanh behavior...")
        visualize_activation_function(tanh, "Tanh", x_range=(-3, 3))
        
        # Show Tanh with real data
        sample_data = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        visualize_activation_on_data(tanh, "Tanh", sample_data)
    
except NotImplementedError:
    print("⚠️  Tanh not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in Tanh: {e}")
    print("   Check your implementation in the forward method")

print()  # Add spacing

# %% [markdown]
"""
## Step 5: Softmax - The Probability Maker

### What is Softmax?
**Softmax** converts a vector of real numbers into a probability distribution. It's essential for multi-class classification.

**Mathematical Definition:**
```
f(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
```

**Key Properties:**
- **Probability distribution**: All outputs sum to 1
- **Non-negative**: All outputs ≥ 0
- **Differentiable**: Smooth for optimization
- **Relative**: Emphasizes the largest input

### Why Softmax is Special
1. **Probability interpretation**: Perfect for classification
2. **Competitive**: Emphasizes the winner (largest input)
3. **Differentiable**: Works well with gradient descent

### Real-World Analogy
Softmax is like **voting with enthusiasm** - not only does the most popular choice win, but the "votes" are weighted by how much more popular it is.

### When to Use Softmax
- **Multi-class classification** (output layer)
- **Attention mechanisms** (in Transformers)
- **When you need probability distributions**

### Numerical Stability Note
For numerical stability, we subtract the maximum value before computing exponentials.
"""

# %%
#| export
class Softmax:
    """
    Softmax Activation Function: f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Converts a vector of real numbers into a probability distribution.
    Essential for multi-class classification.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        TODO: Implement Softmax activation
        
        APPROACH:
        1. For numerical stability, subtract the maximum value from each row
        2. Compute exponentials of the shifted values
        3. Divide each exponential by the sum of exponentials in its row
        4. Return a new Tensor with the results
        
        EXAMPLE:
        Input: Tensor([[1, 2, 3]])
        Expected: Tensor([[0.090, 0.245, 0.665]]) (approximately)
        Sum should be 1.0
        
        HINTS:
        - Use np.max(x.data, axis=1, keepdims=True) to find row maximums
        - Subtract max from x.data for numerical stability
        - Use np.exp() for exponentials
        - Use np.sum(exp_vals, axis=1, keepdims=True) for row sums
        - Remember to return a new Tensor object
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Allow calling the activation like a function: softmax(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Softmax:
    """Softmax Activation: f(x_i) = e^(x_i) / Σ(e^(x_j))"""
    
    def forward(self, x: Tensor) -> Tensor:
        # Subtract max for numerical stability
        shifted = x.data - np.max(x.data, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        result = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Softmax Implementation

Let's test your Softmax implementation to ensure it's working correctly:
"""

# %%
try:
    # Create Softmax activation
    softmax = Softmax()
    
    print("🔧 Testing Softmax Implementation")
    print("=" * 40)
    
    # Test 1: Basic functionality
    test_input = Tensor([[1, 2, 3]])
    result = softmax(test_input)
    
    print(f"Input:  {test_input.data.flatten()}")
    print(f"Output: {result.data.flatten()}")
    
    # Check properties
    # 1. All outputs should be non-negative
    if np.all(result.data >= 0):
        print("✅ Non-negative test passed!")
    else:
        print("❌ Non-negative test failed: all outputs should be ≥ 0")
    
    # 2. Sum should equal 1 (probability distribution)
    row_sums = np.sum(result.data, axis=1)
    if np.allclose(row_sums, 1.0):
        print("✅ Probability distribution test passed!")
    else:
        print(f"❌ Sum test failed: sum should be 1.0, got {row_sums}")
    
    # 3. Test with multiple rows
    multi_input = Tensor([[1, 2, 3], [0, 0, 0], [10, 20, 30]])
    multi_result = softmax(multi_input)
    multi_sums = np.sum(multi_result.data, axis=1)
    
    if np.allclose(multi_sums, 1.0):
        print("✅ Multi-row test passed!")
    else:
        print(f"❌ Multi-row test failed: all row sums should be 1.0, got {multi_sums}")
    
    # 4. Test numerical stability
    large_input = Tensor([[1000, 1001, 1002]])
    large_result = softmax(large_input)
    
    # Should not produce NaN or inf
    if not np.any(np.isnan(large_result.data)) and not np.any(np.isinf(large_result.data)):
        print("✅ Numerical stability test passed!")
    else:
        print("❌ Numerical stability test failed: large values produced NaN/inf")
    
    # 5. Test that largest input gets highest probability
    test_logits = Tensor([[1, 5, 2]])
    test_probs = softmax(test_logits)
    max_idx = np.argmax(test_probs.data)
    
    if max_idx == 1:  # Second element (index 1) should be largest
        print("✅ Max probability test passed!")
    else:
        print("❌ Max probability test failed: largest input should get highest probability")
    
    print("✅ Softmax tests complete!")
    
    # 🎨 Visualize Softmax behavior (development only)
    if _should_show_plots():
        print("\n🎨 Visualizing Softmax behavior...")
        # Note: Softmax is different - it's a vector function, so we show it differently
        sample_logits = Tensor([[1.0, 2.0, 3.0]])  # Simple 3-class example
        softmax_output = softmax(sample_logits)
        
        print(f"   Example: logits {sample_logits.data.flatten()} → probabilities {softmax_output.data.flatten()}")
        print(f"   Sum of probabilities: {softmax_output.data.sum():.6f} (should be 1.0)")
        
        # Show how different input scales affect output
        scale_examples = [
            Tensor([[1.0, 2.0, 3.0]]),    # Original
            Tensor([[2.0, 4.0, 6.0]]),    # Scaled up
            Tensor([[0.1, 0.2, 0.3]]),    # Scaled down
        ]
        
        print("\n   📊 Scale sensitivity:")
        for i, example in enumerate(scale_examples):
            output = softmax(example)
            print(f"   Scale {i+1}: {example.data.flatten()} → {output.data.flatten()}")
    
except NotImplementedError:
    print("⚠️  Softmax not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in Softmax: {e}")
    print("   Check your implementation in the forward method")

print()  # Add spacing

# %% [markdown]
"""
## 🎨 Comprehensive Activation Function Comparison

Now that we've implemented all four activation functions, let's compare them side by side to understand their differences and use cases.
"""

# %%
# Comprehensive comparison of all activation functions
print("🎨 Comprehensive Activation Function Comparison")
print("=" * 60)

try:
    # Create all activation functions
    activations = {
        'ReLU': ReLU(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'Softmax': Softmax()
    }
    
    # Test with sample data
    test_data = Tensor([[-2, -1, 0, 1, 2]])
    
    print("📊 Activation Function Outputs:")
    print(f"Input: {test_data.data.flatten()}")
    print("-" * 40)
    
    for name, activation in activations.items():
        try:
            result = activation(test_data)
            print(f"{name:8}: {result.data.flatten()}")
        except Exception as e:
            print(f"{name:8}: Error - {e}")
    
    print("\n📈 Key Properties Summary:")
    print("-" * 40)
    print("ReLU     : Range [0, ∞), sparse, fast")
    print("Sigmoid  : Range (0, 1), smooth, probability-like")
    print("Tanh     : Range (-1, 1), zero-centered, symmetric")
    print("Softmax  : Probability distribution, sums to 1")
    
    print("\n🎯 When to Use Each:")
    print("-" * 40)
    print("ReLU     : Hidden layers, CNNs, most deep networks")
    print("Sigmoid  : Binary classification, gates, probabilities")
    print("Tanh     : RNNs, when you need zero-centered output")
    print("Softmax  : Multi-class classification, attention")
    
    # Show comprehensive visualization if available
    if _should_show_plots():
        print("\n🎨 Generating comprehensive comparison plot...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Activation Function Comparison', fontsize=16)
            
            x_vals = np.linspace(-5, 5, 100)
            
            # Plot each activation function
            for i, (name, activation) in enumerate(list(activations.items())[:3]):  # Skip Softmax for now
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                y_vals = []
                for x in x_vals:
                    try:
                        input_tensor = Tensor([[x]])
                        output = activation(input_tensor)
                        y_vals.append(output.data.item())
                    except:
                        y_vals.append(0)
                
                ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                ax.set_title(f'{name} Activation')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Input (x)')
                ax.set_ylabel(f'{name}(x)')
            
            # Special handling for Softmax
            ax = axes[1, 1]
            sample_inputs = np.array([[1, 2, 3], [0, 0, 0], [-1, 0, 1]])
            softmax_results = []
            
            for inp in sample_inputs:
                result = softmax(Tensor([inp]))
                softmax_results.append(result.data.flatten())
            
            x_pos = np.arange(len(sample_inputs))
            width = 0.25
            
            for i in range(3):  # 3 classes
                values = [result[i] for result in softmax_results]
                ax.bar(x_pos + i * width, values, width, label=f'Class {i+1}')
            
            ax.set_title('Softmax Activation')
            ax.set_xlabel('Input Examples')
            ax.set_ylabel('Probability')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(['[1,2,3]', '[0,0,0]', '[-1,0,1]'])
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("   📊 Matplotlib not available - skipping comprehensive plot")
        except Exception as e:
            print(f"   ⚠️  Comprehensive plot error: {e}")
    
except Exception as e:
    print(f"❌ Error in comprehensive comparison: {e}")

print("\n" + "=" * 60)
print("🎉 Congratulations! You've implemented all four activation functions!")
print("You now understand the building blocks that make neural networks intelligent.")
print("=" * 60) 