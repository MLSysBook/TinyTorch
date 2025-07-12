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

## Build → Use → Reflect
1. **Build**: Activation functions that add nonlinearity
2. **Use**: Transform tensors and see immediate results
3. **Reflect**: How nonlinearity enables complex pattern learning

## Module Dependencies
This module builds on the **tensor** module:
- **tensor** → **activations** → **layers** → **networks**
- Clean separation: data structures → math functions → building blocks → complete systems
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/activations/activations_dev.py`  
**Building Side:** Code exports to `tinytorch.core.activations`

```python
# Final package structure:
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.functional`
- **Consistency:** All activation functions live together in `core.activations`
"""

# %%
#| default_exp core.activations

# Setup and imports
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Union, List

# Import our Tensor class
from tinytorch.core.tensor import Tensor

print("🔥 TinyTorch Activations Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build activation functions!")

# %%
#| export
import math
import numpy as np
import sys
from typing import Union, List

# Import our Tensor class
from tinytorch.core.tensor import Tensor

# %%
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    return 'pytest' not in sys.modules and 'test' not in sys.argv

# %% [markdown]
"""
## Step 1: What is an Activation Function?

### Definition
An **activation function** is a mathematical function that adds nonlinearity to neural networks. It transforms the output of a layer before passing it to the next layer.

### Why Activation Functions Matter
**Without activation functions, neural networks are just linear transformations!**

```
Linear → Linear → Linear = Still just Linear
Linear → Activation → Linear = Can learn complex patterns!
```

**The fundamental insight**: Activation functions add **nonlinearity**, allowing networks to learn complex patterns that linear functions cannot capture.

### Real-World Examples
- **ReLU**: Detects when features are "active" (positive)
- **Sigmoid**: Outputs probabilities between 0 and 1
- **Tanh**: Outputs values between -1 and 1 (centered)
- **Softmax**: Converts logits to probability distributions

### Visual Intuition
```
Input: [-2, -1, 0, 1, 2]
ReLU:   [0,  0, 0, 1, 2]  (clips negatives to 0)
Sigmoid: [0.1, 0.3, 0.5, 0.7, 0.9]  (squashes to 0-1)
Tanh:    [-0.9, -0.8, 0, 0.8, 0.9]  (squashes to -1 to 1)
```

Let's implement these step by step and **visualize** how they work!
"""

# %%
#| export
import math
import numpy as np
import sys
from typing import Union, List

# Import our Tensor class
from tinytorch.core.tensor import Tensor

# %%
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    return 'pytest' not in sys.modules and 'test' not in sys.argv

# %% [markdown]
"""
## 🎨 Activation Visualization Tools

Let's create visualization functions to see how activation functions transform data.
These visualizations help you understand what each activation function does!
"""

# %%
def visualize_activation_function(activation_func, name: str, x_range=(-5, 5), num_points=100):
    """
    Visualize how an activation function transforms inputs.
    
    Args:
        activation_func: Activation function to visualize
        name: Name of the activation function
        x_range: Range of x values to plot
        num_points: Number of points to plot
        
    NOTE: This is a development/learning tool, not exported to the package.
    """
    if not _should_show_plots():
        print(f"📊 Visualization for {name} disabled during testing")
        return
    
    # Create input range
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    x_tensor = Tensor(x_vals.reshape(1, -1))
    
    # Apply activation function
    y_tensor = activation_func(x_tensor)
    y_vals = y_tensor.data.flatten()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'{name}(x)')
    plt.plot(x_vals, x_vals, 'r--', alpha=0.5, label='y=x (identity)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title(f'{name} Activation Function')
    plt.legend()
    plt.show()
    
    # Show key properties
    print(f"📊 {name} Properties:")
    print(f"   Range: [{y_vals.min():.2f}, {y_vals.max():.2f}]")
    print(f"   At x=0: {activation_func(Tensor([[0.0]])).data[0,0]:.2f}")
    print(f"   At x=1: {activation_func(Tensor([[1.0]])).data[0,0]:.2f}")
    print(f"   At x=-1: {activation_func(Tensor([[-1.0]])).data[0,0]:.2f}")

# %%
def compare_activations(activations_dict, x_range=(-5, 5), num_points=100):
    """
    Compare multiple activation functions side by side.
    
    Args:
        activations_dict: Dictionary of {name: activation_function}
        x_range: Range of x values to plot
        num_points: Number of points to plot
        
    NOTE: This is a development/learning tool, not exported to the package.
    """
    if not _should_show_plots():
        print("📊 Activation comparison disabled during testing")
        return
    
    # Create input range
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    x_tensor = Tensor(x_vals.reshape(1, -1))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (name, activation_func) in enumerate(activations_dict.items()):
        # Apply activation function
        y_tensor = activation_func(x_tensor)
        y_vals = y_tensor.data.flatten()
        
        # Plot
        color = colors[i % len(colors)]
        plt.plot(x_vals, y_vals, color=color, linewidth=2, label=f'{name}(x)')
    
    # Add reference lines
    plt.plot(x_vals, x_vals, 'k--', alpha=0.3, label='y=x (identity)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    plt.axhline(y=1, color='k', linestyle=':', alpha=0.2)
    plt.axhline(y=-1, color='k', linestyle=':', alpha=0.2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Activation Functions Comparison')
    plt.legend()
    plt.show()

# %%
def visualize_activation_on_data(activation_func, name: str, input_data: Tensor):
    """
    Visualize how an activation function transforms actual data.
    
    Args:
        activation_func: Activation function to test
        name: Name of the activation function
        input_data: Input tensor with real data
        
    NOTE: This is a development/learning tool, not exported to the package.
    """
    if not _should_show_plots():
        print(f"📊 Data visualization for {name} disabled during testing")
        return
    
    # Apply activation
    output_data = activation_func(input_data)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input distribution
    input_flat = input_data.data.flatten()
    axes[0].hist(input_flat, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Input Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Output distribution
    output_flat = output_data.data.flatten()
    axes[1].hist(output_flat, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title(f'{name} Output Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Input vs Output scatter
    axes[2].scatter(input_flat, output_flat, alpha=0.6, color='green')
    axes[2].set_xlabel('Input Values')
    axes[2].set_ylabel('Output Values')
    axes[2].set_title(f'Input vs {name} Output')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show statistics
    print(f"📊 {name} Transformation Statistics:")
    print(f"   Input range: [{input_flat.min():.2f}, {input_flat.max():.2f}]")
    print(f"   Output range: [{output_flat.min():.2f}, {output_flat.max():.2f}]")
    print(f"   Input mean: {input_flat.mean():.2f}, Output mean: {output_flat.mean():.2f}")
    print(f"   Input std: {input_flat.std():.2f}, Output std: {output_flat.std():.2f}")

# %% [markdown]
"""
## Step 2: ReLU Activation Function

**ReLU** (Rectified Linear Unit) is the most popular activation function in deep learning.

### What is ReLU?
- **Formula**: `f(x) = max(0, x)`
- **Behavior**: Keeps positive values unchanged, sets negative values to zero
- **Range**: [0, ∞) - unbounded above, bounded below at zero

### Why ReLU is Popular
- **Simple**: Easy to compute and understand
- **Sparse**: Outputs exactly zero for negative inputs (sparsity)
- **Non-saturating**: Doesn't suffer from vanishing gradients
- **Computationally efficient**: Just a max operation

### Real-World Analogy
Think of ReLU as a **threshold detector**:
- If a feature is "active" (positive), let it through
- If a feature is "inactive" (negative), ignore it
- Like a neuron that only fires when stimulated enough
"""

# %%
#| export
class ReLU:
    """
    ReLU Activation: f(x) = max(0, x)
    
    The most popular activation function in deep learning.
    Simple, effective, and computationally efficient.
    
    TODO: Implement ReLU activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Apply element-wise max(0, x) operation
    3. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[-3, -1, 0, 1, 3]])
    Output: Tensor([[0, 0, 0, 1, 3]])
    
    HINTS:
    - Use x.data to get the numpy array
    - Use np.maximum(0, x.data) for element-wise max
    - Return Tensor(result) to wrap the result
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU: f(x) = max(0, x)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU applied element-wise
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
# Test ReLU implementation
print("Testing ReLU Implementation:")
print("=" * 40)

try:
    relu = ReLU()
    
    # Test 1: Basic functionality
    test_input = Tensor([[-3, -1, 0, 1, 3]])
    output = relu(test_input)
    expected = [0, 0, 0, 1, 3]
    
    print(f"✅ Input: {test_input.data.flatten()}")
    print(f"✅ Output: {output.data.flatten()}")
    print(f"✅ Expected: {expected}")
    
    # Check if implementation is correct
    if np.allclose(output.data.flatten(), expected):
        print("🎉 ReLU implementation is CORRECT!")
    else:
        print("❌ ReLU implementation needs fixing")
        print("   Make sure negative values become 0, positive values stay unchanged")
    
    # Test 2: Edge cases
    edge_cases = Tensor([[0.0, -0.0, 1e-10, -1e-10]])
    edge_output = relu(edge_cases)
    print(f"✅ Edge cases: {edge_cases.data.flatten()}")
    print(f"✅ Edge output: {edge_output.data.flatten()}")
    
    print("✅ ReLU tests complete!")
    
    # 🎨 Visualize ReLU behavior (development only)
    if _should_show_plots():
        print("\n🎨 Visualizing ReLU behavior...")
        visualize_activation_function(relu, "ReLU", x_range=(-3, 3))
        
        # Show ReLU with real data
        sample_data = Tensor([[-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5]])
        visualize_activation_on_data(relu, "ReLU", sample_data)
    
except NotImplementedError:
    print("⚠️  ReLU not implemented yet - complete the forward method above!")
except Exception as e:
    print(f"❌ Error in ReLU: {e}")
    print("   Check your implementation in the forward method")

print()  # Add spacing

# %% [markdown]
"""
## Step 3: Sigmoid Activation Function

**Sigmoid** is the classic activation function that squashes values to the range (0, 1).

### What is Sigmoid?
- **Formula**: `f(x) = 1 / (1 + e^(-x))`
- **Behavior**: Smoothly maps any real number to (0, 1)
- **Range**: (0, 1) - always positive, never exactly 0 or 1

### Why Sigmoid is Useful
- **Probability interpretation**: Output can be interpreted as probability
- **Smooth**: Differentiable everywhere (good for gradients)
- **Bounded**: Output is always between 0 and 1
- **S-shaped curve**: Gradual transition from 0 to 1

### Real-World Analogy
Think of Sigmoid as a **smooth switch**:
- Large negative inputs → close to 0 (off)
- Large positive inputs → close to 1 (on)
- Around zero → gradual transition (50% on)
"""

# %%
#| export
class Sigmoid:
    """
    Sigmoid Activation: f(x) = 1 / (1 + e^(-x))
    
    Classic activation function that outputs probabilities.
    Smooth, bounded, and differentiable.
    
    TODO: Implement Sigmoid activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Apply sigmoid formula: 1 / (1 + exp(-x))
    3. Handle numerical stability (clip extreme values)
    4. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[-3, -1, 0, 1, 3]])
    Output: Tensor([[0.047, 0.269, 0.5, 0.731, 0.953]])
    
    HINTS:
    - Use x.data to get the numpy array
    - Use np.exp(-x.data) for the exponential
    - Consider np.clip(x.data, -500, 500) for numerical stability
    - Return Tensor(result) to wrap the result
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Sigmoid: f(x) = 1 / (1 + e^(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid applied element-wise
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
# Test Sigmoid implementation
print("Testing Sigmoid Implementation:")
print("=" * 40)

try:
    sigmoid = Sigmoid()
    
    # Test 1: Basic functionality
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    output = sigmoid(test_input)
    
    print(f"✅ Input: {test_input.data.flatten()}")
    print(f"✅ Output: {output.data.flatten()}")
    
    # Check properties
    all_positive = np.all(output.data > 0)
    all_less_than_one = np.all(output.data < 1)
    zero_maps_to_half = abs(sigmoid(Tensor([0])).data[0] - 0.5) < 1e-6
    
    print(f"✅ All outputs positive: {all_positive}")
    print(f"✅ All outputs < 1: {all_less_than_one}")
    print(f"✅ Sigmoid(0) ≈ 0.5: {zero_maps_to_half}")
    
    if all_positive and all_less_than_one and zero_maps_to_half:
        print("🎉 Sigmoid implementation is CORRECT!")
    else:
        print("❌ Sigmoid implementation needs fixing")
        print("   Make sure: 0 < output < 1 and sigmoid(0) = 0.5")
    
    # Test 2: Numerical stability
    extreme_values = Tensor([[-1000, 1000]])
    extreme_output = sigmoid(extreme_values)
    print(f"✅ Extreme values: {extreme_values.data.flatten()}")
    print(f"✅ Extreme output: {extreme_output.data.flatten()}")
    
    # Should not have NaN or inf
    no_nan_inf = not (np.isnan(extreme_output.data).any() or np.isinf(extreme_output.data).any())
    print(f"✅ No NaN/Inf: {no_nan_inf}")
    
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
## Step 4: Tanh Activation Function

**Tanh** (Hyperbolic Tangent) is like Sigmoid but centered at zero.

### What is Tanh?
- **Formula**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- **Behavior**: Smoothly maps any real number to (-1, 1)
- **Range**: (-1, 1) - symmetric around zero

### Why Tanh is Useful
- **Zero-centered**: Output is centered around 0 (unlike Sigmoid)
- **Stronger gradients**: Steeper slope than Sigmoid
- **Symmetric**: Treats positive and negative inputs equally
- **Bounded**: Output is always between -1 and 1

### Real-World Analogy
Think of Tanh as a **balanced switch**:
- Large negative inputs → close to -1 (strongly negative)
- Large positive inputs → close to +1 (strongly positive)
- Around zero → gradual transition (neutral)
"""

# %%
#| export
class Tanh:
    """
    Tanh Activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Zero-centered activation function with stronger gradients.
    Symmetric and bounded between -1 and 1.
    
    TODO: Implement Tanh activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Apply tanh formula or use np.tanh()
    3. Handle numerical stability if needed
    4. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[-3, -1, 0, 1, 3]])
    Output: Tensor([[-0.995, -0.762, 0, 0.762, 0.995]])
    
    HINTS:
    - Use x.data to get the numpy array
    - Use np.tanh(x.data) for the hyperbolic tangent
    - Or implement manually: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    - Return Tensor(result) to wrap the result
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh applied element-wise
        """
        raise NotImplementedError("Student implementation required")
        
    def __call__(self, x: Tensor) -> Tensor:
        """Allow calling the activation like a function: tanh(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Tanh Implementation

Let's test your Tanh implementation to ensure it's working correctly:
"""

# %%
# Test Tanh implementation
print("Testing Tanh Implementation:")
print("=" * 40)

try:
    tanh = Tanh()
    
    # Test 1: Basic functionality
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    output = tanh(test_input)
    
    print(f"✅ Input: {test_input.data.flatten()}")
    print(f"✅ Output: {output.data.flatten()}")
    
    # Check properties
    in_range = np.all(np.abs(output.data) < 1)
    zero_maps_to_zero = abs(tanh(Tensor([0])).data[0]) < 1e-6
    symmetric = np.allclose(tanh(Tensor([1])).data, -tanh(Tensor([-1])).data)
    
    print(f"✅ All outputs in (-1, 1): {in_range}")
    print(f"✅ Tanh(0) ≈ 0: {zero_maps_to_zero}")
    print(f"✅ Symmetric (tanh(-x) = -tanh(x)): {symmetric}")
    
    if in_range and zero_maps_to_zero and symmetric:
        print("🎉 Tanh implementation is CORRECT!")
    else:
        print("❌ Tanh implementation needs fixing")
        print("   Make sure: -1 < output < 1, tanh(0) = 0, and tanh(-x) = -tanh(x)")
    
    # Test 2: Compare with expected values
    expected_values = {
        0: 0.0,
        1: 0.7616,  # approximately
        -1: -0.7616,  # approximately
    }
    
    for input_val, expected in expected_values.items():
        actual = tanh(Tensor([input_val])).data[0]
        close = abs(actual - expected) < 0.001
        print(f"✅ Tanh({input_val}) ≈ {expected}: {close} (got {actual:.4f})")
    
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
## Step 5: Softmax Activation Function

**Softmax** converts logits into probability distributions - essential for multi-class classification.

### What is Softmax?
- **Formula**: `f(x_i) = e^(x_i) / sum(e^(x_j) for all j)`
- **Behavior**: Converts any vector to a probability distribution
- **Range**: (0, 1) with sum = 1

### Why Softmax is Essential
- **Probability distribution**: Outputs sum to 1.0
- **Multi-class classification**: Each class gets a probability
- **Differentiable**: Smooth gradients for training
- **Competitive**: Emphasizes the largest input (winner-take-all effect)

### Real-World Analogy
Think of Softmax as **voting with confidence**:
- Input: [2, 1, 0] (raw scores)
- Softmax: [0.67, 0.24, 0.09] (probabilities)
- The highest score gets the most probability, but others still get some
"""

# %%
#| export
class Softmax:
    """
    Softmax Activation: f(x_i) = e^(x_i) / sum(e^(x_j) for all j)
    
    Converts logits to probability distributions.
    Essential for multi-class classification.
    
    TODO: Implement Softmax activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Subtract max for numerical stability: x - max(x)
    3. Compute exponentials: exp(x_stable)
    4. Normalize by sum: exp_vals / sum(exp_vals)
    5. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[2, 1, 0]])
    Output: Tensor([[0.665, 0.245, 0.090]]) (sums to 1.0)
    
    HINTS:
    - Use x.data to get the numpy array
    - Use np.max(x.data, axis=-1, keepdims=True) for stability
    - Use np.exp() for exponentials
    - Use np.sum() for normalization
    - Return Tensor(result) to wrap the result
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Softmax: f(x_i) = e^(x_i) / sum(e^(x_j) for all j)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Softmax applied (probabilities sum to 1)
        """
        raise NotImplementedError("Student implementation required")
        
    def __call__(self, x: Tensor) -> Tensor:
        """Allow calling the activation like a function: softmax(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Softmax Implementation

Let's test your Softmax implementation to ensure it's working correctly:
"""

# %%
# Test Softmax implementation
print("Testing Softmax Implementation:")
print("=" * 40)

try:
    softmax = Softmax()
    
    # Test 1: Basic functionality
    test_input = Tensor([[2, 1, 0]])
    output = softmax(test_input)
    
    print(f"✅ Input: {test_input.data.flatten()}")
    print(f"✅ Output: {output.data.flatten()}")
    
    # Check properties
    all_positive = np.all(output.data > 0)
    sums_to_one = abs(np.sum(output.data) - 1.0) < 1e-6
    largest_input_largest_output = np.argmax(test_input.data) == np.argmax(output.data)
    
    print(f"✅ All outputs positive: {all_positive}")
    print(f"✅ Sum equals 1.0: {sums_to_one} (sum = {np.sum(output.data):.6f})")
    print(f"✅ Largest input → largest output: {largest_input_largest_output}")
    
    if all_positive and sums_to_one and largest_input_largest_output:
        print("🎉 Softmax implementation is CORRECT!")
    else:
        print("❌ Softmax implementation needs fixing")
        print("   Make sure: all outputs > 0, sum = 1.0, and largest input gets largest probability")
    
    # Test 2: Numerical stability
    extreme_input = Tensor([[1000, 999, 998]])
    extreme_output = softmax(extreme_input)
    print(f"✅ Extreme input: {extreme_input.data.flatten()}")
    print(f"✅ Extreme output: {extreme_output.data.flatten()}")
    
    # Should not have NaN or inf
    no_nan_inf = not (np.isnan(extreme_output.data).any() or np.isinf(extreme_output.data).any())
    extreme_sums_to_one = abs(np.sum(extreme_output.data) - 1.0) < 1e-6
    
    print(f"✅ No NaN/Inf: {no_nan_inf}")
    print(f"✅ Extreme case sums to 1: {extreme_sums_to_one}")
    
    # Test 3: Equal inputs should give equal probabilities
    equal_input = Tensor([[1, 1, 1]])
    equal_output = softmax(equal_input)
    expected_prob = 1.0 / 3.0
    all_equal = np.allclose(equal_output.data, expected_prob)
    print(f"✅ Equal inputs → equal probabilities: {all_equal}")
    print(f"   Expected: {expected_prob:.3f}, Got: {equal_output.data.flatten()}")
    
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
        
        print("\n   🔍 Scale invariance test:")
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
            if name == 'Softmax':
                # Softmax is special - it's a vector function
                output = activation(test_data)
                print(f"{name:>8}: {output.data.flatten()} (sum={output.data.sum():.3f})")
            else:
                output = activation(test_data)
                print(f"{name:>8}: {output.data.flatten()}")
        except NotImplementedError:
            print(f"{name:>8}: Not implemented yet")
    
    print()
    
    # 🎨 Visual comparison (development only)
    if _should_show_plots():
        print("🎨 Visual Comparison of Activation Functions")
        
        # Compare the three main activation functions (excluding Softmax since it's different)
        comparison_activations = {
            'ReLU': ReLU(),
            'Sigmoid': Sigmoid(),
            'Tanh': Tanh()
        }
        
        compare_activations(comparison_activations, x_range=(-5, 5))
        
        # Show practical data transformation example
        print("\n🔍 Real Data Transformation Example:")
        
        # Create realistic data (like layer outputs)
        realistic_data = Tensor(np.random.randn(1, 20))  # 20 random values
        
        print(f"Sample data statistics:")
        print(f"  Mean: {realistic_data.data.mean():.3f}")
        print(f"  Std: {realistic_data.data.std():.3f}")
        print(f"  Range: [{realistic_data.data.min():.3f}, {realistic_data.data.max():.3f}]")
        
        # Show how each activation transforms this data
        for name, activation in comparison_activations.items():
            try:
                transformed = activation(realistic_data)
                print(f"\n{name} transformation:")
                print(f"  Mean: {transformed.data.mean():.3f}")
                print(f"  Std: {transformed.data.std():.3f}")
                print(f"  Range: [{transformed.data.min():.3f}, {transformed.data.max():.3f}]")
            except NotImplementedError:
                print(f"\n{name}: Not implemented yet")
    
    print("\n✅ Activation function comparison complete!")
    
except Exception as e:
    print(f"❌ Error in comparison: {e}")

print()  # Add spacing

# %% [markdown]
"""
## 🎯 Activation Function Cheat Sheet

### When to Use Each Activation Function

| Activation | Use Case | Advantages | Disadvantages |
|------------|----------|------------|---------------|
| **ReLU** | Hidden layers | Fast, sparse, no vanishing gradients | Dead neurons, not zero-centered |
| **Sigmoid** | Binary classification output | Smooth, interpretable probabilities | Vanishing gradients, not zero-centered |
| **Tanh** | Hidden layers (alternative to ReLU) | Zero-centered, stronger gradients | Vanishing gradients for extreme values |
| **Softmax** | Multi-class classification output | Probability distribution, differentiable | Only for final layer |

### Quick Decision Guide

```python
# For hidden layers:
activation = ReLU()  # Most common choice

# For binary classification (final layer):
activation = Sigmoid()  # Outputs probability

# For multi-class classification (final layer):
activation = Softmax()  # Outputs probability distribution

# For hidden layers with negative values:
activation = Tanh()  # Zero-centered alternative
```

### Key Properties Summary

- **ReLU**: `f(x) = max(0, x)` → Range: [0, ∞)
- **Sigmoid**: `f(x) = 1/(1+e^(-x))` → Range: (0, 1)
- **Tanh**: `f(x) = tanh(x)` → Range: (-1, 1)
- **Softmax**: `f(x_i) = e^(x_i)/Σe^(x_j)` → Range: (0, 1), Sum = 1
"""

# %% [markdown]
"""
## Reflection: The Power of Nonlinearity

Now that you've implemented these activation functions, let's reflect on why they're so important:

### Without Activation Functions
```python
# This is just a linear transformation:
y = W3 @ (W2 @ (W1 @ x + b1) + b2) + b3
# Which simplifies to:
y = W_combined @ x + b_combined
```

### With Activation Functions
```python
# This can learn complex patterns:
h1 = activation(W1 @ x + b1)
h2 = activation(W2 @ h1 + b2)
y = W3 @ h2 + b3
```

### Key Insights
1. **Nonlinearity enables complexity**: Without activations, networks are just linear algebra
2. **Different activations for different purposes**: ReLU for hidden layers, Sigmoid for binary classification, Softmax for multi-class
3. **Activation choice matters**: The right activation can make training faster and more stable
4. **Composition creates power**: Stacking many simple nonlinear transformations creates arbitrarily complex functions

### Next Steps
In the next module (layers), you'll see how these activation functions combine with linear transformations to create the building blocks of neural networks!
"""

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

class Sigmoid:
    """Sigmoid Activation: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        # Clip for numerical stability
        clipped = np.clip(x.data, -500, 500)
        result = 1 / (1 + np.exp(-clipped))
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class Tanh:
    """Tanh Activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class Softmax:
    """Softmax Activation: f(x_i) = e^(x_i) / sum(e^(x_j) for all j)"""
    
    def forward(self, x: Tensor) -> Tensor:
        # Subtract max for numerical stability
        x_stable = x.data - np.max(x.data, axis=-1, keepdims=True)
        exp_vals = np.exp(x_stable)
        result = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# Export list
__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax'] 