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
# 🔥 TinyTorch Activations Module

Welcome to the **Activations** module! This is where you'll implement the mathematical functions that give neural networks their power.

## 🎯 Learning Objectives

By the end of this module, you will:
1. **Understand** why activation functions are essential for neural networks
2. **Implement** the three most important activation functions: ReLU, Sigmoid, and Tanh
3. **Test** your functions with various inputs to understand their behavior
4. **Use** these functions as building blocks for neural networks

## 🧠 Why Activation Functions Matter

**Without activation functions, neural networks are just linear transformations!**

```
Linear → Linear → Linear = Still just Linear
Linear → Activation → Linear = Can learn complex patterns!
```

**Key insight**: Activation functions add **nonlinearity**, allowing networks to learn complex patterns that linear functions cannot capture.

## 📚 What You'll Build

- **ReLU**: `f(x) = max(0, x)` - The workhorse of deep learning
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))` - Squashes to (0, 1)
- **Tanh**: `f(x) = tanh(x)` - Squashes to (-1, 1)

Each function serves different purposes and has different mathematical properties.

---

Let's start building! 🚀
"""

# %%
#| default_exp core.activations

# Standard library imports
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# TinyTorch imports
from tinytorch.core.tensor import Tensor

# %%
# Helper function to detect if we're in a testing environment
def _should_show_plots():
    """
    Determine if we should show plots based on the execution context.
    
    Returns False if:
    - Running in pytest (detected by 'pytest' in sys.modules)
    - Running in test environment (detected by environment variables)
    - Running from command line test runner
    
    Returns True if:
    - Running in Jupyter notebook
    - Running interactively in Python
    """
    # Check if we're running in pytest
    if 'pytest' in sys.modules:
        return False
    
    # Check if we're in a test environment
    if os.environ.get('PYTEST_CURRENT_TEST'):
        return False
    
    # Check if we're running from a test file (more specific check)
    if any(arg.endswith('.py') and 'test_' in os.path.basename(arg) and 'tests/' in arg for arg in sys.argv):
        return False
    
    # Check if we're running from the tito CLI test command
    if len(sys.argv) > 0 and 'tito.py' in sys.argv[0] and 'test' in sys.argv:
        return False
    
    # Default to showing plots (notebook/interactive environment)
    return True

# %% [markdown]
"""
## Step 1: ReLU Activation Function

**ReLU** (Rectified Linear Unit) is the most popular activation function in deep learning.

**Formula**: `f(x) = max(0, x)`

**Properties**:
- **Simple**: Easy to compute and understand
- **Sparse**: Outputs exactly zero for negative inputs
- **Unbounded**: No upper limit on positive outputs
- **Non-saturating**: Doesn't suffer from vanishing gradients

**When to use**: Almost everywhere! It's the default choice for hidden layers.
"""

# %%
#| export
class ReLU:
    """
    ReLU Activation: f(x) = max(0, x)
    
    The most popular activation function in deep learning.
    Simple, effective, and computationally efficient.
    
    TODO: Implement ReLU activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU: f(x) = max(0, x)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU applied element-wise
            
        TODO: Implement element-wise max(0, x) operation
        Hint: Use np.maximum(0, x.data)
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make activation callable: relu(x) same as relu.forward(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class ReLU:
    """ReLU Activation: f(x) = max(0, x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU: f(x) = max(0, x)"""
        return Tensor(np.maximum(0, x.data))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your ReLU Function

Once you implement ReLU above, run this cell to test it:
"""

# %%
# Test ReLU function
try:
    print("=== Testing ReLU Function ===")
    
    # Test data: mix of positive, negative, and zero
    x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    print(f"Input: {x.data}")
    
    # Test ReLU
    relu = ReLU()
    y = relu(x)
    print(f"ReLU output: {y.data}")
    print(f"Expected: [[0. 0. 0. 1. 3.]]")
    
    # Test with different shapes
    x_2d = Tensor([[-2.0, 1.0], [0.5, -0.5]])
    y_2d = relu(x_2d)
    print(f"\n2D Input: {x_2d.data}")
    print(f"2D ReLU output: {y_2d.data}")
    
    print("✅ ReLU working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the ReLU function above!")

# %% [markdown]
"""
### 📊 Visualize ReLU Function

Let's plot the ReLU function to see how it transforms inputs:
"""

# %%
# Plot ReLU function
try:
    print("=== Plotting ReLU Function ===")
    
    # Create a range of input values
    x_range = np.linspace(-5, 5, 100)
    x_tensor = Tensor([x_range])
    
    # Apply ReLU (student implementation)
    relu = ReLU()
    y_tensor = relu(x_tensor)
    y_range = y_tensor.data[0]
    
    # Create ideal ReLU for comparison
    y_ideal = np.maximum(0, x_range)
    
    # Only show plots if we're not in a testing environment
    if _should_show_plots():
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot both student implementation and ideal
        plt.subplot(2, 2, 1)
        plt.plot(x_range, y_range, 'b-', linewidth=3, label='Your ReLU Implementation')
        plt.plot(x_range, y_ideal, 'r--', linewidth=2, alpha=0.7, label='Ideal ReLU')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Output')
        plt.title('ReLU: Your Implementation vs Ideal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-5, 5)
        plt.ylim(-1, 5)
        
        # Mathematical explanation plot
        plt.subplot(2, 2, 2)
        # Show the mathematical definition
        x_math = np.array([-3, -2, -1, 0, 1, 2, 3])
        y_math = np.maximum(0, x_math)
        plt.stem(x_math, y_math, basefmt=' ', linefmt='g-', markerfmt='go')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('max(0, x)')
        plt.title('Mathematical Definition: max(0, x)')
        plt.grid(True, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-0.5, 3.5)
        
        # Show the piecewise nature
        plt.subplot(2, 2, 3)
        x_left = np.linspace(-5, 0, 50)
        x_right = np.linspace(0, 5, 50)
        plt.plot(x_left, np.zeros_like(x_left), 'r-', linewidth=3, label='f(x) = 0 for x < 0')
        plt.plot(x_right, x_right, 'b-', linewidth=3, label='f(x) = x for x ≥ 0')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Output')
        plt.title('Piecewise Function Definition')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-5, 5)
        plt.ylim(-1, 5)
        
        # Error analysis
        plt.subplot(2, 2, 4)
        difference = np.abs(y_range - y_ideal)
        max_error = np.max(difference)
        plt.plot(x_range, difference, 'purple', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('|Your Output - Ideal Output|')
        plt.title(f'Implementation Error (Max: {max_error:.6f})')
        plt.grid(True, alpha=0.3)
        plt.xlim(-5, 5)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print(f"\n📊 Analysis:")
        print(f"✅ Maximum error: {max_error:.10f}")
        if max_error < 1e-10:
            print("🎉 Perfect implementation!")
        elif max_error < 1e-6:
            print("🌟 Excellent implementation!")
        elif max_error < 1e-3:
            print("👍 Good implementation!")
        else:
            print("🔧 Implementation needs work.")
            
        print(f"📈 Function properties:")
        print(f"   • Range: [0, ∞)")
        print(f"   • Piecewise: f(x) = 0 for x < 0, f(x) = x for x ≥ 0")
        print(f"   • Monotonic: Always increasing for x ≥ 0")
        print(f"   • Sparse: Exactly zero for negative inputs")
    else:
        print("📊 Plots disabled during testing - this is normal!")
        
    # Always show the mathematical analysis
    difference = np.abs(y_range - y_ideal)
    max_error = np.max(difference)
    print(f"\n📊 Mathematical Analysis:")
    print(f"✅ Maximum error: {max_error:.10f}")
    if max_error < 1e-10:
        print("🎉 Perfect implementation!")
    elif max_error < 1e-6:
        print("🌟 Excellent implementation!")
    elif max_error < 1e-3:
        print("👍 Good implementation!")
    else:
        print("🔧 Implementation needs work.")
        
except Exception as e:
    print(f"❌ Error in plotting: {e}")
    print("Make sure to implement the ReLU function above!")

# %% [markdown]
"""
## Step 2: Sigmoid Activation Function

**Sigmoid** squashes any input to the range (0, 1), making it useful for probabilities.

**Formula**: `f(x) = 1 / (1 + e^(-x))`

**Properties**:
- **Bounded**: Always outputs between 0 and 1
- **Smooth**: Differentiable everywhere
- **S-shaped**: Smooth transition from 0 to 1
- **Saturating**: Can suffer from vanishing gradients

**When to use**: Binary classification (final layer), gates in RNNs/LSTMs.

**⚠️ Numerical Stability**: Be careful with large inputs to avoid overflow!
"""

# %%
#| export
class Sigmoid:
    """
    Sigmoid Activation: f(x) = 1 / (1 + e^(-x))
    
    Squashes input to range (0, 1). Often used for binary classification.
    
    TODO: Implement Sigmoid activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Sigmoid: f(x) = 1 / (1 + e^(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid applied element-wise
            
        TODO: Implement sigmoid function (be careful with numerical stability!)
        
        Hint: For numerical stability, use:
        - For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
        - For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| hide
#| export
class Sigmoid:
    """Sigmoid Activation: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid with numerical stability"""
        # Use the numerically stable version to avoid overflow
        # For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
        # For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
        x_data = x.data
        result = np.zeros_like(x_data)
        
        # Stable computation
        positive_mask = x_data >= 0
        result[positive_mask] = 1.0 / (1.0 + np.exp(-x_data[positive_mask]))
        result[~positive_mask] = np.exp(x_data[~positive_mask]) / (1.0 + np.exp(x_data[~positive_mask]))
        
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Sigmoid Function

Once you implement Sigmoid above, run this cell to test it:
"""

# %%
# Test Sigmoid function
try:
    print("=== Testing Sigmoid Function ===")
    
    # Test data: mix of positive, negative, and zero
    x = Tensor([[-5.0, -1.0, 0.0, 1.0, 5.0]])
    print(f"Input: {x.data}")
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    y = sigmoid(x)
    print(f"Sigmoid output: {y.data}")
    print("Expected: values between 0 and 1")
    print(f"All values in (0,1)? {np.all((y.data > 0) & (y.data < 1))}")
    
    # Test specific values
    x_zero = Tensor([[0.0]])
    y_zero = sigmoid(x_zero)
    print(f"\nSigmoid(0) = {y_zero.data[0, 0]:.4f} (should be 0.5)")
    
    # Test extreme values (numerical stability)
    x_extreme = Tensor([[-100.0, 100.0]])
    y_extreme = sigmoid(x_extreme)
    print(f"Sigmoid([-100, 100]) = {y_extreme.data}")
    print("Should be close to [0, 1] without overflow errors")
    
    print("✅ Sigmoid working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Sigmoid function above!")

# %% [markdown]
"""
### 📊 Visualize Sigmoid Function

Let's plot the Sigmoid function to see its S-shaped curve:
"""

# %%
# Plot Sigmoid function
try:
    print("=== Plotting Sigmoid Function ===")
    
    # Create a range of input values
    x_range = np.linspace(-10, 10, 100)
    x_tensor = Tensor([x_range])
    
    # Apply Sigmoid (student implementation)
    sigmoid = Sigmoid()
    y_tensor = sigmoid(x_tensor)
    y_range = y_tensor.data[0]
    
    # Create ideal Sigmoid for comparison
    y_ideal = 1.0 / (1.0 + np.exp(-x_range))
    
    # Only show plots if we're not in a testing environment
    if _should_show_plots():
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot both student implementation and ideal
        plt.subplot(2, 2, 1)
        plt.plot(x_range, y_range, 'g-', linewidth=3, label='Your Sigmoid Implementation')
        plt.plot(x_range, y_ideal, 'r--', linewidth=2, alpha=0.7, label='Ideal Sigmoid')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='y = 0.5')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Output')
        plt.title('Sigmoid: Your Implementation vs Ideal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-0.1, 1.1)
        
        # Mathematical explanation plot
        plt.subplot(2, 2, 2)
        # Show key points
        x_key = np.array([-5, -2, -1, 0, 1, 2, 5])
        y_key = 1.0 / (1.0 + np.exp(-x_key))
        plt.stem(x_key, y_key, basefmt=' ', linefmt='orange', markerfmt='o')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('1/(1+e^(-x))')
        plt.title('Mathematical Definition: 1/(1+e^(-x))')
        plt.grid(True, alpha=0.3)
        plt.xlim(-6, 6)
        plt.ylim(-0.1, 1.1)
        
        # Show the S-curve properties
        plt.subplot(2, 2, 3)
        x_detailed = np.linspace(-8, 8, 200)
        y_detailed = 1.0 / (1.0 + np.exp(-x_detailed))
        plt.plot(x_detailed, y_detailed, 'g-', linewidth=3)
        # Add asymptotes
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Lower asymptote: y = 0')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Upper asymptote: y = 1')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Midpoint: y = 0.5')
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Output')
        plt.title('S-Curve Properties')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-8, 8)
        plt.ylim(-0.1, 1.1)
        
        # Error analysis
        plt.subplot(2, 2, 4)
        difference = np.abs(y_range - y_ideal)
        max_error = np.max(difference)
        plt.plot(x_range, difference, 'purple', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('|Your Output - Ideal Output|')
        plt.title(f'Implementation Error (Max: {max_error:.6f})')
        plt.grid(True, alpha=0.3)
        plt.xlim(-10, 10)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print(f"\n📊 Analysis:")
        print(f"✅ Maximum error: {max_error:.10f}")
        if max_error < 1e-10:
            print("🎉 Perfect implementation!")
        elif max_error < 1e-6:
            print("🌟 Excellent implementation!")
        elif max_error < 1e-3:
            print("👍 Good implementation!")
        else:
            print("🔧 Implementation needs work.")
            
        print(f"📈 Function properties:")
        print(f"   • Range: (0, 1)")
        print(f"   • Symmetric around (0, 0.5)")
        print(f"   • Smooth and differentiable everywhere")
        print(f"   • Saturates for large |x| (vanishing gradient problem)")
        print(f"   • Useful for binary classification (outputs probabilities)")
    else:
        print("📊 Plots disabled during testing - this is normal!")
        
    # Always show the mathematical analysis
    difference = np.abs(y_range - y_ideal)
    max_error = np.max(difference)
    print(f"\n📊 Mathematical Analysis:")
    print(f"✅ Maximum error: {max_error:.10f}")
    if max_error < 1e-10:
        print("🎉 Perfect implementation!")
    elif max_error < 1e-6:
        print("🌟 Excellent implementation!")
    elif max_error < 1e-3:
        print("👍 Good implementation!")
    else:
        print("🔧 Implementation needs work.")
        
except Exception as e:
    print(f"❌ Error in plotting: {e}")
    print("Make sure to implement the Sigmoid function above!")

# %% [markdown]
"""
## Step 3: Tanh Activation Function

**Tanh** (Hyperbolic Tangent) squashes inputs to the range (-1, 1).

**Formula**: `f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Properties**:
- **Bounded**: Always outputs between -1 and 1
- **Zero-centered**: Output is centered around 0
- **Smooth**: Differentiable everywhere
- **Stronger gradients**: Than sigmoid around zero

**When to use**: Hidden layers when you want zero-centered outputs, RNNs.

**Advantage over Sigmoid**: Zero-centered outputs help with gradient flow.
"""

# %%
#| export
class Tanh:
    """
    Tanh Activation: f(x) = tanh(x)
    
    Squashes input to range (-1, 1). Zero-centered output.
    
    TODO: Implement Tanh activation function.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh: f(x) = tanh(x)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh applied element-wise
            
        TODO: Implement tanh function
        Hint: Use np.tanh(x.data)
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| hide
#| export
class Tanh:
    """Tanh Activation: f(x) = tanh(x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Tanh"""
        return Tensor(np.tanh(x.data))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Tanh Function

Once you implement Tanh above, run this cell to test it:
"""

# %%
# Test Tanh function
try:
    print("=== Testing Tanh Function ===")
    
    # Test data: mix of positive, negative, and zero
    x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    print(f"Input: {x.data}")
    
    # Test Tanh
    tanh = Tanh()
    y = tanh(x)
    print(f"Tanh output: {y.data}")
    print("Expected: values between -1 and 1")
    print(f"All values in (-1,1)? {np.all((y.data > -1) & (y.data < 1))}")
    
    # Test specific values
    x_zero = Tensor([[0.0]])
    y_zero = tanh(x_zero)
    print(f"\nTanh(0) = {y_zero.data[0, 0]:.4f} (should be 0.0)")
    
    # Test extreme values
    x_extreme = Tensor([[-10.0, 10.0]])
    y_extreme = tanh(x_extreme)
    print(f"Tanh([-10, 10]) = {y_extreme.data}")
    print("Should be close to [-1, 1]")
    
    print("✅ Tanh working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement the Tanh function above!")

# %% [markdown]
"""
### 📊 Visualize Tanh Function

Let's plot the Tanh function to see its zero-centered S-shaped curve:
"""

# %%
# Plot Tanh function
try:
    print("=== Plotting Tanh Function ===")
    
    # Create a range of input values
    x_range = np.linspace(-5, 5, 100)
    x_tensor = Tensor([x_range])
    
    # Apply Tanh (student implementation)
    tanh = Tanh()
    y_tensor = tanh(x_tensor)
    y_range = y_tensor.data[0]
    
    # Create ideal Tanh for comparison
    y_ideal = np.tanh(x_range)
    
    # Only show plots if we're not in a testing environment
    if _should_show_plots():
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot both student implementation and ideal
        plt.subplot(2, 2, 1)
        plt.plot(x_range, y_range, 'orange', linewidth=3, label='Your Tanh Implementation')
        plt.plot(x_range, y_ideal, 'r--', linewidth=2, alpha=0.7, label='Ideal Tanh')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Output')
        plt.title('Tanh: Your Implementation vs Ideal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-5, 5)
        plt.ylim(-1.2, 1.2)
        
        # Mathematical explanation plot
        plt.subplot(2, 2, 2)
        # Show key points
        x_key = np.array([-3, -2, -1, 0, 1, 2, 3])
        y_key = np.tanh(x_key)
        plt.stem(x_key, y_key, basefmt=' ', linefmt='purple', markerfmt='o')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('tanh(x)')
        plt.title('Mathematical Definition: tanh(x)')
        plt.grid(True, alpha=0.3)
        plt.xlim(-4, 4)
        plt.ylim(-1.2, 1.2)
        
        # Show symmetry property
        plt.subplot(2, 2, 3)
        x_sym = np.linspace(-4, 4, 100)
        y_sym = np.tanh(x_sym)
        plt.plot(x_sym, y_sym, 'orange', linewidth=3, label='tanh(x)')
        plt.plot(-x_sym, -y_sym, 'b--', linewidth=2, alpha=0.7, label='-tanh(-x)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Upper asymptote: y = 1')
        plt.axhline(y=-1, color='r', linestyle='--', alpha=0.7, label='Lower asymptote: y = -1')
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Output')
        plt.title('Symmetry: tanh(-x) = -tanh(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-4, 4)
        plt.ylim(-1.2, 1.2)
        
        # Error analysis
        plt.subplot(2, 2, 4)
        difference = np.abs(y_range - y_ideal)
        max_error = np.max(difference)
        plt.plot(x_range, difference, 'purple', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('|Your Output - Ideal Output|')
        plt.title(f'Implementation Error (Max: {max_error:.6f})')
        plt.grid(True, alpha=0.3)
        plt.xlim(-5, 5)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print(f"\n📊 Analysis:")
        print(f"✅ Maximum error: {max_error:.10f}")
        if max_error < 1e-10:
            print("🎉 Perfect implementation!")
        elif max_error < 1e-6:
            print("🌟 Excellent implementation!")
        elif max_error < 1e-3:
            print("👍 Good implementation!")
        else:
            print("🔧 Implementation needs work.")
            
        print(f"📈 Function properties:")
        print(f"   • Range: (-1, 1)")
        print(f"   • Odd function: tanh(-x) = -tanh(x)")
        print(f"   • Symmetric around origin (0, 0)")
        print(f"   • Smooth and differentiable everywhere")
        print(f"   • Stronger gradients than sigmoid around zero")
        print(f"   • Related to sigmoid: tanh(x) = 2*sigmoid(2x) - 1")
    else:
        print("📊 Plots disabled during testing - this is normal!")
        
    # Always show the mathematical analysis
    difference = np.abs(y_range - y_ideal)
    max_error = np.max(difference)
    print(f"\n📊 Mathematical Analysis:")
    print(f"✅ Maximum error: {max_error:.10f}")
    if max_error < 1e-10:
        print("🎉 Perfect implementation!")
    elif max_error < 1e-6:
        print("🌟 Excellent implementation!")
    elif max_error < 1e-3:
        print("👍 Good implementation!")
    else:
        print("🔧 Implementation needs work.")
        
except Exception as e:
    print(f"❌ Error in plotting: {e}")
    print("Make sure to implement the Tanh function above!")

# %% [markdown]
"""
## Step 4: Compare All Activation Functions

Let's see how all three functions behave on the same input:
"""

# %%
# Compare all activation functions
try:
    print("=== Comparing All Activation Functions ===")
    
    # Test data: range from -5 to 5
    x = Tensor([[-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]])
    print(f"Input: {x.data}")
    
    # Apply all activations
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    
    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    
    print(f"\nReLU:    {y_relu.data}")
    print(f"Sigmoid: {y_sigmoid.data}")
    print(f"Tanh:    {y_tanh.data}")
    
    print("\n📊 Key Differences:")
    print("- ReLU: Zeros out negative values, unbounded positive")
    print("- Sigmoid: Squashes to (0, 1), always positive")
    print("- Tanh: Squashes to (-1, 1), zero-centered")
    
    print("\n✅ All activation functions working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement all activation functions above!")

# %% [markdown]
"""
### 📊 Comprehensive Activation Function Comparison

Let's plot all three functions together to see their differences:
"""

# %%
# Plot all activation functions together
try:
    print("=== Plotting All Activation Functions Together ===")
    
    # Create a range of input values
    x_range = np.linspace(-5, 5, 100)
    x_tensor = Tensor([x_range])
    
    # Apply all activations (student implementations)
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    
    y_relu = relu(x_tensor).data[0]
    y_sigmoid = sigmoid(x_tensor).data[0]
    y_tanh = tanh(x_tensor).data[0]
    
    # Create ideal functions for comparison
    y_relu_ideal = np.maximum(0, x_range)
    y_sigmoid_ideal = 1.0 / (1.0 + np.exp(-x_range))
    y_tanh_ideal = np.tanh(x_range)
    
    # Only show plots if we're not in a testing environment
    if _should_show_plots():
        # Create the comprehensive plot
        plt.figure(figsize=(15, 10))
        
        # Main comparison plot
        plt.subplot(2, 3, (1, 2))
        plt.plot(x_range, y_relu, 'b-', linewidth=3, label='Your ReLU')
        plt.plot(x_range, y_sigmoid, 'g-', linewidth=3, label='Your Sigmoid')
        plt.plot(x_range, y_tanh, 'orange', linewidth=3, label='Your Tanh')
        
        # Add ideal functions as dashed lines
        plt.plot(x_range, y_relu_ideal, 'b--', linewidth=1, alpha=0.7, label='Ideal ReLU')
        plt.plot(x_range, y_sigmoid_ideal, 'g--', linewidth=1, alpha=0.7, label='Ideal Sigmoid')
        plt.plot(x_range, y_tanh_ideal, '--', color='orange', linewidth=1, alpha=0.7, label='Ideal Tanh')
        
        # Add reference lines
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Formatting
        plt.xlabel('Input (x)', fontsize=12)
        plt.ylabel('Output f(x)', fontsize=12)
        plt.title('Activation Functions: Your Implementation vs Ideal', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper left')
        plt.xlim(-5, 5)
        plt.ylim(-1.5, 5)
        
        # Mathematical definitions
        plt.subplot(2, 3, 3)
        plt.text(0.05, 0.95, 'Mathematical Definitions:', fontsize=12, fontweight='bold', 
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.85, 'ReLU:', fontsize=11, fontweight='bold', color='blue',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.80, 'f(x) = max(0, x)', fontsize=10, fontfamily='monospace',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.70, 'Sigmoid:', fontsize=11, fontweight='bold', color='green',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.65, 'f(x) = 1/(1+e^(-x))', fontsize=10, fontfamily='monospace',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.55, 'Tanh:', fontsize=11, fontweight='bold', color='orange',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.50, 'f(x) = tanh(x)', fontsize=10, fontfamily='monospace',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.45, '     = (e^x-e^(-x))/(e^x+e^(-x))', fontsize=10, fontfamily='monospace',
                 transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.text(0.05, 0.30, 'Key Properties:', fontsize=12, fontweight='bold',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.25, '• ReLU: Sparse, unbounded', fontsize=10, color='blue',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.20, '• Sigmoid: Bounded (0,1)', fontsize=10, color='green',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(0.05, 0.15, '• Tanh: Zero-centered (-1,1)', fontsize=10, color='orange',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.axis('off')
        
        # Error analysis for ReLU
        plt.subplot(2, 3, 4)
        error_relu = np.abs(y_relu - y_relu_ideal)
        plt.plot(x_range, error_relu, 'b-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Error')
        plt.title(f'ReLU Error (Max: {np.max(error_relu):.2e})')
        plt.grid(True, alpha=0.3)
        plt.xlim(-5, 5)
        
        # Error analysis for Sigmoid
        plt.subplot(2, 3, 5)
        error_sigmoid = np.abs(y_sigmoid - y_sigmoid_ideal)
        plt.plot(x_range, error_sigmoid, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Error')
        plt.title(f'Sigmoid Error (Max: {np.max(error_sigmoid):.2e})')
        plt.grid(True, alpha=0.3)
        plt.xlim(-5, 5)
        
        # Error analysis for Tanh
        plt.subplot(2, 3, 6)
        error_tanh = np.abs(y_tanh - y_tanh_ideal)
        plt.plot(x_range, error_tanh, 'orange', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Input (x)')
        plt.ylabel('Error')
        plt.title(f'Tanh Error (Max: {np.max(error_tanh):.2e})')
        plt.grid(True, alpha=0.3)
        plt.xlim(-5, 5)
        
        plt.tight_layout()
        plt.show()
        
        # Comprehensive analysis
        print("\n📊 Comprehensive Analysis:")
        print("=" * 60)
        
        # Function ranges
        print("📈 Output Ranges:")
        print(f"  ReLU:    [{np.min(y_relu):.3f}, {np.max(y_relu):.3f}]")
        print(f"  Sigmoid: [{np.min(y_sigmoid):.3f}, {np.max(y_sigmoid):.3f}]")
        print(f"  Tanh:    [{np.min(y_tanh):.3f}, {np.max(y_tanh):.3f}]")
        
        # Implementation accuracy
        print("\n🎯 Implementation Accuracy:")
        max_errors = [np.max(error_relu), np.max(error_sigmoid), np.max(error_tanh)]
        functions = ['ReLU', 'Sigmoid', 'Tanh']
        
        for func, error in zip(functions, max_errors):
            if error < 1e-10:
                status = "✅ PERFECT"
            elif error < 1e-6:
                status = "✅ EXCELLENT"
            elif error < 1e-3:
                status = "⚠️  GOOD"
            else:
                status = "❌ NEEDS WORK"
            print(f"  {func:8s}: {status:12s} (error: {error:.2e})")
        
        # Mathematical properties verification
        print("\n🔍 Mathematical Properties:")
        
        # Zero-centered test
        x_zero = Tensor([[0.0]])
        print("  Zero-centered test (f(0) should be 0):")
        for name, func in [("ReLU", relu), ("Sigmoid", sigmoid), ("Tanh", tanh)]:
            output = func(x_zero).data[0, 0]
            is_zero = abs(output) < 1e-6
            expected = 0.0 if name != "Sigmoid" else 0.5
            print(f"    {name:8s}: f(0) = {output:.4f} {'✅' if abs(output - expected) < 1e-6 else '❌'}")
        
        # Monotonicity test
        print("  Monotonicity test (should be increasing):")
        test_vals = np.array([-2, -1, 0, 1, 2])
        x_test = Tensor([test_vals])
        for name, func in [("ReLU", relu), ("Sigmoid", sigmoid), ("Tanh", tanh)]:
            outputs = func(x_test).data[0]
            is_monotonic = np.all(outputs[1:] >= outputs[:-1])
            print(f"    {name:8s}: {'✅ Monotonic' if is_monotonic else '❌ Not monotonic'}")
        
        print("\n🎉 Comparison complete! Use these insights to understand each function's role in neural networks.")
    else:
        print("📊 Plots disabled during testing - this is normal!")
        
except Exception as e:
    print(f"❌ Error in plotting: {e}")
    print("Make sure matplotlib is installed and all functions are implemented!")

# %% [markdown]
"""
## Step 5: Understanding Activation Function Properties

Let's explore the mathematical properties of each function:
"""

# %%
# Explore activation function properties
try:
    print("=== Activation Function Properties ===")
    
    # Create test functions
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    
    # Test with a range of values
    test_values = np.linspace(-5, 5, 11)
    x = Tensor([test_values])
    
    print(f"Input range: {test_values}")
    print(f"ReLU range: [{np.min(relu(x).data):.2f}, {np.max(relu(x).data):.2f}]")
    print(f"Sigmoid range: [{np.min(sigmoid(x).data):.2f}, {np.max(sigmoid(x).data):.2f}]")
    print(f"Tanh range: [{np.min(tanh(x).data):.2f}, {np.max(tanh(x).data):.2f}]")
    
    # Test monotonicity (should all be increasing functions)
    print(f"\n📈 Monotonicity Test:")
    for name, func in [("ReLU", relu), ("Sigmoid", sigmoid), ("Tanh", tanh)]:
        outputs = func(x).data[0]
        is_monotonic = np.all(outputs[1:] >= outputs[:-1])
        print(f"{name}: {'✅ Monotonic' if is_monotonic else '❌ Not monotonic'}")
    
    # Test zero-centered property
    print(f"\n🎯 Zero-Centered Test (f(0) = 0):")
    x_zero = Tensor([[0.0]])
    for name, func in [("ReLU", relu), ("Sigmoid", sigmoid), ("Tanh", tanh)]:
        output = func(x_zero).data[0, 0]
        is_zero_centered = abs(output) < 1e-6
        print(f"{name}: f(0) = {output:.4f} {'✅ Zero-centered' if is_zero_centered else '❌ Not zero-centered'}")
    
    print("\n🎉 Property analysis complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your activation function implementations!")

# %% [markdown]
"""
## Step 6: Practical Usage Examples

Let's see how these functions would be used in practice:
"""

# %%
# Practical usage examples
try:
    print("=== Practical Usage Examples ===")
    
    # Example 1: Binary classification with sigmoid
    print("1. Binary Classification (Sigmoid):")
    logits = Tensor([[2.5, -1.2, 0.8, -0.3]])  # Raw network outputs
    sigmoid = Sigmoid()
    probabilities = sigmoid(logits)
    print(f"   Logits: {logits.data}")
    print(f"   Probabilities: {probabilities.data}")
    print(f"   Predictions: {(probabilities.data > 0.5).astype(int)}")
    
    # Example 2: Feature processing with ReLU
    print("\n2. Feature Processing (ReLU):")
    features = Tensor([[-0.5, 1.2, -2.1, 0.8, -0.1]])  # Mixed positive/negative
    relu = ReLU()
    processed = relu(features)
    print(f"   Raw features: {features.data}")
    print(f"   After ReLU: {processed.data}")
    print(f"   Sparsity: {np.mean(processed.data == 0):.1%} zeros")
    
    # Example 3: Normalized features with Tanh
    print("\n3. Normalized Features (Tanh):")
    raw_features = Tensor([[3.2, -1.8, 0.5, -2.4, 1.1]])
    tanh = Tanh()
    normalized = tanh(raw_features)
    print(f"   Raw features: {raw_features.data}")
    print(f"   Normalized: {normalized.data}")
    print(f"   Mean: {np.mean(normalized.data):.3f} (close to 0)")
    
    print("\n✅ Practical examples complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your activation function implementations!")

# %% [markdown]
"""
## 🎉 Congratulations!

You've successfully implemented the three most important activation functions in deep learning!

### 🧱 What You Built
1. **ReLU**: The workhorse activation that enables deep networks
2. **Sigmoid**: The probability activation for binary classification
3. **Tanh**: The zero-centered activation for better gradient flow

### 🎯 Key Insights
- **Nonlinearity is essential**: Without activations, neural networks are just linear transformations
- **Different functions serve different purposes**: ReLU for hidden layers, Sigmoid for probabilities, Tanh for zero-centered outputs
- **Mathematical properties matter**: Monotonicity, boundedness, and zero-centering affect learning

### 🚀 What's Next
These activation functions will be used in:
- **Layers Module**: Building neural network layers
- **Loss Functions**: Computing training objectives
- **Advanced Architectures**: CNNs, RNNs, and more

### 🔧 Export to Package
Run this to export your activations to the TinyTorch package:
```bash
python bin/tito.py sync
```

Then test your implementation:
```bash
python bin/tito.py test --module activations
```

**Excellent work! You've mastered the mathematical foundations of neural networks!** 🎉

---

## 📚 Further Reading

**Want to learn more about activation functions?**
- **ReLU variants**: Leaky ReLU, ELU, Swish
- **Advanced activations**: GELU, Mish, SiLU
- **Activation choice**: When to use which function
- **Gradient flow**: How activations affect training

**Next modules**: Layers, Loss Functions, Optimization
""" 