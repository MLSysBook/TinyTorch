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
- Implement the three most important activation functions: ReLU, Sigmoid, and Tanh
- Visualize how activations transform data and enable complex learning
- See how activations work with layers to build powerful networks

## Build → Use → Understand
1. **Build**: Activation functions that add nonlinearity
2. **Use**: Transform tensors and see immediate results
3. **Understand**: How nonlinearity enables complex pattern learning

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
from tinytorch.core.activations import ReLU, Sigmoid, Tanh
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense, Conv2D
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.functional`
- **Consistency:** All activation functions live together in `core.activations`
"""

# %%
#| default_exp core.activations

__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']

# Setup and imports
import math
import numpy as np
import matplotlib.pyplot as plt
import os
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
import matplotlib.pyplot as plt
import os
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

### Visual Intuition
```
Input: [-2, -1, 0, 1, 2]
ReLU:   [0,  0, 0, 1, 2]  (clips negatives to 0)
Sigmoid: [0.1, 0.3, 0.5, 0.7, 0.9]  (squashes to 0-1)
Tanh:    [-0.9, -0.8, 0, 0.8, 0.9]  (squashes to -1 to 1)
```

### The Math Behind It
Each activation function has different mathematical properties:
- **ReLU**: `f(x) = max(0, x)` - Simple thresholding
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))` - Smooth squashing
- **Tanh**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))` - Centered squashing

Let's implement these step by step!
"""

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

### Visual Example
```
Input:  [-3, -1, 0, 1, 3]
ReLU:   [0,  0, 0, 1, 3]
```

Let's implement it!
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
            
        TODO: Implement element-wise max(0, x) operation
        
        STEP-BY-STEP:
        1. Get the numpy array: data = x.data
        2. Apply ReLU: result = np.maximum(0, data)
        3. Return Tensor(result)
        
        EXAMPLE:
        Input: Tensor([[-2, 1, 0]])
        Expected: Tensor([[0, 1, 0]])
        
        HINTS:
        - np.maximum(0, x.data) applies max(0, x) to each element
        - This keeps positive values unchanged and sets negatives to 0
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
"""

# %%
# Test ReLU function
print("Testing ReLU function...")

try:
    # Test data: mix of positive, negative, and zero
    x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    print(f"✅ Input: {x.data}")
    
    # Test ReLU
    relu = ReLU()
    y = relu(x)
    print(f"✅ ReLU output: {y.data}")
    print(f"✅ Expected: [[0. 0. 0. 1. 3.]]")
    
    # Verify the result
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 3.0]])
    assert np.allclose(y.data, expected), "❌ ReLU output doesn't match expected!"
    print("🎉 ReLU works correctly!")
    
    # Test with different shapes
    x_2d = Tensor([[-2.0, 1.0], [0.5, -0.5]])
    y_2d = relu(x_2d)
    print(f"✅ 2D Input: {x_2d.data}")
    print(f"✅ 2D ReLU output: {y_2d.data}")
    
    print("\n🎉 All ReLU tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement ReLU above!")

# %% [markdown]
"""
## Step 3: Sigmoid Activation Function

**Sigmoid** is a smooth, S-shaped function that squashes any input to the range (0, 1).

### What is Sigmoid?
- **Formula**: `f(x) = 1 / (1 + e^(-x))`
- **Behavior**: Smoothly transforms any real number to (0, 1)
- **Range**: (0, 1) - always positive, bounded

### Why Sigmoid Matters
- **Probability interpretation**: Output can be interpreted as probability
- **Smooth**: Continuous and differentiable everywhere
- **Bounded**: Output is always between 0 and 1
- **Historical importance**: Was the default choice before ReLU

### Real-World Analogy
Think of Sigmoid as a **probability converter**:
- Takes any input (positive or negative)
- Converts it to a probability between 0 and 1
- Like a confidence score that's always positive

### Visual Example
```
Input:   [-3, -1, 0, 1, 3]
Sigmoid: [0.05, 0.27, 0.5, 0.73, 0.95]
```

### The Math Behind It
The sigmoid function uses the exponential function:
- For large positive x: e^(-x) ≈ 0, so f(x) ≈ 1
- For large negative x: e^(-x) ≈ ∞, so f(x) ≈ 0
- For x = 0: e^0 = 1, so f(x) = 0.5

Let's implement it!
"""

# %%
#| export
class Sigmoid:
    """
    Sigmoid Activation: f(x) = 1 / (1 + e^(-x))
    
    Smooth function that squashes inputs to (0, 1).
    Historically important, still used for probability outputs.
    
    TODO: Implement Sigmoid activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Apply the sigmoid formula: 1 / (1 + e^(-x))
    3. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[-2, 0, 2]])
    Output: Tensor([[0.12, 0.5, 0.88]])
    
    HINTS:
    - Use x.data to get the numpy array
    - Use np.exp(-x.data) for e^(-x)
    - Use 1 / (1 + np.exp(-x.data)) for the full formula
    - Return Tensor(result) to wrap the result
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Sigmoid: f(x) = 1 / (1 + e^(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid applied element-wise
            
        TODO: Implement the sigmoid formula
        
        STEP-BY-STEP:
        1. Get the numpy array: data = x.data
        2. Compute e^(-x): exp_neg = np.exp(-data)
        3. Apply sigmoid: result = 1 / (1 + exp_neg)
        4. Return Tensor(result)
        
        EXAMPLE:
        Input: Tensor([[-1, 0, 1]])
        Expected: Tensor([[0.27, 0.5, 0.73]])
        
        HINTS:
        - np.exp(-x.data) computes e^(-x) for each element
        - 1 / (1 + np.exp(-x.data)) applies the full sigmoid formula
        - This squashes any input to the range (0, 1)
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make activation callable: sigmoid(x) same as sigmoid.forward(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Sigmoid:
    """Sigmoid Activation: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid: f(x) = 1 / (1 + e^(-x))"""
        return Tensor(1 / (1 + np.exp(-x.data)))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Sigmoid Function
"""

# %%
# Test Sigmoid function
print("Testing Sigmoid function...")

try:
    # Test data: mix of negative, zero, and positive
    x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    print(f"✅ Input: {x.data}")
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    y = sigmoid(x)
    print(f"✅ Sigmoid output: {y.data}")
    
    # Verify key properties
    assert np.all(y.data > 0), "❌ Sigmoid should always be positive!"
    assert np.all(y.data < 1), "❌ Sigmoid should always be less than 1!"
    assert np.isclose(y.data[0, 2], 0.5, atol=0.01), "❌ Sigmoid(0) should be 0.5!"
    print("✅ Sigmoid properties verified!")
    
    # Test specific values
    expected_approx = np.array([[0.05, 0.27, 0.5, 0.73, 0.95]])
    assert np.allclose(y.data, expected_approx, atol=0.1), "❌ Sigmoid values don't match expected!"
    print("🎉 Sigmoid works correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement Sigmoid above!")

# %% [markdown]
"""
## Step 4: Tanh Activation Function

**Tanh** (Hyperbolic Tangent) is a centered version of sigmoid that outputs values between -1 and 1.

### What is Tanh?
- **Formula**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- **Behavior**: Smoothly transforms any real number to (-1, 1)
- **Range**: (-1, 1) - centered around zero

### Why Tanh Matters
- **Centered**: Output is centered around zero (unlike sigmoid)
- **Zero-centered**: Better for gradient flow in deep networks
- **Smooth**: Continuous and differentiable everywhere
- **Bounded**: Output is always between -1 and 1

### Real-World Analogy
Think of Tanh as a **centered probability converter**:
- Takes any input (positive or negative)
- Converts it to a value between -1 and 1
- Like a confidence score that can be positive or negative

### Visual Example
```
Input: [-3, -1, 0, 1, 3]
Tanh:  [-0.99, -0.76, 0, 0.76, 0.99]
```

### The Math Behind It
Tanh is related to sigmoid: `tanh(x) = 2 * sigmoid(2x) - 1`
- For large positive x: f(x) ≈ 1
- For large negative x: f(x) ≈ -1
- For x = 0: f(x) = 0

Let's implement it!
"""

# %%
#| export
class Tanh:
    """
    Tanh Activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Centered version of sigmoid that outputs values in (-1, 1).
    Better for gradient flow in deep networks.
    
    TODO: Implement Tanh activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Apply the tanh formula using numpy's tanh function
    3. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[-2, 0, 2]])
    Output: Tensor([[-0.96, 0, 0.96]])
    
    HINTS:
    - Use x.data to get the numpy array
    - Use np.tanh(x.data) for the tanh function
    - Return Tensor(result) to wrap the result
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh applied element-wise
            
        TODO: Implement the tanh function
        
        STEP-BY-STEP:
        1. Get the numpy array: data = x.data
        2. Apply tanh: result = np.tanh(data)
        3. Return Tensor(result)
        
        EXAMPLE:
        Input: Tensor([[-1, 0, 1]])
        Expected: Tensor([[-0.76, 0, 0.76]])
        
        HINTS:
        - np.tanh(x.data) computes tanh for each element
        - This squashes any input to the range (-1, 1)
        - The output is centered around zero
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make activation callable: tanh(x) same as tanh.forward(x)"""
        return self.forward(x)

# %%
#| hide
#| export
class Tanh:
    """Tanh Activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
        return Tensor(np.tanh(x.data))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Tanh Function
"""

# %%
# Test Tanh function
print("Testing Tanh function...")

try:
    # Test data: mix of negative, zero, and positive
    x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    print(f"✅ Input: {x.data}")
    
    # Test Tanh
    tanh = Tanh()
    y = tanh(x)
    print(f"✅ Tanh output: {y.data}")
    
    # Verify key properties
    assert np.all(y.data >= -1), "❌ Tanh should always be >= -1!"
    assert np.all(y.data <= 1), "❌ Tanh should always be <= 1!"
    assert np.isclose(y.data[0, 2], 0.0, atol=0.01), "❌ Tanh(0) should be 0!"
    print("✅ Tanh properties verified!")
    
    # Test specific values
    expected_approx = np.array([[-0.99, -0.76, 0.0, 0.76, 0.99]])
    assert np.allclose(y.data, expected_approx, atol=0.1), "❌ Tanh values don't match expected!"
    print("🎉 Tanh works correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to implement Tanh above!")

# %% [markdown]
"""
## Step 5: Comparing Activation Functions

Now let's compare all three activation functions to understand their differences and when to use each one.
"""

# %%
# Compare activation functions
print("Comparing activation functions...")

try:
    # Test data
    x = Tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    print(f"✅ Input: {x.data}")
    
    # Apply all three activations
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    
    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    
    print(f"✅ ReLU:    {y_relu.data}")
    print(f"✅ Sigmoid: {y_sigmoid.data}")
    print(f"✅ Tanh:    {y_tanh.data}")
    
    print("\n💡 Key Differences:")
    print("   ReLU:    [0, ∞) - unbounded, sparse")
    print("   Sigmoid: (0, 1) - bounded, always positive")
    print("   Tanh:    (-1, 1) - bounded, centered")
    
    print("\n🎉 All activation functions working!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# %% [markdown]
"""
## Step 6: Understanding When to Use Each Activation

### ReLU - The Default Choice
**Use ReLU for:**
- Hidden layers in most neural networks
- When you want computational efficiency
- When you want sparse representations
- When you want to avoid vanishing gradients

**Example**: `Dense → ReLU → Dense → ReLU → Dense`

### Sigmoid - Probability Outputs
**Use Sigmoid for:**
- Binary classification outputs (0 or 1)
- When you need probability interpretation
- When you need outputs between 0 and 1

**Example**: `Dense → ReLU → Dense → Sigmoid` (binary classifier)

### Tanh - Centered Outputs
**Use Tanh for:**
- When you want outputs centered around zero
- When you want better gradient flow
- When you need outputs between -1 and 1

**Example**: `Dense → Tanh → Dense → Tanh` (centered features)

### Visual Comparison
```
Input: [-2, -1, 0, 1, 2]
ReLU:   [0,  0, 0, 1, 2]  (sparse, unbounded)
Sigmoid: [0.1, 0.3, 0.5, 0.7, 0.9]  (smooth, 0-1)
Tanh:    [-0.9, -0.8, 0, 0.8, 0.9]  (smooth, -1 to 1)
```
"""

# %%
# Demonstrate activation usage patterns
print("Demonstrating activation usage patterns...")

try:
    # Create a simple network with different activations
    from tinytorch.core.layers import Dense
    
    # Binary classification network
    network = [
        Dense(input_size=3, output_size=4),
        ReLU(),  # Hidden layer
        Dense(input_size=4, output_size=1),
        Sigmoid()  # Output layer (probability)
    ]
    
    # Test input
    x = Tensor([[1.0, 2.0, 3.0]])
    print(f"✅ Input: {x}")
    
    # Forward pass
    current = x
    for i, layer in enumerate(network):
        current = layer(current)
        print(f"✅ After layer {i+1} ({type(layer).__name__}): {current}")
    
    print("\n💡 This network could classify inputs as 0 or 1!")
    print("   The final Sigmoid output is a probability between 0 and 1.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure your activations and layers are working!")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've built the foundation of neural network nonlinearity:

### What You've Accomplished
✅ **ReLU Activation**: Simple, efficient, and widely used  
✅ **Sigmoid Activation**: Smooth probability converter  
✅ **Tanh Activation**: Centered version for better gradients  
✅ **Activation Comparison**: Understanding when to use each  
✅ **Real-world Usage**: Seeing activations in networks  

### Key Concepts You've Learned
- **Activation functions** add nonlinearity to neural networks
- **ReLU** is the default choice for hidden layers
- **Sigmoid** is used for probability outputs
- **Tanh** is used when you need centered outputs
- **Nonlinearity** is essential for learning complex patterns

### What's Next
In the next modules, you'll build on this foundation:
- **Layers**: Combine activations with linear transformations
- **Networks**: Compose layers and activations into architectures
- **Training**: Learn parameters using gradients and optimization
- **Applications**: Solve real problems with neural networks

### Real-World Connection
Your activation functions are now ready to:
- Add nonlinearity to neural network layers
- Enable learning of complex patterns
- Provide appropriate outputs for different tasks
- Integrate with the rest of the TinyTorch ecosystem

**Ready for the next challenge?** Let's move on to building layers that combine linear transformations with your activation functions!
"""

# %%
# Final verification
print("\n" + "="*50)
print("🎉 ACTIVATIONS MODULE COMPLETE!")
print("="*50)
print("✅ ReLU activation function")
print("✅ Sigmoid activation function")
print("✅ Tanh activation function")
print("✅ Activation comparison and usage")
print("✅ Real-world network integration")
print("\n🚀 Ready to build layers in the next module!") 

# %%
#| export
class Softmax:
    """
    Softmax Activation: f(x) = exp(x) / sum(exp(x))
    
    Converts logits to probability distribution. Used for multi-class classification.
    Output sums to 1.0 across the last dimension.
    
    TODO: Implement Softmax activation function.
    
    APPROACH:
    1. Extract the numpy array from the input tensor
    2. Apply softmax formula: exp(x) / sum(exp(x))
    3. Handle numerical stability (subtract max for stability)
    4. Return a new Tensor with the result
    
    EXAMPLE:
    Input: Tensor([[1.0, 2.0, 3.0]])
    Output: Tensor([[0.09, 0.24, 0.67]]) (sums to 1.0)
    
    HINTS:
    - Use x.data to get the numpy array
    - For stability: x_stable = x - np.max(x, axis=-1, keepdims=True)
    - Then: exp_x = np.exp(x_stable)
    - Finally: softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Softmax: f(x) = exp(x) / sum(exp(x))
        
        Args:
            x: Input tensor (logits)
            
        Returns:
            Output tensor with Softmax applied (probabilities)
            
        TODO: Implement numerically stable softmax
        
        STEP-BY-STEP:
        1. Get the numpy array: data = x.data
        2. Subtract max for stability: stable = data - np.max(data, axis=-1, keepdims=True)
        3. Compute exponentials: exp_vals = np.exp(stable)
        4. Normalize: result = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        5. Return Tensor(result)
        
        EXAMPLE:
        Input: Tensor([[1.0, 2.0, 3.0]])
        Expected: Tensor([[0.09, 0.24, 0.67]]) (approximately, sums to 1.0)
        
        HINTS:
        - axis=-1 means along the last dimension
        - keepdims=True preserves dimensions for broadcasting
        - This creates a probability distribution that sums to 1.0
        """
        raise NotImplementedError("Student implementation required")
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

# %%
#| hide
#| export
class Softmax:
    """Softmax Activation: f(x) = exp(x) / sum(exp(x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Softmax with numerical stability"""
        # Subtract max for numerical stability
        x_stable = x.data - np.max(x.data, axis=-1, keepdims=True)
        
        # Compute exponentials
        exp_vals = np.exp(x_stable)
        
        # Normalize to get probabilities
        result = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x) 