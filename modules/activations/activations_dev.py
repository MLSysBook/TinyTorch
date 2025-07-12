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
## Testing Our Activation Functions

Let's test our implementations with some simple examples to make sure they work correctly.
"""

# %%
# Test our activation functions
if __name__ == "__main__":
    # Create test data
    test_data = Tensor([[-2, -1, 0, 1, 2]])
    
    print("Testing Activation Functions:")
    print(f"Input: {test_data.data}")
    
    # Test ReLU
    relu = ReLU()
    try:
        relu_output = relu(test_data)
        print(f"ReLU: {relu_output.data}")
    except NotImplementedError:
        print("ReLU: Not implemented yet")
    
    # Test Sigmoid
    sigmoid = Sigmoid()
    try:
        sigmoid_output = sigmoid(test_data)
        print(f"Sigmoid: {sigmoid_output.data}")
    except NotImplementedError:
        print("Sigmoid: Not implemented yet")
    
    # Test Tanh
    tanh = Tanh()
    try:
        tanh_output = tanh(test_data)
        print(f"Tanh: {tanh_output.data}")
    except NotImplementedError:
        print("Tanh: Not implemented yet")
    
    # Test Softmax
    softmax = Softmax()
    try:
        softmax_output = softmax(test_data)
        print(f"Softmax: {softmax_output.data}")
        print(f"Softmax sum: {np.sum(softmax_output.data)}")
    except NotImplementedError:
        print("Softmax: Not implemented yet")

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