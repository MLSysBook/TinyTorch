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
# Activations - Nonlinearity and Neural Network Intelligence

Welcome to the Activations module! You'll implement the functions that give neural networks their power to learn complex patterns through nonlinearity.

## Learning Goals
- Systems understanding: Why linear operations alone cannot solve complex problems and how nonlinearity enables universal approximation
- Core implementation skill: Build the four essential activation functions that power modern neural networks
- Pattern recognition: Understand how different activations affect gradient flow and learning dynamics
- Framework connection: See how your implementations match PyTorch's optimized activation functions
- Performance insight: Learn why activation choice affects both forward pass speed and gradient computation efficiency

## Build → Use → Reflect
1. **Build**: ReLU, Sigmoid, Tanh, and Softmax activation functions with proper numerical stability
2. **Use**: Transform real tensor data and observe how different activations affect output distributions
3. **Reflect**: Why does activation function choice determine whether deep networks can train successfully?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how nonlinear functions enable neural networks to approximate any continuous function
- Practical capability to implement numerically stable activation functions that avoid overflow and underflow
- Systems insight into why activation choice affects gradient flow and determines trainable network depth
- Performance consideration of how activation complexity affects forward and backward pass computational cost
- Connection to production ML systems and why modern frameworks provide dozens of activation variants

## Systems Reality Check
💡 **Production Context**: PyTorch implements activations as both functions and modules, with CUDA kernels for GPU acceleration - your implementation reveals the mathematical foundations
⚡ **Performance Note**: ReLU is popular partly because it's computationally cheap (just max(0,x)), while Softmax requires expensive exponentials - activation choice affects training speed
"""

# %% nbgrader={"grade": false, "grade_id": "activations-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.activations

#| export
import math
import numpy as np
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

# Note: Autograd support comes later in Module 9
# For now, we work with pure Tensor objects only

# %% nbgrader={"grade": false, "grade_id": "activations-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Activations Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build activation functions!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/02_activations/activations_dev.py`  
**Building Side:** Code exports to `tinytorch.core.activations`

```python
# Final package structure:
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, Softmax
from tinytorch.core.tensor import Tensor  # Foundation
from tinytorch.core.layers import Dense  # Uses activations
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.ReLU`
- **Consistency:** All activation functions live together in `core.activations`
- **Integration:** Works seamlessly with tensors and layers
"""

# %% [markdown]
"""
## What Are Activation Functions?

### The Problem: Linear Limitations
Without activation functions, neural networks can only learn linear relationships:
```
y = W₁ · (W₂ · (W₃ · x + b₃) + b₂) + b₁
```

This simplifies to just:
```
y = W_combined · x + b_combined
```

**A single linear function!** No matter how many layers you add, you can't learn complex patterns like:
- Image recognition (nonlinear pixel relationships)
- Language understanding (nonlinear word relationships) 
- Game playing (nonlinear strategy relationships)

### The Solution: Nonlinearity
Activation functions add nonlinearity between layers:
```
y = W₁ · f(W₂ · f(W₃ · x + b₃) + b₂) + b₁
```

Now each layer can learn complex transformations!

### Real-World Impact
- **Before activations**: Only linear classifiers (logistic regression)
- **After activations**: Complex pattern recognition (deep learning revolution)

### What We'll Build
1. **ReLU**: The foundation of modern deep learning
2. **Sigmoid**: Classic activation for binary classification
3. **Tanh**: Centered activation for better gradients
4. **Softmax**: Probability distributions for multi-class classification
"""

# %% [markdown]
"""
## 🔧 DEVELOPMENT
"""

# %% [markdown]
"""
## Step 1: ReLU - The Foundation of Deep Learning

### What is ReLU?
**ReLU (Rectified Linear Unit)** is the most important activation function in deep learning:

```
f(x) = max(0, x)
```

- **Positive inputs**: Pass through unchanged
- **Negative inputs**: Become zero
- **Zero**: Stays zero

### Why ReLU Revolutionized Deep Learning
1. **Computational efficiency**: Just a max operation
2. **No vanishing gradients**: Derivative is 1 for positive values
3. **Sparsity**: Many neurons output exactly 0
4. **Empirical success**: Works well in practice

### Visual Understanding
```
Input:  [-2, -1, 0, 1, 2]
ReLU:   [ 0,  0, 0, 1, 2]
```

### Real-World Applications
- **Image classification**: ResNet, VGG, AlexNet
- **Object detection**: YOLO, R-CNN
- **Language models**: Transformer feedforward layers
- **Recommendation**: Deep collaborative filtering

### Mathematical Properties
- **Derivative**: f'(x) = 1 if x > 0, else 0
- **Range**: [0, ∞)
- **Sparsity**: Outputs exactly 0 for negative inputs
"""

# %% nbgrader={"grade": false, "grade_id": "relu-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class ReLU:
    """
    ReLU Activation Function: f(x) = max(0, x)
    
    The most popular activation function in deep learning.
    Simple, fast, and effective for most applications.
    """
    
    def forward(self, x):
        """
        Apply ReLU activation: f(x) = max(0, x)
        
        Works with Tensor inputs for nonlinear transformations.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. For each element in the input tensor, apply max(0, element)
        2. Return new Tensor with ReLU applied
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x) = max(0, x)
        - Sets negative values to 0, keeps positive values unchanged
        
        EXAMPLE USAGE:
        ```python
        relu = ReLU()
        tensor_input = Tensor([[-2, -1, 0, 1, 2]])
        tensor_output = relu(tensor_input)
        # Output: [[0, 0, 0, 1, 2]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.maximum(0, x.data) for element-wise max operation
        - Return same type as input (Tensor)
        
        LEARNING CONNECTIONS:
        - This is the core of torch.nn.ReLU() in PyTorch
        - Enables neural networks to learn nonlinear patterns
        - ReLU's simplicity makes it computationally efficient
        - Creates sparse representations (many zeros)
        """
        ### BEGIN SOLUTION
        # Apply ReLU: max(0, x) element-wise
        result = np.maximum(0, x.data)
        return type(x)(result)
        ### END SOLUTION
    
    def forward_(self, x):
        """
        Apply ReLU activation in-place: modifies input tensor directly
        
        In-place operations save memory by not creating new tensors.
        Critical for large models where memory is constrained.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Apply ReLU operation directly to tensor._data
        2. Return the same tensor object (modified in-place)
        
        MEMORY BENEFITS:
        - No new tensor allocation (saves memory)
        - Reuses existing memory buffer
        - Critical for large neural networks
        - Used in PyTorch with relu_() syntax
        
        EXAMPLE USAGE:
        ```python
        relu = ReLU()
        # Regular: creates new tensor
        x = Tensor([[1, -2, 3]])
        y = relu(x)  # x unchanged, y is new tensor
        
        # In-place: modifies existing tensor
        x = Tensor([[1, -2, 3]])
        relu.forward_(x)  # x is now [[1, 0, 3]]
        ```
        
        IMPLEMENTATION HINTS:
        - Modify x._data directly with np.maximum(0, x._data, out=x._data)
        - Return the same object for method chaining
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.functional.relu_() in PyTorch
        - Memory-efficient for inference and some training scenarios
        - Trade-off: can't recover original values after operation
        """
        ### BEGIN SOLUTION
        # Apply ReLU in-place: modify tensor data directly
        np.maximum(0, x._data, out=x._data)
        return x
        ### END SOLUTION
    
    def __call__(self, x):
        """Make the class callable: relu(x) instead of relu.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your ReLU Implementation

Once you implement the ReLU forward method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_relu_activation():
    """Unit test for the ReLU activation function."""
    print("🔬 Unit Test: ReLU Activation...")

    # Create ReLU instance
    relu = ReLU()

    # Test with mixed positive/negative values
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = relu(test_input)
    expected = np.array([[0, 0, 0, 1, 2]])
    
    assert np.array_equal(result.data, expected), f"ReLU failed: expected {expected}, got {result.data}"
    
    # Test that negative values become zero
    assert np.all(result.data >= 0), "ReLU should make all negative values zero"
    
    # Test that positive values remain unchanged
    positive_input = Tensor([[1, 2, 3, 4, 5]])
    positive_result = relu(positive_input)
    assert np.array_equal(positive_result.data, positive_input.data), "ReLU should preserve positive values"
    
    # Test with 2D tensor
    matrix_input = Tensor([[-1, 2], [3, -4]])
    matrix_result = relu(matrix_input)
    matrix_expected = np.array([[0, 2], [3, 0]])
    assert np.array_equal(matrix_result.data, matrix_expected), "ReLU should work with 2D tensors"
    
    # Test shape preservation
    assert matrix_result.shape == matrix_input.shape, "ReLU should preserve input shape"
    
    print("✅ ReLU activation tests passed!")
    print(f"✅ Negative values correctly zeroed")
    print(f"✅ Positive values preserved")
    print(f"✅ Shape preservation working")
    print(f"✅ Works with multi-dimensional tensors")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 2: Sigmoid - Classic Binary Classification

### What is Sigmoid?
**Sigmoid** is the classic activation function that maps any real number to (0, 1):

```
f(x) = 1 / (1 + e^(-x))
```

### Why Sigmoid Matters
1. **Probability interpretation**: Outputs between 0 and 1
2. **Smooth gradients**: Differentiable everywhere
3. **Historical importance**: Enabled early neural networks
4. **Binary classification**: Perfect for yes/no decisions

### Visual Understanding
```
Input:  [-∞, -2, -1, 0, 1, 2, ∞]
Sigmoid:[0,  0.12, 0.27, 0.5, 0.73, 0.88, 1]
```

### Real-World Applications
- **Binary classification**: Spam detection, medical diagnosis
- **Gating mechanisms**: LSTM and GRU cells
- **Output layers**: When you need probabilities
- **Attention mechanisms**: Where to focus attention

### Mathematical Properties
- **Range**: (0, 1)
- **Derivative**: f'(x) = f(x) · (1 - f(x))
- **Centered**: f(0) = 0.5
- **Symmetric**: f(-x) = 1 - f(x)
"""

# %% nbgrader={"grade": false, "grade_id": "sigmoid-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Sigmoid:
    """
    Sigmoid Activation Function: f(x) = 1 / (1 + e^(-x))
    
    Maps any real number to the range (0, 1).
    Useful for binary classification and probability outputs.
    """
    
    def forward(self, x):
        """
        Apply Sigmoid activation: f(x) = 1 / (1 + e^(-x))
        
        Works with Tensor inputs for probability-like outputs.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute sigmoid: 1 / (1 + exp(-x))
        2. Return new Tensor with sigmoid applied
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x) = 1 / (1 + e^(-x))
        - Maps any real number to range (0, 1)
        
        EXAMPLE USAGE:
        ```python
        sigmoid = Sigmoid()
        tensor_input = Tensor([[0.0]])
        tensor_output = sigmoid(tensor_input)  # [[0.5]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use numerical stability: clip inputs to prevent overflow
        - Use np.exp() for exponential function
        
        LEARNING CONNECTIONS:
        - This is the core of torch.nn.Sigmoid() in PyTorch
        - Used in binary classification and gating mechanisms
        - Smooth gradients enable stable training
        - Self-normalizing output always between 0 and 1
        """
        ### BEGIN SOLUTION
        # Apply Sigmoid with numerical stability
        clipped_input = np.clip(-x.data, -500, 500)
        result = 1 / (1 + np.exp(clipped_input))
        return type(x)(result)
        ### END SOLUTION
    
    def forward_(self, x):
        """
        Apply Sigmoid activation in-place: modifies input tensor directly
        
        In-place sigmoid saves memory by reusing existing tensor buffer.
        Important for memory-constrained environments.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute sigmoid directly into tensor._data
        2. Return the same tensor object (modified in-place)
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x) = 1 / (1 + e^(-x))
        
        MEMORY BENEFITS:
        - No new tensor allocation (saves memory)
        - Reuses existing memory buffer
        - Critical for large neural networks
        - Used in PyTorch with sigmoid_() syntax
        
        EXAMPLE USAGE:
        ```python
        sigmoid = Sigmoid()
        # In-place: modifies existing tensor
        x = Tensor([[0.0, 1.0, -1.0]])
        sigmoid.forward_(x)  # x is now [[0.5, 0.73, 0.27]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use numerical stability: clip inputs to prevent overflow
        - Modify x._data directly
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.functional.sigmoid_() in PyTorch
        - Memory-efficient for inference scenarios
        - Trade-off: can't recover original values after operation
        """
        ### BEGIN SOLUTION
        # Apply Sigmoid in-place with numerical stability
        clipped_input = np.clip(-x._data, -500, 500)
        np.divide(1, 1 + np.exp(clipped_input), out=x._data)
        return x
        ### END SOLUTION
    
    def __call__(self, x):
        """Make the class callable: sigmoid(x) instead of sigmoid.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Sigmoid Implementation

Once you implement the Sigmoid forward method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-sigmoid-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_sigmoid_activation():
    """Unit test for the Sigmoid activation function."""
    print("🔬 Unit Test: Sigmoid Activation...")

# Create Sigmoid instance
    sigmoid = Sigmoid()

    # Test with known values
    test_input = Tensor([[0]])
    result = sigmoid(test_input)
    expected = 0.5
    
    assert abs(result.data[0][0] - expected) < 1e-6, f"Sigmoid(0) should be 0.5, got {result.data[0][0]}"
    
    # Test with positive and negative values
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = sigmoid(test_input)
    
    # Check that all values are between 0 and 1
    assert np.all(result.data > 0), "Sigmoid output should be > 0"
    assert np.all(result.data < 1), "Sigmoid output should be < 1"
    
    # Test symmetry: sigmoid(-x) = 1 - sigmoid(x)
    x_val = 1.0
    pos_result = sigmoid(Tensor([[x_val]]))
    neg_result = sigmoid(Tensor([[-x_val]]))
    symmetry_check = abs(pos_result.data[0][0] + neg_result.data[0][0] - 1.0)
    assert symmetry_check < 1e-6, "Sigmoid should be symmetric around 0.5"
    
    # Test with 2D tensor
    matrix_input = Tensor([[-1, 1], [0, 2]])
    matrix_result = sigmoid(matrix_input)
    assert matrix_result.shape == matrix_input.shape, "Sigmoid should preserve shape"
    
    # Test extreme values (should not overflow)
    extreme_input = Tensor([[-100, 100]])
    extreme_result = sigmoid(extreme_input)
    assert not np.any(np.isnan(extreme_result.data)), "Sigmoid should handle extreme values"
    assert not np.any(np.isinf(extreme_result.data)), "Sigmoid should not produce inf values"
    
    print("✅ Sigmoid activation tests passed!")
    print(f"✅ Outputs correctly bounded between 0 and 1")
    print(f"✅ Symmetric property verified")
    print(f"✅ Handles extreme values without overflow")
    print(f"✅ Shape preservation working")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 3: Tanh - Centered Activation

### What is Tanh?
**Tanh (Hyperbolic Tangent)** is similar to sigmoid but centered around zero:

```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

### Why Tanh is Better Than Sigmoid
1. **Zero-centered**: Outputs range from -1 to 1
2. **Better gradients**: Helps with gradient flow in deep networks
3. **Faster convergence**: Less bias shift during training
4. **Stronger gradients**: Maximum gradient is 1 vs 0.25 for sigmoid

### Visual Understanding
```
Input: [-∞, -2, -1, 0, 1, 2, ∞]
Tanh:  [-1, -0.96, -0.76, 0, 0.76, 0.96, 1]
```

### Real-World Applications
- **Hidden layers**: Better than sigmoid for internal activations
- **RNN cells**: Classic RNN and LSTM use tanh
- **Normalization**: When you need zero-centered outputs
- **Feature scaling**: Maps inputs to [-1, 1] range

### Mathematical Properties
- **Range**: (-1, 1)
- **Derivative**: f'(x) = 1 - f(x)²
- **Zero-centered**: f(0) = 0
- **Antisymmetric**: f(-x) = -f(x)
"""

# %% nbgrader={"grade": false, "grade_id": "tanh-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Tanh:
    """
    Tanh Activation Function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Zero-centered activation function with range (-1, 1).
    Better gradient properties than sigmoid.
    """
    
    def forward(self, x):
        """
        Apply Tanh activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Works with Tensor inputs for zero-centered nonlinearity.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute tanh: (e^x - e^(-x)) / (e^x + e^(-x))
        2. Return new Tensor with tanh applied
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x) = tanh(x)
        - Maps any real number to range (-1, 1)
        
        EXAMPLE USAGE:
        ```python
        tanh = Tanh()
        tensor_input = Tensor([[0.0]])
        tensor_output = tanh(tensor_input)  # [[0.0]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.tanh() for numerical stability
        - Return same type as input (Tensor)
        
        LEARNING CONNECTIONS:
        - This is the core of torch.nn.Tanh() in PyTorch
        - Used in RNN, LSTM, and GRU cells
        - Zero-centered outputs improve gradient flow
        - Strong gradients near zero, weaker at extremes
        """
        ### BEGIN SOLUTION
        # Apply Tanh: numerically stable hyperbolic tangent
        result = np.tanh(x.data)
        return type(x)(result)
        ### END SOLUTION
    
    def forward_(self, x):
        """
        Apply Tanh activation in-place: modifies input tensor directly
        
        In-place tanh saves memory by reusing existing tensor buffer.
        Particularly useful for RNN and LSTM implementations.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute tanh directly into tensor._data
        2. Return the same tensor object (modified in-place)
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x) = tanh(x)
        
        MEMORY BENEFITS:
        - No new tensor allocation (saves memory)
        - Reuses existing memory buffer
        - Critical for RNN/LSTM with many timesteps
        - Used in PyTorch with tanh_() syntax
        
        EXAMPLE USAGE:
        ```python
        tanh = Tanh()
        # In-place: modifies existing tensor
        x = Tensor([[0.0, 1.0, -1.0]])
        tanh.forward_(x)  # x is now [[0.0, 0.76, -0.76]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.tanh() for numerical stability
        - Modify x._data directly
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.functional.tanh_() in PyTorch
        - Memory-efficient for RNN/LSTM implementations
        - Trade-off: can't recover original values after operation
        """
        ### BEGIN SOLUTION
        # Apply Tanh in-place: modify tensor data directly
        np.tanh(x._data, out=x._data)
        return x
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable: tanh(x) instead of tanh.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Tanh Implementation

Once you implement the Tanh forward method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-tanh-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_tanh_activation():
    """Unit test for the Tanh activation function."""
    print("🔬 Unit Test: Tanh Activation...")

# Create Tanh instance
    tanh = Tanh()

    # Test with zero (should be 0)
    test_input = Tensor([[0]])
    result = tanh(test_input)
    expected = 0.0
    
    assert abs(result.data[0][0] - expected) < 1e-6, f"Tanh(0) should be 0, got {result.data[0][0]}"
    
    # Test with positive and negative values
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = tanh(test_input)
    
    # Check that all values are between -1 and 1
    assert np.all(result.data > -1), "Tanh output should be > -1"
    assert np.all(result.data < 1), "Tanh output should be < 1"
    
    # Test antisymmetry: tanh(-x) = -tanh(x)
    x_val = 1.5
    pos_result = tanh(Tensor([[x_val]]))
    neg_result = tanh(Tensor([[-x_val]]))
    antisymmetry_check = abs(pos_result.data[0][0] + neg_result.data[0][0])
    assert antisymmetry_check < 1e-6, "Tanh should be antisymmetric"
    
    # Test with 2D tensor
    matrix_input = Tensor([[-1, 1], [0, 2]])
    matrix_result = tanh(matrix_input)
    assert matrix_result.shape == matrix_input.shape, "Tanh should preserve shape"
    
    # Test extreme values (should not overflow)
    extreme_input = Tensor([[-100, 100]])
    extreme_result = tanh(extreme_input)
    assert not np.any(np.isnan(extreme_result.data)), "Tanh should handle extreme values"
    assert not np.any(np.isinf(extreme_result.data)), "Tanh should not produce inf values"
    
    # Test that extreme values approach ±1
    assert abs(extreme_result.data[0][0] - (-1)) < 1e-6, "Tanh(-∞) should approach -1"
    assert abs(extreme_result.data[0][1] - 1) < 1e-6, "Tanh(∞) should approach 1"
    
    print("✅ Tanh activation tests passed!")
    print(f"✅ Outputs correctly bounded between -1 and 1")
    print(f"✅ Antisymmetric property verified")
    print(f"✅ Zero-centered (tanh(0) = 0)")
    print(f"✅ Handles extreme values correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
## Step 4: Softmax - Probability Distributions

### What is Softmax?
**Softmax** converts a vector of real numbers into a probability distribution:

```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```

### Why Softmax is Essential
1. **Probability distribution**: Outputs sum to 1
2. **Multi-class classification**: Choose one class from many
3. **Interpretable**: Each output is a probability
4. **Differentiable**: Enables gradient-based learning

### Visual Understanding
```
Input:  [1, 2, 3]
Softmax:[0.09, 0.24, 0.67]  # Sums to 1.0
```

### Real-World Applications
- **Classification**: Image classification, text classification
- **Language models**: Next word prediction
- **Attention mechanisms**: Where to focus attention
- **Reinforcement learning**: Action selection probabilities

### Mathematical Properties
- **Range**: (0, 1) for each output
- **Constraint**: Σ(f(x_i)) = 1
- **Argmax preservation**: Doesn't change relative ordering
- **Temperature scaling**: Can be made sharper or softer
"""

# %% nbgrader={"grade": false, "grade_id": "softmax-class", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Softmax:
    """
    Softmax Activation Function: f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Converts a vector of real numbers into a probability distribution.
    Essential for multi-class classification.
    """
    
    def forward(self, x):
        """
        Apply Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        Works with Tensor inputs for probability distributions.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute softmax with numerical stability
        2. Return new Tensor with softmax applied
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x_i) = e^(x_i) / Σ(e^(x_j))
        - Converts any real vector to probability distribution (sums to 1)
        
        EXAMPLE USAGE:
        ```python
        softmax = Softmax()
        tensor_input = Tensor([[1.0, 2.0, 3.0]])
        tensor_output = softmax(tensor_input)
        # Output: [[0.09, 0.24, 0.67]] (approximately)
        ```
        
        IMPLEMENTATION HINTS:
        - Use numerical stability: subtract max before exponential
        - Normalize by sum to get probabilities
        
        LEARNING CONNECTIONS:
        - This is the core of torch.nn.Softmax() in PyTorch
        - Used in classification and attention mechanisms
        - Converts logits to probability distributions
        - Always outputs positive values that sum to 1
        """
        ### BEGIN SOLUTION
        # Apply Softmax with numerical stability
        input_data = x.data
        
        # Handle empty input
        if input_data.size == 0:
            return type(x)(input_data.copy())
        
        # Subtract max for numerical stability
        x_shifted = input_data - np.max(input_data, axis=-1, keepdims=True)
        
        # Compute exponentials
        exp_values = np.exp(x_shifted)
        
        # Sum along last axis
        sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
        
        # Divide to get probabilities
        output_data = exp_values / sum_exp
        
        return type(x)(output_data)
        ### END SOLUTION
    
    def forward_(self, x):
        """
        Apply Softmax activation in-place: modifies input tensor directly
        
        In-place softmax saves memory by reusing existing tensor buffer.
        Important for classification layers with large vocabulary.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute softmax directly into tensor._data
        2. Return the same tensor object (modified in-place)
        
        MATHEMATICAL FOUNDATION:
        - Forward: f(x_i) = e^(x_i) / Σ(e^(x_j))
        
        MEMORY BENEFITS:
        - No new tensor allocation (saves memory)
        - Reuses existing memory buffer
        - Critical for large vocabulary language models
        - Used in PyTorch with softmax_() syntax
        
        EXAMPLE USAGE:
        ```python
        softmax = Softmax()
        # In-place: modifies existing tensor
        x = Tensor([[1.0, 2.0, 3.0]])
        softmax.forward_(x)  # x is now [[0.09, 0.24, 0.67]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use numerical stability: subtract max before exponential
        - Modify x._data directly
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.functional.softmax_() in PyTorch
        - Memory-efficient for large classification problems
        - Trade-off: can't recover original logits after operation
        """
        ### BEGIN SOLUTION
        # Apply Softmax in-place with numerical stability
        # Handle empty input
        if x._data.size == 0:
            return x
        
        # Subtract max for numerical stability
        max_vals = np.max(x._data, axis=-1, keepdims=True)
        x._data -= max_vals
        
        # Compute exponentials in-place
        np.exp(x._data, out=x._data)
        
        # Sum along last axis
        sum_exp = np.sum(x._data, axis=-1, keepdims=True)
        
        # Divide to get probabilities in-place
        x._data /= sum_exp
        
        return x
        ### END SOLUTION
    
    def __call__(self, x):
        """Make the class callable: softmax(x) instead of softmax.forward(x)"""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Test Your Softmax Implementation

Once you implement the Softmax forward method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-softmax-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_softmax_activation():
    """Unit test for the Softmax activation function."""
    print("🔬 Unit Test: Softmax Activation...")

# Create Softmax instance
    softmax = Softmax()

    # Test with simple input
    test_input = Tensor([[1, 2, 3]])
    result = softmax(test_input)
    
    # Check that outputs sum to 1
    output_sum = np.sum(result.data)
    assert abs(output_sum - 1.0) < 1e-6, f"Softmax outputs should sum to 1, got {output_sum}"
    
    # Check that all outputs are positive
    assert np.all(result.data > 0), "Softmax outputs should be positive"
    assert np.all(result.data < 1), "Softmax outputs should be less than 1"
    
    # Test with uniform input (should give equal probabilities)
    uniform_input = Tensor([[1, 1, 1]])
    uniform_result = softmax(uniform_input)
    expected_prob = 1.0 / 3.0
    
    for prob in uniform_result.data[0]:
        assert abs(prob - expected_prob) < 1e-6, f"Uniform input should give equal probabilities"
    
    # Test with batch input (multiple samples)
    batch_input = Tensor([[1, 2, 3], [4, 5, 6]])
    batch_result = softmax(batch_input)
    
    # Check that each row sums to 1
    for i in range(batch_input.shape[0]):
        row_sum = np.sum(batch_result.data[i])
        assert abs(row_sum - 1.0) < 1e-6, f"Each row should sum to 1, row {i} sums to {row_sum}"
    
    # Test numerical stability with large values
    large_input = Tensor([[1000, 1001, 1002]])
    large_result = softmax(large_input)
    
    assert not np.any(np.isnan(large_result.data)), "Softmax should handle large values"
    assert not np.any(np.isinf(large_result.data)), "Softmax should not produce inf values"
    
    large_sum = np.sum(large_result.data)
    assert abs(large_sum - 1.0) < 1e-6, "Large values should still sum to 1"

# Test shape preservation
    assert batch_result.shape == batch_input.shape, "Softmax should preserve shape"
    
    print("✅ Softmax activation tests passed!")
    print(f"✅ Outputs sum to 1 (probability distribution)")
    print(f"✅ All outputs are positive")
    print(f"✅ Handles uniform inputs correctly")
    print(f"✅ Works with batch inputs")
    print(f"✅ Numerically stable with large values")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🎯 Comprehensive Test: All Activations Working Together

### Real-World Scenario
Let us test how all activation functions work together in a realistic neural network scenario:

- **Input processing**: Raw data transformation
- **Hidden layers**: ReLU for internal processing
- **Output layer**: Softmax for classification
- **Comparison**: See how different activations transform the same data
"""

# %% nbgrader={"grade": true, "grade_id": "test-activations-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_activations_comprehensive():
    """Comprehensive unit test for all activation functions working together."""
    print("🔬 Unit Test: Activation Functions Comprehensive Test...")
    
    # Create instances of all activation functions
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    softmax = Softmax()
    
    # Test data: simulating neural network layer outputs
    test_data = Tensor([[-2, -1, 0, 1, 2]])
    
    # Apply each activation function
    relu_result = relu(test_data)
    sigmoid_result = sigmoid(test_data)
    tanh_result = tanh(test_data)
    softmax_result = softmax(test_data)
    
    # Test that all functions preserve input shape
    assert relu_result.shape == test_data.shape, "ReLU should preserve shape"
    assert sigmoid_result.shape == test_data.shape, "Sigmoid should preserve shape"
    assert tanh_result.shape == test_data.shape, "Tanh should preserve shape"
    assert softmax_result.shape == test_data.shape, "Softmax should preserve shape"
    
    # Test that all functions return Tensor objects
    assert isinstance(relu_result, Tensor), "ReLU should return Tensor"
    assert isinstance(sigmoid_result, Tensor), "Sigmoid should return Tensor"
    assert isinstance(tanh_result, Tensor), "Tanh should return Tensor"
    assert isinstance(softmax_result, Tensor), "Softmax should return Tensor"
    
    # Test ReLU properties
    assert np.all(relu_result.data >= 0), "ReLU output should be non-negative"
    
    # Test Sigmoid properties
    assert np.all(sigmoid_result.data > 0), "Sigmoid output should be positive"
    assert np.all(sigmoid_result.data < 1), "Sigmoid output should be less than 1"
    
    # Test Tanh properties
    assert np.all(tanh_result.data > -1), "Tanh output should be > -1"
    assert np.all(tanh_result.data < 1), "Tanh output should be < 1"
    
    # Test Softmax properties
    softmax_sum = np.sum(softmax_result.data)
    assert abs(softmax_sum - 1.0) < 1e-6, "Softmax outputs should sum to 1"
    
    # Test chaining activations (realistic neural network scenario)
    # Hidden layer with ReLU
    hidden_output = relu(test_data)
    
    # Add some weights simulation (element-wise multiplication)
    weights = Tensor([[0.5, 0.3, 0.8, 0.2, 0.7]])
    weighted_output = hidden_output * weights
    
    # Final layer with Softmax
    final_output = softmax(weighted_output)
    
    # Test that chained operations work
    assert isinstance(final_output, Tensor), "Chained operations should return Tensor"
    assert abs(np.sum(final_output.data) - 1.0) < 1e-6, "Final output should be valid probability"
    
    # Test with batch data (multiple samples)
    batch_data = Tensor([
    [-2, -1, 0, 1, 2],
    [1, 2, 3, 4, 5],
    [-1, 0, 1, 2, 3]
    ])
    
    batch_softmax = softmax(batch_data)
    
    # Each row should sum to 1
    for i in range(batch_data.shape[0]):
        row_sum = np.sum(batch_softmax.data[i])
        assert abs(row_sum - 1.0) < 1e-6, f"Batch row {i} should sum to 1"
    
    print("✅ Activation functions comprehensive tests passed!")
    print(f"✅ All functions work together seamlessly")
    print(f"✅ Shape preservation across all activations")
    print(f"✅ Chained operations work correctly")
    print(f"✅ Batch processing works for all activations")
    print(f"✅ Ready for neural network integration!")

# Test function defined (called in main block)

# %%
def test_module_activation_tensor_integration():
    """
    Integration test for activation functions with Tensor operations.
    
    Tests that activation functions properly integrate with the Tensor class
    and maintain compatibility for neural network operations.
    """
    print("🔬 Running Integration Test: Activation-Tensor Integration...")
    
    # Test 1: Activation functions preserve Tensor types
    input_tensor = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    relu_fn = ReLU()
    sigmoid_fn = Sigmoid()
    tanh_fn = Tanh()
    
    relu_result = relu_fn(input_tensor)
    sigmoid_result = sigmoid_fn(input_tensor) 
    tanh_result = tanh_fn(input_tensor)
    
    assert isinstance(relu_result, Tensor), "ReLU should return Tensor"
    assert isinstance(sigmoid_result, Tensor), "Sigmoid should return Tensor"
    assert isinstance(tanh_result, Tensor), "Tanh should return Tensor"
    
    # Test 2: Activations work with matrix Tensors (neural network layers)
    layer_output = Tensor([[1.0, -2.0, 3.0], 
                          [-1.0, 2.0, -3.0]])  # Simulating dense layer output
    
    relu_fn = ReLU()
    activated = relu_fn(layer_output)
    expected = np.array([[1.0, 0.0, 3.0], 
                        [0.0, 2.0, 0.0]])
    
    assert isinstance(activated, Tensor), "Matrix activation should return Tensor"
    assert np.array_equal(activated.data, expected), "Matrix ReLU should work correctly"
    
    # Test 3: Softmax with classification scenario
    logits = Tensor([[2.0, 1.0, 0.1],  # Batch of 2 samples
                    [1.0, 3.0, 0.2]])   # Each with 3 classes
    
    softmax_fn = Softmax()
    probabilities = softmax_fn(logits)
    
    assert isinstance(probabilities, Tensor), "Softmax should return Tensor"
    assert probabilities.shape == logits.shape, "Softmax should preserve shape"
    
    # Each row should sum to 1 (probability distribution)
    for i in range(logits.shape[0]):
        row_sum = np.sum(probabilities.data[i])
        assert abs(row_sum - 1.0) < 1e-6, f"Probability row {i} should sum to 1"
    
    # Test 4: Chaining tensor operations with activations
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    
    # Simulate: dense layer output -> activation -> more operations
    dense_sim = x * y  # Element-wise multiplication (simulating dense layer)
    relu_fn = ReLU()
    activated = relu_fn(dense_sim)  # Apply activation
    final = activated + Tensor([1.0, 1.0, 1.0])  # More tensor operations
    
    expected_final = np.array([5.0, 11.0, 19.0])  # [4,10,18] -> relu -> +1 = [5,11,19]
    
    assert isinstance(final, Tensor), "Chained operations should maintain Tensor type"
    assert np.array_equal(final.data, expected_final), "Chained operations should work correctly"
    
    print("✅ Integration Test Passed: Activation-Tensor integration works correctly.")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🧪 Comprehensive Testing Suite

Let's test that our activation functions work correctly with Tensor inputs and produce mathematically correct outputs.
"""


# %% nbgrader={"grade": true, "grade_id": "test-activations-tensor-compatibility", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_activations_tensor_compatibility():
    """Test that activation functions work correctly with Tensor inputs."""
    print("🔬 Unit Test: Activation Functions Tensor Compatibility...")
    
    # Create instances of all activation functions
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    softmax = Softmax()
    
    # Test with Tensor inputs
    tensor_input = Tensor([[-2, -1, 0, 1, 2]])
    
    # Test that all activations return Tensors when given Tensors
    relu_result = relu(tensor_input)
    sigmoid_result = sigmoid(tensor_input)
    tanh_result = tanh(tensor_input)
    softmax_result = softmax(tensor_input)
    
    assert isinstance(relu_result, Tensor), "ReLU should return Tensor when input is Tensor"
    assert isinstance(sigmoid_result, Tensor), "Sigmoid should return Tensor when input is Tensor"
    assert isinstance(tanh_result, Tensor), "Tanh should return Tensor when input is Tensor"
    assert isinstance(softmax_result, Tensor), "Softmax should return Tensor when input is Tensor"
    
    # Test that results are mathematically correct
    expected_relu = np.array([[0, 0, 0, 1, 2]])
    assert np.array_equal(relu_result.data, expected_relu), "ReLU with Tensor should produce correct results"
    
    assert np.all(sigmoid_result.data > 0), "Sigmoid should produce positive values"
    assert np.all(sigmoid_result.data < 1), "Sigmoid should produce values less than 1"
    
    assert np.all(tanh_result.data > -1), "Tanh should produce values > -1"
    assert np.all(tanh_result.data < 1), "Tanh should produce values < 1"
    
    assert abs(np.sum(softmax_result.data) - 1.0) < 1e-6, "Softmax should sum to 1"
    
    print("✅ Tensor compatibility tests passed!")
    print(f"✅ All activations work with Tensors")
    print(f"✅ Mathematical correctness preserved")

# Test function defined (called in main block)

# Test function defined (called in main block)

# %% [markdown]
"""
## 🔄 In-Place Operations: Memory-Efficient Activations

Now that you have working activation functions, let's implement **in-place operations** that modify tensors directly instead of creating new ones. This is crucial for memory efficiency in large neural networks.

### **Learning Outcome**: *"I understand how in-place operations save memory and when to use them"*

---

## Why In-Place Operations Matter

### Memory Efficiency Problem
In large neural networks, creating new tensors for every operation consumes enormous memory:

```python
# Memory-hungry approach (creates new tensors)
x = Tensor(large_data)  # 1GB
y = relu(x)            # 2GB total (x + y)
z = sigmoid(y)         # 3GB total (x + y + z)
```

### In-Place Solution
Modify existing tensors directly to save memory:

```python
# Memory-efficient approach (reuses tensors)
x = Tensor(large_data)  # 1GB
relu.forward_(x)        # 1GB total (x modified in-place)
sigmoid.forward_(x)     # 1GB total (x modified again)
```

### Production Context
- **PyTorch**: Uses `relu_()`, `sigmoid_()`, `tanh_()` for in-place operations
- **Memory savings**: Critical for training large models (GPT, ResNet, etc.)
- **Trade-offs**: Can't recover original values, affects gradient computation
"""

# %% [markdown]
"""
## 🧪 Comprehensive Test: In-Place Activation Functions

Let's test that our in-place implementations work correctly and actually save memory.
"""

# %% nbgrader={"grade": true, "grade_id": "test-inplace-operations", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_inplace_operations():
    """Test in-place activation functions for correctness and memory efficiency."""
    print("🔬 Unit Test: In-Place Activation Functions...")
    
    # Test 1: ReLU in-place operation
    print("  Testing ReLU in-place...")
    relu = ReLU()
    
    # Test that operation modifies tensor in-place
    x_relu = Tensor([[-2, -1, 0, 1, 2]])
    original_id = id(x_relu._data)
    original_data = x_relu.data.copy()
    
    result = relu.forward_(x_relu)
    
    # Verify same object returned
    assert result is x_relu, "In-place operation should return same object"
    assert id(x_relu._data) == original_id, "In-place operation should not create new data array"
    
    # Verify correct computation
    expected = np.maximum(0, original_data)
    assert np.array_equal(x_relu.data, expected), "ReLU in-place computation incorrect"
    
    # Test 2: Sigmoid in-place operation
    print("  Testing Sigmoid in-place...")
    sigmoid = Sigmoid()
    
    x_sigmoid = Tensor([[0.0, 1.0, -1.0]])
    original_id = id(x_sigmoid._data)
    original_data = x_sigmoid.data.copy()
    
    result = sigmoid.forward_(x_sigmoid)
    
    # Verify same object returned
    assert result is x_sigmoid, "Sigmoid in-place should return same object"
    assert id(x_sigmoid._data) == original_id, "Sigmoid in-place should not create new data array"
    
    # Verify correct computation (approximately)
    expected = 1 / (1 + np.exp(-original_data))
    assert np.allclose(x_sigmoid.data, expected), "Sigmoid in-place computation incorrect"
    
    # Test 3: Tanh in-place operation
    print("  Testing Tanh in-place...")
    tanh = Tanh()
    
    x_tanh = Tensor([[0.0, 1.0, -1.0]])
    original_id = id(x_tanh._data)
    original_data = x_tanh.data.copy()
    
    result = tanh.forward_(x_tanh)
    
    # Verify same object returned
    assert result is x_tanh, "Tanh in-place should return same object"
    assert id(x_tanh._data) == original_id, "Tanh in-place should not create new data array"
    
    # Verify correct computation
    expected = np.tanh(original_data)
    assert np.allclose(x_tanh.data, expected), "Tanh in-place computation incorrect"
    
    # Test 4: Softmax in-place operation
    print("  Testing Softmax in-place...")
    softmax = Softmax()
    
    x_softmax = Tensor([[1.0, 2.0, 3.0]])
    original_id = id(x_softmax._data)
    original_data = x_softmax.data.copy()
    
    result = softmax.forward_(x_softmax)
    
    # Verify same object returned
    assert result is x_softmax, "Softmax in-place should return same object"
    assert id(x_softmax._data) == original_id, "Softmax in-place should not create new data array"
    
    # Verify correct computation (probability distribution)
    assert abs(np.sum(x_softmax.data) - 1.0) < 1e-6, "Softmax in-place should sum to 1"
    assert np.all(x_softmax.data > 0), "Softmax in-place should be positive"
    
    # Test 5: Batch processing with in-place operations
    print("  Testing batch in-place operations...")
    batch_tensor = Tensor([[-1, 2], [3, -4], [0, 5]])
    original_id = id(batch_tensor._data)
    
    relu.forward_(batch_tensor)
    
    assert id(batch_tensor._data) == original_id, "Batch in-place should preserve data identity"
    expected_batch = np.array([[0, 2], [3, 0], [0, 5]])
    assert np.array_equal(batch_tensor.data, expected_batch), "Batch in-place computation incorrect"
    
    print("✅ In-place operations tests passed!")
    print(f"✅ All operations modify tensors in-place")
    print(f"✅ Memory addresses preserved during operations")
    print(f"✅ Mathematical correctness maintained")
    print(f"✅ Batch processing works correctly")

# Test function defined (called in main block)

# %% nbgrader={"grade": true, "grade_id": "test-memory-comparison", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_memory_comparison():
    """Compare memory usage between regular and in-place operations."""
    print("🔬 Unit Test: Memory Usage Comparison...")
    
    import sys
    
    # Create test data
    test_size = (100, 100)  # Moderate size for memory testing
    original_data = np.random.randn(*test_size)
    
    # Test 1: Regular operations memory usage
    print("  Testing regular operations memory...")
    x_regular = Tensor(original_data.copy())
    original_memory = sys.getsizeof(x_regular.data)
    
    relu = ReLU()
    y_regular = relu.forward(x_regular)
    
    # Regular operation creates new tensor
    regular_total_memory = sys.getsizeof(x_regular.data) + sys.getsizeof(y_regular.data)
    
    # Test 2: In-place operations memory usage
    print("  Testing in-place operations memory...")
    x_inplace = Tensor(original_data.copy())
    inplace_memory_before = sys.getsizeof(x_inplace.data)
    
    relu.forward_(x_inplace)
    inplace_memory_after = sys.getsizeof(x_inplace.data)
    
    # In-place operation should use same memory
    assert inplace_memory_before == inplace_memory_after, "In-place should not change memory usage"
    assert regular_total_memory > inplace_memory_after, "Regular operations should use more memory"
    
    # Test 3: Memory efficiency calculation
    memory_savings = regular_total_memory - inplace_memory_after
    efficiency_ratio = regular_total_memory / inplace_memory_after
    
    print(f"    Regular operations: {regular_total_memory} bytes")
    print(f"    In-place operations: {inplace_memory_after} bytes")
    print(f"    Memory savings: {memory_savings} bytes ({efficiency_ratio:.1f}x more efficient)")
    
    # Test 4: Chained in-place operations
    print("  Testing chained in-place operations...")
    x_chain = Tensor(np.random.randn(50, 50))
    chain_memory_start = sys.getsizeof(x_chain.data)
    
    # Apply multiple in-place operations
    relu.forward_(x_chain)
    Sigmoid().forward_(x_chain)
    Tanh().forward_(x_chain)
    
    chain_memory_end = sys.getsizeof(x_chain.data)
    
    assert chain_memory_start == chain_memory_end, "Chained in-place should maintain memory usage"
    
    print("✅ Memory comparison tests passed!")
    print(f"✅ In-place operations use ~50% less memory")
    print(f"✅ Memory usage remains constant during chained operations")
    print(f"✅ Significant memory savings for large tensors")

# Test function defined (called in main block)


# %% [markdown]
"""
## ⚡ ML Systems: Performance Analysis & Optimization

Now that you have working activation functions, let us develop **performance engineering skills**. This section teaches you to measure computational costs, understand scaling patterns, and think about production optimization.

### **Learning Outcome**: *"I understand performance trade-offs between different activation functions"*

---

## Performance Profiling Tools (Light Implementation)

As an ML systems engineer, you need to understand which activation functions are fast vs slow, and why. Let us build simple tools to measure and compare performance.
"""

# %% nbgrader={"grade": false, "grade_id": "activation-profiler", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import time

class ActivationProfiler:
    """
    Performance profiling toolkit for activation functions.
    
    Helps ML engineers understand computational costs and optimize
    neural network performance for production deployment.
    """
    
    def __init__(self):
        self.results = {}
        
    def time_activation(self, activation_fn, tensor, activation_name, iterations=100):
        """
        Time how long an activation function takes to run.
        
        TODO: Implement activation timing.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Record start time using time.time()
        2. Run the activation function for specified iterations
        3. Record end time
        4. Calculate average time per iteration
        5. Return the average time in milliseconds
        
        EXAMPLE:
        profiler = ActivationProfiler()
        relu = ReLU()
        test_tensor = Tensor(np.random.randn(1000, 1000))
        avg_time = profiler.time_activation(relu, test_tensor, "ReLU")
        print(f"ReLU took {avg_time:.3f} ms on average")
        
        HINTS:
        - Use time.time() for timing
        - Run multiple iterations for better accuracy
        - Calculate: (end_time - start_time) / iterations * 1000 for ms
        - Return the average time per call in milliseconds
        """
        ### BEGIN SOLUTION
        start_time = time.time()
        
        for _ in range(iterations):
            result = activation_fn(tensor)
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) / iterations * 1000
        
        return avg_time_ms
        ### END SOLUTION
    
    def compare_activations(self, tensor_size=(1000, 1000), iterations=50):
        """
        Compare performance of all activation functions.
        
        This function is PROVIDED to show systems analysis.
        Students run it to understand performance differences.
        """
        print(f"⚡ ACTIVATION PERFORMANCE COMPARISON")
        print(f"=" * 50)
        print(f"Tensor size: {tensor_size}, Iterations: {iterations}")
        
        # Create test tensor
        test_tensor = Tensor(np.random.randn(*tensor_size))
        tensor_mb = test_tensor.data.nbytes / (1024 * 1024)
        print(f"Test tensor: {tensor_mb:.2f} MB")
        
        # Test all activation functions
        activations = {
            'ReLU': ReLU(),
            'Sigmoid': Sigmoid(),
            'Tanh': Tanh(),
            'Softmax': Softmax()
        }
        
        results = {}
        for name, activation_fn in activations.items():
            avg_time = self.time_activation(activation_fn, test_tensor, name, iterations)
            results[name] = avg_time
            print(f"   {name:8}: {avg_time:.3f} ms")
        
        # Calculate speed ratios relative to fastest
        fastest_time = min(results.values())
        fastest_name = min(results, key=results.get)
        
        print(f"\n📊 SPEED ANALYSIS:")
        for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
            speed_ratio = time_ms / fastest_time
            if name == fastest_name:
                print(f"   {name:8}: {speed_ratio:.1f}x (fastest)")
            else:
                print(f"   {name:8}: {speed_ratio:.1f}x slower than {fastest_name}")
        
        return results
    
    def compare_inplace_vs_regular(self, tensor_size=(1000, 1000), iterations=30):
        """
        Compare performance and memory usage between regular and in-place operations.
        
        This function is PROVIDED to demonstrate in-place operation benefits.
        Students use it to understand memory vs performance trade-offs.
        """
        print(f"\n🔄 IN-PLACE vs REGULAR OPERATIONS COMPARISON")
        print(f"=" * 60)
        print(f"Tensor size: {tensor_size}, Iterations: {iterations}")
        
        # Test memory usage
        test_data = np.random.randn(*tensor_size)
        tensor_mb = test_data.nbytes / (1024 * 1024)
        print(f"Test tensor: {tensor_mb:.2f} MB")
        
        activations_to_test = [
            ('ReLU', ReLU()),
            ('Sigmoid', Sigmoid()),
            ('Tanh', Tanh())
        ]
        
        print(f"\n📊 PERFORMANCE & MEMORY COMPARISON:")
        print(f"{'Activation':<10} {'Regular (ms)':<12} {'In-place (ms)':<14} {'Memory Regular (MB)':<18} {'Memory In-place (MB)':<19} {'Speedup':<8}")
        print(f"-" * 90)
        
        for name, activation_fn in activations_to_test:
            # Test regular operations
            test_tensor_regular = Tensor(test_data.copy())
            regular_time = self.time_activation(activation_fn, test_tensor_regular, f"{name}_regular", iterations)
            
            # Create copy for regular operation to measure memory
            x_regular = Tensor(test_data.copy())
            y_regular = activation_fn.forward(x_regular)
            regular_memory = (x_regular.data.nbytes + y_regular.data.nbytes) / (1024 * 1024)
            
            # Test in-place operations
            def time_inplace(activation_fn, tensor, iterations):
                start_time = time.time()
                for _ in range(iterations):
                    # Create fresh tensor for each iteration
                    test_tensor = Tensor(test_data.copy())
                    activation_fn.forward_(test_tensor)
                end_time = time.time()
                return (end_time - start_time) / iterations * 1000
            
            inplace_time = time_inplace(activation_fn, test_tensor_regular, iterations)
            
            # Measure in-place memory usage
            x_inplace = Tensor(test_data.copy())
            activation_fn.forward_(x_inplace)
            inplace_memory = x_inplace.data.nbytes / (1024 * 1024)
            
            # Calculate speedup and memory savings
            speedup = regular_time / inplace_time if inplace_time > 0 else float('inf')
            memory_ratio = regular_memory / inplace_memory
            
            print(f"{name:<10} {regular_time:<12.3f} {inplace_time:<14.3f} {regular_memory:<18.2f} {inplace_memory:<19.2f} {speedup:<8.2f}x")
        
        print(f"\n💡 IN-PLACE OPERATION INSIGHTS:")
        print(f"   - Memory savings: ~50% reduction (no duplicate tensors)")
        print(f"   - Performance: Often faster due to cache locality")
        print(f"   - Trade-off: Original values are lost (can't be recovered)")
        print(f"   - Use case: Inference, memory-constrained training")
        print(f"   - Production: Critical for large model deployment")
    
    def analyze_scaling(self, activation_fn, activation_name, sizes=[100, 500, 1000]):
        """
        Analyze how activation performance scales with tensor size.
        
        This function is PROVIDED to demonstrate scaling patterns.
        Students use it to understand computational complexity.
        """
        print(f"\n🔍 SCALING ANALYSIS: {activation_name}")
        print(f"=" * 40)
        
        scaling_results = []
        
        for size in sizes:
            test_tensor = Tensor(np.random.randn(size, size))
            avg_time = self.time_activation(activation_fn, test_tensor, activation_name, iterations=20)
            
            elements = size * size
            time_per_element = avg_time / elements * 1e6  # microseconds per element
            
            result = {
                'size': size,
                'elements': elements,
                'time_ms': avg_time,
                'time_per_element_us': time_per_element
            }
            scaling_results.append(result)
            
            print(f"   {size}x{size}: {avg_time:.3f}ms ({time_per_element:.3f}μs/element)")
        
        # Analyze scaling pattern
        if len(scaling_results) >= 2:
            small = scaling_results[0]
            large = scaling_results[-1]
            
            size_ratio = large['size'] / small['size']
            time_ratio = large['time_ms'] / small['time_ms']
            
            print(f"\n📈 Scaling Pattern:")
            print(f"   Size increased {size_ratio:.1f}x ({small['size']} → {large['size']})")
            print(f"   Time increased {time_ratio:.1f}x")
            
            if abs(time_ratio - size_ratio**2) < abs(time_ratio - size_ratio):
                print(f"   Pattern: O(n^2) - linear in tensor size")
            else:
                print(f"   Pattern: ~O(n) - very efficient scaling")
        
        return scaling_results

def benchmark_activation_suite():
    """
    Comprehensive benchmark of all activation functions.
    
    This function is PROVIDED to show complete systems analysis.
    Students run it to understand production performance implications.
    """
    profiler = ActivationProfiler()
    
    print("🏆 COMPREHENSIVE ACTIVATION BENCHMARK")
    print("=" * 60)
    
    # Test 1: Performance comparison
    comparison_results = profiler.compare_activations(tensor_size=(800, 800), iterations=30)
    
    # Test 1.5: In-place vs Regular operations comparison
    profiler.compare_inplace_vs_regular(tensor_size=(500, 500), iterations=20)
    
    # Test 2: Scaling analysis for each activation
    activations_to_test = [
        (ReLU(), "ReLU"),
        (Sigmoid(), "Sigmoid"),
        (Tanh(), "Tanh")
    ]
    
    for activation_fn, name in activations_to_test:
        profiler.analyze_scaling(activation_fn, name, sizes=[200, 400, 600])
    
    # Test 3: Memory vs Performance trade-offs
    print(f"\n💾 MEMORY vs PERFORMANCE ANALYSIS:")
    print(f"=" * 40)
    
    test_tensor = Tensor(np.random.randn(500, 500))
    original_memory = test_tensor.data.nbytes / (1024 * 1024)
    
    for name, activation_fn in [("ReLU", ReLU()), ("Sigmoid", Sigmoid())]:
        start_time = time.time()
        result = activation_fn(test_tensor)
        end_time = time.time()
        
        result_memory = result.data.nbytes / (1024 * 1024)
        time_ms = (end_time - start_time) * 1000
        
        print(f"   {name}:")
        print(f"     Input: {original_memory:.2f} MB")
        print(f"     Output: {result_memory:.2f} MB")
        print(f"     Memory overhead: {result_memory - original_memory:.2f} MB")
        print(f"     Time: {time_ms:.3f} ms")
    
    print(f"\n🎯 PRODUCTION INSIGHTS:")
    print(f"   - ReLU is typically fastest (simple max operation)")
    print(f"   - Sigmoid/Tanh slower due to exponential calculations")
    print(f"   - All operations scale linearly with tensor size")
    print(f"   - Memory usage doubles (input + output tensors)")
    print(f"   - Choose activation based on accuracy vs speed trade-offs")
    
    return comparison_results

# %% [markdown]
"""
### 🧪 Test: Activation Performance Profiling

Let us test our activation profiler with realistic performance analysis.
"""

# %% nbgrader={"grade": false, "grade_id": "test-activation-profiler", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_activation_profiler():
    """Test activation profiler with comprehensive scenarios."""
    print("🔬 Unit Test: Activation Performance Profiler...")
    
    profiler = ActivationProfiler()
    
    # Create test tensor
    test_tensor = Tensor(np.random.randn(100, 100))
    relu = ReLU()
    
    # Test timing functionality
    avg_time = profiler.time_activation(relu, test_tensor, "ReLU", iterations=10)
    
    # Verify timing results
    assert isinstance(avg_time, (int, float)), "Should return numeric time"
    assert avg_time > 0, "Time should be positive"
    assert avg_time < 1000, "Time should be reasonable (< 1000ms)"
    
    print("✅ Basic timing functionality test passed")
    
    # Test comparison functionality
    comparison_results = profiler.compare_activations(tensor_size=(50, 50), iterations=5)
    
    # Verify comparison results
    assert isinstance(comparison_results, dict), "Should return dictionary of results"
    assert len(comparison_results) == 4, "Should test all 4 activation functions"
    
    expected_activations = ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']
    for activation in expected_activations:
        assert activation in comparison_results, f"Should include {activation}"
        assert comparison_results[activation] > 0, f"{activation} time should be positive"
    
    print("✅ Activation comparison test passed")
    
    # Test scaling analysis
    scaling_results = profiler.analyze_scaling(relu, "ReLU", sizes=[50, 100])
    
    # Verify scaling results
    assert isinstance(scaling_results, list), "Should return list of scaling results"
    assert len(scaling_results) == 2, "Should test both sizes"
    
    for result in scaling_results:
        assert 'size' in result, "Should include size"
        assert 'time_ms' in result, "Should include timing"
        assert result['time_ms'] > 0, "Time should be positive"
    
    print("✅ Scaling analysis test passed")
    
    print("🎯 Activation Profiler: All tests passed!")

# Test function defined (called in main block)

# %% [markdown]
"""
### 🎯 Learning Activity: Activation Performance Analysis

**Goal**: Learn to measure activation function performance and understand which operations are fast vs slow in production ML systems.
"""

# %% nbgrader={"grade": false, "grade_id": "activation-performance-analysis", "locked": false, "schema_version": 3, "solution": false, "task": false}
# Activation profiler initialization moved to main block

if __name__ == "__main__":
    # Initialize the activation profiler
    profiler = ActivationProfiler()
    
    # Run all activation tests
    test_unit_relu_activation()
    test_unit_sigmoid_activation()
    test_unit_tanh_activation()
    test_unit_softmax_activation()
    test_unit_activations_comprehensive()
    test_module_activation_tensor_integration()
    
    # Run tensor compatibility tests
    test_unit_activations_tensor_compatibility()
    
    # Run in-place operation tests
    test_unit_inplace_operations()
    test_unit_memory_comparison()
    
    test_activation_profiler()
    
    print("⚡ ACTIVATION PERFORMANCE ANALYSIS")
    print("=" * 50)

    # Create test data
    test_tensor = Tensor(np.random.randn(500, 500))  # Medium-sized tensor for testing
    print(f"Test tensor size: {test_tensor.shape}")
    print(f"Memory footprint: {test_tensor.data.nbytes/(1024*1024):.2f} MB")

    # Test individual activation timing
    print(f"\n🎯 Individual Activation Timing:")
    activations_to_test = [
        (ReLU(), "ReLU"),
        (Sigmoid(), "Sigmoid"), 
        (Tanh(), "Tanh"),
        (Softmax(), "Softmax")
    ]

    individual_results = {}
    for activation_fn, name in activations_to_test:
        # Students implement this timing call
        avg_time = profiler.time_activation(activation_fn, test_tensor, name, iterations=50)
        individual_results[name] = avg_time
        print(f"   {name:8}: {avg_time:.3f} ms average")

    # Analyze the results  
    fastest = min(individual_results, key=individual_results.get)
    slowest = max(individual_results, key=individual_results.get)
    speed_ratio = individual_results[slowest] / individual_results[fastest]

    print(f"\n📊 PERFORMANCE INSIGHTS:")
    print(f"   Fastest: {fastest} ({individual_results[fastest]:.3f} ms)")
    print(f"   Slowest: {slowest} ({individual_results[slowest]:.3f} ms)")
    print(f"   Speed difference: {speed_ratio:.1f}x")

    print(f"\n💡 WHY THE DIFFERENCE?")
    print(f"   - ReLU: Just max(0, x) - simple comparison")
    print(f"   - Sigmoid: Requires exponential calculation")
    print(f"   - Tanh: Also exponential, but often optimized")
    print(f"   - Softmax: Exponentials + division")

    print(f"\n🏭 PRODUCTION IMPLICATIONS:")
    print(f"   - ReLU dominates modern deep learning (speed + effectiveness)")
    print(f"   - Sigmoid/Tanh used where probability interpretation needed")
    print(f"   - Speed matters: 1000 layers × speed difference = major impact")
    
    print("All tests passed!")
    print("Activations module complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built the nonlinear functions that enable neural network intelligence, let's connect this foundational work to broader ML systems challenges. These questions help you think critically about how activation functions scale to production ML environments.

Take time to reflect thoughtfully on each question - your insights will help you understand how the activation concepts you've implemented connect to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Computational Efficiency and Numerical Stability

**Context**: Your activation implementations handle basic operations like ReLU's max(0, x) and Softmax's exponential computations. In production ML systems, these operations run billions of times during training and inference, making computational efficiency and numerical stability critical for system reliability.

**Reflection Question**: Design a production-grade activation function system that balances computational efficiency with numerical stability. How would you optimize ReLU for sparse computation, implement numerically stable Softmax for large vocabulary language models, and handle precision requirements across different hardware platforms? Consider scenarios where numerical instability in activation functions could cascade through deep networks and cause training failures.

Think about: vectorization strategies, overflow/underflow protection, sparse computation optimization, and precision trade-offs between speed and accuracy.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-computational-efficiency", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON COMPUTATIONAL EFFICIENCY AND NUMERICAL STABILITY:

TODO: Replace this text with your thoughtful response about production-grade activation function design.

Consider addressing:
- How would you optimize activation functions for both efficiency and numerical stability?
- What strategies would you use to handle large-scale sparse computation in ReLU?
- How would you implement numerically stable Softmax for large vocabulary models?
- What precision trade-offs would you make across different hardware platforms?
- How would you prevent numerical instability from cascading through deep networks?

Write a technical analysis connecting your activation implementations to real production optimization challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of efficiency vs stability trade-offs (3 points)
- Addresses numerical stability concerns in large-scale systems (3 points)
- Shows practical knowledge of optimization strategies (2 points)
- Demonstrates systems thinking about activation function design (2 points)
- Clear technical reasoning and practical considerations (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring technical analysis of activation optimization
# Students should demonstrate understanding of efficiency and numerical stability in production systems
### END SOLUTION

# %% [markdown]
"""
### Question 2: Hardware Optimization and Parallelization

**Context**: Your activation functions perform element-wise operations that are ideal for parallel computation. Production ML systems deploy these functions across diverse hardware: CPUs, GPUs, TPUs, and edge devices, each with different computational characteristics and optimization opportunities.

**Reflection Question**: Architect a hardware-aware activation function system that automatically optimizes for different compute platforms. How would you leverage ReLU's sparsity for GPU memory optimization, implement vectorized operations for CPU SIMD instructions, and design activation kernels for specialized AI accelerators? Consider the challenges of maintaining consistent numerical behavior across platforms while maximizing hardware-specific performance.

Think about: SIMD vectorization, GPU kernel fusion, sparse computation patterns, and platform-specific optimization techniques.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-hardware-optimization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON HARDWARE OPTIMIZATION AND PARALLELIZATION:

TODO: Replace this text with your thoughtful response about hardware-aware activation function design.

Consider addressing:
- How would you design activation functions that optimize for different hardware platforms?
- What strategies would you use to leverage GPU parallelism for activation computations?
- How would you implement SIMD vectorization for CPU-based activation functions?
- What role would kernel fusion play in optimizing activation performance?
- How would you maintain numerical consistency across different hardware platforms?

Write an architectural analysis connecting your activation implementations to real hardware optimization challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of hardware-specific optimization strategies (3 points)
- Designs practical approaches to parallel activation computation (3 points)
- Addresses platform consistency and performance trade-offs (2 points)
- Demonstrates systems thinking about hardware-software optimization (2 points)
- Clear architectural reasoning with hardware insights (bonus points for comprehensive understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of hardware optimization challenges
# Students should demonstrate knowledge of parallel computation and platform-specific optimization
### END SOLUTION

# %% [markdown]
"""
### Question 3: Integration with Training Systems and Gradient Flow

**Context**: Your activation functions will integrate with automatic differentiation systems for training neural networks. The choice and implementation of activation functions significantly impacts gradient flow, training stability, and convergence speed in large-scale ML training systems.

**Reflection Question**: Design an activation function integration system for large-scale neural network training that optimizes gradient flow and training stability. How would you implement activation functions that support efficient gradient computation, handle the vanishing gradient problem in deep networks, and integrate with distributed training systems? Consider the challenges of maintaining training stability when activation choices affect gradient magnitude and direction across hundreds of layers.

Think about: gradient flow characteristics, backpropagation efficiency, training stability, and distributed training considerations.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-training-integration", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON INTEGRATION WITH TRAINING SYSTEMS:

TODO: Replace this text with your thoughtful response about activation function integration with training systems.

Consider addressing:
- How would you design activation functions to optimize gradient flow in deep networks?
- What strategies would you use to handle vanishing/exploding gradient problems?
- How would you integrate activation functions with automatic differentiation systems?
- What role would activation choices play in distributed training stability?
- How would you balance activation complexity with training efficiency?

Write a design analysis connecting your activation functions to automatic differentiation and training optimization.

GRADING RUBRIC (Instructor Use):
- Understands activation function impact on gradient flow and training (3 points)
- Designs practical approaches to training integration and stability (3 points)
- Addresses distributed training and efficiency considerations (2 points)
- Shows systems thinking about training system architecture (2 points)
- Clear design reasoning with training optimization insights (bonus points for deep understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of training system integration
# Students should demonstrate knowledge of gradient flow and training optimization challenges
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Activation Functions

    Congratulations! You have successfully implemented all four essential activation functions:

### ✅ What You have Built
    - **ReLU**: The foundation of modern deep learning with sparsity and efficiency
    - **Sigmoid**: Classic activation for binary classification and probability outputs
    - **Tanh**: Zero-centered activation with better gradient properties
    - **Softmax**: Probability distribution for multi-class classification
    - **💾 Memory Efficiency**: In-place operations to minimize memory usage
    - **⚡ Performance Analysis**: Tools to measure and optimize computational costs
    - **🆕 In-Place Operations**: Memory-efficient versions (forward_) that modify tensors directly
    - **🆕 Memory Optimization**: Understand and measure memory vs performance trade-offs

### ✅ Key Learning Outcomes
    - **Understanding**: Why nonlinearity is essential for neural networks
    - **Implementation**: Built activation functions from scratch using NumPy
    - **Testing**: Progressive validation with immediate feedback after each function
    - **Integration**: Saw how activations work together in neural networks
    - **Real-world context**: Understanding where each activation is used
    - **🆕 Autograd Integration**: Learned how to make functions work with automatic differentiation
    - **🆕 Gradient Computation**: Implemented mathematically correct backward passes
    - **🆕 Memory Efficiency**: Implemented in-place operations for memory optimization
    - **🆕 Performance Analysis**: Measured and compared activation performance characteristics

### ✅ Mathematical Mastery
    - **ReLU**: f(x) = max(0, x), f'(x) = 1 if x > 0 else 0
    - **Sigmoid**: f(x) = 1/(1 + e^(-x)), f'(x) = f(x)(1 - f(x))
    - **Tanh**: f(x) = tanh(x), f'(x) = 1 - f(x)²
    - **Softmax**: f(x_i) = e^(x_i)/Σ(e^(x_j)), complex Jacobian for backprop
    - **🆕 Gradient Functions**: All derivatives implemented for automatic differentiation

### ✅ Professional Skills Developed
    - **Numerical stability**: Handling overflow and underflow
    - **API design**: Consistent interfaces across all functions
    - **Testing discipline**: Immediate validation after each implementation
    - **Integration thinking**: Understanding how components work together
    - **🆕 Autograd Design**: Making functions compatible with automatic differentiation
    - **🆕 Backward Pass Implementation**: Writing gradient functions for training
    - **🆕 Memory Engineering**: Implementing in-place operations for efficiency
    - **🆕 Performance Profiling**: Measuring and optimizing computational performance

### ✅ Ready for Next Steps
    Your activation functions are now ready to power:
    - **Dense layers**: Linear transformations with nonlinear activations
    - **Convolutional layers**: Spatial feature extraction with ReLU
    - **Network architectures**: Complete neural networks with proper activations
    - **🧠 Neural Network Building Blocks**: Foundation components for network architecture
    - **🚀 Production Ready**: Numerically stable implementations

### 🔗 Connection to Real ML Systems
    Your implementations mirror production systems:
    - **PyTorch**: `torch.nn.ReLU()`, `torch.nn.Sigmoid()`, `torch.nn.Tanh()`, `torch.nn.Softmax()`
    - **TensorFlow**: `tf.nn.relu()`, `tf.nn.sigmoid()`, `tf.nn.tanh()`, `tf.nn.softmax()`
    - **Industry applications**: Every major deep learning model uses these functions

### 🎯 The Power of Nonlinearity
    You have unlocked the key to deep learning:
    - **Before**: Linear models limited to simple patterns
    - **After**: Nonlinear models can learn any pattern (universal approximation)

    **Next Module**: Layers - Building blocks that combine your tensors and activations into powerful transformations!

    Your activation functions are the key to neural network intelligence. Now let us build the layers that use them!
""" 