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
        
        TODO: Implement ReLU activation function.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. For each element in the input tensor, apply max(0, element)
        2. Use NumPy's maximum function for efficient element-wise operation
        3. Return a new tensor of the same type with the results
        4. Preserve the input tensor's shape
        
        EXAMPLE USAGE:
        ```python
        relu = ReLU()
        input_tensor = Tensor([[-2, -1, 0, 1, 2]])
        output = relu(input_tensor)
        print(output.data)  # [[0, 0, 0, 1, 2]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.maximum(0, x.data) for element-wise max with 0
        - Return the same type as input: return type(x)(result)
        - The shape should remain the same as input
        - Don't modify the input tensor (immutable operations)
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.ReLU() in PyTorch
        - Used in virtually every modern neural network
        - Enables deep networks by preventing vanishing gradients
        - Creates sparse representations (many zeros)
        """
        ### BEGIN SOLUTION
        result = np.maximum(0, x.data)
        return type(x)(result)
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
def test_relu_activation():
    """Test ReLU activation function"""
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

# Run the test
test_relu_activation()

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
        
        TODO: Implement Sigmoid activation function.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute the negative of input: -x.data
        2. Compute the exponential: np.exp(-x.data)
        3. Add 1 to the exponential: 1 + np.exp(-x.data)
        4. Take the reciprocal: 1 / (1 + np.exp(-x.data))
        5. Return as new Tensor
        
        EXAMPLE USAGE:
        ```python
        sigmoid = Sigmoid()
        input_tensor = Tensor([[-2, -1, 0, 1, 2]])
        output = sigmoid(input_tensor)
        print(output.data)  # [[0.119, 0.269, 0.5, 0.731, 0.881]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.exp() for exponential function
        - Formula: 1 / (1 + np.exp(-x.data))
        - Handle potential overflow with np.clip(-x.data, -500, 500)
        - Return Tensor(result)
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.Sigmoid() in PyTorch
        - Used in binary classification output layers
        - Key component in LSTM and GRU gating mechanisms
        - Historically important for early neural networks
        """
        ### BEGIN SOLUTION
        # Clip to prevent overflow
        clipped_input = np.clip(-x.data, -500, 500)
        result = 1 / (1 + np.exp(clipped_input))
        return type(x)(result)
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
def test_sigmoid_activation():
    """Test Sigmoid activation function"""
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

# Run the test
test_sigmoid_activation()

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
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Tanh activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        TODO: Implement Tanh activation function.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Use NumPy's built-in tanh function: np.tanh(x.data)
        2. Alternatively, implement manually:
           - Compute e^x and e^(-x)
           - Calculate (e^x - e^(-x)) / (e^x + e^(-x))
        3. Return as new Tensor
        
        EXAMPLE USAGE:
        ```python
        tanh = Tanh()
        input_tensor = Tensor([[-2, -1, 0, 1, 2]])
        output = tanh(input_tensor)
        print(output.data)  # [[-0.964, -0.762, 0, 0.762, 0.964]]
        ```
        
        IMPLEMENTATION HINTS:
        - Use np.tanh(x.data) for simplicity
        - Manual implementation: (np.exp(x.data) - np.exp(-x.data)) / (np.exp(x.data) + np.exp(-x.data))
        - Handle overflow by clipping inputs: np.clip(x.data, -500, 500)
        - Return Tensor(result)
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.Tanh() in PyTorch
        - Used in RNN, LSTM, and GRU cells
        - Better than sigmoid for hidden layers
        - Zero-centered outputs help with gradient flow
        """
        ### BEGIN SOLUTION
        # Use NumPy's built-in tanh function
        result = np.tanh(x.data)
        return type(x)(result)
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
def test_tanh_activation():
    """Test Tanh activation function"""
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

# Run the test
test_tanh_activation()

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
        
        TODO: Implement Softmax activation function.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Handle empty input case
        2. Subtract max value for numerical stability: x - max(x)
        3. Compute exponentials: np.exp(x - max(x))
        4. Compute sum of exponentials: np.sum(exp_values)
        5. Divide each exponential by the sum: exp_values / sum
        6. Return as same tensor type as input
        
        EXAMPLE USAGE:
        ```python
        softmax = Softmax()
        input_tensor = Tensor([[1, 2, 3]])
        output = softmax(input_tensor)
        print(output.data)  # [[0.09, 0.24, 0.67]]
        print(np.sum(output.data))  # 1.0
        ```
        
        IMPLEMENTATION HINTS:
        - Handle empty case: if x.data.size == 0: return type(x)(x.data.copy())
        - Subtract max for numerical stability: x_shifted = x.data - np.max(x.data, axis=-1, keepdims=True)
        - Compute exponentials: exp_values = np.exp(x_shifted)
        - Sum along last axis: sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
        - Divide: result = exp_values / sum_exp
        - Return same type as input: return type(x)(result)
        
        LEARNING CONNECTIONS:
        - This is like torch.nn.Softmax() in PyTorch
        - Used in classification output layers
        - Key component in attention mechanisms
        - Enables probability-based decision making
        """
        ### BEGIN SOLUTION
        # Handle empty input
        if x.data.size == 0:
            return type(x)(x.data.copy())
        
        # Subtract max for numerical stability
        x_shifted = x.data - np.max(x.data, axis=-1, keepdims=True)
        
        # Compute exponentials
        exp_values = np.exp(x_shifted)
        
        # Sum along last axis
        sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
        
        # Divide to get probabilities
        result = exp_values / sum_exp
        
        return type(x)(result)
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
def test_softmax_activation():
    """Test Softmax activation function"""
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

# Run the test
test_softmax_activation()

# %% [markdown]
"""
## 🎯 Comprehensive Test: All Activations Working Together

### Real-World Scenario
Let's test how all activation functions work together in a realistic neural network scenario:

- **Input processing**: Raw data transformation
- **Hidden layers**: ReLU for internal processing
- **Output layer**: Softmax for classification
- **Comparison**: See how different activations transform the same data
"""

# %% nbgrader={"grade": true, "grade_id": "test-activations-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_activations():
    """Test all activation functions working together"""
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

# Run the comprehensive test
test_activations()

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

if __name__ == "__main__":
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("Activations")

# %% [markdown]
"""
## 🎯 Module Summary: Activation Functions Mastery!

    Congratulations! You've successfully implemented all four essential activation functions:

### ✅ What You've Built
    - **ReLU**: The foundation of modern deep learning with sparsity and efficiency
    - **Sigmoid**: Classic activation for binary classification and probability outputs
    - **Tanh**: Zero-centered activation with better gradient properties
    - **Softmax**: Probability distribution for multi-class classification

### ✅ Key Learning Outcomes
    - **Understanding**: Why nonlinearity is essential for neural networks
    - **Implementation**: Built activation functions from scratch using NumPy
    - **Testing**: Progressive validation with immediate feedback after each function
    - **Integration**: Saw how activations work together in neural networks
    - **Real-world context**: Understanding where each activation is used

### ✅ Mathematical Mastery
    - **ReLU**: f(x) = max(0, x) - Simple but powerful
    - **Sigmoid**: f(x) = 1/(1 + e^(-x)) - Maps to (0,1)
    - **Tanh**: f(x) = tanh(x) - Zero-centered, maps to (-1,1)
    - **Softmax**: f(x_i) = e^(x_i)/Σ(e^(x_j)) - Probability distribution

### ✅ Professional Skills Developed
    - **Numerical stability**: Handling overflow and underflow
    - **API design**: Consistent interfaces across all functions
    - **Testing discipline**: Immediate validation after each implementation
    - **Integration thinking**: Understanding how components work together

### ✅ Ready for Next Steps
    Your activation functions are now ready to power:
    - **Dense layers**: Linear transformations with nonlinear activations
    - **Convolutional layers**: Spatial feature extraction with ReLU
    - **Network architectures**: Complete neural networks with proper activations
    - **Training**: Gradient computation through activation functions

### 🔗 Connection to Real ML Systems
    Your implementations mirror production systems:
    - **PyTorch**: `torch.nn.ReLU()`, `torch.nn.Sigmoid()`, `torch.nn.Tanh()`, `torch.nn.Softmax()`
    - **TensorFlow**: `tf.nn.relu()`, `tf.nn.sigmoid()`, `tf.nn.tanh()`, `tf.nn.softmax()`
    - **Industry applications**: Every major deep learning model uses these functions

### 🎯 The Power of Nonlinearity
    You've unlocked the key to deep learning:
    - **Before**: Linear models limited to simple patterns
    - **After**: Nonlinear models can learn any pattern (universal approximation)

    **Next Module**: Layers - Building blocks that combine your tensors and activations into powerful transformations!

    Your activation functions are the key to neural network intelligence. Now let's build the layers that use them!
""" 