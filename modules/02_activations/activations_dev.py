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
# Activations - Nonlinear Intelligence for Neural Networks

Welcome to Activations! You'll implement the essential functions that enable neural networks to learn complex patterns.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 01 (Tensor): N-dimensional arrays with broadcasting

**The Gap**: Linear operations stacked together remain linear - limiting networks to simple patterns.

**This Module's Solution**: Implement ReLU and Softmax activation functions that add nonlinearity, enabling complex learning.

**Connection Map**:
```
Tensor → Activations → Neural Networks
(data)    (intelligence)  (complex learning)
```

## Learning Objectives
1. **Core Implementation**: Build ReLU and Softmax activation functions
2. **Conceptual Understanding**: How nonlinearity enables complex pattern learning
3. **Testing Skills**: Validate activation functions with comprehensive tests
4. **Integration Knowledge**: Connect activations to neural network systems

## Build → Test → Use
1. **Build**: Implement essential activation functions
2. **Test**: Validate correctness and properties
3. **Use**: Apply in neural network contexts
"""

# In[ ]:

#| default_exp core.activations

#| export
import numpy as np
import os
import sys

# Import our tensor foundation
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development - import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# In[ ]:

print("🔥 TinyTorch Activations Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build essential activation functions!")

# %% [markdown]
"""
## Why Activation Functions Matter

Activation functions inject nonlinearity into neural networks, enabling them to learn complex patterns beyond simple linear relationships.

### ReLU: The Modern Standard

ReLU (Rectified Linear Unit) applies f(x) = max(0, x):
- Zeros out negative values
- Preserves positive values unchanged
- Computationally simple and efficient
- Enables training of very deep networks

### Softmax: Converting Scores to Probabilities

Softmax transforms any vector into a probability distribution:
- All outputs sum to 1.0
- All outputs are non-negative
- Larger inputs get larger probabilities
- Essential for classification tasks
"""

# %% [markdown]
"""
## Part 1: ReLU - The Foundation of Modern Deep Learning
"""

# %% nbgrader={"grade": false, "grade_id": "relu-class", "solution": true}

#| export
class ReLU:
    """
    ReLU Activation Function: f(x) = max(0, x)

    Zeros out negative values, preserves positive values.
    Essential for modern deep learning.
    """
    
    def forward(self, x):
        """
        Apply ReLU activation: f(x) = max(0, x)

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output with negatives zeroed

        TODO: Implement ReLU using numpy's maximum function

        APPROACH:
        1. Validate input is a Tensor
        2. Use np.maximum(0, x.data) for vectorized operation
        3. Return new Tensor with result

        EXAMPLE:
            >>> relu = ReLU()
            >>> x = Tensor([[-1.0, 1.0]])
            >>> y = relu.forward(x)
            >>> print(y.data)  # [[0.0, 1.0]]
        """
        ### BEGIN SOLUTION
        # Input validation
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)}")

        # Check for empty tensor
        if x.data.size == 0:
            return Tensor(np.array([]))

        # Check for NaN or infinite values
        if np.any(np.isnan(x.data)) or np.any(np.isinf(x.data)):
            raise ValueError("Input tensor contains NaN or infinite values")

        # Vectorized element-wise maximum with 0
        # This is the exact operation that revolutionized deep learning!
        result = np.maximum(0, x.data)
        return Tensor(result)
        ### END SOLUTION
    
    def forward_(self, x):
        """
        Apply ReLU in-place (modifies original tensor).

        Args:
            x (Tensor): Input tensor to modify

        Returns:
            Tensor: Same tensor object (modified)
        """
        ### BEGIN SOLUTION
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)}")
        if x.data.size == 0:
            return x
        if np.any(np.isnan(x.data)) or np.any(np.isinf(x.data)):
            raise ValueError("Input tensor contains NaN or infinite values")
        np.maximum(0, x.data, out=x.data)
        return x
        ### END SOLUTION
    
    def __call__(self, x):
        """Make ReLU callable: relu(x) instead of relu.forward(x)"""
        return self.forward(x)

# ✅ IMPLEMENTATION CHECKPOINT: ReLU class complete

# %% [markdown]
"""
## Testing ReLU Implementation

### 🧪 Unit Test: ReLU Activation
This test validates our ReLU implementation with various input scenarios
"""

def test_unit_relu_activation():
    """
    Test ReLU activation function.

    Validates that ReLU zeros negatives and preserves positives.
    """
    print("🔬 Unit Test: ReLU Activation...")

    relu = ReLU()

    # Basic functionality test
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = relu(test_input)
    expected = np.array([[0, 0, 0, 1, 2]])

    assert np.array_equal(result.data, expected), f"ReLU failed: expected {expected}, got {result.data}"

    # 2D tensor test
    matrix_input = Tensor([[-1, 2], [3, -4]])
    matrix_result = relu(matrix_input)
    expected_matrix = np.array([[0, 2], [3, 0]])

    assert np.array_equal(matrix_result.data, expected_matrix), "ReLU should work with 2D tensors"

    # In-place operation test
    inplace_input = Tensor([[-1, 0, 1]])
    relu.forward_(inplace_input)
    expected_inplace = np.array([[0, 0, 1]])

    assert np.array_equal(inplace_input.data, expected_inplace), "In-place ReLU should modify original tensor"

    print("✅ ReLU activation tests passed!")

# Test immediately after implementation
test_unit_relu_activation()

# %% [markdown]
"""
## Part 2: Softmax - Converting Scores to Probabilities

Softmax transforms any real-valued vector into a probability distribution.
Essential for classification and attention mechanisms.
"""

# %% nbgrader={"grade": false, "grade_id": "softmax-class", "solution": true}

#| export
class Softmax:
    """
    Softmax Activation Function: f(x_i) = e^(x_i) / Σ(e^(x_j))

    Converts any vector into a probability distribution.
    Essential for classification tasks.
    """
    
    def __init__(self, dim=-1):
        """
        Initialize Softmax with dimension specification.
        
        Args:
            dim (int): Dimension along which to apply softmax.
                      -1 means last dimension (most common)
                      0 means first dimension, etc.
                      
        Examples:
            Softmax(dim=-1)  # Apply along last dimension (default)
            Softmax(dim=0)   # Apply along first dimension
            Softmax(dim=1)   # Apply along second dimension
        """
        self.dim = dim
    
    def forward(self, x):
        """
        Apply Softmax activation with numerical stability.

        Args:
            x (Tensor): Input tensor containing scores

        Returns:
            Tensor: Probability distribution (sums to 1)

        TODO: Implement numerically stable softmax

        APPROACH:
        1. Validate input is a Tensor
        2. Subtract max for numerical stability
        3. Compute exponentials: np.exp(x_stable)
        4. Normalize by sum to create probabilities

        EXAMPLE:
            >>> softmax = Softmax()
            >>> x = Tensor([[1.0, 2.0, 3.0]])
            >>> y = softmax.forward(x)
            >>> print(np.sum(y.data))  # 1.0
        """
        ### BEGIN SOLUTION
        # Input validation
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)}")

        # Check for empty tensor
        if x.data.size == 0:
            raise ValueError("Cannot apply softmax to empty tensor")

        # Check for NaN values (infinite values are handled by max subtraction)
        if np.any(np.isnan(x.data)):
            raise ValueError("Input tensor contains NaN values")

        # Step 1: Numerical stability - subtract maximum value
        # This prevents exp(large_number) from overflowing to infinity
        max_vals = np.max(x.data, axis=self.dim, keepdims=True)
        x_stable = x.data - max_vals

        # Step 2: Compute exponentials of stable values
        exp_vals = np.exp(x_stable)

        # Step 3: Normalize to create probability distribution
        sum_exp = np.sum(exp_vals, axis=self.dim, keepdims=True)

        # Handle edge case where sum is zero (shouldn't happen with valid input)
        if np.any(sum_exp == 0):
            raise ValueError("Softmax normalization resulted in zero sum")

        result = exp_vals / sum_exp

        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x):
        """Make Softmax callable: softmax(x) instead of softmax.forward(x)"""
        return self.forward(x)

# ✅ IMPLEMENTATION CHECKPOINT: Softmax class complete

# %% [markdown]
"""
## Testing Softmax Implementation

### 🧪 Unit Test: Softmax Activation
This test validates our Softmax implementation for correctness and numerical stability
"""

def test_unit_softmax_activation():
    """
    Test Softmax activation function.

    Validates that Softmax creates valid probability distributions.
    """
    print("🔬 Unit Test: Softmax Activation...")

    softmax = Softmax()

    # Basic probability distribution test
    test_input = Tensor([[1.0, 2.0, 3.0]])
    result = softmax(test_input)

    # Check outputs sum to 1
    sum_result = np.sum(result.data, axis=-1)
    assert np.allclose(sum_result, 1.0), f"Softmax should sum to 1, got {sum_result}"
    assert np.all(result.data >= 0), "Softmax outputs should be non-negative"

    # Numerical stability test with large values
    large_input = Tensor([[1000.0, 1001.0, 1002.0]])
    large_result = softmax(large_input)

    assert not np.any(np.isnan(large_result.data)), "Should handle large values without NaN"
    assert np.allclose(np.sum(large_result.data, axis=-1), 1.0), "Large values should still sum to 1"

    # Batch processing test
    batch_input = Tensor([[1.0, 2.0], [3.0, 4.0]])
    batch_result = softmax(batch_input)
    row_sums = np.sum(batch_result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "Each batch item should sum to 1"

    print("✅ Softmax activation tests passed!")

# Test immediately after implementation
test_unit_softmax_activation()

# ✅ IMPLEMENTATION CHECKPOINT: Both ReLU and Softmax complete

# In[ ]:

# %% [markdown]
"""
## Integration Testing: Activations in Neural Network Context

Let's test these activations in realistic neural network scenarios
"""

def test_unit_activation_pipeline():
    """Test activations working together in a neural network pipeline."""
    print("🔬 Unit Test: Activation Pipeline...")

    relu = ReLU()
    softmax = Softmax()

    # Test neural network pipeline
    hidden_output = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    hidden_activated = relu(hidden_output)
    expected_relu = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])

    assert np.array_equal(hidden_activated.data, expected_relu), "ReLU should zero negatives"

    # Classification with Softmax
    class_logits = Tensor([[2.0, 1.0, 0.1]])
    class_probabilities = softmax(class_logits)

    assert np.allclose(np.sum(class_probabilities.data, axis=-1), 1.0), "Softmax should sum to 1"
    assert np.all(class_probabilities.data >= 0), "Probabilities should be non-negative"

    print("✅ Activation pipeline works correctly!")

# Test pipeline functionality
test_unit_activation_pipeline()

# In[ ]:

# %% [markdown]
"""
## Integration Test: Realistic Neural Network Pipeline

Test activations in a complete neural network forward pass simulation
"""

def test_module():
    """Complete module test covering all activation functionality."""
    print("🔬 Complete Module Test: All Activations...")

    # Test individual components
    test_unit_relu_activation()
    test_unit_softmax_activation()
    test_unit_activation_pipeline()

    # Test error handling
    relu = ReLU()
    try:
        relu("not a tensor")
        assert False, "Should raise TypeError"
    except TypeError:
        pass  # Expected

    print("\n✅ Complete module test passed!")
    print("✅ All activation functions working correctly")
    print("✅ Ready for neural network integration")

# Test complete module
test_module()

# In[ ]:

# Main execution block - all tests run when module is executed directly
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 RUNNING ACTIVATION TESTS")
    print("="*50)

    # Run complete module test
    test_module()

    print("\n" + "="*50)
    print("🎉 ACTIVATION MODULE COMPLETE!")
    print("="*50)
    print("✅ ReLU: Simple and effective nonlinearity")
    print("✅ Softmax: Converts scores to probabilities")
    print("💡 Ready to build neural network layers!")

    print(f"\n🎯 Module 02 (Activations) Complete!")
    print(f"Next: Module 03 - Neural Network Layers!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

### Question 1: Activation Function Choice

**Context**: You implemented ReLU (simple max operation) and Softmax (exponentials + normalization).

**Question**: For a mobile neural network with limited compute, analyze the trade-offs between ReLU and Softmax. Consider computational cost, memory usage, and when each is essential.

**YOUR ANALYSIS:**

[Student response area]

### Question 2: Numerical Stability

**Context**: Your Softmax subtracts the maximum value before computing exponentials.

**Question**: Why is this numerical stability crucial? How do small errors in activations affect deep network training?

**YOUR ANALYSIS:**

[Student response area]
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Essential Activations

Congratulations! You've implemented the essential activation functions for neural networks:

### What You've Accomplished
✅ **ReLU Implementation**: The activation function that revolutionized deep learning
✅ **Softmax Implementation**: Converts any vector to a probability distribution
✅ **Testing Framework**: Comprehensive validation of activation properties
✅ **Pipeline Integration**: Demonstrated activations working in neural network contexts

### Key Learning Outcomes
- **Nonlinearity Understanding**: How activation functions enable complex pattern learning
- **Numerical Implementation**: Building mathematically correct and stable algorithms
- **Error Handling**: Robust implementations that handle edge cases gracefully
- **Systems Integration**: Components that work together in larger systems

### Mathematical Foundations Mastered
- **ReLU**: f(x) = max(0, x) - simple yet powerful nonlinearity
- **Softmax**: Converting scores to probabilities with numerical stability
- **Probability Theory**: Understanding valid probability distributions

### Ready for Next Steps
Your activation implementations enable:
- **Neural Network Layers**: Combining with linear transformations
- **Classification**: Converting network outputs to interpretable probabilities
- **Deep Learning**: Training networks with many layers

### Connection to Real Systems
- **PyTorch**: Your implementations mirror `torch.nn.ReLU()` and `torch.nn.Softmax()`
- **Production**: Same mathematical foundations with hardware optimizations

### Next Steps
Ready for Module 03: Neural Network Layers - combining your activations with linear transformations!

**Forward Momentum**: You've built the nonlinear intelligence that makes neural networks powerful!
"""