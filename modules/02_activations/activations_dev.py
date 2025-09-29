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
## The Intelligence Layer: How Nonlinearity Enables Learning

Without activation functions, neural networks are just fancy linear algebra. No matter how many layers you stack, they can only learn straight lines. Activation functions add the "intelligence" that enables neural networks to learn curves, patterns, and complex relationships.

### The Linearity Problem

```
Linear Network (No Activations):
Input → Linear → Linear → Linear → Output
  x   →  Ax    →  B(Ax) →C(B(Ax)) = (CBA)x

Result: Still just a linear function!
Cannot learn: curves, XOR, complex patterns
```

### The Nonlinearity Solution

```
Nonlinear Network (With Activations):
Input → Linear → ReLU → Linear → ReLU → Output
  x   →  Ax    → max(0,Ax) → B(·) → max(0,B(·))

Result: Can approximate ANY function!
Can learn: curves, XOR, images, language
```

### ReLU: The Intelligence Function

ReLU (Rectified Linear Unit) is the most important function in modern AI:

```
ReLU Function: f(x) = max(0, x)

   y
   ▲
   │   ╱
   │  ╱  (positive values unchanged)
   │ ╱
───┼─────────▶ x
   │ 0      (negative values → 0)
   │

Key Properties:
• Computationally cheap: just comparison and zero
• Gradient friendly: derivative is 0 or 1
• Solves vanishing gradients: keeps signal strong
• Enables deep networks: 100+ layers possible
```

### Softmax: The Probability Converter

Softmax transforms any numbers into valid probabilities:

```
Raw Scores → Softmax → Probabilities
[2.0, 1.0, 0.1] → [0.66, 0.24, 0.10]
                   ↑    ↑    ↑
                   Sum = 1.0 ✓
                   All ≥ 0   ✓
                   Larger in → Larger out ✓

Formula: softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

Use Case: Classification ("What percentage dog vs cat?")
```
"""

# %% [markdown]
"""
## Part 1: ReLU - The Foundation of Modern Deep Learning

ReLU transformed deep learning from a curiosity to the technology powering modern AI. Before ReLU, deep networks suffered from vanishing gradients and couldn't learn effectively beyond a few layers. ReLU's simple yet brilliant design solved this problem.

### ReLU in Action: Element-wise Processing

```
Input Tensor:           After ReLU:
┌─────────────────┐    ┌─────────────────┐
│ -2.1   0.5   3.2│    │  0.0   0.5   3.2│
│  1.7  -0.8   2.1│ →  │  1.7   0.0   2.1│
│ -1.0   4.0  -0.3│    │  0.0   4.0   0.0│
└─────────────────┘    └─────────────────┘
      ↓                      ↓
Negative → 0            Positive → unchanged
```

### The Dead Neuron Problem

```
ReLU can "kill" neurons permanently:

Neuron with weights that produce only negative outputs:
Input: [1, 2, 3] → Linear: weights*input = -5.2 → ReLU: 0
Input: [4, 1, 2] → Linear: weights*input = -2.8 → ReLU: 0
Input: [0, 5, 1] → Linear: weights*input = -1.1 → ReLU: 0

Result: Neuron outputs 0 forever (no learning signal)
This is why proper weight initialization matters!
```

### Why ReLU Works Better Than Alternatives

```
Sigmoid: f(x) = 1/(1 + e^(-x))
Problem: Gradients vanish for |x| > 3

Tanh: f(x) = tanh(x)
Problem: Gradients vanish for |x| > 2

ReLU: f(x) = max(0, x)
Solution: Gradient is exactly 1 for x > 0 (no vanishing!)
```

Now let's implement this game-changing function:
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

**What we're testing**: ReLU's core behavior - zero negatives, preserve positives
**Why it matters**: ReLU must work perfectly for neural networks to learn
**Expected**: All negative values become 0, positive values unchanged

### ReLU Test Cases Visualization

```
Test Case 1 - Basic Functionality:
Input:  [-2, -1,  0,  1,  2]
Output: [ 0,  0,  0,  1,  2]
         ↑   ↑   ↑   ↑   ↑
         ✓   ✓   ✓   ✓   ✓
      (all negatives → 0, positives preserved)

Test Case 2 - Matrix Processing:
Input:  [[-1.5,  2.3],    Output: [[0.0, 2.3],
         [ 0.0, -3.7]]             [0.0, 0.0]]

Test Case 3 - Edge Cases:
• Very large positive: 1e6 → 1e6 (no overflow)
• Very small negative: -1e-6 → 0 (proper handling)
• Zero exactly: 0.0 → 0.0 (boundary condition)
```
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

Softmax is the bridge between raw neural network outputs and human-interpretable probabilities. It takes any vector of real numbers and transforms it into a valid probability distribution where all values sum to 1.0.

### The Probability Transformation Process

```
Step 1: Raw Neural Network Outputs (can be any values)
Raw scores: [2.0, 1.0, 0.1]

Step 2: Exponentiation (makes everything positive)
exp([2.0, 1.0, 0.1]) = [7.39, 2.72, 1.10]

Step 3: Normalization (makes sum = 1.0)
[7.39, 2.72, 1.10] / (7.39+2.72+1.10) = [0.66, 0.24, 0.10]
                     ↑                      ↑     ↑     ↑
                   Sum: 11.21              Total: 1.00 ✓
```

### Softmax in Classification

```
Neural Network for Image Classification:
                    Raw Scores      Softmax      Interpretation
Input: Dog Image → [2.1, 0.3, -0.8] → [0.75, 0.18, 0.07] → 75% Dog
                    ↑    ↑     ↑        ↑     ↑     ↑         18% Cat
                   Dog  Cat   Bird     Dog   Cat   Bird       7% Bird

Key Properties:
• Larger inputs get exponentially larger probabilities
• Never produces negative probabilities
• Always sums to exactly 1.0
• Differentiable (can backpropagate gradients)
```

### The Numerical Stability Problem

```
Raw Softmax Formula: softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

Problem with large numbers:
Input: [1000, 999, 998]
exp([1000, 999, 998]) = [∞, ∞, ∞]  ← Overflow!

Solution - Subtract max before exp:
x_stable = x - max(x)
Input: [1000, 999, 998] - 1000 = [0, -1, -2]
exp([0, -1, -2]) = [1.00, 0.37, 0.14] ← Stable!
```

Now let's implement this essential function:
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

**What we're testing**: Softmax probability distribution properties
**Why it matters**: Softmax must create valid probabilities for classification
**Expected**: All outputs ≥ 0, sum to 1.0, numerically stable with large inputs

### Softmax Test Cases Visualization

```
Test Case 1 - Basic Probability Distribution:
Input:  [1.0, 2.0, 3.0]
Output: [0.09, 0.24, 0.67]  ← Sum = 1.00 ✓, All ≥ 0 ✓
         ↑     ↑     ↑
      e^1/Σ e^2/Σ e^3/Σ    (largest input gets largest probability)

Test Case 2 - Numerical Stability:
Input:  [1000, 999, 998]     ← Would cause overflow without stability trick
Output: [0.67, 0.24, 0.09]   ← Still produces valid probabilities!

Test Case 3 - Edge Cases:
• All equal inputs: [1, 1, 1] → [0.33, 0.33, 0.33] (uniform distribution)
• One dominant: [10, 0, 0] → [≈1.0, ≈0.0, ≈0.0] (winner-take-all)
• Negative inputs: [-1, -2, -3] → [0.67, 0.24, 0.09] (still works!)

Test Case 4 - Batch Processing:
Input Matrix:  [[1, 2, 3],     Output Matrix: [[0.09, 0.24, 0.67],
                [4, 5, 6]]  →                  [0.09, 0.24, 0.67]]
                ↑                               ↑
            Each row processed independently   Each row sums to 1.0
```
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