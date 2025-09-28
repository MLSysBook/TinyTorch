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

Welcome to Activations! You'll implement the functions that break linearity and enable neural networks to learn complex patterns.

## 🔗 Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): N-dimensional arrays with broadcasting

**The Problem**: Your current tensors only support linear operations. Multiple linear layers stacked together create... more linear operations. This means your "deep" network can only learn patterns that a single linear layer could learn - essentially expensive linear regression.

**This Module's Solution**: Implement ReLU and Softmax activation functions that inject nonlinearity between layers, enabling your networks to learn complex patterns like image recognition and natural language understanding.

**Connection Map**:
```
Tensor → Activations → Neural Networks
(data)    (intelligence)  (complex learning)
```

## Learning Goals
- Systems understanding: How activation choice affects memory, computation, and hardware utilization
- Core implementation skill: Build production-grade activation functions with proper error handling
- Pattern/abstraction mastery: Understand the computational trade-offs between different activation types
- Framework connections: Your implementations mirror PyTorch's core activation functions
- Optimization trade-offs: Experience memory bottlenecks and discover why ReLU dominates modern architectures

## Build → Use → Reflect
1. **Build**: ReLU and Softmax with validation, error handling, and systems analysis
"""
# 2. **Use**: Test in realistic neural network pipelines with edge cases
# 3. **Reflect**: Connect your implementation measurements to production ML systems design

# ## Systems Reality Check
# 💡 **Production Context**: Your ReLU implementation uses the same algorithm as PyTorch's CUDA kernels
# ⚡ **Performance Insight**: You'll experience firsthand why ReLU's computational simplicity revolutionized deep learning

# In[ ]:

#| default_exp core.activations

#| export
import math
import numpy as np
import os
import sys
from typing import Union, List

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
## Visual Guide: Understanding Activation Functions Through Diagrams

### Why Nonlinearity Matters: A Visual Journey

```
Linear vs Nonlinear Decision Boundaries:

Linear (WITHOUT Activations):     Nonlinear (WITH Activations):

  Class A  │  Class B                Class A ╭─╮  Class B
          │                                  │ │
          │                                  │ ╰─╮ Class A
          │                                  │   │
          │                                  ╰───╯
   ───────┼───────                    ─────────────────
          │                              Complex boundary
   Simple line boundary                  enabled by ReLU!

Key Insight: Linear combinations of linear functions = still linear
            Activation functions break linearity → enable complex patterns
```

### ReLU: The Breakthrough That Enabled Deep Learning

```
ReLU Function Visualization:

        Output
          ▲
       2  │     ╱
          │    ╱
       1  │   ╱
          │  ╱
        0 │ ╱─────────►  Input
          │╱ -2 -1  1  2
         ╱│
        ╱ │
       ╱  │

Mathematical: f(x) = max(0, x)

Why Revolutionary:
┌─────────────────┬────────────────┬─────────────────┐
│   Old Problem   │   ReLU Solves  │  ML Impact      │
├─────────────────┼────────────────┼─────────────────┤
│ Vanishing Grads │ ∂f/∂x = 1 or 0 │ Deep networks   │
"""
# │ Slow computation│ Just max(0,x)  │ 6x training     │
# │ Complex math    │ Simple compare  │ Hardware-friendly│
# │ Always active   │ 50% sparse     │ Efficient memory│
# └─────────────────┴────────────────┴─────────────────┘
# ```

# ### Softmax: Converting Scores to Probabilities
# 
# ```
# Softmax Transformation:
# 
# Raw Logits:         Softmax Probabilities:
# [2.0, 1.0, 0.1] ──> [0.67, 0.24, 0.09]
#                         ↓
#                    Sum = 1.0 ✓
#                    All ≥ 0   ✓
#                    Proper probability!
# 
# Attention Mechanism Pattern:
# 
# Query-Key Similarities:  Attention Weights:
# [0.8, 1.2, 0.4, 0.9] ──> [0.19, 0.42, 0.12, 0.27]
#                              ↓
#                         Weighted sum of values
#                         Focus on important parts!
# 
# Why Essential:
# • Classification: Convert network outputs to class probabilities
# • Attention: Focus mechanism in transformers
# • Sampling: Probability-based token generation
# • Interpretability: Understand model confidence
# ```

# ### Computational Complexity: Why ReLU Dominates
# 
# ```
# Performance Analysis (per element):
# 
# ReLU:        Softmax:
# ┌─────────┐  ┌──────────────────────────────┐
# │ Compare │  │ 1. Subtract max (stability)  │
# │   +     │  │ 2. Exponential computation   │
# │ Select  │  │ 3. Sum all exponentials      │
# └─────────┘  │ 4. Divide each by sum        │
#              └──────────────────────────────┘
# 
# 2 operations   vs   4N + 3 operations (N = vector size)
# 
# GPU Parallelization:
# ReLU:    [Perfect] Each element independent
# Softmax: [Good]   Element-wise ops + reduction step
# 
# Memory Pattern:
# ReLU:    [Optimal] Can compute in-place
# Softmax: [Good]    Needs temporary storage for stability
# ```

# %% nbgrader={"grade": false, "grade_id": "relu-class", "solution": true}

# ## Part 1: ReLU - The Foundation of Modern Deep Learning

# ReLU (Rectified Linear Unit) revolutionized deep learning by solving the vanishing gradient problem while being computationally trivial.

#| export
class ReLU:
    """
    ReLU Activation Function: f(x) = max(0, x)
    
    The most important activation function in modern deep learning.
    Computationally efficient and enables training very deep networks.
    
    Key Properties:
    - Zero for negative inputs, identity for positive inputs
    - Gradient is 1 (positive) or 0 (negative) - prevents vanishing gradients
    - Computationally trivial: just comparison and selection
    - Creates sparse representations (many zeros)
    """
    
    def forward(self, x):
        """
        Apply ReLU activation: f(x) = max(0, x)

        The core function that enabled the deep learning revolution.
        Simple yet powerful - sets negative values to zero, preserves positive values.

        Args:
            x (Tensor): Input tensor of any shape

        Returns:
            Tensor: Output tensor with ReLU applied element-wise

        Raises:
            TypeError: If input is not a Tensor
            ValueError: If tensor contains NaN or infinite values

        Mathematical Foundation:
            f(x) = max(0, x) = { x if x > 0
                               { 0 if x ≤ 0

            Gradient: f'(x) = { 1 if x > 0
                               { 0 if x ≤ 0

        Implementation Approach:
        1. Validate input tensor and check for edge cases
        2. Use numpy's maximum function for vectorized operation
        3. Return new tensor with results

        Example:
            >>> relu = ReLU()
            >>> x = Tensor([[-1.0, 0.0, 1.0, 2.0]])
            >>> y = relu.forward(x)
            >>> print(y.data)  # [[0.0, 0.0, 1.0, 2.0]]
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
        Apply ReLU activation in-place for memory efficiency.

        In-place operations are crucial for large models where memory is constrained.
        Modifies the input tensor directly rather than creating a new one.

        Args:
            x (Tensor): Input tensor to modify in-place

        Returns:
            Tensor: The same tensor object (modified)

        Raises:
            TypeError: If input is not a Tensor
            ValueError: If tensor contains NaN or infinite values

        Memory Benefits:
        - No additional memory allocation
        - Critical for training very large networks
        - Used in production systems for memory efficiency

        Trade-offs:
        - Destroys original input (can't backpropagate through)
        - Used in inference or when gradients aren't needed

        Example:
            >>> relu = ReLU()
            >>> x = Tensor([[-1.0, 2.0]])
            >>> relu.forward_(x)  # Modifies x directly
            >>> print(x.data)    # [[0.0, 2.0]]
        """
        ### BEGIN SOLUTION
        # Input validation
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)}")

        # Check for empty tensor
        if x.data.size == 0:
            return x

        # Check for NaN or infinite values
        if np.any(np.isnan(x.data)) or np.any(np.isinf(x.data)):
            raise ValueError("Input tensor contains NaN or infinite values")

        # In-place operation: modify tensor data directly
        # 'out' parameter writes result back to input array
        np.maximum(0, x.data, out=x.data)
        return x
        ### END SOLUTION
    
    def __call__(self, x):
        """Make ReLU callable: relu(x) instead of relu.forward(x)"""
        return self.forward(x)

# ✅ IMPLEMENTATION CHECKPOINT: ReLU class complete

# 🤔 PREDICTION: How much faster is ReLU compared to traditional activations like sigmoid?
# Your guess: ___x faster

# 🔍 SYSTEMS INSIGHT #1: Memory Bottleneck Experience - Hit Your Hardware Limits!
def experience_memory_bottleneck():
    """Experience memory limitations firsthand - see when your system breaks!"""
    try:
        import psutil
        import time

        print("Memory Bottleneck Experience:")
        print("=" * 50)

        # Get system memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print(f"Available system memory: {available_gb:.1f} GB")

        # Start with manageable size and scale up
        print(f"\nTesting activation memory usage (increasing size until bottleneck):")

        relu = ReLU()
        sizes = []
        times = []
        memories = []

        # Scale up until we hit memory pressure
        base_size = 1_000_000  # 1M elements
        multiplier = 1

        while True:
            size = base_size * multiplier
            memory_needed_gb = size * 4 * 3 / (1024**3)  # 3x memory for processing

            if memory_needed_gb > available_gb * 0.7:  # Don't crash the system
                print(f"\n⚠️ Stopping at {size:,} elements ({memory_needed_gb:.1f} GB needed)")
                print(f"This would exceed 70% of available memory ({available_gb:.1f} GB)")
                break

            # Create large tensor and measure processing
            print(f"Size {size:,}: ", end="")

            try:
                start_time = time.perf_counter()
                large_tensor = Tensor(np.random.randn(size) - 0.5)
                memory_before = psutil.virtual_memory().used

                # Apply ReLU
                result = relu(large_tensor)

                memory_after = psutil.virtual_memory().used
                elapsed = time.perf_counter() - start_time
                memory_used = (memory_after - memory_before) / (1024**2)  # MB

                sizes.append(size)
                times.append(elapsed)
                memories.append(memory_used)

                print(f"{elapsed:.3f}s, {memory_used:.1f}MB")

                # Clean up
                del large_tensor, result

                multiplier *= 2
                if multiplier > 32:  # Safety limit
                    break

            except MemoryError:
                print("💥 Memory limit reached!")
                break

        if len(times) >= 2:
            print(f"\n💡 Key Insights:")
            print(f"• Largest tensor processed: {sizes[-1]:,} elements")
            print(f"• Memory scaling: ~{memories[-1]/memories[0]:.1f}x for {sizes[-1]/sizes[0]:.0f}x data")
            print(f"• This experience shows real hardware constraints in ML!")
            print(f"• In production, this is why gradient checkpointing exists")

    except Exception as e:
        print(f"⚠️ Error in memory experience: {e}")
        print("Note: This exercise shows real system limits!")

# Run the memory bottleneck experience
experience_memory_bottleneck()

# 🔍 SYSTEMS INSIGHT #2: Why ReLU Revolutionized Deep Learning
def analyze_relu_performance():
    """Measure why ReLU became the dominant activation function."""
    try:
        import time
        
        # Create test data (simulating a large neural network layer)
        size = 1_000_000
        test_data = np.random.randn(size) - 0.5  # Mix of positive/negative
        
        # Test ReLU performance
        start = time.perf_counter()
        relu_result = np.maximum(0, test_data)
        relu_time = time.perf_counter() - start
        
        # Compare with sigmoid (traditional activation)
        start = time.perf_counter()
        sigmoid_result = 1 / (1 + np.exp(-test_data))
        sigmoid_time = time.perf_counter() - start
        
        # Compare with tanh (another traditional activation)
        start = time.perf_counter()
        tanh_result = np.tanh(test_data)
        tanh_time = time.perf_counter() - start
        
        print(f"Performance Comparison ({size:,} elements):")
        print(f"ReLU:    {relu_time:.4f}s")
        print(f"Sigmoid: {sigmoid_time:.4f}s ({sigmoid_time/relu_time:.1f}x slower)")
        print(f"Tanh:    {tanh_time:.4f}s ({tanh_time/relu_time:.1f}x slower)")
        
        # Memory analysis
        print(f"\nMemory Usage Analysis:")
        print(f"ReLU sparsity: {np.mean(relu_result == 0):.1%} zeros")
        print(f"Memory savings from sparsity: ~{np.mean(relu_result == 0)*100:.0f}%")
        
        # Gradient analysis
        relu_grad = (test_data > 0).astype(float)
        sigmoid_grad = sigmoid_result * (1 - sigmoid_result)
        
        print(f"\nGradient Health Analysis:")
        print(f"ReLU: {np.mean(relu_grad == 1):.1%} active gradients (1.0)")
        print(f"Sigmoid: max gradient = {np.max(sigmoid_grad):.3f} (vanishing!)")
        
        # 💡 WHY THIS MATTERS: ReLU's simplicity and gradient properties
        # enabled training of very deep networks (100+ layers)
        print(f"\n💡 Key Insights:")
        print(f"• ReLU is {sigmoid_time/relu_time:.0f}x faster than sigmoid")
        print(f"• {np.mean(relu_result == 0):.0%} sparsity saves memory and computation")
        print(f"• Gradients are 1.0 (not vanishing) for {np.mean(relu_grad == 1):.0%} of neurons")
        print(f"• This enabled the deep learning revolution!")
        
    except Exception as e:
        print(f"⚠️ Error in ReLU analysis: {e}")
        print("Make sure ReLU implementation is complete")

# Run the analysis
analyze_relu_performance()

# ## Testing ReLU Implementation

# ### 🧪 Unit Test: ReLU Activation
# This test validates our ReLU implementation with various input scenarios

def test_unit_relu_activation():
    """
    Test ReLU activation function comprehensively.
    
    Validates that ReLU correctly:
    1. Zeros out negative values
    2. Preserves positive values
    3. Handles zero correctly
    4. Works with multi-dimensional tensors
    5. Provides in-place operation
    """
    print("🔬 Unit Test: ReLU Activation...")
    
    # Create ReLU instance
    relu = ReLU()
    
    # Test 1: Basic functionality with mixed values
    test_input = Tensor([[-2, -1, 0, 1, 2]])
    result = relu(test_input)
    expected = np.array([[0, 0, 0, 1, 2]])
    
    assert np.array_equal(result.data, expected), f"ReLU failed: expected {expected}, got {result.data}"
    
    # Test 2: All negative values should become zero
    negative_input = Tensor([[-5, -3, -1]])
    negative_result = relu(negative_input)
    expected_negative = np.array([[0, 0, 0]])
    
    assert np.array_equal(negative_result.data, expected_negative), "ReLU should zero out all negative values"
    
    # Test 3: All positive values should be unchanged
    positive_input = Tensor([[1, 3, 5]])
    positive_result = relu(positive_input)
    
    assert np.array_equal(positive_result.data, positive_input.data), "ReLU should preserve positive values"
    
    # Test 4: 2D tensor processing
    matrix_input = Tensor([[-1, 2], [3, -4]])
    matrix_result = relu(matrix_input)
    expected_matrix = np.array([[0, 2], [3, 0]])
    
    assert np.array_equal(matrix_result.data, expected_matrix), "ReLU should work with 2D tensors"
    assert matrix_result.shape == matrix_input.shape, "ReLU should preserve tensor shape"
    
    # Test 5: In-place operation
    inplace_input = Tensor([[-1, 0, 1]])
    original_data = inplace_input.data.copy()
    relu.forward_(inplace_input)
    expected_inplace = np.array([[0, 0, 1]])
    
    assert np.array_equal(inplace_input.data, expected_inplace), "In-place ReLU should modify original tensor"
    
    print("✅ ReLU activation tests passed!")
    print(f"✅ Correctly zeros out negative values")
    print(f"✅ Preserves positive values unchanged")
    print(f"✅ Shape preservation working correctly")
    print(f"✅ In-place operation functioning properly")

# Test immediately after implementation
test_unit_relu_activation()

# %% nbgrader={"grade": false, "grade_id": "softmax-class", "solution": true}

# ## Part 2: Softmax - Converting Scores to Probabilities

# Softmax transforms any real-valued vector into a probability distribution.
# Essential for classification and attention mechanisms.

#| export
class Softmax:
    """
    Softmax Activation Function: f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Converts any real-valued vector into a valid probability distribution.
    Essential for classification outputs and attention mechanisms.
    
    Key Properties:
    - Outputs sum to 1.0 (probability constraint)
    - All outputs are positive (probability constraint)
    - Differentiable (enables gradient-based learning)
    - Emphasizes largest inputs (winner-take-more effect)
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

        Converts logits (raw scores) into probabilities using the softmax function.
        Includes numerical stability measures to prevent overflow.

        Args:
            x (Tensor): Input tensor containing logits/scores

        Returns:
            Tensor: Output tensor with probabilities (sums to 1 along specified dim)

        Raises:
            TypeError: If input is not a Tensor
            ValueError: If tensor contains NaN values or is empty

        Mathematical Foundation:
            Standard: f(x_i) = e^(x_i) / Σ(e^(x_j))
            Stable:   f(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x)))

            The stable version prevents overflow by subtracting the maximum value.

        Implementation Approach:
        1. Validate input and handle edge cases
        2. Find maximum value along specified dimension (for stability)
        3. Subtract max from all values (prevents exp overflow)
        4. Compute exponentials and normalize

        Example:
            >>> softmax = Softmax()
            >>> x = Tensor([[1.0, 2.0, 3.0]])
            >>> y = softmax.forward(x)
            >>> print(y.data)  # [[0.09, 0.24, 0.67]] (approximately)
            >>> print(np.sum(y.data))  # 1.0 (probability distribution)
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

# 🤔 PREDICTION: How does softmax computational cost scale with vector size?
# O(N)? O(N²)? O(N log N)? Your answer: _______

# 🔍 SYSTEMS INSIGHT #3: Softmax Computational Complexity and Numerical Stability
def analyze_softmax_complexity():
    """Analyze Softmax performance characteristics and numerical stability."""
    try:
        import time
        
        print("Softmax Scaling Analysis:")
        print("=" * 50)
        
        sizes = [100, 1000, 10000, 100000]
        times = []
        
        for size in sizes:
            # Create test data with large values (numerical challenge)
            test_data = np.random.randn(size) * 10 + 50  # Large values
            
            # Measure softmax computation time
            start = time.perf_counter()
            
            # Numerically stable softmax
            max_val = np.max(test_data)
            shifted = test_data - max_val
            exp_vals = np.exp(shifted)
            result = exp_vals / np.sum(exp_vals)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            print(f"Size {size:6,}: {elapsed*1000:.2f}ms")
        
        # Analyze scaling behavior
        print(f"\nScaling Analysis:")
        if len(times) >= 2:
            scale_factor = times[-1] / times[0]
            size_factor = sizes[-1] / sizes[0]
            complexity_order = np.log(scale_factor) / np.log(size_factor)
            print(f"Time scaling: ~O(N^{complexity_order:.1f})")
        
        # Test numerical stability
        print(f"\nNumerical Stability Test:")
        
        # Without stability (would overflow)
        large_vals = np.array([1000.0, 1001.0, 1002.0])
        try:
            # This would overflow without stability measures
            raw_exp = np.exp(large_vals)
            if np.any(np.isinf(raw_exp)):
                print("❌ Raw exponentials overflow to infinity")
            else:
                print("✅ Raw exponentials computed successfully")
        except:
            print("❌ Raw exponentials failed completely")
        
        # With stability
        max_val = np.max(large_vals)
        stable_vals = large_vals - max_val  # [-2, -1, 0]
        stable_exp = np.exp(stable_vals)
        stable_softmax = stable_exp / np.sum(stable_exp)
        
        print(f"✅ Stable softmax: {stable_softmax}")
        print(f"✅ Sum check: {np.sum(stable_softmax):.6f} (should be 1.0)")
        
        # Memory analysis
        print(f"\nMemory Usage Pattern:")
        print(f"Input tensor:      {size * 4 / 1024:.1f} KB (float32)")
        print(f"Intermediate max:  {4 / 1024:.3f} KB")
        print(f"Shifted values:    {size * 4 / 1024:.1f} KB")
        print(f"Exponentials:      {size * 4 / 1024:.1f} KB")
        print(f"Sum result:        {4 / 1024:.3f} KB")
        print(f"Total peak memory: {size * 12 / 1024:.1f} KB (~3x input)")
        
        # 💡 WHY THIS MATTERS: Softmax is computationally expensive
        # but essential for interpretable probability outputs
        print(f"\n💡 Key Insights:")
        print(f"• Softmax is O(N) but with high constant factors")
        print(f"• Requires careful numerical implementation")
        print(f"• Uses ~3x memory during computation")
        print(f"• Critical for classification and attention mechanisms")
        
    except Exception as e:
        print(f"⚠️ Error in Softmax analysis: {e}")
        print("Make sure Softmax implementation is complete")

# Run the analysis
analyze_softmax_complexity()

# ## Testing Softmax Implementation

# ### 🧪 Unit Test: Softmax Activation
# This test validates our Softmax implementation for correctness and numerical stability

def test_unit_softmax_activation():
    """
    Test Softmax activation function comprehensively.
    
    Validates that Softmax correctly:
    1. Creates valid probability distributions (sum to 1)
    2. Produces only non-negative outputs
    3. Handles numerical stability with large inputs
    4. Works with multi-dimensional tensors
    5. Preserves tensor shapes
    """
    print("🔬 Unit Test: Softmax Activation...")
    
    # Create Softmax instance
    softmax = Softmax()
    
    # Test 1: Basic probability distribution properties
    test_input = Tensor([[1.0, 2.0, 3.0]])
    result = softmax(test_input)
    
    # Check that outputs sum to 1 (probability distribution)
    sum_result = np.sum(result.data, axis=-1)
    assert np.allclose(sum_result, 1.0), f"Softmax should sum to 1, got {sum_result}"
    
    # Check that all values are positive
    assert np.all(result.data >= 0), "Softmax outputs should be non-negative"
    
    # Test 2: Uniform input should give uniform distribution
    zero_input = Tensor([[0.0, 0.0, 0.0]])
    zero_result = softmax(zero_input)
    expected_uniform = np.array([[1/3, 1/3, 1/3]])
    
    assert np.allclose(zero_result.data, expected_uniform, atol=1e-6), "Equal inputs should give uniform distribution"
    
    # Test 3: Numerical stability with large values
    large_input = Tensor([[1000.0, 1001.0, 1002.0]])
    large_result = softmax(large_input)
    
    # Should not produce NaN or Inf
    assert not np.any(np.isnan(large_result.data)), "Softmax should handle large values without NaN"
    assert not np.any(np.isinf(large_result.data)), "Softmax should handle large values without Inf"
    assert np.allclose(np.sum(large_result.data, axis=-1), 1.0), "Large value softmax should still sum to 1"
    
    # Test 4: Batch processing (2D tensor)
    batch_input = Tensor([[1.0, 2.0], [3.0, 4.0]])
    batch_result = softmax(batch_input)
    
    # Each row should sum to 1
    row_sums = np.sum(batch_result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "Each batch item should sum to 1"
    
    # Test 5: Shape preservation
    assert batch_result.shape == batch_input.shape, "Softmax should preserve tensor shape"
    
    # Test 6: Order preservation (larger input → larger probability)
    ordered_input = Tensor([[1.0, 2.0, 3.0]])
    ordered_result = softmax(ordered_input)
    
    # Should maintain order: prob(3.0) > prob(2.0) > prob(1.0)
    probs = ordered_result.data[0]
    assert probs[2] > probs[1] > probs[0], "Softmax should preserve input ordering"
    
    print("✅ Softmax activation tests passed!")
    print(f"✅ Outputs form valid probability distributions")
    print(f"✅ All outputs are non-negative")
    print(f"✅ Numerically stable with large inputs")
    print(f"✅ Batch processing works correctly")
    print(f"✅ Shape preservation functioning")
    print(f"✅ Order preservation maintained")

# Test immediately after implementation
test_unit_softmax_activation()

# ✅ IMPLEMENTATION CHECKPOINT: Both ReLU and Softmax complete

# 🤔 PREDICTION: Which activation uses more memory during computation?
# ReLU or Softmax? Why? Your answer: _______

# 🔍 SYSTEMS INSIGHT #4: Activation Function Memory Comparison
def analyze_activation_memory():
    """Compare memory usage patterns between ReLU and Softmax."""
    try:
        import sys
        
        print("Activation Function Memory Analysis:")
        print("=" * 50)
        
        # Test with different tensor sizes
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            print(f"\nTensor size: {size:,} elements")
            
            # Create test data
            test_data = np.random.randn(size)
            base_memory = test_data.nbytes
            
            print(f"Input memory: {base_memory / 1024:.1f} KB")
            
            # ReLU memory analysis
            relu_result = np.maximum(0, test_data)
            relu_memory = relu_result.nbytes
            relu_total = base_memory + relu_memory
            
            print(f"ReLU:")
            print(f"  Output memory: {relu_memory / 1024:.1f} KB")
            print(f"  Total memory:  {relu_total / 1024:.1f} KB ({relu_total / base_memory:.1f}x input)")
            
            # Softmax memory analysis (tracking peak usage)
            max_val = np.max(test_data)  # Scalar: 8 bytes
            shifted = test_data - max_val  # Same size as input
            exp_vals = np.exp(shifted)     # Same size as input
            sum_exp = np.sum(exp_vals)     # Scalar: 8 bytes
            softmax_result = exp_vals / sum_exp  # Reuses exp_vals memory
            
            # Peak memory: input + shifted + exp_vals
            softmax_peak = base_memory + test_data.nbytes + exp_vals.nbytes
            softmax_final = base_memory + softmax_result.nbytes
            
            print(f"Softmax:")
            print(f"  Peak memory:   {softmax_peak / 1024:.1f} KB ({softmax_peak / base_memory:.1f}x input)")
            print(f"  Final memory:  {softmax_final / 1024:.1f} KB ({softmax_final / base_memory:.1f}x input)")
            
            # In-place potential
            print(f"In-place potential:")
            print(f"  ReLU: ✅ Can modify input directly")
            print(f"  Softmax: ❌ Needs intermediate storage")
        
        # Real-world scenario analysis
        print(f"\nReal-world Impact Example:")
        print(f"Large language model layer (2048 hidden units, batch size 32):")
        
        layer_size = 2048 * 32
        layer_memory = layer_size * 4  # float32
        
        print(f"Base tensor: {layer_memory / 1024 / 1024:.1f} MB")
        print(f"ReLU peak:   {layer_memory * 2 / 1024 / 1024:.1f} MB")
        print(f"Softmax peak: {layer_memory * 3 / 1024 / 1024:.1f} MB")
        
        # GPU memory impact
        gpu_memory = 24 * 1024  # 24GB GPU
        print(f"\nGPU Memory Usage (24GB total):")
        print(f"ReLU impact:   {layer_memory * 2 / 1024 / 1024 / 1024 * 100:.2f}% of GPU memory")
        print(f"Softmax impact: {layer_memory * 3 / 1024 / 1024 / 1024 * 100:.2f}% of GPU memory")
        
        # 💡 WHY THIS MATTERS: Memory usage affects model size limits
        print(f"\n💡 Key Insights:")
        print(f"• ReLU: 2x memory (can be optimized to 1x with in-place)")
        print(f"• Softmax: 3x memory peak (needs intermediate storage)")
        print(f"• ReLU enables larger models in same memory")
        print(f"• Softmax memory cost limits attention scale")
        
    except Exception as e:
        print(f"⚠️ Error in memory analysis: {e}")
        print("Make sure both activation implementations are complete")

# Run the analysis
analyze_activation_memory()

# In[ ]:

# ## Integration Testing: Activations in Neural Network Context

# Let's test these activations in realistic neural network scenarios

def test_unit_activations_comprehensive():
    """Comprehensive test of both activation functions working together."""
    print("🔬 Comprehensive Test: ReLU + Softmax Pipeline...")

    # Create activation instances
    relu = ReLU()
    softmax = Softmax()

    # Test Case 1: Hidden layer processing with ReLU
    print("\nTest 1: Hidden Layer Processing")
    # Simulate output from a linear layer (can be negative)
    hidden_output = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    hidden_activated = relu(hidden_output)
    expected_relu = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])

    assert np.array_equal(hidden_activated.data, expected_relu), "ReLU should zero negatives"
    print("✅ ReLU correctly processes hidden layer outputs")

    # Test Case 2: Classification output with Softmax
    print("\nTest 2: Classification Output")
    class_logits = Tensor([[2.0, 1.0, 0.1]])
    class_probabilities = softmax(class_logits)

    # Verify probability properties
    assert np.allclose(np.sum(class_probabilities.data, axis=-1), 1.0), "Softmax should create probability distribution"
    assert np.all(class_probabilities.data >= 0), "Probabilities should be non-negative"

    # Highest logit should get highest probability
    max_logit_idx = np.argmax(class_logits.data)
    max_prob_idx = np.argmax(class_probabilities.data)
    assert max_logit_idx == max_prob_idx, "Highest logit should get highest probability"
    print("✅ Softmax correctly creates probability distributions")

    # Test Case 3: Error handling validation
    print("\nTest 3: Error Handling")
    try:
        relu("not a tensor")
        assert False, "Should raise TypeError"
    except TypeError:
        print("✅ ReLU properly validates input type")

    try:
        nan_tensor = Tensor([np.nan, 1.0, 2.0])
        relu(nan_tensor)
        assert False, "Should raise ValueError for NaN"
    except ValueError:
        print("✅ ReLU properly handles NaN inputs")

    try:
        empty_tensor = Tensor([])
        softmax(empty_tensor)
        assert False, "Should raise ValueError for empty tensor"
    except ValueError:
        print("✅ Softmax properly handles empty tensors")

    # Test Case 4: Batch processing
    print("\nTest 4: Batch Processing")
    batch_logits = Tensor([
        [1.0, 2.0, 0.5],  # Sample 1
        [0.1, 0.2, 0.9],  # Sample 2
        [2.0, 1.0, 1.5]   # Sample 3
    ])

    batch_probs = softmax(batch_logits)

    # Each row should sum to 1
    row_sums = np.sum(batch_probs.data, axis=1)
    assert np.allclose(row_sums, [1.0, 1.0, 1.0]), "Each batch item should form probability distribution"
    print("✅ Batch processing works correctly")

    print("\n✅ Comprehensive activation tests passed!")
    print(f"✅ ReLU enables nonlinear hidden representations")
    print(f"✅ Softmax provides interpretable classification outputs")
    print(f"✅ Both functions handle edge cases and errors properly")
    print(f"✅ Both functions handle batch processing correctly")

# Test comprehensive functionality
test_unit_activations_comprehensive()

# In[ ]:

# ## Integration Test: Realistic Neural Network Pipeline

# Test activations in a complete neural network forward pass simulation

def test_module_activation_integration():
    """Enhanced integration test: activations in realistic neural network pipeline with edge cases."""
    print("🔬 Enhanced Integration Test: Neural Network Pipeline with Edge Cases...")

    # Setup test components
    relu = ReLU()
    softmax = Softmax()

    # Test Case 1: Standard forward pass
    print("\nTest 1: Standard Forward Pass")

    # Step 1: Input data (batch of 3 samples, 4 features each)
    input_data = Tensor([
        [0.5, -0.3, 1.2, -0.8],  # Sample 1
        [-1.0, 0.8, 0.0, 1.5],   # Sample 2
        [0.2, -0.5, -0.9, 0.3]   # Sample 3
    ])
    print(f"Input shape: {input_data.shape}")

    # Step 2: Simulate hidden layer output (after linear transformation)
    hidden_output = Tensor([
        [-1.5, 0.8, 2.1],   # Sample 1 hidden activations
        [0.3, -0.6, 1.2],   # Sample 2 hidden activations
        [-0.8, 1.5, -0.3]   # Sample 3 hidden activations
    ])

    # Step 3: Apply ReLU to hidden layer
    hidden_activated = relu(hidden_output)
    expected_relu = np.array([
        [0.0, 0.8, 2.1],   # Negatives zeroed
        [0.3, 0.0, 1.2],   # Negatives zeroed
        [0.0, 1.5, 0.0]    # Negatives zeroed
    ])

    assert np.allclose(hidden_activated.data, expected_relu), "ReLU should zero negatives in hidden layer"
    print("✅ Hidden layer ReLU activation successful")

    # Step 4: Apply Softmax for classification
    final_logits = Tensor([
        [2.1, 0.5, 1.2],   # Sample 1 class scores
        [0.8, 1.5, 0.3],   # Sample 2 class scores
        [1.0, 2.0, 0.1]    # Sample 3 class scores
    ])

    class_probabilities = softmax(final_logits)

    # Verify softmax properties
    batch_sums = np.sum(class_probabilities.data, axis=1)
    assert np.allclose(batch_sums, [1.0, 1.0, 1.0]), "Each sample should have probabilities summing to 1"
    print("✅ Standard forward pass successful")

    # Test Case 2: Large batch processing (realistic production scale)
    print("\nTest 2: Large Batch Processing")
    large_batch_size = 128
    num_classes = 1000

    # Create large batch of logits
    large_logits = Tensor(np.random.randn(large_batch_size, num_classes) * 10)

    # Process with softmax
    large_probs = softmax(large_logits)

    # Verify all properties hold at scale
    large_sums = np.sum(large_probs.data, axis=1)
    assert np.allclose(large_sums, 1.0), "Large batch softmax should maintain probability properties"
    assert np.all(large_probs.data >= 0), "Large batch probabilities should be non-negative"
    print(f"✅ Large batch processing ({large_batch_size} samples, {num_classes} classes) successful")

    # Test Case 3: Extreme values (numerical stability)
    print("\nTest 3: Extreme Values Handling")
    extreme_logits = Tensor([
        [1000.0, 999.0, 998.0],  # Very large values
        [-1000.0, -999.0, -998.0],  # Very small values
        [0.0, 0.0, 0.0]  # All zeros
    ])

    extreme_probs = softmax(extreme_logits)
    extreme_sums = np.sum(extreme_probs.data, axis=1)

    assert np.allclose(extreme_sums, 1.0), "Extreme values should still create valid probability distributions"
    assert not np.any(np.isnan(extreme_probs.data)), "No NaN values should be produced"
    assert not np.any(np.isinf(extreme_probs.data)), "No infinite values should be produced"
    print("✅ Extreme values handling successful")

    # Test Case 4: Memory efficiency with in-place ReLU
    print("\nTest 4: Memory Efficiency Test")
    memory_test_tensor = Tensor(np.random.randn(1000, 1000) - 0.5)  # 50% negative

    # Test in-place operation
    original_id = id(memory_test_tensor.data)
    relu.forward_(memory_test_tensor)

    assert id(memory_test_tensor.data) == original_id, "In-place operation should modify original tensor"
    assert np.all(memory_test_tensor.data >= 0), "In-place ReLU should zero all negatives"
    print("✅ Memory efficient in-place operations successful")

    print("\n✅ Enhanced integration test passed!")
    print(f"✅ Standard neural network pipeline works correctly")
    print(f"✅ Large-scale batch processing handles production workloads")
    print(f"✅ Numerical stability maintained under extreme conditions")
    print(f"✅ Memory-efficient operations function properly")
    print(f"✅ Ready for real-world deployment scenarios!")

# Test integration functionality
test_module_activation_integration()

# In[ ]:

# Main execution block - all tests run when module is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 RUNNING ALL ACTIVATION TESTS")
    print("="*60)
    
    # Run all activation tests in sequence
    test_unit_relu_activation()
    print()
    test_unit_softmax_activation()
    print()
    test_unit_activations_comprehensive()
    print()
    test_module_activation_integration()
    
    print("\n" + "="*60)
    print("🎉 ALL ACTIVATION TESTS PASSED!")
    print("="*60)
    print("✅ ReLU: The foundation of modern deep learning")
    print("✅ Softmax: The key to interpretable classifications")
    print("💡 Ready to build neural networks with essential nonlinearity!")
    
    # Display module completion message
    print(f"\n🎯 Module 03 (Activations) Complete!")
    print(f"You've implemented the activation functions that:")
    print(f"  • Enable neural networks to learn complex patterns")
    print(f"  • Provide numerical stability in production systems")
    print(f"  • Form the foundation of 90%+ of modern architectures")
    print(f"\nNext: Use these activations to build neural network layers!")

# ## 🤔 ML Systems Thinking: Interactive Questions

# Now that you've built ReLU and Softmax activation functions and analyzed their performance characteristics, let's connect this work to broader ML systems challenges.

# ### Question 1: Memory Bottleneck Analysis and Hardware Trade-offs

# **Context**: In your memory bottleneck experience, you saw how activation memory usage scales with tensor size. Your ReLU analysis showed 2x memory usage while Softmax peaked at 3x. You also measured performance differences between ReLU's simple comparison vs Softmax's exponential computations.

# **Reflection Question**: Based on your measurements, design a memory-efficient activation strategy for training a large language model with 7B parameters where GPU memory is the primary constraint. How would you modify your ReLU and Softmax implementations to reduce memory overhead? Consider when to use in-place operations, how to handle gradient computation, and specific optimizations for different parts of the network (hidden layers vs attention vs output layers).

# Think about: gradient checkpointing integration, when in-place operations break backpropagation, memory vs recomputation trade-offs, and how activation sparsity affects subsequent layer memory usage.

# *Target length: 200-400 words*

# **YOUR ANALYSIS OF MEMORY-EFFICIENT ACTIVATION STRATEGIES:**

# [Student response area - replace this text with your analysis]

# ### Question 2: Numerical Stability and Error Propagation

# **Context**: Your Softmax implementation includes numerical stability measures (subtracting max values) and error handling for NaN/infinite inputs. You tested extreme values like [1000.0, 999.0, 998.0] and verified the stability measures work correctly.

# **Reflection Question**: Analyze how numerical errors in activations propagate through deep networks during training. If small floating-point inconsistencies occur in your ReLU or Softmax implementations (due to hardware differences or precision settings), how do these errors compound across 100+ layers? Design specific error detection and mitigation strategies for your activation functions that could be integrated into a production training loop.

# Think about: error accumulation patterns, early error detection strategies, precision monitoring during training, and how to balance numerical accuracy with computational efficiency.

# *Target length: 200-400 words*

# **YOUR ANALYSIS OF NUMERICAL STABILITY AND ERROR PROPAGATION:**

# [Student response area - replace this text with your analysis]

# ### Question 3: Production Integration and Framework Evolution

# **Context**: Your implementations mirror PyTorch's ReLU and Softmax algorithms, but production frameworks add optimizations like CUDA kernels, kernel fusion, and automatic mixed precision. You've built the core algorithms that these optimizations build upon.

# **Reflection Question**: Design an evolution path for your activation implementations to support advanced production features. How would you extend your current ReLU and Softmax classes to support automatic mixed precision, gradient checkpointing, and kernel fusion? What interfaces would you add to make your implementations compatible with advanced optimizers and distributed training systems while maintaining the simplicity of your current API?

# Think about: backward compatibility, performance monitoring hooks, optimization hint interfaces, and how to abstract hardware-specific optimizations while keeping the mathematical core unchanged.

# *Target length: 200-400 words*

# **YOUR ANALYSIS OF PRODUCTION EVOLUTION STRATEGIES:**

# [Student response area - replace this text with your analysis]

# ## 🎯 MODULE SUMMARY: Essential Activations

# Congratulations! You've successfully implemented the two most crucial activation functions in modern deep learning:

# ### What You've Accomplished
# ✅ **ReLU Implementation**: 25+ lines of the activation that revolutionized deep learning
# ✅ **Softmax Implementation**: 30+ lines of numerically stable probability distribution creation  
# ✅ **Performance Analysis**: Comprehensive benchmarking revealing why ReLU dominates hidden layers
# ✅ **Memory Profiling**: Discovered that Softmax uses 3x peak memory vs ReLU's 2x
# ✅ **Integration Testing**: Validated activations work in realistic neural network pipelines

# ### Key Learning Outcomes
# - **Nonlinearity Mastery**: Understanding how activation functions enable neural networks to learn complex patterns
# - **Numerical Stability**: Implementing mathematically correct algorithms that handle edge cases
# - **Performance Awareness**: Connecting computational complexity to hardware capabilities and architecture choices
# - **Systems Integration**: Building components that work seamlessly in larger neural network systems

# ### Mathematical Foundations Mastered
# - **ReLU Mathematics**: f(x) = max(0, x) and its gradient properties that solved vanishing gradients
# - **Softmax Mathematics**: f(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x))) with numerical stability
# - **Probability Theory**: Converting arbitrary scores to valid probability distributions
# - **Computational Complexity**: O(N) operations with different constant factors and memory patterns

# ### Professional Skills Developed
# - **Numerical Programming**: Implementing mathematically stable algorithms for production use
# - **Performance Analysis**: Measuring and understanding computational bottlenecks in ML systems
# - **Systems Design**: Considering memory usage, hardware constraints, and scalability in implementation choices
# - **Integration Testing**: Validating components work correctly in realistic system contexts

# ### Ready for Advanced Applications
# Your activation implementations now enable:
# - **Neural Network Layers**: Combining linear transformations with nonlinear activations
# - **Deep Architectures**: Using ReLU to train networks with 100+ layers without vanishing gradients
# - **Classification Systems**: Converting network outputs to interpretable probability distributions
# - **Attention Mechanisms**: Using Softmax for attention weight computation in transformers

# ### Connection to Real ML Systems
# Your implementations mirror production systems:
# - **PyTorch**: `torch.nn.ReLU()` and `torch.nn.Softmax(dim=-1)` implement identical mathematics with hardware optimizations
# - **TensorFlow**: `tf.nn.relu()` and `tf.nn.softmax()` follow the same algorithmic approaches with CUDA acceleration
# - **Hardware Acceleration**: Modern GPUs have specialized tensor cores optimized for these exact operations
# - **Industry Standard**: Every major ML framework prioritizes optimizing these specific activation functions

# ### Next Steps
# 1. **Export your module**: `tito module complete 02_activations`
# 2. **Validate integration**: `tito test --module activations`
# 3. **Explore activation variants**: Experiment with Leaky ReLU or GELU implementations
# 4. **Ready for Module 04**: Layers - combining your activations with linear transformations!

# **Forward Momentum**: Your activation functions provide the nonlinear intelligence that transforms simple linear operations into powerful learning systems capable of solving complex real-world problems!