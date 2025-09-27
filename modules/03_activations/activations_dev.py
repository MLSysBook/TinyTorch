#!/usr/bin/env python
# coding: utf-8

# # Activations - Making Networks Learn Efficiently

# Welcome to Activations! You'll implement the essential nonlinear functions that enable neural networks to learn complex patterns.

# ## 🔗 Building on Previous Learning
# **What You Built Before**:
# - Module 01 (Setup): Python environment and development tools
# - Module 02 (Tensor): N-dimensional arrays with shape management and broadcasting

# **What's Working**: You can create and manipulate tensors with vectorized operations and memory-efficient broadcasting!

# **The Gap**: Your tensors can only perform linear operations. Without nonlinearity, neural networks can't learn complex patterns - they're just expensive linear regression.

# **This Module's Solution**: Implement ReLU and Softmax - the activation functions that provide the nonlinear intelligence enabling deep learning breakthroughs.

# **Connection Map**:
# ```
# Tensor → Activations → Layers
# (data)    (intelligence)  (architecture)
# ```

# ## Learning Goals
# - Systems understanding: How activation choice affects memory usage, computational complexity, and training stability
# - Core implementation skill: Build numerically stable activation functions used in 90%+ of production systems
# - Pattern/abstraction mastery: Understand when to use ReLU (hidden layers) vs Softmax (classification outputs)
# - Framework connections: See how your implementations mirror PyTorch's essential activation functions
# - Optimization trade-offs: Learn why ReLU's simplicity revolutionized deep learning and why Softmax requires careful numerical implementation

# ## Build → Use → Reflect
# 1. **Build**: ReLU and Softmax activation functions with proper numerical stability and performance characteristics
# 2. **Use**: Apply these activations in realistic neural network scenarios with batch processing
# 3. **Reflect**: How do activation functions affect both learning capacity and computational efficiency in ML systems?

# ## Systems Reality Check
# 💡 **Production Context**: PyTorch implements ReLU with highly optimized CUDA kernels while Softmax requires careful numerical stability - your implementation reveals these design trade-offs
# ⚡ **Performance Insight**: ReLU is popular partly because it's computationally cheap (just max(0,x)), while Softmax requires expensive exponentials - hardware considerations drive ML algorithm choice

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
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    from tensor_dev import Tensor

# In[ ]:

print("🔥 TinyTorch Activations Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build essential activation functions!")

# ## Visual Guide: Understanding Activation Functions Through Diagrams

# ### Why Nonlinearity Matters: A Visual Journey
# 
# ```
# Linear vs Nonlinear Decision Boundaries:
# 
# Linear (WITHOUT Activations):     Nonlinear (WITH Activations):
# 
#   Class A  │  Class B                Class A ╭─╮  Class B
#           │                                  │ │
#           │                                  │ ╰─╮ Class A
#           │                                  │   │
#           │                                  ╰───╯
#    ───────┼───────                    ─────────────────
#           │                              Complex boundary
#    Simple line boundary                  enabled by ReLU!
# 
# Key Insight: Linear combinations of linear functions = still linear
#             Activation functions break linearity → enable complex patterns
# ```

# ### ReLU: The Breakthrough That Enabled Deep Learning
# 
# ```
# ReLU Function Visualization:
# 
#         Output
#           ▲
#        2  │     ╱
#           │    ╱
#        1  │   ╱
#           │  ╱
#         0 │ ╱─────────►  Input
#           │╱ -2 -1  1  2
#          ╱│
#         ╱ │
#        ╱  │
# 
# Mathematical: f(x) = max(0, x)
# 
# Why Revolutionary:
# ┌─────────────────┬────────────────┬─────────────────┐
# │   Old Problem   │   ReLU Solves  │  ML Impact      │
# ├─────────────────┼────────────────┼─────────────────┤
# │ Vanishing Grads │ ∂f/∂x = 1 or 0 │ Deep networks   │
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

# In[ ]:

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
            
        Mathematical Foundation:
            f(x) = max(0, x) = { x if x > 0
                               { 0 if x ≤ 0
            
            Gradient: f'(x) = { 1 if x > 0
                               { 0 if x ≤ 0
        
        Implementation Approach:
        1. Use numpy's maximum function for vectorized operation
        2. Compare each element with 0 simultaneously
        3. Return new tensor with results
        
        Why This Works:
        - np.maximum broadcasts 0 across entire tensor
        - Vectorized operation leverages CPU/GPU parallelism
        - Creates new tensor (preserves input for backpropagation)
        
        Example:
            >>> relu = ReLU()
            >>> x = Tensor([[-1.0, 0.0, 1.0, 2.0]])
            >>> y = relu.forward(x)
            >>> print(y.data)  # [[0.0, 0.0, 1.0, 2.0]]
        """
        ### BEGIN SOLUTION
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
            
        Memory Benefits:
        - No additional memory allocation
        - Critical for training very large networks
        - Used in production systems for memory efficiency
        
        Trade-offs:
        - Destroys original input (can't backpropagate through)
        - Used in inference or when gradients aren't needed
        - PyTorch notation: relu_() vs relu()
        
        Example:
            >>> relu = ReLU()
            >>> x = Tensor([[-1.0, 2.0]])
            >>> relu.forward_(x)  # Modifies x directly
            >>> print(x.data)    # [[0.0, 2.0]]
        """
        ### BEGIN SOLUTION
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

# 🔍 SYSTEMS INSIGHT #1: Why ReLU Revolutionized Deep Learning
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

# In[ ]:

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
            
        Mathematical Foundation:
            Standard: f(x_i) = e^(x_i) / Σ(e^(x_j))
            Stable:   f(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x)))
            
            The stable version prevents overflow by subtracting the maximum value.
        
        Implementation Approach:
        1. Find maximum value along specified dimension (for stability)
        2. Subtract max from all values (prevents exp overflow)
        3. Compute exponentials of shifted values
        4. Normalize by sum to create probability distribution
        
        Why Numerical Stability Matters:
        - e^(large_number) can overflow to infinity
        - Subtracting max shifts values to reasonable range
        - Mathematically equivalent but numerically stable
        
        Example:
            >>> softmax = Softmax()
            >>> x = Tensor([[1.0, 2.0, 3.0]])
            >>> y = softmax.forward(x)
            >>> print(y.data)  # [[0.09, 0.24, 0.67]] (approximately)
            >>> print(np.sum(y.data))  # 1.0 (probability distribution)
        """
        ### BEGIN SOLUTION
        # Step 1: Numerical stability - subtract maximum value
        # This prevents exp(large_number) from overflowing to infinity
        max_vals = np.max(x.data, axis=self.dim, keepdims=True)
        x_stable = x.data - max_vals
        
        # Step 2: Compute exponentials of stable values
        exp_vals = np.exp(x_stable)
        
        # Step 3: Normalize to create probability distribution
        sum_exp = np.sum(exp_vals, axis=self.dim, keepdims=True)
        result = exp_vals / sum_exp
        
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, x):
        """Make Softmax callable: softmax(x) instead of softmax.forward(x)"""
        return self.forward(x)

# ✅ IMPLEMENTATION CHECKPOINT: Softmax class complete

# 🤔 PREDICTION: How does softmax computational cost scale with vector size?
# O(N)? O(N²)? O(N log N)? Your answer: _______

# 🔍 SYSTEMS INSIGHT #2: Softmax Computational Complexity and Numerical Stability
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

# 🔍 SYSTEMS INSIGHT #3: Activation Function Memory Comparison
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
    
    # Test Case 3: Batch processing
    print("\nTest 3: Batch Processing")
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
    print(f"✅ Both functions handle batch processing correctly")

# Test comprehensive functionality
test_unit_activations_comprehensive()

# In[ ]:

# ## Integration Test: Realistic Neural Network Pipeline

# Test activations in a complete neural network forward pass simulation

def test_module_activation_integration():
    """Integration test: activations in a realistic neural network pipeline."""
    print("🔬 Integration Test: Neural Network Pipeline...")
    
    # Setup test components
    relu = ReLU()
    softmax = Softmax()
    
    # Simulate a complete forward pass
    
    # Step 1: Input data (batch of 3 samples, 4 features each)
    input_data = Tensor([
        [0.5, -0.3, 1.2, -0.8],  # Sample 1
        [-1.0, 0.8, 0.0, 1.5],   # Sample 2
        [0.2, -0.5, -0.9, 0.3]   # Sample 3
    ])
    print(f"Input shape: {input_data.shape}")
    
    # Step 2: Simulate hidden layer output (after linear transformation)
    # In reality, this would be: output = input @ weights + bias
    hidden_output = Tensor([
        [-1.5, 0.8, 2.1],   # Sample 1 hidden activations
        [0.3, -0.6, 1.2],   # Sample 2 hidden activations
        [-0.8, 1.5, -0.3]   # Sample 3 hidden activations
    ])
    print(f"Hidden layer output shape: {hidden_output.shape}")
    
    # Step 3: Apply ReLU to hidden layer
    hidden_activated = relu(hidden_output)
    expected_relu = np.array([
        [0.0, 0.8, 2.1],   # Negatives zeroed
        [0.3, 0.0, 1.2],   # Negatives zeroed
        [0.0, 1.5, 0.0]    # Negatives zeroed
    ])
    
    assert np.allclose(hidden_activated.data, expected_relu), "ReLU should zero negatives in hidden layer"
    print("✅ Hidden layer ReLU activation successful")
    
    # Step 4: Simulate final layer output (classification logits)
    final_logits = Tensor([
        [2.1, 0.5, 1.2],   # Sample 1 class scores
        [0.8, 1.5, 0.3],   # Sample 2 class scores
        [1.0, 2.0, 0.1]    # Sample 3 class scores
    ])
    print(f"Final logits shape: {final_logits.shape}")
    
    # Step 5: Apply Softmax for classification
    class_probabilities = softmax(final_logits)
    
    # Verify softmax properties
    batch_sums = np.sum(class_probabilities.data, axis=1)
    assert np.allclose(batch_sums, [1.0, 1.0, 1.0]), "Each sample should have probabilities summing to 1"
    
    # Verify predictions make sense (highest logit → highest probability)
    for i in range(3):
        max_logit_class = np.argmax(final_logits.data[i])
        max_prob_class = np.argmax(class_probabilities.data[i])
        assert max_logit_class == max_prob_class, f"Sample {i}: highest logit should get highest probability"
    
    print("✅ Classification Softmax successful")
    
    # Display predictions
    print(f"\n📊 Sample Predictions:")
    for i in range(3):
        probs = class_probabilities.data[i]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        print(f"   Sample {i+1}: Class {predicted_class} (confidence: {confidence:.3f})")
    
    print("\n✅ Integration test passed!")
    print(f"✅ Complete forward pass simulation successful")
    print(f"✅ ReLU enables nonlinear hidden representations")
    print(f"✅ Softmax provides interpretable classification outputs")
    print(f"✅ Pipeline handles realistic batch processing")

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

# ### Question 1: Hardware Optimization Strategy

# **Context**: Your ReLU implementation uses `np.maximum(0, x)` while Softmax requires exponentials, divisions, and reductions. In production systems, these functions are called billions of times during training and inference.

# **Reflection Question**: Design a comprehensive optimization strategy for activation functions across different hardware platforms. How would you optimize ReLU and Softmax differently for CPU vs GPU vs TPU execution? Consider the specific characteristics of each activation (ReLU's sparsity, Softmax's reduction operations) and how they interact with different memory hierarchies, instruction sets, and parallelization capabilities.

# Think about: SIMD vectorization opportunities, memory bandwidth vs compute intensity, kernel fusion strategies, precision trade-offs between training and inference, and the impact of activation sparsity on subsequent layer computations.

# *Target length: 200-400 words*

# **YOUR REFLECTION ON HARDWARE OPTIMIZATION:**

# [Student response area - replace this text with your analysis]

# ### Question 2: Numerical Stability in Distributed Systems

# **Context**: Your Softmax implementation includes numerical stability measures, but distributed training introduces additional challenges: gradient synchronization across nodes, mixed precision arithmetic, and maintaining consistency when different hardware has different floating-point behaviors.

# **Reflection Question**: Architect a robust activation system for distributed training that maintains numerical stability across heterogeneous hardware. How would you handle scenarios where ReLU creates different sparsity patterns on different nodes, or where Softmax produces slightly different probability distributions due to hardware-specific floating-point precision? Design specific mechanisms to ensure training convergence isn't affected by these numerical inconsistencies.

# Think about: gradient synchronization protocols, deterministic activation implementations, precision hierarchies for different training phases, and error accumulation across distributed components.

# *Target length: 200-400 words*

# **YOUR REFLECTION ON DISTRIBUTED NUMERICAL STABILITY:**

# [Student response area - replace this text with your analysis]

# ### Question 3: Activation Function Evolution and Framework Design

# **Context**: You implemented the current standard activations (ReLU, Softmax), but the field continues evolving (GELU, Swish, etc.). Production frameworks must support established functions optimally while enabling experimentation with new activations.

# **Reflection Question**: Design an extensible activation system that balances optimal performance for established functions with flexibility for research. How would you architect a plugin system that maintains zero-overhead for ReLU/Softmax while supporting experimental activations? Consider automatic differentiation integration, performance profiling systems, and how to handle activations that might need different optimization strategies than current ones.

# Think about: plugin architectures, performance regression detection, automatic code generation for new activations, backward compatibility guarantees, and integration with existing optimization pipelines.

# *Target length: 200-400 words*

# **YOUR REFLECTION ON EXTENSIBLE ACTIVATION SYSTEMS:**

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
# 1. **Export your module**: `tito module complete 03_activations`
# 2. **Validate integration**: `tito test --module activations`
# 3. **Explore activation variants**: Experiment with Leaky ReLU or GELU implementations
# 4. **Ready for Module 04**: Layers - combining your activations with linear transformations!

# **Forward Momentum**: Your activation functions provide the nonlinear intelligence that transforms simple linear operations into powerful learning systems capable of solving complex real-world problems!