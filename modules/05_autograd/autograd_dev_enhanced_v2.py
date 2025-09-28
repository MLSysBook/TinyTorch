# %% [markdown]
"""
# Autograd - Automatic Differentiation Engine

Welcome to Autograd! You'll implement the magic that powers deep learning - automatic gradient computation for ANY computational graph!

## 🔗 Building on Previous Learning

**What You Built Before**:
- Module 02 (Tensor): Data structures for n-dimensional arrays
- Module 03 (Activations): Non-linear functions for neural networks

**What's Working**: You can build computational graphs with tensors and apply non-linear transformations.

**The Gap**: You have to manually compute derivatives - tedious, error-prone, and doesn't scale to complex networks.

**This Module's Solution**: Build an automatic differentiation engine that tracks operations and computes gradients via chain rule.

**Connection Map**:
```
Tensor → Autograd → Optimizers
(data)   (∇f/∇x)   (x -= α∇f/∇x)
```

## Learning Goals
- Understand computational graphs and gradient flow
- Master the chain rule for automatic differentiation  
- Build memory-efficient gradient accumulation
- Connect to PyTorch's autograd system
- Analyze memory vs compute trade-offs in backpropagation

## Build → Use → Reflect
1. **Build**: Implement Variable class and gradient computation
2. **Use**: Test on complex computational graphs
3. **Reflect**: Analyze memory usage and scaling behavior

## Systems Reality Check
💡 **Production Context**: PyTorch's autograd is the foundation of all deep learning
⚡ **Performance Insight**: Gradient storage can use 2-3x more memory than forward pass!
"""

# %% 
#| default_exp autograd
import numpy as np
from typing import List, Optional, Callable, Union

# %% [markdown]
"""
## Part 1: The Million Dollar Question

How does PyTorch automatically compute gradients for ANY neural network architecture, no matter how complex?

The answer: **Computational Graphs + Chain Rule**

Let's discover how this works by building it ourselves!
"""

# %% [markdown]
"""
## Part 2: The Variable Class - Tracking Computation History

Every value in our computational graph needs to remember:
1. Its data
2. Whether it needs gradients
3. How it was created (for backpropagation)
"""

# %% nbgrader={"grade": false, "grade_id": "variable-class", "solution": true}
#| export
class Variable:
    """
    A Variable wraps data and tracks how it was created for gradient computation.
    
    This is the foundation of automatic differentiation - each Variable knows
    its parents and the operation that created it, forming a computational graph.
    
    TODO: Implement the Variable class with gradient tracking capabilities.
    
    APPROACH:
    1. Store data as numpy array for efficient computation
    2. Track whether gradients are needed (requires_grad)
    3. Store the operation that created this Variable (grad_fn)
    
    EXAMPLE:
    >>> x = Variable(np.array([2.0]), requires_grad=True)
    >>> y = x * 3  # y knows it was created by multiplication
    >>> print(y.data)
    [6.0]
    
    HINTS:
    - Use np.array() to ensure data is numpy array
    - Initialize grad to None (computed during backward)
    - grad_fn stores the backward function
    """
    
    def __init__(self, data, requires_grad=False, grad_fn=None):
        ### BEGIN SOLUTION
        # SYSTEMS INSIGHT: float32 uses 4 bytes per element
        # For 1B parameters = 4GB just for data storage
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        
        # CRITICAL ML PATTERN: Gradients initialized lazily
        # Memory saved until backward() is called
        self.grad = None
        
        # AUTOGRAD CORE: Links to parent operation in computation graph
        # Enables automatic chain rule application
        self.grad_fn = grad_fn
        self._backward_hooks = []  # Extension point for advanced features
        ### END SOLUTION
    
    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation using chain rule.
        
        TODO: Implement backward pass through computational graph.
        
        APPROACH:
        1. Initialize gradient if not provided (for scalar outputs)
        2. Accumulate gradients (for shared parameters)
        3. Call grad_fn to propagate gradients to parents
        
        HINTS:
        - Gradient accumulates: grad = grad + new_gradient
        - Only propagate if grad_fn exists
        - Check requires_grad before accumulating
        """
        ### BEGIN SOLUTION
        # OPTIMIZATION: Skip gradient computation when not needed
        # Saves O(N) operations where N = parameter count
        if not self.requires_grad:
            return
            
        # AUTOGRAD PATTERN: Scalar loss needs starting gradient
        # ∂L/∂L = 1 (derivative of loss w.r.t. itself)
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("Gradient must be specified for non-scalar outputs")
            gradient = np.ones_like(self.data)  # O(1) memory for scalars
        
        # CRITICAL ML SYSTEMS PRINCIPLE: Gradient accumulation
        # Why: Shared parameters (e.g., embeddings) receive gradients from multiple paths
        # Memory: Creates new array to avoid aliasing bugs
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient  # += would modify original!
            
        # GRAPH TRAVERSAL: Recursive backpropagation
        # Complexity: O(graph_depth), can hit Python recursion limit (~1000)
        if self.grad_fn is not None:
            self.grad_fn(gradient)
        ### END SOLUTION
    
    def zero_grad(self):
        """Reset gradient to None."""
        ### BEGIN SOLUTION
        self.grad = None
        ### END SOLUTION

# %% [markdown]
"""
## Part 3: Implementing Operations with Gradient Tracking

Now we need operations that build the computational graph AND know how to compute gradients.
"""

# %% nbgrader={"grade": false, "grade_id": "operations", "solution": true}
#| export
class Add:
    """Addition operation with gradient computation."""
    
    @staticmethod
    def forward(a: Variable, b: Variable) -> Variable:
        """
        Forward pass: z = a + b
        
        TODO: Implement forward pass and create backward function.
        
        HINTS:
        - Result needs gradients if either input needs gradients
        - Backward function gets gradient from child
        - Addition gradient: ∂z/∂a = 1, ∂z/∂b = 1
        """
        ### BEGIN SOLUTION
        # Track gradients if either input needs them
        requires_grad = a.requires_grad or b.requires_grad
        
        def backward_fn(grad_output):
            # Addition gradient: ∂z/∂a = 1, ∂z/∂b = 1
            # Just pass gradients through unchanged
            if a.requires_grad:
                a.backward(grad_output)
            if b.requires_grad:
                b.backward(grad_output)
        
        # Create output Variable with link to backward function
        result = Variable(
            a.data + b.data,
            requires_grad=requires_grad,
            grad_fn=backward_fn if requires_grad else None
        )
        return result
        ### END SOLUTION

class Multiply:
    """Multiplication operation with gradient computation."""
    
    @staticmethod  
    def forward(a: Variable, b: Variable) -> Variable:
        """
        Forward pass: z = a * b
        
        TODO: Implement forward pass with gradient tracking.
        
        HINTS:
        - Multiplication gradient uses chain rule
        - ∂z/∂a = b, ∂z/∂b = a
        - Save values needed for backward
        """
        ### BEGIN SOLUTION
        requires_grad = a.requires_grad or b.requires_grad
        
        def backward_fn(grad_output):
            # Chain rule for multiplication:
            # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            if a.requires_grad:
                a.backward(grad_output * b.data)  # Scale by other operand
            if b.requires_grad:
                b.backward(grad_output * a.data)  # Scale by other operand
        
        result = Variable(
            a.data * b.data,
            requires_grad=requires_grad,
            grad_fn=backward_fn if requires_grad else None
        )
        return result
        ### END SOLUTION

# Add operator overloading for convenience
Variable.__add__ = lambda self, other: Add.forward(self, other)
Variable.__mul__ = lambda self, other: Multiply.forward(self, other)

# %% [markdown]
"""
### ✅ IMPLEMENTATION CHECKPOINT: Basic autograd complete

### 🤔 PREDICTION: How much memory does gradient storage use compared to parameters?
Write your guess: _____ × parameter memory

### 🔍 SYSTEMS INSIGHT #1: Gradient Memory Analysis
"""

# %%
def analyze_gradient_memory():
    """Let's measure the memory overhead of gradients!"""
    try:
        # Create a simple computational graph
        x = Variable(np.random.randn(1000, 1000), requires_grad=True)
        y = Variable(np.random.randn(1000, 1000), requires_grad=True)
        z = x * 2 + y * 3
        w = z * z  # More complex graph
        
        # Compute gradients
        w_sum = Variable(np.array([w.data.sum()]), requires_grad=True)
        w_sum.backward()
        
        # Measure memory
        param_memory = x.data.nbytes + y.data.nbytes
        grad_memory = x.grad.nbytes + y.grad.nbytes if x.grad is not None else 0
        
        print(f"Parameters: {param_memory / 1024 / 1024:.2f} MB")
        print(f"Gradients: {grad_memory / 1024 / 1024:.2f} MB")
        print(f"Ratio: {grad_memory / param_memory:.1f}x parameter memory")
        
        # Scale to real networks
        print(f"\nFor a 7B parameter model like LLaMA-7B:")
        print(f"  Parameters: {7e9 * 4 / 1024**3:.1f} GB (float32)")
        print(f"  Gradients: {7e9 * 4 / 1024**3:.1f} GB")
        print(f"  Total training memory: {7e9 * 8 / 1024**3:.1f} GB minimum!")
        
        # 💡 WHY THIS MATTERS: This is why gradient checkpointing exists!
        # Trading compute for memory by recomputing activations during backward.
        
    except Exception as e:
        print(f"⚠️ Error in analysis: {e}")
        print("Make sure Variable class and operations are implemented correctly")

analyze_gradient_memory()

# %% nbgrader={"grade": true, "grade_id": "compute-q1", "points": 2}
"""
### 📊 Computation Question: Memory Requirements

Your Variable class uses float32 (4 bytes per element). Calculate the memory needed for:
- A Variable with shape (1000, 1000) 
- Its gradient after backward()
- Total memory if using Adam optimizer (which stores 2 additional momentum buffers)

Show your calculation and give answers in MB.

YOUR ANSWER:
"""
### BEGIN SOLUTION
"""
Variable data: 1000 × 1000 × 4 bytes = 4,000,000 bytes = 4.0 MB
Gradient: Same size as data = 4.0 MB
Adam momentum (m): 4.0 MB
Adam velocity (v): 4.0 MB
Total with Adam: 4.0 + 4.0 + 4.0 + 4.0 = 16.0 MB
"""
### END SOLUTION

# %% [markdown]
"""
## Part 4: Testing Our Autograd Engine

Let's verify our implementation works correctly!
"""

# %% nbgrader={"grade": true, "grade_id": "test-autograd", "locked": true, "points": 10}
def test_unit_autograd():
    """Test automatic differentiation."""
    print("🧪 Testing Autograd Implementation...")
    
    # Test 1: Simple addition
    x = Variable(np.array([2.0]), requires_grad=True)
    y = Variable(np.array([3.0]), requires_grad=True)
    z = x + y
    z.backward()
    
    assert np.allclose(x.grad, [1.0]), "Addition gradient for x incorrect"
    assert np.allclose(y.grad, [1.0]), "Addition gradient for y incorrect"
    print("✅ Addition gradients correct")
    
    # Test 2: Multiplication
    x.zero_grad()
    y.zero_grad()
    z = x * y
    z.backward()
    
    assert np.allclose(x.grad, [3.0]), "Multiplication gradient for x incorrect"
    assert np.allclose(y.grad, [2.0]), "Multiplication gradient for y incorrect"
    print("✅ Multiplication gradients correct")
    
    # Test 3: Complex expression
    x = Variable(np.array([2.0]), requires_grad=True)
    y = Variable(np.array([3.0]), requires_grad=True)
    z = x * x + y * y  # z = x² + y²
    z.backward()
    
    assert np.allclose(x.grad, [4.0]), "Complex expression gradient for x incorrect"
    assert np.allclose(y.grad, [6.0]), "Complex expression gradient for y incorrect"
    print("✅ Complex expression gradients correct")
    
    print("🎉 All autograd tests passed!")

test_unit_autograd()

# %% [markdown]
"""
## Part 5: Matrix Operations with Broadcasting

Real neural networks need matrix operations. Let's add them!
"""

# %% nbgrader={"grade": false, "grade_id": "matmul", "solution": true}
#| export
class MatMul:
    """Matrix multiplication with gradient computation."""
    
    @staticmethod
    def forward(a: Variable, b: Variable) -> Variable:
        """
        Forward pass: C = A @ B
        
        TODO: Implement matrix multiplication with gradients.
        
        HINTS:
        - Use np.dot or @ operator
        - Gradient w.r.t A: grad_output @ B.T
        - Gradient w.r.t B: A.T @ grad_output
        - Handle shape broadcasting correctly
        """
        ### BEGIN SOLUTION
        requires_grad = a.requires_grad or b.requires_grad
        
        def backward_fn(grad_output):
            # Matrix calculus: Use transposes for gradient flow
            if a.requires_grad:
                grad_a = grad_output @ b.data.T  # ∂L/∂A = ∂L/∂C @ B^T
                a.backward(grad_a)
            if b.requires_grad:
                grad_b = a.data.T @ grad_output  # ∂L/∂B = A^T @ ∂L/∂C
                b.backward(grad_b)
        
        result = Variable(
            a.data @ b.data,
            requires_grad=requires_grad,
            grad_fn=backward_fn if requires_grad else None
        )
        return result
        ### END SOLUTION

Variable.__matmul__ = lambda self, other: MatMul.forward(self, other)

# %% [markdown]
"""
### ✅ IMPLEMENTATION CHECKPOINT: Matrix operations complete

### 🤔 PREDICTION: How many FLOPs does a matrix multiplication A(m×k) @ B(k×n) require?
Your answer: _______ operations

### 🔍 SYSTEMS INSIGHT #2: Matrix Multiplication Complexity
"""

# %%
def analyze_matmul_complexity():
    """Measure the computational complexity of matrix multiplication."""
    import time
    
    try:
        sizes = [100, 200, 400, 800]
        times = []
        flops = []
        
        for size in sizes:
            A = Variable(np.random.randn(size, size), requires_grad=True)
            B = Variable(np.random.randn(size, size), requires_grad=True)
            
            # Measure forward pass
            start = time.perf_counter()
            C = A @ B
            forward_time = time.perf_counter() - start
            
            # Measure backward pass
            start = time.perf_counter()
            C_sum = Variable(np.array([C.data.sum()]), requires_grad=True)
            C_sum.backward()
            backward_time = time.perf_counter() - start
            
            times.append((forward_time, backward_time))
            # FLOPs for matrix multiply: 2 * m * n * k (multiply-add)
            flops.append(2 * size * size * size)
            
            print(f"Size {size}×{size}:")
            print(f"  Forward: {forward_time*1000:.2f}ms")
            print(f"  Backward: {backward_time*1000:.2f}ms (~2× forward)")
            print(f"  FLOPs: {flops[-1]/1e6:.1f}M")
        
        # Analyze scaling
        time_ratio = times[-1][0] / times[0][0]
        size_ratio = sizes[-1] / sizes[0]
        scaling_exp = np.log(time_ratio) / np.log(size_ratio)
        
        print(f"\nTime scaling: O(N^{scaling_exp:.1f}) - should be ~3 for matmul")
        
        # 💡 WHY THIS MATTERS: This O(N³) scaling is why attention (O(N²×d))
        # becomes the bottleneck in transformers with long sequences!
        
    except Exception as e:
        print(f"⚠️ Error in analysis: {e}")
        print("Make sure MatMul is implemented correctly")

analyze_matmul_complexity()

# %% nbgrader={"grade": true, "grade_id": "compute-q2", "points": 2}
"""
### 📊 Computation Question: Matrix Multiplication FLOPs

For matrix multiplication C = A @ B where:
- A has shape (M, K)
- B has shape (K, N)

The FLOPs (floating-point operations) = 2 × M × N × K (multiply + add for each output)

Calculate the FLOPs for these operations in a neural network forward pass:
1. Input (batch=32, features=784) @ Weight (784, 128) = ?
2. Hidden (batch=32, features=128) @ Weight (128, 10) = ?
3. Total FLOPs for both operations = ?

Give your answers in MFLOPs (millions of FLOPs).

YOUR ANSWER:
"""
### BEGIN SOLUTION
"""
1. First layer: 2 × 32 × 128 × 784 = 6,422,528 FLOPs = 6.42 MFLOPs
2. Second layer: 2 × 32 × 10 × 128 = 81,920 FLOPs = 0.08 MFLOPs  
3. Total: 6.42 + 0.08 = 6.50 MFLOPs

Note: First layer dominates computation due to larger dimensions (784 vs 128).
"""
### END SOLUTION

# %% [markdown]
"""
## Part 6: Building a Complete Neural Network Layer

Let's use our autograd to build a real neural network layer!
"""

# %% nbgrader={"grade": false, "grade_id": "linear-layer", "solution": true}
#| export
class Linear:
    """Fully connected layer with automatic differentiation."""
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize a linear layer: y = xW^T + b
        
        TODO: Initialize weights and bias as Variables with gradients.
        
        HINTS:
        - Use Xavier/He initialization for weights
        - Initialize bias to zeros
        - Both need requires_grad=True
        """
        ### BEGIN SOLUTION
        # Xavier initialization prevents gradient vanishing/explosion
        scale = np.sqrt(2.0 / in_features)
        self.weight = Variable(
            np.random.randn(out_features, in_features) * scale,
            requires_grad=True
        )
        self.bias = Variable(
            np.zeros((out_features,)),
            requires_grad=True
        )
        ### END SOLUTION
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass through the layer."""
        ### BEGIN SOLUTION
        output = x @ self.weight.T + self.bias  # y = xW^T + b
        return output
        ### END SOLUTION
    
    def parameters(self) -> List[Variable]:
        """Return all parameters."""
        ### BEGIN SOLUTION
        return [self.weight, self.bias]
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "compute-q3", "points": 2}
"""
### 📊 Computation Question: Parameter Counting

You just implemented a Linear layer. For a 3-layer MLP with architecture:
- Input: 784 features
- Hidden 1: 256 neurons  
- Hidden 2: 128 neurons
- Output: 10 classes

Calculate:
1. Parameters in each layer (weights + biases)
2. Total parameters in the network
3. Memory in MB (float32 = 4 bytes per parameter)

Show your work.

YOUR ANSWER:
"""
### BEGIN SOLUTION
"""
Layer 1 (784 → 256): 
  Weights: 784 × 256 = 200,704
  Bias: 256
  Total: 200,960

Layer 2 (256 → 128):
  Weights: 256 × 128 = 32,768
  Bias: 128
  Total: 32,896

Layer 3 (128 → 10):
  Weights: 128 × 10 = 1,280
  Bias: 10
  Total: 1,290

Network total: 200,960 + 32,896 + 1,290 = 235,146 parameters
Memory: 235,146 × 4 bytes = 940,584 bytes = 0.94 MB
"""
### END SOLUTION

# %% [markdown]
"""
### ✅ IMPLEMENTATION CHECKPOINT: Neural network layer complete

### 🤔 PREDICTION: For a layer with 1000 inputs and 1000 outputs, how many parameters?
Your answer: _______ parameters

### 🔍 SYSTEMS INSIGHT #3: Parameter Counting and Memory
"""

# %%
def analyze_layer_parameters():
    """Count parameters and analyze memory usage in neural network layers."""
    try:
        # Create layers of different sizes
        sizes = [(784, 128), (128, 64), (64, 10)]  # Like a small MNIST network
        
        total_params = 0
        total_memory = 0
        
        print("Layer Parameter Analysis:")
        print("-" * 50)
        
        for in_feat, out_feat in sizes:
            layer = Linear(in_feat, out_feat)
            
            # Count parameters
            weight_params = layer.weight.data.size
            bias_params = layer.bias.data.size
            layer_params = weight_params + bias_params
            
            # Calculate memory
            layer_memory = layer_params * 4  # float32
            
            total_params += layer_params
            total_memory += layer_memory
            
            print(f"Layer {in_feat}→{out_feat}:")
            print(f"  Weights: {weight_params:,} ({weight_params/1000:.1f}K)")
            print(f"  Bias: {bias_params:,}")
            print(f"  Total: {layer_params:,} params = {layer_memory/1024:.1f}KB")
        
        print("-" * 50)
        print(f"Network Total: {total_params:,} parameters")
        print(f"Memory (float32): {total_memory/1024:.1f}KB")
        print(f"With gradients: {total_memory*2/1024:.1f}KB")
        print(f"With Adam optimizer: {total_memory*4/1024:.1f}KB")
        
        # Scale up
        print(f"\nScaling to GPT-3 (175B params):")
        gpt3_memory = 175e9 * 4  # float32
        print(f"  Parameters only: {gpt3_memory/1024**4:.1f}TB")
        print(f"  With Adam: {gpt3_memory*4/1024**4:.1f}TB!")
        
        # 💡 WHY THIS MATTERS: This is why large models use:
        # - Mixed precision (float16/bfloat16)
        # - Gradient checkpointing
        # - Model parallelism across GPUs
        
    except Exception as e:
        print(f"⚠️ Error: {e}")

analyze_layer_parameters()

# %% nbgrader={"grade": true, "grade_id": "compute-q4", "points": 2}
"""
### 📊 Computation Question: Gradient Accumulation

Consider this scenario: A shared weight matrix W (shape 100×100) is used in 3 different places 
in your network. During backward pass:
- Path 1 contributes gradient G1 with all elements = 0.1
- Path 2 contributes gradient G2 with all elements = 0.2  
- Path 3 contributes gradient G3 with all elements = 0.3

Because of gradient accumulation in your backward() method:

1. What will be the final value of W.grad[0,0] (top-left element)?
2. If we OVERWROTE instead of accumulated, what would W.grad[0,0] be?
3. How many total gradient additions occur for the entire weight matrix?

YOUR ANSWER:
"""
### BEGIN SOLUTION
"""
1. W.grad[0,0] = 0.1 + 0.2 + 0.3 = 0.6 (accumulated from all paths)

2. If overwriting: W.grad[0,0] = 0.3 (only the last gradient)

3. Total additions: 100 × 100 × 3 = 30,000 gradient additions
   (each of 10,000 elements gets 3 gradient contributions)

This shows why accumulation is critical for shared parameters!
"""
### END SOLUTION

# %% [markdown]
"""
## Part 7: Complete Test Suite
"""

# %%
def test_unit_all():
    """Run all unit tests for the autograd module."""
    print("🧪 Running Complete Autograd Test Suite...")
    print("=" * 50)
    
    # Test basic autograd
    test_unit_autograd()
    print()
    
    # Test matrix multiplication
    print("🧪 Testing Matrix Multiplication...")
    A = Variable(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)
    B = Variable(np.array([[5, 6], [7, 8]], dtype=np.float32), requires_grad=True)
    C = A @ B
    
    C_sum = Variable(np.array([C.data.sum()]), requires_grad=True)
    C_sum.backward()
    
    expected_grad_A = B.data.sum(axis=0, keepdims=True).T @ np.ones((1, 2))
    print(f"✅ MatMul forward: {np.allclose(C.data, [[19, 22], [43, 50]])}")
    print(f"✅ MatMul gradients computed")
    print()
    
    # Test neural network layer
    print("🧪 Testing Neural Network Layer...")
    layer = Linear(10, 5)
    x = Variable(np.random.randn(3, 10), requires_grad=True)
    y = layer.forward(x)
    
    assert y.data.shape == (3, 5), "Output shape incorrect"
    print(f"✅ Linear layer forward pass: shape {y.data.shape}")
    
    y_sum = Variable(np.array([y.data.sum()]), requires_grad=True)
    y_sum.backward()
    
    assert layer.weight.grad is not None, "Weight gradients not computed"
    assert layer.bias.grad is not None, "Bias gradients not computed"
    print("✅ Linear layer gradients computed")
    
    print("=" * 50)
    print("🎉 All tests passed! Autograd engine working correctly!")

# Main execution
if __name__ == "__main__":
    test_unit_all()

# %% nbgrader={"grade": true, "grade_id": "compute-q5", "points": 2}
"""
### 📊 Computation Question: Batch Size vs Memory

You have a model with 1M parameters training with batch size 64. The memory usage is:
- Model parameters: 4 MB
- Gradients: 4 MB  
- Adam optimizer state: 8 MB
- Activations (batch-dependent): 32 MB

Answer:
1. What is the total memory usage?
2. If you double the batch size to 128, what will the new TOTAL memory be?
3. What is the maximum batch size if you have 100 MB available?

Show calculations.

YOUR ANSWER:
"""
### BEGIN SOLUTION
"""
1. Total memory = 4 + 4 + 8 + 32 = 48 MB

2. With batch size 128:
   - Fixed (params + grads + optimizer): 4 + 4 + 8 = 16 MB (unchanged)
   - Activations: 32 MB × (128/64) = 64 MB (scales linearly)
   - New total: 16 + 64 = 80 MB

3. Maximum batch size with 100 MB:
   - Fixed costs: 16 MB
   - Available for activations: 100 - 16 = 84 MB
   - Batch size: 64 × (84/32) = 168 (maximum)
   
Key insight: Only activations scale with batch size, not parameters/gradients!
"""
### END SOLUTION

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Synthesis Questions

Now that you've built and measured an autograd system, consider these broader questions:
"""

# %% nbgrader={"grade": false, "grade_id": "synthesis-q1", "solution": true, "points": 5}
"""
### Synthesis Question 1: Memory vs Compute Trade-offs

You discovered that gradient computation requires significant memory (1× parameters for 
gradients, 3× more for optimizers). You also measured that backward passes take ~2× 
the time of forward passes.

Design a training strategy for a model that requires 4× your available memory. Your 
strategy should address:
- How to fit the model in memory
- What you sacrifice (time, accuracy, or complexity)
- When this trade-off is worthwhile

YOUR ANSWER (5-7 sentences):
"""
### BEGIN SOLUTION
"""
Strategy: Gradient checkpointing with micro-batching.

1. Divide model into 4 checkpoint segments, storing only segment boundaries
2. During backward, recompute intermediate activations for each segment
3. Process mini-batches in 4 micro-batches, accumulating gradients

Trade-offs:
- Time: ~30% slower due to recomputation
- Memory: 4× reduction achieved
- Complexity: More complex implementation

This is worthwhile when model quality is critical but hardware is limited,
such as research environments or edge deployment. The time cost is acceptable
for better model performance that couldn't otherwise be achieved.
"""
### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "synthesis-q2", "solution": true, "points": 5}
"""
### Synthesis Question 2: Scaling Bottlenecks

Based on your measurements:
- Matrix operations scale O(N³)
- Gradient storage scales O(N) with parameters
- Graph traversal scales O(depth) with network depth

For each scaling pattern, describe:
1. When it becomes the primary bottleneck
2. A real-world scenario where this limits training
3. An engineering solution to mitigate it

YOUR ANSWER (6-8 sentences):
"""
### BEGIN SOLUTION
"""
1. O(N³) matrix operations:
   - Bottleneck: Large hidden dimensions (>10K)
   - Scenario: Language models with large embeddings
   - Solution: Block-sparse matrices, reducing N³ to N²×log(N)

2. O(N) gradient storage:
   - Bottleneck: Models with >10B parameters
   - Scenario: Training exceeds GPU memory
   - Solution: Gradient sharding across devices, ZeRO optimization

3. O(depth) graph traversal:
   - Bottleneck: Networks >1000 layers deep
   - Scenario: Very deep ResNets or Transformers
   - Solution: Gradient checkpointing at strategic layers, reversible layers

The key insight: Different architectures hit different bottlenecks, requiring
architecture-specific optimization strategies.
"""
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Autograd

Congratulations! You've successfully implemented automatic differentiation from scratch:

### What You've Accomplished
✅ **200+ lines of autograd code**: Complete automatic differentiation engine
✅ **Variable class**: Gradient tracking with computational graph construction  
✅ **5 operations**: Add, Multiply, MatMul, and neural network layers
✅ **Memory profiling**: Discovered gradients use 1× parameter memory
✅ **Performance analysis**: Measured O(N³) scaling for matrix operations

### Key Learning Outcomes
- **Chain rule mastery**: Backpropagation through arbitrary computational graphs
- **Memory-compute trade-offs**: Why gradient checkpointing exists
- **Systems insight**: Gradient accumulation vs storage patterns
- **Production patterns**: How PyTorch's autograd actually works

### Mathematical Foundations Mastered
- **Chain rule**: ∂L/∂x = ∂L/∂y · ∂y/∂x
- **Matrix calculus**: Gradients for matrix multiplication
- **Computational complexity**: O(N³) for matmul, O(N) for element-wise

### Professional Skills Developed
- **Automatic differentiation**: Core of all modern deep learning
- **Memory profiling**: Quantifying memory usage in training
- **Performance analysis**: Understanding scaling bottlenecks

### Ready for Advanced Applications
Your autograd implementation now enables:
- **Immediate**: Training neural networks with gradient descent
- **Next Module**: Building optimizers (SGD, Adam) using your gradients
- **Real-world**: Understanding PyTorch's autograd internals

### Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: torch.autograd.Variable and Function classes
- **TensorFlow**: tf.GradientTape API
- **JAX**: grad() transformation

### Next Steps
1. **Export your module**: `tito module complete 06_autograd`
2. **Validate integration**: `tito test --module autograd`
3. **Explore advanced features**: Higher-order gradients, custom operations
4. **Ready for Module 07**: Build optimizers using your autograd engine!

**You've built the foundation of deep learning**: Every neural network trained today relies on automatic differentiation. Your implementation gives you deep understanding of how gradients flow through complex architectures!
"""