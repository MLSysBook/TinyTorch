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
# Module 11: Kernels - Hardware-Optimized ML Operations

Welcome to the Kernels module! This is where we move beyond NumPy to understand how ML operations are optimized for modern hardware. You'll implement custom kernels that run faster than standard library functions.

## Learning Goals
- Understand why custom kernels matter for ML performance
- Implement vectorized operations using SIMD principles
- Master memory-efficient algorithms for better cache utilization
- Build parallel processing patterns for CPU and GPU-style computing
- Create performance profiling tools to measure and optimize code
- Apply kernel optimizations to compressed model operations

## Build → Use → Optimize
1. **Build**: Custom operations, vectorization, and memory optimization
2. **Use**: Apply optimized kernels to real ML workloads
3. **Optimize**: Profile, measure, and improve performance systematically
"""

# %% nbgrader={"grade": false, "grade_id": "kernels-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.kernels

#| export
import numpy as np
import sys
import os
import time
import tracemalloc
import psutil
from typing import Callable, Dict, Any, Optional, Tuple, List
from functools import wraps
from pathlib import Path

# Import our existing components
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import matmul_naive as matmul
    from tinytorch.core.activations import ReLU, Sigmoid, Tanh
    from tinytorch.core.cnn import Conv2D
except ImportError:
    # For development, import from local modules
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.extend([
        os.path.join(base_dir, '01_tensor'),
        os.path.join(base_dir, '02_activations'),
        os.path.join(base_dir, '03_layers'),
        os.path.join(base_dir, '05_cnn'),
        os.path.join(base_dir, 'utils')
    ])
    
    try:
        from tensor_dev import Tensor
        from layers_dev import matmul_naive as matmul
        from activations_dev import ReLU, Sigmoid, Tanh
        from cnn_dev import Conv2D
    except ImportError:
        # Create minimal mock for development
        class Tensor:
            def __init__(self, data):
                self.data = np.array(data)
                self.shape = self.data.shape
            def __str__(self):
                return f"Tensor({self.data})"

# Simple timing utility for kernel performance measurement
def time_kernel(func, *args, **kwargs):
    """
    Simple timing function for measuring kernel performance.
    
    Returns:
        tuple: (result, time_in_microseconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    microseconds = (end - start) * 1_000_000
    return result, microseconds

# %% nbgrader={"grade": false, "grade_id": "kernels-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Kernels Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total // (1024**3):.1f}GB RAM")
print("Ready to optimize ML operations!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/11_kernels/kernels_dev.py`  
**Building Side:** Code exports to `tinytorch.core.kernels`

```python
# Final package structure:
from tinytorch.core.kernels import vectorized_matmul, parallel_relu, cached_conv2d
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
```

**Why this matters:**
- **Performance:** Custom kernels can be 2-10x faster than naive implementations
- **Understanding:** Learn how PyTorch, TensorFlow achieve their speed
- **Real-world:** Modern ML frameworks rely heavily on optimized kernels
- **Hardware:** Bridge the gap between algorithms and computer architecture
"""

# %% [markdown]
"""
## What are ML Kernels?

### The Performance Gap
Your neural network training is slow. A simple matrix multiplication that should take milliseconds takes seconds. Why?

**The problem:** NumPy operations, while convenient, aren't optimized for your specific hardware or use case.

**The solution:** Custom kernels - specialized functions written to extract maximum performance from your hardware.

### What is a Kernel?
A **kernel** is a highly optimized function that performs a specific computation:

```python
# Standard approach - easy but slow
def slow_matmul(A, B):
    return np.dot(A, B)

# Kernel approach - harder but fast
def fast_matmul(A, B):
    # Optimized for your CPU's cache hierarchy
    # Uses SIMD instructions for parallel operations
    # Minimizes memory allocations
    return optimized_result
```

### Why Kernels Matter for ML
Modern ML frameworks achieve their speed through thousands of optimized kernels:

- **PyTorch**: 2000+ CUDA kernels, 500+ CPU kernels
- **TensorFlow**: XLA compiler generates optimized kernels
- **JAX**: JIT compilation creates specialized kernels
- **Hardware**: GPUs have 1000s of cores, TPUs have specialized ML units

### The Performance Hierarchy
```
Python loops:        1x speed    (baseline)
NumPy operations:    10x speed   (vectorized)
Optimized kernels:   100x speed  (hardware-aware)
GPU kernels:         1000x speed (massive parallelism)
```

### Real-World Impact
- **Training time**: 10-hour training → 1-hour training
- **Inference cost**: $1000/month → $100/month
- **Model size**: Enable larger models through efficiency
- **Energy**: 90% reduction in power consumption

### What You'll Learn
1. **Custom operations** - Moving beyond NumPy limitations
2. **Vectorization** - Using SIMD for parallel computation
3. **Memory optimization** - Cache-friendly algorithms
4. **Parallel processing** - CPU and GPU-style parallelism
5. **Performance measurement** - Professional profiling tools
6. **Compressed kernels** - Optimizations for quantized models

Let's build the optimizations that power modern AI!
"""

# %% [markdown]
"""
## Step 1: Custom Operations - Beyond NumPy

### Why Custom Operations?
NumPy is great for prototyping, but has limitations:
- **Generic**: Optimized for general use, not your specific case
- **Memory**: Creates temporary arrays, wastes memory
- **Control**: Can't control memory layout, algorithm choice
- **Specialization**: Can't optimize for your data patterns

### The Philosophy
Instead of using general-purpose functions, we write **specialized** functions:

```python
# Generic NumPy approach
def generic_activation(x):
    return np.maximum(0, x)  # ReLU

# Specialized kernel approach  
def fast_relu_kernel(x):
    # Optimized for your specific use case
    # No unnecessary memory allocations
    # Optimized for your data sizes
    return result
```

### Design Principles
- **Specialization**: Optimize for specific input patterns
- **Memory efficiency**: Minimize allocations and copies
- **Algorithmic choice**: Pick the best algorithm for your data
- **Measurement**: Always profile before and after

### Real-World Context
This is how:
- **PyTorch**: Custom autograd functions override standard operations
- **TensorFlow**: tf.function compiles optimized graphs
- **JAX**: jax.jit creates specialized kernels
- **CUDA**: Every GPU operation is a custom kernel
"""

# %% nbgrader={"grade": false, "grade_id": "custom-matmul", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul_baseline(A: Tensor, B: Tensor) -> Tensor:
    """
    Baseline matrix multiplication using TinyTorch's proven implementation.
    
    This function demonstrates how to build on existing TinyTorch components
    rather than reinventing the wheel. We use the standard matmul from Module 03
    as our baseline for comparison with optimized kernels.
    
    This is NOT a custom implementation - it's the standard TinyTorch matmul
    wrapped for use in kernel comparisons and benchmarking.
    
    TODO: Use TinyTorch's standard matmul implementation as a baseline.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Import the standard matmul function from tinytorch.core.layers
    2. Extract numpy arrays from input Tensors
    3. Use the proven implementation from TinyTorch
    4. Wrap result back in Tensor format
    5. Return the result
    
    CODE REUSE PRINCIPLES:
    1. Always use the packaged version for reliability
    2. Don't duplicate working code - reference the source
    3. Use descriptive names that indicate what the function actually does
    4. Keep dependencies simple and reliable
    
    EXAMPLE USAGE:
    ```python
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    C = matmul_baseline(A, B)
    # Expected: [[19, 22], [43, 50]]
    ```
    
    LEARNING CONNECTIONS:
    - This shows how to use TinyTorch as a library
    - Demonstrates reliable dependency management
    - Serves as baseline for kernel performance comparisons
    - Shows proper software engineering practices
    """
    ### BEGIN SOLUTION
    # Extract numpy arrays from Tensors
    A_data = A.data if hasattr(A, 'data') else A
    B_data = B.data if hasattr(B, 'data') else B
    
    # Use NumPy's matrix multiplication as our baseline
    # This is our baseline - reliable, tested, and consistent
    result_data = np.dot(A_data, B_data)
    
    # Wrap the result back in a Tensor for consistency
    result = Tensor(result_data)
    
    return result
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "test-custom-matmul", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Baseline Matrix Multiplication

def test_matmul_baseline():
    """Test baseline matrix multiplication implementation."""
    print("🔬 Unit Test: Baseline Matrix Multiplication...")
    
    # Test case 1: Small matrices (2x2)
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    C = matmul_baseline(A, B)
    expected = Tensor([[19, 22], [43, 50]])  # Hand-computed
    
    assert np.allclose(C.data, expected.data), f"Expected {expected.data}, got {C.data}"
    print("✅ Small matrix multiplication works")
    
    # Test case 2: Rectangular matrices
    A = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    B = Tensor([[7, 8], [9, 10], [11, 12]])  # 3x2
    C = matmul_baseline(A, B)
    expected = Tensor([[58, 64], [139, 154]])
    
    assert np.allclose(C.data, expected.data), f"Expected {expected.data}, got {C.data}"
    print("✅ Rectangular matrix multiplication works")
    
    # Test case 3: Compare with NumPy (medium size - should use TinyTorch implementation)
    np.random.seed(42)
    A = Tensor(np.random.randn(32, 32))
    B = Tensor(np.random.randn(32, 32))
    
    C_baseline = matmul_baseline(A, B)
    C_numpy = Tensor(np.dot(A.data, B.data))
    
    assert np.allclose(C_baseline.data, C_numpy.data, rtol=1e-10), "Baseline implementation differs from NumPy"
    print("✅ Baseline implementation matches NumPy")
    
    # Test case 4: Large matrix
    A = Tensor(np.random.randn(100, 100))
    B = Tensor(np.random.randn(100, 100))
    C = matmul_baseline(A, B)
    
    assert C.shape == (100, 100), f"Expected shape (100, 100), got {C.shape}"
    print("✅ Large matrix multiplication works")
    
    print("📈 Progress: Baseline Matrix Multiplication ✓")

# Run the test
test_matmul_baseline()

# %% [markdown]
"""
## Step 2: Vectorized Operations - SIMD Principles

### What is Vectorization?
**Vectorization** means processing multiple data elements in parallel using SIMD (Single Instruction, Multiple Data) operations.

### The Problem with Loops
```python
# Scalar processing - one element at a time
def slow_relu(x):
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = max(0, x[i])  # One operation per cycle
    return result
```

### The Vectorization Solution
```python
# Vector processing - multiple elements at once
def fast_relu(x):
    return np.maximum(0, x)  # Many operations per cycle
```

### Why Vectorization Matters
- **CPU SIMD**: Modern CPUs can process 4-8 floats simultaneously
- **GPU parallelism**: GPUs have thousands of cores for parallel processing
- **Memory bandwidth**: Better utilization of memory transfers
- **Compiler optimization**: Enables automatic vectorization

### SIMD Principles
1. **Data parallelism**: Same operation on multiple data elements
2. **Memory alignment**: Aligned data enables faster SIMD instructions
3. **Batch processing**: Process data in chunks that fit SIMD registers
4. **Avoid branches**: Conditional operations break SIMD efficiency

### Real-World Context
- **NumPy**: All operations are vectorized using BLAS/LAPACK
- **PyTorch**: Vectorized operations compile to SIMD instructions
- **GPU kernels**: Thousands of parallel threads process data
- **AVX-512**: Intel's latest SIMD can process 16 floats at once
"""

# %% nbgrader={"grade": false, "grade_id": "vectorized-relu", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def vectorized_relu(x: Tensor) -> Tensor:
    """
    Vectorized ReLU implementation demonstrating SIMD principles.
    
    This function shows how to write operations that take advantage of
    CPU vectorization capabilities for better performance.
    
    TODO: Implement a vectorized ReLU that's optimized for performance.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy array from Tensor
    2. Use NumPy's vectorized operations (these compile to SIMD instructions)
    3. Apply ReLU: f(x) = max(0, x) for all elements simultaneously
    4. Return result as Tensor
    
    VECTORIZATION TECHNIQUES:
    1. Use np.maximum instead of loops - this is vectorized
    2. Ensure input is contiguous in memory for better SIMD performance
    3. Consider using specific dtypes (float32 vs float64) for SIMD alignment
    4. Avoid conditional operations that break vectorization
    
    EXAMPLE USAGE:
    ```python
    x = Tensor([-2, -1, 0, 1, 2])
    y = vectorized_relu(x)
    # Expected: [0, 0, 0, 1, 2]
    ```
    
    PERFORMANCE CONSIDERATIONS:
    - np.maximum is vectorized and uses SIMD instructions
    - Memory layout matters: contiguous arrays are faster
    - Data type matters: float32 allows more SIMD parallelism than float64
    - Avoid Python loops - they can't be vectorized
    
    LEARNING CONNECTIONS:
    - This is how PyTorch's ReLU is implemented under the hood
    - GPU kernels use similar principles with thousands of parallel threads
    - Modern CPUs can process 4-16 floats simultaneously with SIMD
    """
    ### BEGIN SOLUTION
    # Extract numpy array
    x_data = x.data if hasattr(x, 'data') else x
    
    # Ensure contiguous memory layout for better SIMD performance
    if not x_data.flags.c_contiguous:
        x_data = np.ascontiguousarray(x_data)
    
    # Vectorized ReLU using NumPy's maximum function
    # This compiles to SIMD instructions on modern CPUs
    result = np.maximum(0, x_data)
    
    return Tensor(result)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "vectorized-operations", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def vectorized_operations(x: Tensor, y: Tensor) -> Dict[str, Tensor]:
    """
    Demonstration of various vectorized operations.
    
    Shows how multiple operations can be vectorized for better performance.
    
    TODO: Implement a collection of vectorized operations.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy arrays from input Tensors
    2. Implement vectorized versions of common operations
    3. Use NumPy's built-in vectorized functions
    4. Return dictionary of results
    
    OPERATIONS TO IMPLEMENT:
    - element_wise_multiply: x * y (element-wise)
    - element_wise_add: x + y (element-wise)
    - squared_difference: (x - y)^2
    - euclidean_distance: sqrt(sum((x - y)^2))
    - dot_product: sum(x * y)
    
    VECTORIZATION PRINCIPLES:
    - Use NumPy operations instead of Python loops
    - Combine operations when possible: (x - y)**2 instead of subtract then square
    - Consider memory layout and data types
    - Measure performance improvements
    
    EXAMPLE USAGE:
    ```python
    x = Tensor([1, 2, 3, 4])
    y = Tensor([2, 3, 4, 5])
    results = vectorized_operations(x, y)
    # Returns dict with all vectorized operation results
    ```
    """
    ### BEGIN SOLUTION
    # Extract numpy arrays
    x_data = x.data if hasattr(x, 'data') else x
    y_data = y.data if hasattr(y, 'data') else y
    
    # Ensure arrays are the same shape for element-wise operations
    assert x_data.shape == y_data.shape, f"Shape mismatch: {x_data.shape} vs {y_data.shape}"
    
    # Vectorized operations
    results = {
        'element_wise_multiply': Tensor(x_data * y_data),
        'element_wise_add': Tensor(x_data + y_data),
        'squared_difference': Tensor((x_data - y_data) ** 2),
        'euclidean_distance': Tensor(np.sqrt(np.sum((x_data - y_data) ** 2))),
        'dot_product': Tensor(np.dot(x_data.flatten(), y_data.flatten()))
    }
    
    return results
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "test-vectorized-operations", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Vectorized Operations

def test_vectorized_operations():
    """Test vectorized operations implementation."""
    print("🔬 Unit Test: Vectorized Operations...")
    
    # Test vectorized ReLU
    x = Tensor([-2, -1, 0, 1, 2])
    y = vectorized_relu(x)
    expected = [0, 0, 0, 1, 2]
    
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print("✅ Vectorized ReLU works")
    
    # Test vectorized operations
    x = Tensor([1, 2, 3, 4])
    y = Tensor([2, 3, 4, 5])
    results = vectorized_operations(x, y)
    
    # Check element-wise multiply
    expected_mul = [2, 6, 12, 20]
    assert np.allclose(results['element_wise_multiply'].data, expected_mul), \
        f"Expected {expected_mul}, got {results['element_wise_multiply'].data}"
    print("✅ Element-wise multiply works")
    
    # Check element-wise add
    expected_add = [3, 5, 7, 9]
    assert np.allclose(results['element_wise_add'].data, expected_add), \
        f"Expected {expected_add}, got {results['element_wise_add'].data}"
    print("✅ Element-wise add works")
    
    # Check squared difference
    expected_sq_diff = [1, 1, 1, 1]  # (1-2)^2, (2-3)^2, etc.
    assert np.allclose(results['squared_difference'].data, expected_sq_diff), \
        f"Expected {expected_sq_diff}, got {results['squared_difference'].data}"
    print("✅ Squared difference works")
    
    # Check dot product
    expected_dot = 40  # 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert np.allclose(results['dot_product'].data, expected_dot), \
        f"Expected {expected_dot}, got {results['dot_product'].data}"
    print("✅ Dot product works")
    
    print("📈 Progress: Vectorized Operations ✓")

# Run the test
test_vectorized_operations()

# %% [markdown]
"""
## Step 3: Memory Layout Optimization - Cache-Friendly Algorithms

### Why Memory Layout Matters
Modern CPUs are **memory-bound**, not compute-bound. The bottleneck isn't how fast you can multiply numbers—it's how fast you can get data from memory.

### The Memory Hierarchy
```
CPU Registers:    1 cycle     (fastest, tiny)
L1 Cache:         3 cycles    (fast, small)
L2 Cache:         10 cycles   (medium, medium)
L3 Cache:         40 cycles   (slow, large)
Main Memory:      200+ cycles (slowest, huge)
```

### Cache-Friendly Principles
1. **Spatial locality**: Access nearby memory locations
2. **Temporal locality**: Reuse recently accessed data
3. **Cache lines**: Memory is loaded in 64-byte chunks
4. **Cache blocking**: Process data in cache-sized chunks

### Real-World Impact
- **Matrix multiplication**: Cache-friendly algorithms are 10x faster
- **Image processing**: Row-major vs column-major access patterns
- **Neural networks**: Memory layout affects training speed significantly

### The Problem with Naive Algorithms
```python
# Cache-unfriendly: jumps around memory
def slow_transpose(A):
    for i in range(rows):
        for j in range(cols):
            B[j, i] = A[i, j]  # Poor cache locality
```

### Cache-Friendly Solution
```python
# Cache-friendly: processes data in blocks
def fast_transpose(A):
    # Process in cache-sized blocks
    for block_i in range(0, rows, BLOCK_SIZE):
        for block_j in range(0, cols, BLOCK_SIZE):
            # Process block - good cache locality
            for i in range(block_i, min(block_i + BLOCK_SIZE, rows)):
                for j in range(block_j, min(block_j + BLOCK_SIZE, cols)):
                    B[j, i] = A[i, j]
```
"""

# %% nbgrader={"grade": false, "grade_id": "cache-friendly-matmul", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def cache_friendly_matmul(A: Tensor, B: Tensor, block_size: int = 32) -> Tensor:
    """
    Cache-friendly matrix multiplication using blocking technique.
    
    This implementation uses cache blocking to improve memory access patterns
    and achieve better performance on modern CPUs.
    
    TODO: Implement cache-friendly matrix multiplication using blocking.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy arrays and get dimensions
    2. Pre-allocate output matrix
    3. Use three nested loops for blocks: block_i, block_j, block_k
    4. Within each block, use three nested loops for elements: i, j, k
    5. Process data in cache-sized blocks for better locality
    
    BLOCKING ALGORITHM:
    1. Divide matrices into blocks of size block_size x block_size
    2. For each block of C, compute contribution from corresponding A and B blocks
    3. This keeps data in cache longer, reducing memory access time
    
    CACHE OPTIMIZATION PRINCIPLES:
    - Process data in small blocks that fit in cache
    - Reuse data as much as possible while it's in cache
    - Access memory in predictable patterns
    - Minimize cache misses
    
    EXAMPLE USAGE:
    ```python
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    C = cache_friendly_matmul(A, B, block_size=2)
    # Expected: [[19, 22], [43, 50]]
    ```
    
    PERFORMANCE HINTS:
    - block_size should be chosen based on cache size
    - Typical L1 cache: 32KB, so block_size=32 for float32 matrices
    - Experiment with different block sizes for your hardware
    - This algorithm is O(n^3) but with much better constants
    
    LEARNING CONNECTIONS:
    - This is how BLAS libraries achieve high performance
    - GPUs use similar tiling strategies for shared memory
    - Modern compilers can sometimes do this automatically
    """
    ### BEGIN SOLUTION
    # Extract numpy arrays
    A_data = A.data if hasattr(A, 'data') else A
    B_data = B.data if hasattr(B, 'data') else B
    
    # Get dimensions
    m, k = A_data.shape
    k2, n = B_data.shape
    assert k == k2, f"Cannot multiply {A_data.shape} and {B_data.shape}"
    
    # Pre-allocate output matrix
    C = np.zeros((m, n), dtype=A_data.dtype)
    
    # Cache-friendly blocked matrix multiplication
    for block_i in range(0, m, block_size):
        for block_j in range(0, n, block_size):
            for block_k in range(0, k, block_size):
                # Define block boundaries
                end_i = min(block_i + block_size, m)
                end_j = min(block_j + block_size, n)
                end_k = min(block_k + block_size, k)
                
                # Process block - good cache locality
                for i in range(block_i, end_i):
                    for j in range(block_j, end_j):
                        for k_idx in range(block_k, end_k):
                            C[i, j] += A_data[i, k_idx] * B_data[k_idx, j]
    
    return Tensor(C)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "test-cache-friendly", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Cache-Friendly Matrix Multiplication

def test_cache_friendly_matmul():
    """Test cache-friendly matrix multiplication implementation."""
    print("🔬 Unit Test: Cache-Friendly Matrix Multiplication...")
    
    # Test case 1: Small matrices
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    C = cache_friendly_matmul(A, B, block_size=2)
    expected = [[19, 22], [43, 50]]
    
    assert np.allclose(C.data, expected), f"Expected {expected}, got {C.data}"
    print("✅ Small matrix cache-friendly multiplication works")
    
    # Test case 2: Larger matrices with different block sizes
    np.random.seed(42)
    A = Tensor(np.random.randn(64, 64))
    B = Tensor(np.random.randn(64, 64))
    
    C_blocked = cache_friendly_matmul(A, B, block_size=16)
    C_numpy = Tensor(np.dot(A.data, B.data))
    
    assert np.allclose(C_blocked.data, C_numpy.data, rtol=1e-10), \
        "Cache-friendly implementation differs from NumPy"
    print("✅ Cache-friendly implementation matches NumPy")
    
    # Test case 3: Non-square matrices
    A = Tensor(np.random.randn(48, 32))
    B = Tensor(np.random.randn(32, 48))
    
    C_blocked = cache_friendly_matmul(A, B, block_size=8)
    C_numpy = Tensor(np.dot(A.data, B.data))
    
    assert np.allclose(C_blocked.data, C_numpy.data, rtol=1e-10), \
        "Non-square cache-friendly implementation differs from NumPy"
    print("✅ Non-square matrix cache-friendly multiplication works")
    
    print("📈 Progress: Cache-Friendly Algorithms ✓")

# Run the test
test_cache_friendly_matmul()

# %% [markdown]
"""
## Step 4: Parallel Processing - CPU and GPU-Style Computing

### Why Parallel Processing?
Modern hardware has multiple cores, and ML workloads are inherently parallel. We need to use all available compute resources.

### Types of Parallelism
1. **Data parallelism**: Split data across processors
2. **Task parallelism**: Split operations across processors
3. **Pipeline parallelism**: Different stages on different processors
4. **Model parallelism**: Split model across processors

### CPU vs GPU Parallelism
- **CPU**: Few cores (4-64), complex operations, low latency
- **GPU**: Many cores (1000s), simple operations, high throughput

### Parallel Processing Patterns
```python
# Sequential processing
for i in range(n):
    result[i] = expensive_operation(data[i])

# Parallel processing
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(expensive_operation, data[i]) for i in range(n)]
    results = [f.result() for f in futures]
```

### Real-World Context
- **PyTorch**: Parallel data loading, distributed training
- **TensorFlow**: tf.data for parallel preprocessing
- **NumPy**: Multithreaded BLAS operations
- **GPU kernels**: Thousands of parallel threads
"""

# %% nbgrader={"grade": false, "grade_id": "parallel-relu", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def parallel_relu(x: Tensor, num_workers: int = 4) -> Tensor:
    """
    Parallel ReLU implementation using multiple CPU cores.
    
    This function demonstrates data parallelism by splitting the input
    across multiple worker processes.
    
    TODO: Implement parallel ReLU using multiprocessing or threading.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy array from Tensor
    2. Split array into chunks for parallel processing
    3. Define worker function that applies ReLU to a chunk
    4. Use ThreadPoolExecutor to process chunks in parallel
    5. Combine results from all workers
    6. Return result as Tensor
    
    PARALLELIZATION STRATEGY:
    1. Split input into num_workers chunks
    2. Each worker processes its chunk independently
    3. Apply ReLU: max(0, x) to each chunk
    4. Combine results preserving original order
    
    EXAMPLE USAGE:
    ```python
    x = Tensor(np.random.randn(1000))
    y = parallel_relu(x, num_workers=4)
    # Processes data using 4 parallel workers
    ```
    
    PERFORMANCE CONSIDERATIONS:
    - Overhead of parallel processing may not be worth it for small arrays
    - Threading vs multiprocessing trade-offs
    - Chunk size should be large enough to amortize overhead
    - Consider memory bandwidth limitations
    
    LEARNING CONNECTIONS:
    - This is how PyTorch processes batches in parallel
    - GPUs naturally do this with thousands of parallel threads
    - Modern deep learning frameworks heavily use parallelism
    """
    ### BEGIN SOLUTION
    from concurrent.futures import ThreadPoolExecutor
    
    # Extract numpy array
    x_data = x.data if hasattr(x, 'data') else x
    
    # For small arrays, parallel processing isn't worth the overhead
    if x_data.size < 1000:
        return Tensor(np.maximum(0, x_data))
    
    # Split array into chunks
    chunk_size = max(1, x_data.size // num_workers)
    chunks = []
    flat_data = x_data.flatten()
    
    for i in range(0, len(flat_data), chunk_size):
        chunks.append(flat_data[i:i + chunk_size])
    
    # Worker function
    def relu_chunk(chunk):
        return np.maximum(0, chunk)
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(relu_chunk, chunk): i for i, chunk in enumerate(chunks)}
        results = [None] * len(chunks)
        
        for future in future_to_chunk:
            chunk_idx = future_to_chunk[future]
            results[chunk_idx] = future.result()
    
    # Combine results
    combined_result = np.concatenate(results)
    
    # Reshape back to original shape
    result = combined_result.reshape(x_data.shape)
    
    return Tensor(result)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "parallel-batch-processing", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def parallel_batch_processing(batch_data: List[Tensor], operation: Callable, num_workers: int = 4) -> List[Tensor]:
    """
    Process a batch of tensors in parallel using multiple workers.
    
    This function demonstrates how to parallelize operations across
    multiple data samples, similar to how modern ML frameworks work.
    
    TODO: Implement parallel batch processing.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Take a list of Tensors and an operation function
    2. Use ThreadPoolExecutor to process multiple tensors simultaneously
    3. Apply the operation to each tensor in parallel
    4. Return list of results in original order
    
    PARALLELIZATION STRATEGY:
    1. Each worker processes one tensor at a time
    2. Multiple workers can process different tensors simultaneously
    3. Preserve order of results to match input order
    
    EXAMPLE USAGE:
    ```python
    batch = [Tensor(np.random.randn(100, 100)) for _ in range(8)]
    relu_op = lambda x: vectorized_relu(x)
    results = parallel_batch_processing(batch, relu_op, num_workers=4)
    # Processes 8 tensors using 4 parallel workers
    ```
    
    PERFORMANCE CONSIDERATIONS:
    - Each tensor should be large enough to justify parallel overhead
    - Balance number of workers with available CPU cores
    - Consider memory usage with multiple workers
    - Thread vs process pool trade-offs
    
    LEARNING CONNECTIONS:
    - This is how PyTorch's DataLoader processes batches
    - Similar to how GPUs process multiple samples simultaneously
    - Foundation for distributed training across multiple nodes
    """
    ### BEGIN SOLUTION
    from concurrent.futures import ThreadPoolExecutor
    
    # For small batches, parallel processing might not be worth it
    if len(batch_data) < num_workers:
        return [operation(tensor) for tensor in batch_data]
    
    # Process batch in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(operation, tensor): i for i, tensor in enumerate(batch_data)}
        
        # Collect results in original order
        results = [None] * len(batch_data)
        for future in future_to_index:
            index = future_to_index[future]
            results[index] = future.result()
    
    return results
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "test-parallel-processing", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Parallel Processing

def test_parallel_processing():
    """Test parallel processing implementations."""
    print("🔬 Unit Test: Parallel Processing...")
    
    # Test parallel ReLU
    x = Tensor(np.array([-2, -1, 0, 1, 2]))
    y = parallel_relu(x, num_workers=2)
    expected = [0, 0, 0, 1, 2]
    
    assert np.allclose(y.data, expected), f"Expected {expected}, got {y.data}"
    print("✅ Parallel ReLU works")
    
    # Test parallel ReLU with larger data
    x_large = Tensor(np.random.randn(2000))
    y_large = parallel_relu(x_large, num_workers=4)
    y_sequential = vectorized_relu(x_large)
    
    assert np.allclose(y_large.data, y_sequential.data), \
        "Parallel ReLU differs from sequential version"
    print("✅ Parallel ReLU matches sequential version")
    
    # Test parallel batch processing
    batch = [Tensor(np.random.randn(100)) for _ in range(8)]
    relu_op = lambda x: vectorized_relu(x)
    
    results_parallel = parallel_batch_processing(batch, relu_op, num_workers=4)
    results_sequential = [relu_op(tensor) for tensor in batch]
    
    assert len(results_parallel) == len(results_sequential), \
        f"Expected {len(results_sequential)} results, got {len(results_parallel)}"
    
    for i, (parallel, sequential) in enumerate(zip(results_parallel, results_sequential)):
        assert np.allclose(parallel.data, sequential.data), \
            f"Batch item {i}: parallel differs from sequential"
    
    print("✅ Parallel batch processing works")
    print("📈 Progress: Parallel Processing ✓")

# Run the test
test_parallel_processing()

# %% [markdown]
"""
## Step 5: Simple Performance Measurement - Timing Your Kernels

### Why Timing Matters
> "Premature optimization is the root of all evil" - Donald Knuth

But **measured optimization** based on simple timing is essential for understanding kernel performance.

### What We'll Measure
1. **Execution time**: How long does each kernel take?
2. **Relative performance**: Which implementation is faster?
3. **Scale effects**: How does performance change with data size?
4. **Optimization impact**: Did our changes actually help?

### The Simple Timing Process
1. **Measure baseline**: Time the standard implementation
2. **Time optimizations**: Measure your improved versions
3. **Compare results**: See which is faster
4. **Verify correctness**: Ensure optimized code produces correct results

### Our Simple Timing Tool
We use `time.perf_counter()` for microsecond-precision timing:
- **Precise**: Measures actual execution time
- **Simple**: Easy to understand and use
- **Realistic**: Shows kernel performance at the right scale
- **Educational**: Immediate feedback on optimization impact

### Real-World Context
- **Kernel operations**: Typically take 10-1000 microseconds
- **Optimization impact**: Good kernels are 2-10x faster
- **Professional tools**: Production systems use sophisticated profilers
- **Foundation**: Simple timing teaches measurement principles
"""

# %% nbgrader={"grade": false, "grade_id": "test-profiling", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Simple Kernel Timing

def test_simple_kernel_timing():
    """Test simple kernel timing capabilities."""
    print("🔬 Unit Test: Simple Kernel Timing...")
    
    # Test timing different matrix multiplication methods
    np.random.seed(42)
    A = Tensor(np.random.randn(100, 100))
    B = Tensor(np.random.randn(100, 100))
    
    # Time NumPy matmul
    result_numpy, time_numpy = time_kernel(lambda: Tensor(np.dot(A.data, B.data)))
    print(f"🔍 NumPy matmul: {time_numpy:.1f} μs")
    
    # Time baseline matmul  
    result_baseline, time_baseline = time_kernel(matmul_baseline, A, B)
    print(f"🔍 Baseline matmul: {time_baseline:.1f} μs")
    
    # Time cache-friendly matmul
    result_cache, time_cache = time_kernel(cache_friendly_matmul, A, B, 16)
    print(f"🔍 Cache-friendly matmul: {time_cache:.1f} μs")
    
    # Verify results are similar
    assert np.allclose(result_numpy.data, result_baseline.data, rtol=1e-10), \
        "NumPy and baseline results differ"
    assert np.allclose(result_numpy.data, result_cache.data, rtol=1e-10), \
        "NumPy and cache-friendly results differ"
    
    print("✅ All matrix multiplication methods produce correct results")
    
    # Test timing parallel vs sequential ReLU
    x_large = Tensor(np.random.randn(10000))
    
    result_seq, time_seq = time_kernel(vectorized_relu, x_large)
    result_par, time_par = time_kernel(parallel_relu, x_large, 4)
    
    print(f"🔍 Sequential ReLU: {time_seq:.1f} μs")
    print(f"🔍 Parallel ReLU: {time_par:.1f} μs")
    
    # Verify results are the same
    assert np.allclose(result_seq.data, result_par.data), \
        "Sequential and parallel ReLU results differ"
    
    print("✅ Simple timing works correctly")
    print("📈 Progress: Simple Kernel Timing ✓")

# Run the test
test_simple_kernel_timing()

# %% [markdown]
"""
## Step 6: Compressed Model Kernels - Optimizing Quantized Operations

### Why Compressed Model Kernels?
Modern deployment requires smaller, faster models:
- **Mobile devices**: Limited compute and memory
- **Edge computing**: Real-time inference requirements
- **Cloud costs**: Reduce computational expenses
- **Energy efficiency**: Lower power consumption

### Types of Model Compression
1. **Quantization**: Reduce precision (float32 → int8)
2. **Pruning**: Remove unimportant weights
3. **Knowledge distillation**: Train smaller models
4. **Low-rank approximation**: Factorize weight matrices

### Quantization Fundamentals
```python
# Original: 32-bit floating point
weights_fp32 = np.array([1.234, -0.567, 2.891])

# Quantized: 8-bit integer
scale = max(weights_fp32) / 127
weights_int8 = np.round(weights_fp32 / scale).astype(np.int8)

# Dequantized for computation
weights_dequant = weights_int8 * scale
```

### Why Custom Kernels for Compression?
- **Integer arithmetic**: Faster than floating-point on many devices
- **Memory bandwidth**: 4x less data to transfer
- **Specialized instructions**: CPUs have optimized int8 operations
- **Accumulation**: Need to handle precision carefully

### Real-World Context
- **TensorFlow Lite**: Quantized inference kernels
- **PyTorch Mobile**: Optimized int8 operations
- **ONNX Runtime**: Hardware-specific quantized kernels
- **Hardware accelerators**: TPUs, Neural Processing Units
"""

# %% nbgrader={"grade": false, "grade_id": "quantized-matmul", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def quantized_matmul(A: Tensor, B: Tensor, scale_A: float = 1.0, scale_B: float = 1.0) -> Tensor:
    """
    Quantized matrix multiplication kernel for compressed models.
    
    This function demonstrates how to perform matrix multiplication
    with quantized (int8) weights while maintaining numerical accuracy.
    
    TODO: Implement quantized matrix multiplication.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy arrays from Tensors
    2. Quantize inputs to int8 using provided scales
    3. Perform integer matrix multiplication
    4. Rescale result back to appropriate range
    5. Return result as Tensor
    
    QUANTIZATION PROCESS:
    1. Quantize: int8_value = round(float_value / scale)
    2. Compute: int8_result = int8_A @ int8_B
    3. Rescale: float_result = int8_result * scale_A * scale_B
    
    EXAMPLE USAGE:
    ```python
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[0.5, 1.5], [2.5, 3.5]])
    C = quantized_matmul(A, B, scale_A=1.0/127, scale_B=1.0/127)
    # Should approximate regular matrix multiplication
    ```
    
    PERFORMANCE CONSIDERATIONS:
    - int8 operations are often faster than float32
    - Memory usage is 4x lower
    - Accumulation in int32 to prevent overflow
    - Careful handling of scales to maintain precision
    
    LEARNING CONNECTIONS:
    - This is how TensorFlow Lite performs quantized inference
    - Similar to how mobile ML accelerators work
    - Foundation for edge deployment of neural networks
    """
    ### BEGIN SOLUTION
    # Extract numpy arrays
    A_data = A.data if hasattr(A, 'data') else A
    B_data = B.data if hasattr(B, 'data') else B
    
    # Quantize inputs to int8
    A_int8 = np.round(A_data / scale_A).astype(np.int8)
    B_int8 = np.round(B_data / scale_B).astype(np.int8)
    
    # Perform integer matrix multiplication
    # Use int32 for accumulation to prevent overflow
    C_int32 = np.dot(A_int8.astype(np.int32), B_int8.astype(np.int32))
    
    # Rescale result back to float
    C_float = C_int32 * scale_A * scale_B
    
    return Tensor(C_float)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "quantized-relu", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def quantized_relu(x: Tensor, scale: float = 1.0) -> Tensor:
    """
    Quantized ReLU implementation for compressed models.
    
    This function shows how to apply ReLU activation to quantized values
    while maintaining the quantization format.
    
    TODO: Implement quantized ReLU activation.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Extract numpy array from Tensor
    2. Quantize input to int8 using provided scale
    3. Apply ReLU in integer domain: max(0, x)
    4. Keep result in int8 format (no rescaling needed for ReLU)
    5. Convert back to float using scale
    6. Return result as Tensor
    
    QUANTIZED RELU PROCESS:
    1. Quantize: int8_value = round(float_value / scale)
    2. Apply ReLU: int8_result = max(0, int8_value)
    3. Dequantize: float_result = int8_result * scale
    
    EXAMPLE USAGE:
    ```python
    x = Tensor([-1.0, 0.0, 1.0, 2.0])
    y = quantized_relu(x, scale=1.0/127)
    # Should produce [0.0, 0.0, 1.0, 2.0] (approximately)
    ```
    
    OPTIMIZATION NOTES:
    - ReLU in int8 is just max(0, x) - very fast
    - No floating-point operations needed during activation
    - Maintains quantization format throughout
    - Can be vectorized efficiently
    
    LEARNING CONNECTIONS:
    - This is how quantized neural networks maintain speed
    - Similar to how mobile processors optimize ML inference
    - Foundation for real-time edge computing applications
    """
    ### BEGIN SOLUTION
    # Extract numpy array
    x_data = x.data if hasattr(x, 'data') else x
    
    # Quantize input to int8
    x_int8 = np.round(x_data / scale).astype(np.int8)
    
    # Apply ReLU in integer domain
    x_relu_int8 = np.maximum(0, x_int8)
    
    # Convert back to float
    x_relu_float = x_relu_int8 * scale
    
    return Tensor(x_relu_float)
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "test-compressed-kernels", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Compressed Model Kernels

def test_compressed_kernels():
    """Test compressed model kernel implementations."""
    print("🔬 Unit Test: Compressed Model Kernels...")
    
    # Test quantized matrix multiplication
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[0.5, 1.5], [2.5, 3.5]])
    
    # Regular matrix multiplication
    C_regular = matmul_baseline(A, B)
    
    # Quantized matrix multiplication
    scale_A = 1.0 / 127
    scale_B = 1.0 / 127
    C_quantized = quantized_matmul(A, B, scale_A, scale_B)
    
    # Should be approximately equal (some quantization error expected)
    assert np.allclose(C_regular.data, C_quantized.data, rtol=0.1), \
        f"Regular: {C_regular.data}, Quantized: {C_quantized.data}"
    print("✅ Quantized matrix multiplication works")
    
    # Test quantized ReLU
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Regular ReLU
    y_regular = vectorized_relu(x)
    
    # Quantized ReLU
    scale = 1.0 / 127
    y_quantized = quantized_relu(x, scale)
    
    # Should be approximately equal
    assert np.allclose(y_regular.data, y_quantized.data, rtol=0.1), \
        f"Regular: {y_regular.data}, Quantized: {y_quantized.data}"
    print("✅ Quantized ReLU works")
    
    # Test that quantized operations can be timed
    # This shows the performance characteristics of quantized vs regular operations
    x_large = Tensor(np.random.randn(1000))
    
    # Time regular ReLU
    _, time_regular = time_kernel(vectorized_relu, x_large)
    
    # Time quantized ReLU
    _, time_quantized = time_kernel(quantized_relu, x_large, 1.0/127)
    
    print(f"🔍 Regular ReLU: {time_regular:.1f} μs")
    print(f"🔍 Quantized ReLU: {time_quantized:.1f} μs")
    
    print("✅ Quantized operations timing works")
    print("📈 Progress: Compressed Model Kernels ✓")

# Run the test
test_compressed_kernels()

# %% nbgrader={"grade": false, "grade_id": "final-performance-test", "locked": false, "schema_version": 3, "solution": false, "task": false}
### 🧪 Unit Test: Comprehensive Kernel Performance Comparison

def final_performance_test():
    """Comprehensive performance test of all implemented kernels."""
    print("🔬 Final Performance Test: Comprehensive Kernel Comparison")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    A = Tensor(np.random.randn(256, 256))
    B = Tensor(np.random.randn(256, 256))
    x = Tensor(np.random.randn(10000))
    
    print("\n📊 Matrix Multiplication Performance:")
    print("-" * 40)
    
    # Test different matrix multiplication methods
    methods = [
        ("NumPy", lambda: Tensor(np.dot(A.data, B.data))),
        ("Baseline", lambda: matmul_baseline(A, B)),
        ("Cache-friendly", lambda: cache_friendly_matmul(A, B, 32)),
        ("Quantized", lambda: quantized_matmul(A, B, 1.0/127, 1.0/127))
    ]
    
    results = {}
    for name, method in methods:
        result, time_us = time_kernel(method)
        results[name] = (result, time_us)
        print(f"{name:15}: {time_us:.1f} μs")
    
    print("\n📊 ReLU Activation Performance:")
    print("-" * 40)
    
    # Test different ReLU methods
    relu_methods = [
        ("Vectorized", lambda: vectorized_relu(x)),
        ("Parallel", lambda: parallel_relu(x, 4)),
        ("Quantized", lambda: quantized_relu(x, 1.0/127))
    ]
    
    relu_results = {}
    for name, method in relu_methods:
        result, time_us = time_kernel(method)
        relu_results[name] = (result, time_us)
        print(f"{name:15}: {time_us:.1f} μs")
    
    print("\n✅ All kernels implemented successfully!")
    print("📈 Progress: Complete Kernels Module ✓")
    
    # Verify correctness
    print("\n🔍 Correctness Verification:")
    print("-" * 40)
    
    # Check that all matrix multiplication methods produce similar results
    base_result = results["NumPy"][0]
    for name, (result, _) in results.items():
        if name != "NumPy":
            if name == "Quantized":
                # Quantized results have some error tolerance
                assert np.allclose(base_result.data, result.data, rtol=0.1), \
                    f"{name} differs significantly from NumPy"
            else:
                assert np.allclose(base_result.data, result.data, rtol=1e-10), \
                    f"{name} differs from NumPy"
    
    # Check that all ReLU methods produce similar results
    base_relu = relu_results["Vectorized"][0]
    for name, (result, _) in relu_results.items():
        if name != "Vectorized":
            if name == "Quantized":
                # Quantized results have some error tolerance
                assert np.allclose(base_relu.data, result.data, rtol=0.1), \
                    f"{name} ReLU differs significantly from vectorized"
            else:
                assert np.allclose(base_relu.data, result.data, rtol=1e-10), \
                    f"{name} ReLU differs from vectorized"
    
    print("✅ All implementations produce correct results!")
    
    print("\n🎉 CONGRATULATIONS! 🎉")
    print("You've successfully implemented hardware-optimized ML kernels!")
    print("You now understand the performance optimizations that power modern AI frameworks.")

# Run the final test
final_performance_test()

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
    success = run_module_tests_auto("Kernels")

# %% [markdown]
"""
## 🎯 Module Summary: Hardware-Optimized ML Operations

### What You've Built
You've implemented a complete set of hardware-optimized ML kernels:

1. **Custom Operations**: Specialized matrix multiplication beyond NumPy
2. **Vectorized Operations**: SIMD-optimized ReLU and element-wise operations
3. **Cache-Friendly Algorithms**: Blocked matrix multiplication for better memory access
4. **Parallel Processing**: Multi-core CPU utilization for large operations
5. **Performance Profiling**: Tools to measure and optimize kernel performance
6. **Compressed Kernels**: Quantized operations for mobile deployment

### Key Insights
- **Specialization beats generalization**: Custom kernels outperform generic libraries
- **Memory is the bottleneck**: Cache-friendly algorithms are crucial
- **Parallelism is everywhere**: From SIMD to multi-core to GPU-style processing
- **Measurement drives optimization**: Profile first, optimize second
- **Compression enables deployment**: Quantized models run faster with less memory

### Real-World Connections
- **PyTorch**: Uses thousands of optimized kernels for speed
- **TensorFlow**: XLA compiler generates specialized kernels
- **Mobile ML**: Quantized kernels enable edge deployment
- **Cloud computing**: Kernel optimization reduces server costs
- **Research**: Custom kernels enable larger models and faster experimentation

### Next Steps
In real ML systems, you'd:
1. **GPU kernels**: Implement CUDA/OpenCL versions
2. **Auto-tuning**: Automatically find optimal parameters
3. **Hardware specialization**: Optimize for specific processors
4. **Kernel fusion**: Combine multiple operations into single kernels
5. **Distributed computing**: Scale kernels across multiple machines

### 🏆 Achievement Unlocked
You've mastered the performance optimization techniques that power modern ML frameworks. You understand how to move beyond high-level libraries to extract maximum performance from hardware!

**You've completed the TinyTorch Kernels module!** 🎉
"""