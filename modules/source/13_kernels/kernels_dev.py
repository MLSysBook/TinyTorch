# %% [markdown]
"""
# Kernels - High-Performance Computational Kernels

Welcome to Kernels! You'll implement high-performance computational kernels that power modern ML systems!

## LINK Building on Previous Learning
**What You Built Before**:
- Module 11 (Training): Complete training loops with gradient computation
- Module 12 (Regularization): Advanced training techniques for robust models

**What's Working**: You can train neural networks end-to-end with sophisticated optimization and regularization!

**The Gap**: Your implementations work correctly but may not be optimized for real-world performance demands.

**This Module's Solution**: Implement high-performance computational kernels that optimize memory access, leverage parallelism, and achieve production-grade performance.

**Connection Map**:
```
Training -> Kernels -> Benchmarking
(correct)   (fast)    (measured)
```

## Learning Goals (Your 5-Point Framework)
- **Systems understanding**: Memory layout, cache optimization, and vectorization for ML operations
- **Core implementation skill**: Building high-performance computational kernels from scratch
- **Pattern/abstraction mastery**: Recognizing optimization patterns across different hardware architectures
- **Framework connections**: Understanding how PyTorch and TensorFlow achieve high performance
- **Optimization trade-offs**: Balancing memory usage, computational complexity, and parallelism

## Build -> Use -> Reflect
1. **Build**: Implement optimized kernels for matrix operations, activations, and memory management
2. **Use**: Apply kernels to real ML workloads and measure performance improvements
3. **Reflect**: Analyze optimization patterns and design production-grade kernel architectures

## Systems Reality Check
TIP **Production Context**: PyTorch uses custom CUDA kernels and CPU vectorization for 10-100x speedups
SPEED **Performance Insight**: Memory bandwidth is often the limiting factor, not compute - optimize data movement first
"""

# %% [markdown]
"""
## What Are High-Performance Kernels?

High-performance kernels are optimized computational functions that leverage hardware-specific features like:

```
CPU Kernels:
+-------------------------------------+
| SIMD Instructions (AVX, SSE)       | <- Process 4-16 floats simultaneously
| Cache-Friendly Memory Patterns     | <- Minimize cache misses
| Loop Unrolling & Vectorization     | <- Eliminate loop overhead
+-------------------------------------+

GPU Kernels:
+-------------------------------------+
| Thread Blocks & Shared Memory      | <- Parallel processing with fast memory
| Memory Coalescing                   | <- Efficient global memory access
| Warp-Level Operations               | <- 32 threads execute together
+-------------------------------------+
```

**Why This Matters for ML Systems:**
- **Training Speed**: 10-100x faster matrix operations enable larger models
- **Inference Latency**: Optimized kernels reduce serving costs and improve user experience
- **Memory Efficiency**: Better data layouts reduce memory bandwidth requirements
- **Energy Efficiency**: Optimized code reduces power consumption in data centers
"""

# %% [markdown]
"""
## Mathematical Foundations

### Cache-Friendly Matrix Multiplication

Standard algorithm is O(n³) but cache-unfriendly:
```python
# Cache-unfriendly (random memory access)
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i,j] += A[i,k] * B[k,j]  # B[k,j] jumps around memory
```

Blocked algorithm improves cache locality:
```python
# Cache-friendly (blocked access)
for bi in range(0, n, block_size):
    for bj in range(0, n, block_size):
        for bk in range(0, n, block_size):
            # Process block that fits in cache
            for i in range(bi, min(bi+block_size, n)):
                for j in range(bj, min(bj+block_size, n)):
                    for k in range(bk, min(bk+block_size, n)):
                        C[i,j] += A[i,k] * B[k,j]
```

### SIMD Vectorization

Single Instruction, Multiple Data (SIMD) processes multiple elements simultaneously:

```
Scalar ReLU (1 element at a time):
for i in range(n):
    y[i] = max(0, x[i])  # 1 operation per cycle

Vectorized ReLU (8 elements at a time with AVX):
y = np.maximum(0, x)  # 8 operations per cycle
```

### Memory Access Patterns

```
Row-Major Access (Fast):
A[0,0] A[0,1] A[0,2] A[0,3] ...  <- Sequential memory access

Column-Major Access (Slow):
A[0,0] A[1,0] A[2,0] A[3,0] ...  <- Strided memory access

Cache Line Impact:
+-----+-----+-----+-----+
| A[0,0:4] loaded together | <- 64-byte cache line
+-----+-----+-----+-----+
```
"""

# %% [markdown]
"""
## Why Build High-Performance Kernels?

### Production Performance Requirements
Modern ML systems require optimized kernels for:

1. **Real-Time Inference**: Self-driving cars need <10ms response times
2. **Large-Scale Training**: Training GPT-scale models requires maximum hardware utilization
3. **Edge Deployment**: Mobile and IoT devices have limited compute and memory
4. **Cost Optimization**: Cloud compute costs scale with execution time

### Learning Through Implementation
Building kernels teaches you:

- **Hardware-Software Interface**: How software maps to CPU/GPU architecture
- **Performance Engineering**: Systematic optimization methodology
- **Production Debugging**: Why ML models are slow and how to fix them
- **System Design**: How to build scalable ML infrastructure

### Connection to Frameworks
Every major ML framework uses custom kernels:
- **PyTorch**: ATen library with CUDA kernels and CPU vectorization
- **TensorFlow**: XLA compiler with hardware-specific optimizations
- **JAX**: JIT compilation with automatic kernel fusion
"""

# %% [markdown]
"""
## Production Context - How Real Systems Work

### PyTorch Kernel Architecture
```python
# High-level PyTorch operation
result = torch.matmul(A, B)

# Maps to optimized kernel based on:
# - Hardware: CPU (MKL-DNN) vs GPU (cuBLAS)
# - Data type: float32, float16, int8
# - Tensor size: Small (custom) vs Large (BLAS)
# - Memory layout: Contiguous vs Strided
```

### Performance Hierarchy
```
1. Specialized Hardware: TPUs, Tensor Cores    (100-1000x)
2. Optimized Libraries: cuBLAS, MKL           (10-100x)
3. Vectorized Code: SIMD, OpenMP             (2-10x)
4. Cache-Friendly: Blocked algorithms         (1.5-3x)
5. Naive Implementation: Baseline             (1x)
```

### Real-World Impact
- **Training Cost**: Optimized kernels reduce AWS training costs by 50-90%
- **Serving Latency**: Fast inference enables real-time applications
- **Model Size**: Quantization kernels enable deployment on mobile devices
- **Energy Usage**: Efficient kernels reduce data center power consumption
"""

# %%
#| default_exp core.kernels
import numpy as np
import sys
import os
import time
import psutil
from typing import Callable, Dict, Any, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor

# Import our existing components
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # Create minimal mock for development
    class Tensor:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
        def __str__(self):
            return f"Tensor({self.data})"

# %% [markdown]
"""
## Architecture - Building High-Performance Kernels

Our kernel optimization strategy follows a systematic hierarchy:

```
TARGET Optimization Strategy:
+-------------------------------------+
| 1. Correctness: Get the right answer |
| 2. Cache Optimization: Memory patterns |
| 3. Vectorization: SIMD instructions  |
| 4. Parallelization: Multi-core      |
| 5. Quantization: Reduced precision  |
+-------------------------------------+

🔧 Implementation Layers:
+-------------------------------------+
| Higher Level: Kernel Composition    | <- Combine optimizations
| Mid Level: Algorithm Optimization   | <- Cache blocking, tiling
| Lower Level: Hardware Primitives    | <- SIMD, memory layout
+-------------------------------------+
```

**Design Principles:**
1. **Measure First**: Profile before optimizing
2. **Systematic Approach**: One optimization at a time
3. **Hardware Awareness**: Understand the target architecture
4. **Composability**: Build higher-level optimizations from primitives
"""

# %% [markdown]
"""
## Implementation - Building High-Performance Kernels

### Core Timing Infrastructure
"""

# %%
def time_kernel(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Precision timing function for measuring kernel performance.

    This is the foundation for all performance analysis - accurate timing
    that accounts for CPU frequency scaling and system noise.

    Args:
        func: The kernel function to time
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        tuple: (function_result, execution_time_microseconds)

    TODO: Implement high-precision kernel timing with noise reduction.

    APPROACH:
    1. Use time.perf_counter() for high precision timing
    2. Warm up CPU to stable frequency before measurement
    3. Handle OS scheduling noise with multiple measurements
    4. Return both result and timing for validation

    EXAMPLE:
    >>> result, time_us = time_kernel(np.matmul, A, B)
    >>> print(f"Matrix multiply took {time_us:.2f} microseconds")

    PERFORMANCE CONSIDERATIONS:
    - perf_counter() has nanosecond precision on modern systems
    - CPU frequency scaling can affect measurements
    - OS scheduling introduces timing noise
    - Cache state affects first vs subsequent runs
    """
    ### BEGIN SOLUTION
    # Warm-up run to stabilize CPU frequency
    _ = func(*args, **kwargs)

    # High-precision timing
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    # Convert to microseconds for better readability
    execution_time_us = (end - start) * 1_000_000

    return result, execution_time_us
    ### END SOLUTION

# PASS IMPLEMENTATION CHECKPOINT: Timing infrastructure complete

# THINK PREDICTION: How much timing overhead does our measurement add?
# Your guess: _____ microseconds

# MAGNIFY SYSTEMS INSIGHT: Timing Overhead Analysis
def analyze_timing_overhead():
    """Measure the overhead of our timing infrastructure."""
    try:
        # Test with minimal operation
        def minimal_op():
            return 42

        # Time the timing overhead
        measurements = []
        for _ in range(100):
            _, timing = time_kernel(minimal_op)
            measurements.append(timing)

        avg_overhead = np.mean(measurements)
        std_overhead = np.std(measurements)
        min_overhead = np.min(measurements)

        print(f"Timing overhead analysis:")
        print(f"  Average: {avg_overhead:.3f} μs")
        print(f"  Std dev: {std_overhead:.3f} μs")
        print(f"  Minimum: {min_overhead:.3f} μs")
        print(f"  Relative precision: ±{std_overhead/avg_overhead*100:.1f}%")

        # TIP WHY THIS MATTERS: Timing overhead must be much smaller than
        # the operations we're measuring, or results will be meaningless.
        # Modern CPUs: ~1-10 μs overhead, so measure operations >100 μs

        return {
            'avg_overhead_us': avg_overhead,
            'precision_percent': std_overhead/avg_overhead*100,
            'reliable_for_operations_above_us': avg_overhead * 10
        }
    except Exception as e:
        print(f"WARNING️ Timing analysis error: {e}")
        return None

# Run the analysis
timing_analysis = analyze_timing_overhead()

# %% [markdown]
"""
### TEST Unit Test: Timing Infrastructure
This test validates `time_kernel`, ensuring accurate performance measurement
"""

# %%
def test_unit_timing_infrastructure():
    """Test timing infrastructure with known operations."""
    print("TEST Unit Test: Timing Infrastructure")

    # Test 1: Basic timing functionality
    def test_operation():
        time.sleep(0.001)  # 1ms sleep
        return "done"

    result, elapsed_us = time_kernel(test_operation)

    assert result == "done", "Function result should be preserved"
    assert 800 <= elapsed_us <= 2000, f"1ms sleep should take ~1000μs, got {elapsed_us:.1f}μs"
    print(f"PASS Basic timing: {elapsed_us:.1f}μs for 1ms operation")

    # Test 2: Timing precision
    def fast_operation():
        return sum(range(1000))

    measurements = []
    for _ in range(10):
        _, timing = time_kernel(fast_operation)
        measurements.append(timing)

    cv = np.std(measurements) / np.mean(measurements)
    assert cv < 0.5, f"Timing precision should be reasonable, CV={cv:.3f}"
    print(f"PASS Timing precision: CV={cv:.3f} across 10 measurements")

    # Test 3: Argument passing
    def add_operation(a, b, c=0):
        return a + b + c

    result, _ = time_kernel(add_operation, 5, 10, c=2)
    assert result == 17, f"Arguments should pass correctly, got {result}"
    print("PASS Argument passing works correctly")

# Run the test
test_unit_timing_infrastructure()

# %% [markdown]
"""
### Matrix Multiplication Optimization
"""

# %%
def matmul_baseline(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Baseline matrix multiplication using NumPy's optimized implementation.

    This serves as our reference implementation and performance baseline.
    NumPy uses highly optimized BLAS libraries (Intel MKL, OpenBLAS).

    Args:
        A: Left matrix (M x K)
        B: Right matrix (K x N)

    Returns:
        np.ndarray: Result matrix (M x N)

    TODO: Use NumPy's optimized matrix multiplication as baseline.

    APPROACH:
    1. Validate input shapes for compatibility
    2. Use np.dot() which calls optimized BLAS
    3. This is our "ground truth" for correctness and baseline for performance

    EXAMPLE:
    >>> A = np.random.randn(100, 50)
    >>> B = np.random.randn(50, 75)
    >>> C = matmul_baseline(A, B)
    >>> print(C.shape)  # (100, 75)

    PERFORMANCE NOTES:
    - NumPy calls optimized BLAS: Intel MKL or OpenBLAS
    - These libraries use vectorization, threading, and cache optimization
    - Typical performance: 100+ GFLOPS on modern CPUs
    """
    ### BEGIN SOLUTION
    # Validate shapes
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply {A.shape} and {B.shape}: inner dimensions don't match")

    # Use NumPy's optimized matrix multiplication
    result = np.dot(A, B)

    return result
    ### END SOLUTION

# PASS IMPLEMENTATION CHECKPOINT: Baseline matrix multiplication complete

# MAGNIFY SYSTEMS INSIGHT: Matrix Multiplication Performance Scaling
def analyze_matmul_scaling():
    """Analyze how matrix multiplication performance scales with size."""
    try:
        sizes = [64, 128, 256, 512]
        results = []

        for size in sizes:
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Time the operation
            _, time_us = time_kernel(matmul_baseline, A, B)

            # Calculate metrics
            flops = 2 * size**3  # Multiply-accumulate operations
            gflops = flops / (time_us / 1_000_000) / 1e9

            results.append({
                'size': size,
                'time_us': time_us,
                'gflops': gflops,
                'memory_mb': (A.nbytes + B.nbytes + A.nbytes) / 1024 / 1024
            })

            print(f"Size {size:3d}: {time_us:8.1f}μs, {gflops:6.1f} GFLOPS, {results[-1]['memory_mb']:5.1f}MB")

        # Analyze scaling behavior
        time_scaling = results[-1]['time_us'] / results[0]['time_us']
        size_scaling = (results[-1]['size'] / results[0]['size']) ** 3
        efficiency = time_scaling / size_scaling

        print(f"\nScaling analysis:")
        print(f"  Time scaling: {time_scaling:.1f}x")
        print(f"  Theoretical (O(n³)): {size_scaling:.1f}x")
        print(f"  Efficiency: {efficiency:.3f} (1.0 = perfect scaling)")

        # TIP WHY THIS MATTERS: Matrix multiplication is O(n³), but cache effects
        # and memory bandwidth limits mean real performance doesn't scale perfectly.
        # Understanding these limits helps size operations for optimal performance.

        return results

    except Exception as e:
        print(f"WARNING️ Scaling analysis error: {e}")
        return None

# Run the analysis
matmul_scaling = analyze_matmul_scaling()

# %%
def cache_friendly_matmul(A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Cache-friendly matrix multiplication using blocking technique.

    This implementation improves memory access patterns by processing
    matrices in cache-sized blocks, reducing cache misses.

    Args:
        A: Left matrix (M x K)
        B: Right matrix (K x N)
        block_size: Size of cache blocks (default 64)

    Returns:
        np.ndarray: Result matrix (M x N)

    TODO: Implement cache-friendly matrix multiplication using blocking.

    APPROACH:
    1. Divide matrices into block_size x block_size blocks
    2. Process blocks in order that maximizes data reuse
    3. Inner loops work on cache-friendly sub-matrices
    4. Accumulate partial results in output blocks

    BLOCKING ALGORITHM:
    ```
    for each block row of A:
        for each block column of B:
            for each block column of A / block row of B:
                multiply sub-blocks and accumulate
    ```

    EXAMPLE:
    >>> A = np.random.randn(128, 128)
    >>> B = np.random.randn(128, 128)
    >>> C = cache_friendly_matmul(A, B, block_size=32)

    CACHE OPTIMIZATION:
    - block_size should fit in L1 cache (~32KB)
    - For float32: block_size=64 uses ~16KB per block
    - Reduces cache misses from O(n³) to O(n³/B) where B=block_size
    """
    ### BEGIN SOLUTION
    M, K = A.shape
    K2, N = B.shape

    if K != K2:
        raise ValueError(f"Cannot multiply {A.shape} and {B.shape}")

    # Initialize result matrix
    C = np.zeros((M, N), dtype=A.dtype)

    # Cache-friendly blocked multiplication
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            for k in range(0, K, block_size):
                # Define block boundaries
                end_i = min(i + block_size, M)
                end_j = min(j + block_size, N)
                end_k = min(k + block_size, K)

                # Extract blocks
                A_block = A[i:end_i, k:end_k]
                B_block = B[k:end_k, j:end_j]

                # Multiply blocks and accumulate
                C[i:end_i, j:end_j] += np.dot(A_block, B_block)

    return C
    ### END SOLUTION

# %% [markdown]
"""
### TEST Unit Test: Cache-Friendly Matrix Multiplication
This test validates `cache_friendly_matmul`, ensuring correctness and performance improvement
"""

# %%
def test_unit_cache_friendly_matmul():
    """Test cache-friendly matrix multiplication."""
    print("TEST Unit Test: Cache-Friendly Matrix Multiplication")

    # Test 1: Correctness
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)

    result_cache = cache_friendly_matmul(A, B, block_size=1)
    result_baseline = matmul_baseline(A, B)

    assert np.allclose(result_cache, result_baseline), "Cache-friendly result should match baseline"
    print("PASS Correctness: Matches baseline implementation")

    # Test 2: Performance comparison
    size = 256
    A_large = np.random.randn(size, size).astype(np.float32)
    B_large = np.random.randn(size, size).astype(np.float32)

    _, baseline_time = time_kernel(matmul_baseline, A_large, B_large)
    _, cache_time = time_kernel(cache_friendly_matmul, A_large, B_large, 64)

    print(f"PASS Performance: Baseline={baseline_time:.1f}μs, Cache-friendly={cache_time:.1f}μs")

    # Test 3: Different block sizes
    block_sizes = [32, 64, 128]
    for bs in block_sizes:
        result = cache_friendly_matmul(A, B, block_size=bs)
        assert np.allclose(result, result_baseline), f"Block size {bs} should be correct"

    print(f"PASS Block sizes: Tested {block_sizes}")

# Run the test
test_unit_cache_friendly_matmul()

# %% [markdown]
"""
### Vectorized Operations
"""

# %%
def vectorized_relu(x: np.ndarray) -> np.ndarray:
    """
    Vectorized ReLU implementation using SIMD principles.

    This function demonstrates how to write operations that leverage
    CPU vectorization for better performance than scalar loops.

    Args:
        x: Input array

    Returns:
        np.ndarray: ReLU applied element-wise

    TODO: Implement vectorized ReLU optimized for SIMD execution.

    APPROACH:
    1. Ensure input array is contiguous for vectorization
    2. Use NumPy's vectorized operations (compile to SIMD)
    3. Handle different data types appropriately
    4. Return result maintaining input shape

    VECTORIZATION TECHNIQUES:
    - np.maximum() uses SIMD instructions when possible
    - Contiguous memory layout enables efficient vectorization
    - Proper data types (float32) maximize SIMD lane utilization

    EXAMPLE:
    >>> x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    >>> y = vectorized_relu(x)
    >>> print(y)  # [0, 0, 0, 1, 2]

    PERFORMANCE BENEFITS:
    - AVX2: 8 float32 operations per instruction
    - AVX-512: 16 float32 operations per instruction
    - Typical speedup: 4-16x over scalar loops
    """
    ### BEGIN SOLUTION
    # Ensure contiguous memory layout for best SIMD performance
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)

    # Vectorized ReLU using NumPy's maximum function
    # This compiles to SIMD instructions on modern CPUs
    result = np.maximum(0, x)

    return result
    ### END SOLUTION

# %%
def vectorized_operations(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Collection of vectorized operations demonstrating SIMD principles.

    Shows how multiple operations can be vectorized efficiently.

    Args:
        x: First input array
        y: Second input array (must be same shape as x)

    Returns:
        Dict[str, np.ndarray]: Dictionary of vectorized operation results

    TODO: Implement vectorized versions of common operations.

    OPERATIONS TO IMPLEMENT:
    - Element-wise addition, multiplication
    - Squared difference
    - Euclidean distance
    - Dot product

    APPROACH:
    1. Validate input shapes match
    2. Use NumPy vectorized functions
    3. Combine operations when beneficial
    4. Return comprehensive results dictionary

    EXAMPLE:
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([2, 3, 4, 5])
    >>> results = vectorized_operations(x, y)
    >>> print(results['element_wise_add'])  # [3, 5, 7, 9]

    VECTORIZATION BENEFITS:
    - Single instruction processes multiple elements
    - Reduced loop overhead
    - Better CPU pipeline utilization
    """
    ### BEGIN SOLUTION
    # Validate shapes
    if x.shape != y.shape:
        raise ValueError(f"Input shapes don't match: {x.shape} vs {y.shape}")

    # Ensure contiguous arrays for best performance
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    if not y.flags.c_contiguous:
        y = np.ascontiguousarray(y)

    # Vectorized operations
    results = {
        'element_wise_add': x + y,
        'element_wise_multiply': x * y,
        'squared_difference': (x - y) ** 2,
        'euclidean_distance': np.sqrt(np.sum((x - y) ** 2)),
        'dot_product': np.dot(x.flatten(), y.flatten()),
        'cosine_similarity': np.dot(x.flatten(), y.flatten()) / (np.linalg.norm(x) * np.linalg.norm(y))
    }

    return results
    ### END SOLUTION

# PASS IMPLEMENTATION CHECKPOINT: Vectorized operations complete

# MAGNIFY SYSTEMS INSIGHT: Vectorization Performance Analysis
def analyze_vectorization_performance():
    """Compare vectorized vs scalar performance."""
    try:
        size = 100000
        x = np.random.randn(size).astype(np.float32)
        y = np.random.randn(size).astype(np.float32)

        # Time vectorized ReLU
        _, vec_time = time_kernel(vectorized_relu, x)

        # Time scalar ReLU (simulated)
        def scalar_relu_simulation(arr):
            # Simulate scalar processing with numpy operations
            # (Real scalar would be much slower)
            result = np.zeros_like(arr)
            for i in range(min(1000, len(arr))):  # Sample to avoid timeout
                result[i] = max(0, arr[i])
            return result

        _, scalar_time = time_kernel(scalar_relu_simulation, x[:1000])

        # Estimate full scalar time
        estimated_scalar_time = scalar_time * (size / 1000)
        speedup = estimated_scalar_time / vec_time

        print(f"Vectorization performance analysis:")
        print(f"  Array size: {size:,} elements")
        print(f"  Vectorized ReLU: {vec_time:.1f}μs")
        print(f"  Estimated scalar: {estimated_scalar_time:.1f}μs")
        print(f"  Speedup: {speedup:.1f}x")

        # Test vectorized operations
        _, ops_time = time_kernel(vectorized_operations, x, y)
        operations_per_second = 6 * size / (ops_time / 1_000_000)  # 6 operations

        print(f"  Vectorized operations: {ops_time:.1f}μs")
        print(f"  Throughput: {operations_per_second/1e6:.1f}M ops/sec")

        # TIP WHY THIS MATTERS: Vectorization provides 4-16x speedups on modern CPUs.
        # This is essential for real-time inference and efficient training.
        # ML frameworks like PyTorch rely heavily on vectorized operations.

        return {
            'vectorized_speedup': speedup,
            'throughput_mops': operations_per_second / 1e6
        }

    except Exception as e:
        print(f"WARNING️ Vectorization analysis error: {e}")
        return None

# Run the analysis
vectorization_analysis = analyze_vectorization_performance()

# %% [markdown]
"""
### TEST Unit Test: Vectorized Operations
This test validates vectorized implementations for correctness and performance
"""

# %%
def test_unit_vectorized_operations():
    """Test vectorized operations."""
    print("TEST Unit Test: Vectorized Operations")

    # Test 1: Vectorized ReLU correctness
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    result = vectorized_relu(x)
    expected = np.array([0, 0, 0, 1, 2], dtype=np.float32)

    assert np.allclose(result, expected), "Vectorized ReLU should be correct"
    print("PASS ReLU correctness: Produces expected outputs")

    # Test 2: Vectorized operations correctness
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    y = np.array([2, 3, 4, 5], dtype=np.float32)

    results = vectorized_operations(x, y)

    assert np.allclose(results['element_wise_add'], [3, 5, 7, 9]), "Addition should be correct"
    assert np.allclose(results['element_wise_multiply'], [2, 6, 12, 20]), "Multiplication should be correct"
    assert np.allclose(results['dot_product'], 40), "Dot product should be correct"

    print("PASS Operations correctness: All operations produce expected results")

    # Test 3: Performance with larger arrays
    large_x = np.random.randn(10000).astype(np.float32)
    large_y = np.random.randn(10000).astype(np.float32)

    _, relu_time = time_kernel(vectorized_relu, large_x)
    _, ops_time = time_kernel(vectorized_operations, large_x, large_y)

    assert relu_time < 1000, f"ReLU should be fast, took {relu_time:.1f}μs"
    assert ops_time < 5000, f"Operations should be fast, took {ops_time:.1f}μs"

    print(f"PASS Performance: ReLU={relu_time:.1f}μs, Operations={ops_time:.1f}μs")

# Run the test
test_unit_vectorized_operations()

# %% [markdown]
"""
### Parallel Processing
"""

# %%
def parallel_relu(x: np.ndarray, num_workers: int = 4) -> np.ndarray:
    """
    Parallel ReLU implementation using multiple CPU cores.

    Demonstrates data parallelism by distributing computation
    across multiple worker threads.

    Args:
        x: Input array
        num_workers: Number of parallel workers

    Returns:
        np.ndarray: ReLU applied in parallel

    TODO: Implement parallel ReLU using threading or multiprocessing.

    APPROACH:
    1. Split input array into chunks for each worker
    2. Process chunks in parallel using ThreadPoolExecutor
    3. Combine results maintaining original order
    4. Handle edge cases (small arrays, uneven splits)

    PARALLELIZATION STRATEGY:
    - Thread-based for I/O bound or small computations
    - Process-based for CPU-bound large computations
    - Chunk size should balance overhead vs parallelism

    EXAMPLE:
    >>> x = np.random.randn(100000)
    >>> y = parallel_relu(x, num_workers=8)

    PERFORMANCE CONSIDERATIONS:
    - Overhead of thread creation and coordination
    - Memory bandwidth limitations
    - Thread synchronization costs
    - Optimal for large arrays where parallelism benefits exceed overhead
    """
    ### BEGIN SOLUTION
    # For small arrays, parallel processing overhead isn't worth it
    if x.size < 10000:
        return vectorized_relu(x)

    # Split array into chunks
    chunk_size = max(1, x.size // num_workers)
    chunks = []
    flat_x = x.flatten()

    for i in range(0, len(flat_x), chunk_size):
        chunks.append(flat_x[i:i + chunk_size])

    # Worker function
    def relu_chunk(chunk):
        return vectorized_relu(chunk)

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(relu_chunk, chunk) for chunk in chunks]

        # Collect results in order
        results = [future.result() for future in futures]

    # Combine results and reshape
    combined = np.concatenate(results)
    return combined.reshape(x.shape)
    ### END SOLUTION

# %%
def parallel_batch_processing(batch_data: np.ndarray, operation: Callable = None, num_workers: int = 4) -> np.ndarray:
    """
    Process batches of data in parallel across multiple workers.

    Demonstrates how ML frameworks parallelize batch processing
    for improved throughput.

    Args:
        batch_data: Input batch (batch_size, ...)
        operation: Operation to apply (default: ReLU)
        num_workers: Number of parallel workers

    Returns:
        np.ndarray: Processed batch data

    TODO: Implement parallel batch processing.

    APPROACH:
    1. Split batch across workers (each worker gets some samples)
    2. Apply operation to each worker's subset
    3. Combine results maintaining batch order
    4. Default to ReLU if no operation specified

    PARALLELIZATION PATTERN:
    - Each worker processes complete samples
    - Good for independent operations on batch elements
    - Scales well with batch size

    EXAMPLE:
    >>> batch = np.random.randn(128, 784)  # 128 samples, 784 features
    >>> result = parallel_batch_processing(batch, vectorized_relu, 4)

    ML SYSTEMS CONNECTION:
    - PyTorch DataLoader uses similar parallelization
    - GPU tensor operations naturally parallel across batch dimension
    - Critical for large batch training and inference
    """
    ### BEGIN SOLUTION
    if operation is None:
        operation = vectorized_relu

    batch_size = batch_data.shape[0]

    # For small batches, parallel processing overhead isn't worth it
    if batch_size < num_workers:
        return operation(batch_data)

    # Split batch into chunks
    chunk_size = max(1, batch_size // num_workers)
    chunks = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunks.append(batch_data[i:end_idx])

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(operation, chunk) for chunk in chunks]

        # Collect results in order
        results = [future.result() for future in futures]

    # Combine results
    return np.concatenate(results, axis=0)
    ### END SOLUTION

# PASS IMPLEMENTATION CHECKPOINT: Parallel processing complete

# MAGNIFY SYSTEMS INSIGHT: Parallel Processing Scaling Analysis
def analyze_parallel_scaling():
    """Analyze how parallel processing scales with worker count."""
    try:
        # Test data
        large_array = np.random.randn(50000).astype(np.float32)
        batch_data = np.random.randn(64, 1000).astype(np.float32)

        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        results = []

        print("Parallel processing scaling analysis:")
        print("Worker Count | ReLU Time | Batch Time | ReLU Speedup | Batch Speedup")
        print("-" * 70)

        baseline_relu_time = None
        baseline_batch_time = None

        for workers in worker_counts:
            # Time parallel ReLU
            _, relu_time = time_kernel(parallel_relu, large_array, workers)

            # Time parallel batch processing
            _, batch_time = time_kernel(parallel_batch_processing, batch_data, vectorized_relu, workers)

            # Calculate speedups
            if baseline_relu_time is None:
                baseline_relu_time = relu_time
                baseline_batch_time = batch_time
                relu_speedup = 1.0
                batch_speedup = 1.0
            else:
                relu_speedup = baseline_relu_time / relu_time
                batch_speedup = baseline_batch_time / batch_time

            results.append({
                'workers': workers,
                'relu_time': relu_time,
                'batch_time': batch_time,
                'relu_speedup': relu_speedup,
                'batch_speedup': batch_speedup
            })

            print(f"{workers:11d} | {relu_time:8.1f}μs | {batch_time:9.1f}μs | "
                  f"{relu_speedup:11.2f}x | {batch_speedup:12.2f}x")

        # Analyze scaling efficiency
        max_speedup_relu = max(r['relu_speedup'] for r in results)
        max_speedup_batch = max(r['batch_speedup'] for r in results)

        print(f"\nScaling analysis:")
        print(f"  Max ReLU speedup: {max_speedup_relu:.2f}x")
        print(f"  Max batch speedup: {max_speedup_batch:.2f}x")
        print(f"  ReLU efficiency: {max_speedup_relu/8:.2f} (theoretical max: 1.0)")
        print(f"  Batch efficiency: {max_speedup_batch/8:.2f} (theoretical max: 1.0)")

        # TIP WHY THIS MATTERS: Parallel processing has diminishing returns due to:
        # 1. Thread overhead and synchronization costs
        # 2. Memory bandwidth limitations
        # 3. Amdahl's law - sequential portions limit speedup
        # Understanding these limits helps choose optimal parallelism levels.

        return results

    except Exception as e:
        print(f"WARNING️ Parallel scaling analysis error: {e}")
        return None

# Run the analysis
parallel_scaling = analyze_parallel_scaling()

# %% [markdown]
"""
### TEST Unit Test: Parallel Processing
This test validates parallel implementations for correctness and performance scaling
"""

# %%
def test_unit_parallel_processing():
    """Test parallel processing implementations."""
    print("TEST Unit Test: Parallel Processing")

    # Test 1: Parallel ReLU correctness
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

    result_parallel = parallel_relu(x, num_workers=2)
    result_sequential = vectorized_relu(x)

    assert np.allclose(result_parallel, result_sequential), "Parallel ReLU should match sequential"
    print("PASS ReLU correctness: Parallel matches sequential result")

    # Test 2: Parallel batch processing correctness
    batch = np.random.randn(16, 10).astype(np.float32)

    result_parallel = parallel_batch_processing(batch, vectorized_relu, num_workers=4)
    result_sequential = vectorized_relu(batch)

    assert np.allclose(result_parallel, result_sequential), "Parallel batch should match sequential"
    assert result_parallel.shape == batch.shape, "Output shape should match input"
    print("PASS Batch correctness: Parallel matches sequential result")

    # Test 3: Performance with larger data
    large_x = np.random.randn(20000).astype(np.float32)
    large_batch = np.random.randn(32, 1000).astype(np.float32)

    _, sequential_time = time_kernel(vectorized_relu, large_x)
    _, parallel_time = time_kernel(parallel_relu, large_x, 4)

    print(f"PASS Performance: Sequential={sequential_time:.1f}μs, Parallel={parallel_time:.1f}μs")

    # Test 4: Edge cases
    small_x = np.array([1, 2, 3])
    result_small = parallel_relu(small_x, num_workers=8)
    expected_small = vectorized_relu(small_x)

    assert np.allclose(result_small, expected_small), "Small arrays should work correctly"
    print("PASS Edge cases: Small arrays handled correctly")

# Run the test
test_unit_parallel_processing()

# %% [markdown]
"""
### Quantization Kernels
"""

# %%
def quantized_matmul(A: np.ndarray, B: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantized matrix multiplication for memory and compute efficiency.

    Implements quantization to reduce memory usage and enable
    efficient inference on edge devices.

    Args:
        A: Left matrix (float32)
        B: Right matrix (float32)
        bits: Quantization bits (default 8)

    Returns:
        np.ndarray: Dequantized result matrix

    TODO: Implement quantized matrix multiplication.

    APPROACH:
    1. Calculate quantization scales based on data range
    2. Quantize inputs to int8/int16 format
    3. Perform integer matrix multiplication
    4. Dequantize result back to float32

    QUANTIZATION PROCESS:
    ```
    scale = max(abs(data)) / (2^(bits-1) - 1)
    quantized = round(data / scale).clip(-128, 127)  # for 8-bit
    result = quantized_A @ quantized_B
    dequantized = result * scale_A * scale_B
    ```

    EXAMPLE:
    >>> A = np.random.randn(64, 32).astype(np.float32)
    >>> B = np.random.randn(32, 48).astype(np.float32)
    >>> C = quantized_matmul(A, B, bits=8)

    PERFORMANCE BENEFITS:
    - 4x memory reduction (float32 -> int8)
    - Faster integer arithmetic on some hardware
    - Enables deployment on memory-constrained devices
    """
    ### BEGIN SOLUTION
    # Calculate quantization scales
    max_val = 2**(bits-1) - 1  # e.g., 127 for 8-bit

    scale_A = np.max(np.abs(A)) / max_val if np.max(np.abs(A)) > 0 else 1.0
    scale_B = np.max(np.abs(B)) / max_val if np.max(np.abs(B)) > 0 else 1.0

    # Quantize inputs
    if bits == 8:
        dtype = np.int8
        min_val, max_val = -128, 127
    elif bits == 16:
        dtype = np.int16
        min_val, max_val = -32768, 32767
    else:
        raise ValueError(f"Unsupported quantization: {bits} bits")

    A_quantized = np.round(A / scale_A).clip(min_val, max_val).astype(dtype)
    B_quantized = np.round(B / scale_B).clip(min_val, max_val).astype(dtype)

    # Perform integer matrix multiplication
    # Use int32 accumulation to prevent overflow
    C_quantized = np.dot(A_quantized.astype(np.int32), B_quantized.astype(np.int32))

    # Dequantize result
    C_dequantized = C_quantized.astype(np.float32) * scale_A * scale_B

    return C_dequantized
    ### END SOLUTION

# %%
def quantized_relu(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantized ReLU activation for efficient inference.

    Applies ReLU in quantized domain to maintain precision
    while reducing computational overhead.

    Args:
        x: Input array (float32)
        bits: Quantization bits (default 8)

    Returns:
        np.ndarray: Quantized ReLU result (dequantized to float32)

    TODO: Implement quantized ReLU activation.

    APPROACH:
    1. Calculate quantization scale from input range
    2. Quantize input to integer representation
    3. Apply ReLU in integer domain (max(0, x))
    4. Dequantize result back to float32

    QUANTIZED RELU PROCESS:
    ```
    scale = max(abs(x)) / (2^(bits-1) - 1)
    x_quantized = round(x / scale).clip(-128, 127)
    relu_quantized = max(0, x_quantized)
    result = relu_quantized * scale
    ```

    EXAMPLE:
    >>> x = np.array([-1.0, 0.0, 1.0, 2.0])
    >>> y = quantized_relu(x, bits=8)
    >>> print(y)  # [0.0, 0.0, ~1.0, ~2.0]

    OPTIMIZATION BENEFITS:
    - ReLU in integer domain is just max(0, x)
    - No floating-point operations during activation
    - Maintains quantization format for subsequent operations
    """
    ### BEGIN SOLUTION
    # Calculate quantization scale
    max_val = 2**(bits-1) - 1  # e.g., 127 for 8-bit
    scale = np.max(np.abs(x)) / max_val if np.max(np.abs(x)) > 0 else 1.0

    # Quantize input
    if bits == 8:
        dtype = np.int8
        min_val, max_val = -128, 127
    elif bits == 16:
        dtype = np.int16
        min_val, max_val = -32768, 32767
    else:
        raise ValueError(f"Unsupported quantization: {bits} bits")

    x_quantized = np.round(x / scale).clip(min_val, max_val).astype(dtype)

    # Apply ReLU in quantized domain
    relu_quantized = np.maximum(0, x_quantized)

    # Dequantize result
    result = relu_quantized.astype(np.float32) * scale

    return result
    ### END SOLUTION

# PASS IMPLEMENTATION CHECKPOINT: Quantization kernels complete

# MAGNIFY SYSTEMS INSIGHT: Quantization Analysis
def analyze_quantization_impact():
    """Analyze the impact of quantization on accuracy and performance."""
    try:
        # Test matrices
        A = np.random.randn(128, 64).astype(np.float32) * 10  # Scale for visible quantization
        B = np.random.randn(64, 96).astype(np.float32) * 10
        x = np.random.randn(1000).astype(np.float32) * 5

        # Compare quantized vs full precision
        print("Quantization impact analysis:")
        print("Operation      | Bits | Accuracy (MSE) | Memory | Time")
        print("-" * 55)

        # Matrix multiplication analysis
        baseline_matmul = matmul_baseline(A, B)
        baseline_size = A.nbytes + B.nbytes + baseline_matmul.nbytes
        _, baseline_time = time_kernel(matmul_baseline, A, B)

        for bits in [8, 16]:
            quant_result = quantized_matmul(A, B, bits=bits)
            mse = np.mean((baseline_matmul - quant_result) ** 2)

            # Estimate quantized memory usage
            if bits == 8:
                quant_size = A.size + B.size + baseline_matmul.size  # int8 = 1 byte
            else:
                quant_size = (A.size + B.size + baseline_matmul.size) * 2  # int16 = 2 bytes

            memory_ratio = quant_size / baseline_size

            _, quant_time = time_kernel(quantized_matmul, A, B, bits)
            time_ratio = quant_time / baseline_time

            print(f"MatMul         | {bits:4d} | {mse:13.6f} | {memory_ratio:5.2f}x | {time_ratio:5.2f}x")

        # ReLU analysis
        baseline_relu = vectorized_relu(x)
        _, baseline_relu_time = time_kernel(vectorized_relu, x)

        for bits in [8, 16]:
            quant_relu = quantized_relu(x, bits=bits)
            mse_relu = np.mean((baseline_relu - quant_relu) ** 2)

            _, quant_relu_time = time_kernel(quantized_relu, x, bits)
            time_ratio_relu = quant_relu_time / baseline_relu_time

            print(f"ReLU           | {bits:4d} | {mse_relu:13.6f} | {0.25:5.2f}x | {time_ratio_relu:5.2f}x")

        print(f"\nBaseline performance:")
        print(f"  MatMul: {baseline_time:.1f}μs, {baseline_size/1024:.1f}KB")
        print(f"  ReLU: {baseline_relu_time:.1f}μs, {x.nbytes/1024:.1f}KB")

        # TIP WHY THIS MATTERS: Quantization trades accuracy for memory and speed.
        # 8-bit quantization: 4x memory reduction, variable performance impact
        # Critical for edge deployment where memory is constrained
        # Modern ML accelerators (TPUs, mobile chips) heavily use quantization

        return {
            'matmul_accuracy_8bit': np.mean((baseline_matmul - quantized_matmul(A, B, 8)) ** 2),
            'memory_reduction': baseline_size / (A.size + B.size),  # Approximate
            'deployment_ready': True
        }

    except Exception as e:
        print(f"WARNING️ Quantization analysis error: {e}")
        return None

# Run the analysis
quantization_analysis = analyze_quantization_impact()

# %% [markdown]
"""
### TEST Unit Test: Quantization Kernels
This test validates quantization implementations for correctness and efficiency trade-offs
"""

# %%
def test_unit_quantization_kernels():
    """Test quantization kernel implementations."""
    print("TEST Unit Test: Quantization Kernels")

    # Test 1: Quantized matrix multiplication correctness
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

    result_quant = quantized_matmul(A, B, bits=8)
    result_baseline = matmul_baseline(A, B)

    # Should be approximately correct (quantization introduces error)
    relative_error = np.mean(np.abs(result_quant - result_baseline) / np.abs(result_baseline + 1e-8))
    assert relative_error < 0.1, f"Quantization error too high: {relative_error:.3f}"
    print(f"PASS MatMul quantization: relative error {relative_error:.3f}")

    # Test 2: Quantized ReLU correctness
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    result_quant_relu = quantized_relu(x, bits=8)
    result_baseline_relu = vectorized_relu(x)

    # Check that negative values become zero and positive values remain positive
    assert np.all(result_quant_relu >= 0), "Quantized ReLU should be non-negative"
    assert np.allclose(result_quant_relu[x <= 0], 0, atol=0.1), "Negative inputs should become zero"
    print("PASS ReLU quantization: maintains ReLU properties")

    # Test 3: Different bit depths
    for bits in [8, 16]:
        result_8bit = quantized_matmul(A, B, bits=bits)
        assert result_8bit.shape == result_baseline.shape, f"{bits}-bit result shape should match"

        result_relu_bits = quantized_relu(x, bits=bits)
        assert result_relu_bits.shape == x.shape, f"{bits}-bit ReLU shape should match"

    print("PASS Bit depths: 8-bit and 16-bit quantization work correctly")

    # Test 4: Performance characteristics
    large_A = np.random.randn(64, 64).astype(np.float32)
    large_B = np.random.randn(64, 64).astype(np.float32)

    _, baseline_time = time_kernel(matmul_baseline, large_A, large_B)
    _, quant_time = time_kernel(quantized_matmul, large_A, large_B, 8)

    print(f"PASS Performance: Baseline={baseline_time:.1f}μs, Quantized={quant_time:.1f}μs")

# Run the test
test_unit_quantization_kernels()

# %% [markdown]
"""
## Advanced Systems Analysis Framework

Now you'll implement the Progressive Analysis Framework at the **Advanced Level**.

At this level, you design comprehensive analyses from scratch - no scaffolding provided.
"""

# %% [markdown]
"""
### TARGET ADVANCED ANALYSIS CHALLENGE: Comprehensive Kernel Optimization Analysis

**CHALLENGE**: Design and implement a complete kernel optimization analysis system that:

1. **Performance Profiling**: Measures execution time, throughput, and resource utilization
2. **Memory Pattern Analysis**: Analyzes cache behavior, memory bandwidth, and access patterns
3. **Optimization Opportunities**: Identifies bottlenecks and recommends improvements
4. **Hardware Adaptation**: Adapts recommendations based on target hardware architecture
5. **Production Readiness**: Assesses readiness for deployment in production ML systems

**YOUR MISSION**: Implement `KernelOptimizationAnalyzer` class with methods for comprehensive analysis.

**TODO: Design comprehensive kernel optimization analysis from scratch.**

**DESIGN REQUIREMENTS**:
- Analyze cache efficiency and memory bandwidth utilization
- Identify vectorization opportunities and parallel processing potential
- Measure quantization impact on accuracy vs performance trade-offs
- Generate actionable optimization recommendations for production deployment
- Support analysis across different hardware architectures (CPU, GPU, edge devices)

**ANALYSIS FRAMEWORK**:
```python
class KernelOptimizationAnalyzer:
    def analyze_cache_efficiency(self, kernel_func, data_sizes):
        # TODO: Measure cache hit rates and memory access patterns
        pass

    def analyze_vectorization_potential(self, operation_sequence):
        # TODO: Identify SIMD optimization opportunities
        pass

    def analyze_parallel_scaling(self, workload, worker_counts):
        # TODO: Measure parallel processing efficiency
        pass

    def analyze_quantization_trade_offs(self, precision_levels):
        # TODO: Accuracy vs performance analysis
        pass

    def generate_optimization_roadmap(self, target_hardware):
        # TODO: Prioritized recommendations for production deployment
        pass
```

**EXPECTED INSIGHTS**:
- Cache miss rates and optimal block sizes
- Vectorization speedup potential and SIMD utilization
- Parallel processing efficiency and scaling bottlenecks
- Quantization accuracy degradation vs memory/speed benefits
- Hardware-specific optimization strategies

**PRODUCTION FOCUS**: Your analysis should guide real optimization decisions for production ML systems.
"""

# %%
class KernelOptimizationAnalyzer:
    """
    Advanced kernel optimization analysis system for production ML systems.

    TODO: Design comprehensive analysis from scratch.

    This class should provide complete optimization analysis including:
    - Cache efficiency and memory bandwidth analysis
    - Vectorization potential and SIMD utilization assessment
    - Parallel processing scaling analysis and bottleneck identification
    - Quantization impact analysis for accuracy vs performance trade-offs
    - Hardware-specific optimization recommendations for production deployment

    Your implementation should guide real optimization decisions for production ML systems.
    """

    def __init__(self, hardware_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer with hardware configuration.

        TODO: Design initialization strategy that detects or accepts hardware specs.

        Should handle:
        - CPU specifications (cores, cache sizes, SIMD capabilities)
        - Memory hierarchy (L1/L2/L3 cache, RAM bandwidth)
        - GPU specifications (if available)
        - Target deployment environment (cloud, edge, mobile)
        """
        ### BEGIN SOLUTION
        self.hardware_config = hardware_config or self._detect_hardware()
        self.analysis_results = {}
        self.optimization_recommendations = []
        self.baseline_measurements = {}

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect current hardware configuration."""
        return {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total // (1024**3),
            'cache_sizes': {
                'l1_data': 32768,    # 32KB typical L1 data cache
                'l1_instruction': 32768,  # 32KB typical L1 instruction cache
                'l2': 262144,        # 256KB typical L2 cache
                'l3': 8388608        # 8MB typical L3 cache
            },
            'cpu_frequency': 2.4,  # GHz - would detect actual frequency
            'memory_bandwidth': 25.6,  # GB/s - would measure actual bandwidth
            'simd_width': 8,       # AVX2 - 8 float32 per instruction
            'gpu_available': False,
            'deployment_target': 'cloud'  # vs 'edge' or 'mobile'
        }
        ### END SOLUTION

    def analyze_cache_efficiency(self, kernel_func: Callable, data_sizes: List[int],
                               access_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze cache efficiency and memory access patterns.

        TODO: Design comprehensive cache analysis that measures:
        - Cache hit/miss rates for different data sizes
        - Memory bandwidth utilization
        - Optimal block sizes for cache-friendly algorithms
        - Impact of different access patterns (sequential, strided, random)

        Should return actionable insights about memory optimization opportunities.
        """
        ### BEGIN SOLUTION
        if access_patterns is None:
            access_patterns = ['sequential', 'strided', 'random']

        cache_analysis = {
            'data_sizes_tested': data_sizes,
            'access_patterns': access_patterns,
            'cache_efficiency': {},
            'bandwidth_utilization': {},
            'optimal_block_sizes': {},
            'recommendations': []
        }

        l1_size = self.hardware_config['cache_sizes']['l1_data']
        l2_size = self.hardware_config['cache_sizes']['l2']
        l3_size = self.hardware_config['cache_sizes']['l3']

        for size in data_sizes:
            # Generate test data
            test_data = np.random.randn(size, size).astype(np.float32)
            data_size_bytes = test_data.nbytes

            # Time the kernel operation
            _, execution_time = time_kernel(kernel_func, test_data, test_data)

            # Estimate cache behavior
            if data_size_bytes <= l1_size:
                cache_level = 'L1'
                efficiency = 0.95
            elif data_size_bytes <= l2_size:
                cache_level = 'L2'
                efficiency = 0.85
            elif data_size_bytes <= l3_size:
                cache_level = 'L3'
                efficiency = 0.70
            else:
                cache_level = 'RAM'
                efficiency = 0.30

            # Calculate bandwidth utilization
            bytes_accessed = data_size_bytes * 2  # Read A, B
            bandwidth_used = bytes_accessed / (execution_time / 1_000_000) / (1024**3)  # GB/s
            peak_bandwidth = self.hardware_config['memory_bandwidth']
            bandwidth_util = bandwidth_used / peak_bandwidth

            cache_analysis['cache_efficiency'][size] = {
                'cache_level': cache_level,
                'efficiency_estimate': efficiency,
                'data_size_mb': data_size_bytes / (1024**2),
                'execution_time_us': execution_time
            }

            cache_analysis['bandwidth_utilization'][size] = {
                'bandwidth_gb_s': bandwidth_used,
                'utilization_percent': bandwidth_util * 100,
                'bottleneck': 'memory' if bandwidth_util > 0.8 else 'compute'
            }

        # Determine optimal block sizes
        for cache_level, cache_size in [('L1', l1_size), ('L2', l2_size)]:
            # Optimal block size fits in cache with room for temporaries
            optimal_elements = int((cache_size * 0.7) / 4)  # 70% of cache, float32 = 4 bytes
            optimal_block_size = int(np.sqrt(optimal_elements))
            cache_analysis['optimal_block_sizes'][cache_level] = optimal_block_size

        # Generate recommendations
        if any(analysis['bottleneck'] == 'memory' for analysis in cache_analysis['bandwidth_utilization'].values()):
            cache_analysis['recommendations'].append("Memory bandwidth limited - consider cache blocking")

        if max(data_sizes)**2 * 4 > l3_size:
            cache_analysis['recommendations'].append(f"Large matrices exceed L3 cache - use block size <= {cache_analysis['optimal_block_sizes']['L2']}")

        self.analysis_results['cache_efficiency'] = cache_analysis
        return cache_analysis
        ### END SOLUTION

    def analyze_vectorization_potential(self, operation_sequence: List[str],
                                      data_shapes: List[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """
        Analyze vectorization potential and SIMD optimization opportunities.

        TODO: Design analysis that identifies:
        - Operations that can benefit from SIMD vectorization
        - Data layout requirements for optimal vectorization
        - Expected speedup from vectorization
        - Vectorization-friendly algorithm modifications

        Should provide specific recommendations for SIMD optimization.
        """
        ### BEGIN SOLUTION
        if data_shapes is None:
            data_shapes = [(1000,), (1000, 1000), (100, 100, 100)]

        vectorization_analysis = {
            'operations_analyzed': operation_sequence,
            'simd_opportunities': {},
            'data_layout_requirements': {},
            'speedup_estimates': {},
            'algorithm_modifications': [],
            'recommendations': []
        }

        simd_width = self.hardware_config['simd_width']

        # Analyze each operation for vectorization potential
        vectorizable_ops = {
            'add': {'potential': 'high', 'speedup': simd_width * 0.9},
            'multiply': {'potential': 'high', 'speedup': simd_width * 0.9},
            'relu': {'potential': 'high', 'speedup': simd_width * 0.8},
            'matmul': {'potential': 'medium', 'speedup': 3.0},  # More complex, less perfect vectorization
            'conv2d': {'potential': 'medium', 'speedup': 4.0},
            'softmax': {'potential': 'low', 'speedup': 1.5},   # Has sequential dependencies
            'batchnorm': {'potential': 'high', 'speedup': simd_width * 0.7}
        }

        for op in operation_sequence:
            if op in vectorizable_ops:
                vectorization_analysis['simd_opportunities'][op] = vectorizable_ops[op]
            else:
                vectorization_analysis['simd_opportunities'][op] = {
                    'potential': 'unknown',
                    'speedup': 1.0
                }

        # Analyze data layout requirements
        for i, shape in enumerate(data_shapes):
            layout_analysis = {
                'shape': shape,
                'memory_layout': 'contiguous_required',
                'alignment': 'simd_aligned',
                'stride_pattern': 'unit_stride_optimal'
            }

            # For multi-dimensional arrays, analyze optimal access patterns
            if len(shape) > 1:
                layout_analysis['access_pattern'] = 'row_major_optimal'
                layout_analysis['vectorization_dimension'] = 'last_dimension'

            vectorization_analysis['data_layout_requirements'][f'shape_{i}'] = layout_analysis

        # Calculate overall speedup potential
        total_speedup = 1.0
        for op in operation_sequence:
            if op in vectorization_analysis['simd_opportunities']:
                speedup = vectorization_analysis['simd_opportunities'][op]['speedup']
                total_speedup *= speedup ** (1.0 / len(operation_sequence))  # Geometric mean

        vectorization_analysis['speedup_estimates']['overall'] = total_speedup
        vectorization_analysis['speedup_estimates']['best_case'] = max(
            vectorization_analysis['simd_opportunities'][op]['speedup']
            for op in operation_sequence
            if op in vectorization_analysis['simd_opportunities']
        )

        # Algorithm modification suggestions
        if 'matmul' in operation_sequence:
            vectorization_analysis['algorithm_modifications'].append(
                "Use BLAS libraries (MKL, OpenBLAS) for vectorized matrix operations"
            )

        if any(op in ['add', 'multiply', 'relu'] for op in operation_sequence):
            vectorization_analysis['algorithm_modifications'].append(
                "Ensure contiguous memory layout and use NumPy vectorized operations"
            )

        # Generate recommendations
        high_potential_ops = [op for op in operation_sequence
                            if vectorization_analysis['simd_opportunities'].get(op, {}).get('potential') == 'high']

        if high_potential_ops:
            vectorization_analysis['recommendations'].append(
                f"High vectorization potential: {', '.join(high_potential_ops)}"
            )

        if total_speedup > 2.0:
            vectorization_analysis['recommendations'].append(
                f"Significant speedup possible: {total_speedup:.1f}x with full vectorization"
            )

        self.analysis_results['vectorization_potential'] = vectorization_analysis
        return vectorization_analysis
        ### END SOLUTION

    def analyze_parallel_scaling(self, workload_func: Callable, worker_counts: List[int],
                               data_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Analyze parallel processing efficiency and scaling bottlenecks.

        TODO: Design analysis that measures:
        - Parallel processing speedup across different worker counts
        - Scaling efficiency and diminishing returns
        - Thread overhead and synchronization costs
        - Optimal parallelism level for different workload sizes

        Should identify when parallel processing is beneficial vs overhead costs.
        """
        ### BEGIN SOLUTION
        if data_sizes is None:
            data_sizes = [1000, 10000, 100000]

        parallel_analysis = {
            'worker_counts_tested': worker_counts,
            'data_sizes_tested': data_sizes,
            'scaling_results': {},
            'efficiency_analysis': {},
            'overhead_analysis': {},
            'optimal_parallelism': {},
            'recommendations': []
        }

        max_cores = self.hardware_config['cpu_cores']

        for data_size in data_sizes:
            test_data = np.random.randn(data_size).astype(np.float32)
            size_results = {}

            # Measure performance for different worker counts
            baseline_time = None
            for workers in worker_counts:
                if workers > max_cores:
                    continue  # Skip if more workers than cores

                try:
                    _, execution_time = time_kernel(workload_func, test_data, workers)

                    if baseline_time is None:
                        baseline_time = execution_time
                        speedup = 1.0
                        efficiency = 1.0
                    else:
                        speedup = baseline_time / execution_time
                        efficiency = speedup / workers

                    size_results[workers] = {
                        'execution_time_us': execution_time,
                        'speedup': speedup,
                        'efficiency': efficiency
                    }

                except Exception as e:
                    size_results[workers] = {
                        'execution_time_us': None,
                        'speedup': 0,
                        'efficiency': 0,
                        'error': str(e)
                    }

            parallel_analysis['scaling_results'][data_size] = size_results

            # Analyze scaling efficiency
            if size_results:
                max_speedup = max(result['speedup'] for result in size_results.values() if result['speedup'] > 0)
                best_workers = max(size_results.keys(), key=lambda w: size_results[w]['speedup'])

                parallel_analysis['efficiency_analysis'][data_size] = {
                    'max_speedup': max_speedup,
                    'best_worker_count': best_workers,
                    'scaling_efficiency': max_speedup / best_workers,
                    'diminishing_returns_threshold': best_workers
                }

            # Estimate overhead
            if len(size_results) >= 2:
                single_thread_time = size_results.get(1, {}).get('execution_time_us', 0)
                two_thread_time = size_results.get(2, {}).get('execution_time_us', single_thread_time)

                if single_thread_time > 0 and two_thread_time > 0:
                    theoretical_two_thread = single_thread_time / 2
                    overhead_factor = two_thread_time / theoretical_two_thread

                    parallel_analysis['overhead_analysis'][data_size] = {
                        'overhead_factor': overhead_factor,
                        'overhead_percent': (overhead_factor - 1) * 100,
                        'worthwhile_threshold': single_thread_time * 10  # 10x overhead minimum
                    }

        # Determine optimal parallelism
        for data_size in data_sizes:
            if data_size in parallel_analysis['scaling_results']:
                results = parallel_analysis['scaling_results'][data_size]
                optimal_workers = max(results.keys(),
                                    key=lambda w: results[w]['speedup'] if results[w]['speedup'] > 0 else 0)

                parallel_analysis['optimal_parallelism'][data_size] = {
                    'optimal_workers': optimal_workers,
                    'speedup_at_optimal': results[optimal_workers]['speedup'],
                    'efficiency_at_optimal': results[optimal_workers]['efficiency']
                }

        # Generate recommendations
        avg_efficiency = np.mean([
            analysis['scaling_efficiency']
            for analysis in parallel_analysis['efficiency_analysis'].values()
        ])

        if avg_efficiency > 0.7:
            parallel_analysis['recommendations'].append(
                "Excellent parallel scaling - parallel processing highly beneficial"
            )
        elif avg_efficiency > 0.4:
            parallel_analysis['recommendations'].append(
                "Good parallel scaling - parallel processing beneficial for large workloads"
            )
        else:
            parallel_analysis['recommendations'].append(
                "Poor parallel scaling - overhead exceeds benefits, avoid parallel processing"
            )

        # Workload size recommendations
        small_workloads = [size for size in data_sizes if size < 10000]
        if small_workloads and any(
            parallel_analysis['overhead_analysis'].get(size, {}).get('overhead_percent', 0) > 50
            for size in small_workloads
        ):
            parallel_analysis['recommendations'].append(
                "Small workloads have high overhead - use sequential processing"
            )

        self.analysis_results['parallel_scaling'] = parallel_analysis
        return parallel_analysis
        ### END SOLUTION

    def analyze_quantization_trade_offs(self, operations: List[Callable],
                                      precision_levels: List[int] = None,
                                      accuracy_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Analyze quantization impact on accuracy vs performance trade-offs.

        TODO: Design analysis that measures:
        - Accuracy degradation at different quantization levels
        - Performance improvement from reduced precision
        - Memory usage reduction
        - Optimal quantization strategy for production deployment

        Should provide guidance on quantization deployment decisions.
        """
        ### BEGIN SOLUTION
        if precision_levels is None:
            precision_levels = [32, 16, 8]  # float32, float16/int16, int8

        quantization_analysis = {
            'precision_levels_tested': precision_levels,
            'operations_analyzed': [op.__name__ for op in operations],
            'accuracy_analysis': {},
            'performance_analysis': {},
            'memory_analysis': {},
            'deployment_recommendations': {},
            'recommendations': []
        }

        # Test data
        test_sizes = [64, 128, 256]

        for op_func in operations:
            op_name = op_func.__name__
            operation_results = {}

            for size in test_sizes:
                if 'matmul' in op_name.lower():
                    test_data_a = np.random.randn(size, size).astype(np.float32)
                    test_data_b = np.random.randn(size, size).astype(np.float32)
                    baseline_result = op_func(test_data_a, test_data_b)
                    baseline_time = time_kernel(op_func, test_data_a, test_data_b)[1]
                    baseline_memory = (test_data_a.nbytes + test_data_b.nbytes + baseline_result.nbytes)
                else:
                    test_data = np.random.randn(size, size).astype(np.float32)
                    baseline_result = op_func(test_data)
                    baseline_time = time_kernel(op_func, test_data)[1]
                    baseline_memory = test_data.nbytes + baseline_result.nbytes

                size_results = {
                    'baseline': {
                        'precision': 32,
                        'accuracy_mse': 0.0,
                        'execution_time_us': baseline_time,
                        'memory_bytes': baseline_memory,
                        'relative_performance': 1.0,
                        'relative_memory': 1.0
                    }
                }

                # Test different precision levels
                for bits in precision_levels:
                    if bits == 32:
                        continue  # Already have baseline

                    try:
                        if 'matmul' in op_name.lower() and hasattr(op_func, '__name__'):
                            # Use quantized version if available
                            if bits in [8, 16]:
                                quant_result = quantized_matmul(test_data_a, test_data_b, bits=bits)
                                quant_time = time_kernel(quantized_matmul, test_data_a, test_data_b, bits)[1]
                        elif 'relu' in op_name.lower():
                            if bits in [8, 16]:
                                quant_result = quantized_relu(test_data, bits=bits)
                                quant_time = time_kernel(quantized_relu, test_data, bits)[1]
                        else:
                            # Simulate quantization effect
                            max_val = 2**(bits-1) - 1
                            scale = np.max(np.abs(baseline_result)) / max_val
                            quantized = np.round(baseline_result / scale) * scale
                            quant_result = quantized
                            quant_time = baseline_time * 0.8  # Assume some speedup

                        # Calculate accuracy metrics
                        mse = np.mean((baseline_result - quant_result) ** 2)
                        relative_error = mse / (np.mean(baseline_result ** 2) + 1e-8)

                        # Estimate memory usage
                        memory_factor = bits / 32.0
                        quant_memory = int(baseline_memory * memory_factor)

                        size_results[bits] = {
                            'precision': bits,
                            'accuracy_mse': mse,
                            'relative_error': relative_error,
                            'execution_time_us': quant_time,
                            'memory_bytes': quant_memory,
                            'relative_performance': baseline_time / quant_time,
                            'relative_memory': baseline_memory / quant_memory,
                            'acceptable_accuracy': relative_error < accuracy_threshold
                        }

                    except Exception as e:
                        size_results[bits] = {
                            'precision': bits,
                            'error': str(e),
                            'acceptable_accuracy': False
                        }

                operation_results[size] = size_results

            quantization_analysis['accuracy_analysis'][op_name] = operation_results

        # Aggregate analysis across operations and sizes
        for precision in precision_levels:
            if precision == 32:
                continue

            accuracy_scores = []
            performance_gains = []
            memory_reductions = []

            for op_name, op_results in quantization_analysis['accuracy_analysis'].items():
                for size, size_results in op_results.items():
                    if precision in size_results and 'relative_error' in size_results[precision]:
                        accuracy_scores.append(size_results[precision]['acceptable_accuracy'])
                        performance_gains.append(size_results[precision]['relative_performance'])
                        memory_reductions.append(size_results[precision]['relative_memory'])

            if accuracy_scores:
                quantization_analysis['deployment_recommendations'][precision] = {
                    'accuracy_success_rate': np.mean(accuracy_scores),
                    'avg_performance_gain': np.mean(performance_gains),
                    'avg_memory_reduction': np.mean(memory_reductions),
                    'recommended_for_production': np.mean(accuracy_scores) > 0.8 and np.mean(performance_gains) > 1.1
                }

        # Generate recommendations
        for precision, metrics in quantization_analysis['deployment_recommendations'].items():
            if metrics['recommended_for_production']:
                quantization_analysis['recommendations'].append(
                    f"{precision}-bit quantization: {metrics['avg_performance_gain']:.1f}x speedup, "
                    f"{metrics['avg_memory_reduction']:.1f}x memory reduction, "
                    f"{metrics['accuracy_success_rate']*100:.0f}% accuracy success rate"
                )

        if not any(metrics['recommended_for_production']
                  for metrics in quantization_analysis['deployment_recommendations'].values()):
            quantization_analysis['recommendations'].append(
                "Quantization not recommended - accuracy degradation exceeds threshold"
            )

        self.analysis_results['quantization_trade_offs'] = quantization_analysis
        return quantization_analysis
        ### END SOLUTION

    def generate_optimization_roadmap(self, target_hardware: str = 'cloud',
                                    priority_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Generate prioritized optimization roadmap for production deployment.

        TODO: Design roadmap generation that synthesizes all analyses into:
        - Prioritized optimization opportunities
        - Implementation difficulty vs impact assessment
        - Hardware-specific recommendations
        - Deployment timeline and resource requirements

        Should provide actionable guidance for ML system optimization in production.
        """
        ### BEGIN SOLUTION
        if priority_metrics is None:
            priority_metrics = ['performance', 'memory', 'accuracy']

        roadmap = {
            'target_hardware': target_hardware,
            'priority_metrics': priority_metrics,
            'optimization_opportunities': [],
            'implementation_plan': {},
            'resource_requirements': {},
            'expected_outcomes': {},
            'recommendations': []
        }

        # Hardware-specific considerations
        hardware_profiles = {
            'cloud': {
                'cpu_cores': 16,
                'memory_gb': 64,
                'performance_priority': 'high',
                'cost_sensitivity': 'medium',
                'deployment_complexity': 'low'
            },
            'edge': {
                'cpu_cores': 4,
                'memory_gb': 8,
                'performance_priority': 'medium',
                'cost_sensitivity': 'high',
                'deployment_complexity': 'high'
            },
            'mobile': {
                'cpu_cores': 8,
                'memory_gb': 4,
                'performance_priority': 'medium',
                'cost_sensitivity': 'high',
                'deployment_complexity': 'very_high'
            }
        }

        target_profile = hardware_profiles.get(target_hardware, hardware_profiles['cloud'])

        # Analyze optimization opportunities from all analyses
        opportunities = []

        # From cache analysis
        if 'cache_efficiency' in self.analysis_results:
            cache_results = self.analysis_results['cache_efficiency']
            for size, analysis in cache_results['bandwidth_utilization'].items():
                if analysis['bottleneck'] == 'memory':
                    opportunities.append({
                        'type': 'cache_optimization',
                        'impact': 'high',
                        'difficulty': 'medium',
                        'description': 'Implement cache-friendly blocking algorithms',
                        'expected_improvement': '2-4x performance gain',
                        'implementation_effort': '2-3 weeks'
                    })
                    break

        # From vectorization analysis
        if 'vectorization_potential' in self.analysis_results:
            vec_results = self.analysis_results['vectorization_potential']
            overall_speedup = vec_results['speedup_estimates'].get('overall', 1.0)
            if overall_speedup > 2.0:
                opportunities.append({
                    'type': 'vectorization',
                    'impact': 'high',
                    'difficulty': 'low',
                    'description': 'Implement SIMD vectorization for element-wise operations',
                    'expected_improvement': f'{overall_speedup:.1f}x performance gain',
                    'implementation_effort': '1-2 weeks'
                })

        # From parallel analysis
        if 'parallel_scaling' in self.analysis_results:
            parallel_results = self.analysis_results['parallel_scaling']
            avg_efficiency = np.mean([
                analysis['scaling_efficiency']
                for analysis in parallel_results['efficiency_analysis'].values()
            ]) if parallel_results['efficiency_analysis'] else 0

            if avg_efficiency > 0.5 and target_profile['cpu_cores'] > 4:
                opportunities.append({
                    'type': 'parallelization',
                    'impact': 'medium',
                    'difficulty': 'medium',
                    'description': f'Implement parallel processing for {target_profile["cpu_cores"]} cores',
                    'expected_improvement': f'{avg_efficiency * target_profile["cpu_cores"]:.1f}x speedup',
                    'implementation_effort': '2-4 weeks'
                })

        # From quantization analysis
        if 'quantization_trade_offs' in self.analysis_results:
            quant_results = self.analysis_results['quantization_trade_offs']
            for precision, metrics in quant_results['deployment_recommendations'].items():
                if metrics['recommended_for_production']:
                    impact_level = 'high' if metrics['avg_memory_reduction'] > 2.0 else 'medium'
                    opportunities.append({
                        'type': 'quantization',
                        'impact': impact_level,
                        'difficulty': 'high',
                        'description': f'Deploy {precision}-bit quantization',
                        'expected_improvement': f'{metrics["avg_performance_gain"]:.1f}x speedup, {metrics["avg_memory_reduction"]:.1f}x memory reduction',
                        'implementation_effort': '3-6 weeks'
                    })
                    break

        # Sort opportunities by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        difficulty_penalty = {'low': 0, 'medium': -0.5, 'high': -1, 'very_high': -2}

        def opportunity_score(opp):
            impact_score = priority_order.get(opp['impact'], 1)
            difficulty_score = difficulty_penalty.get(opp['difficulty'], 0)

            # Hardware-specific adjustments
            if target_hardware == 'mobile' and opp['type'] == 'quantization':
                impact_score += 1  # Quantization more important for mobile
            elif target_hardware == 'cloud' and opp['type'] == 'parallelization':
                impact_score += 0.5  # Parallelization more beneficial in cloud

            return impact_score + difficulty_score

        opportunities.sort(key=opportunity_score, reverse=True)
        roadmap['optimization_opportunities'] = opportunities[:5]  # Top 5 opportunities

        # Create implementation plan
        phases = ['Phase 1 (0-1 months)', 'Phase 2 (1-3 months)', 'Phase 3 (3-6 months)']
        current_phase = 0

        for i, opportunity in enumerate(roadmap['optimization_opportunities']):
            if i < 2:
                phase = phases[0]
            elif i < 4:
                phase = phases[1]
            else:
                phase = phases[2]

            if phase not in roadmap['implementation_plan']:
                roadmap['implementation_plan'][phase] = []

            roadmap['implementation_plan'][phase].append({
                'optimization': opportunity['type'],
                'description': opportunity['description'],
                'effort': opportunity['implementation_effort']
            })

        # Resource requirements
        roadmap['resource_requirements'] = {
            'engineering_time': '3-6 months for full implementation',
            'hardware_requirements': f"Target: {target_hardware} with {target_profile['cpu_cores']} cores, {target_profile['memory_gb']}GB RAM",
            'testing_infrastructure': 'Performance testing and regression testing framework',
            'deployment_complexity': target_profile['deployment_complexity']
        }

        # Expected outcomes
        total_performance_gain = 1.0
        total_memory_reduction = 1.0

        for opp in roadmap['optimization_opportunities']:
            # Extract numerical improvements (simplified)
            if 'x performance gain' in opp['expected_improvement']:
                try:
                    gain = float(opp['expected_improvement'].split('x')[0])
                    total_performance_gain *= gain ** 0.5  # Assume some compounding
                except:
                    pass

            if 'x memory reduction' in opp['expected_improvement']:
                try:
                    reduction = float(opp['expected_improvement'].split('x memory reduction')[0].split()[-1])
                    total_memory_reduction *= reduction ** 0.5
                except:
                    pass

        roadmap['expected_outcomes'] = {
            'performance_improvement': f'{total_performance_gain:.1f}x overall speedup',
            'memory_efficiency': f'{total_memory_reduction:.1f}x memory reduction',
            'deployment_readiness': 'Production-ready optimized kernels',
            'maintenance_overhead': 'Low (well-structured optimization patterns)'
        }

        # Generate final recommendations
        roadmap['recommendations'] = [
            f"Prioritize {roadmap['optimization_opportunities'][0]['type']} optimization first (highest impact)",
            f"Target hardware ({target_hardware}) well-suited for planned optimizations",
            f"Expected overall improvement: {total_performance_gain:.1f}x performance, {total_memory_reduction:.1f}x memory efficiency",
            "Implement comprehensive performance testing before production deployment"
        ]

        if target_hardware in ['edge', 'mobile']:
            roadmap['recommendations'].append(
                "Quantization critical for resource-constrained deployment"
            )

        self.analysis_results['optimization_roadmap'] = roadmap
        return roadmap
        ### END SOLUTION

# PASS IMPLEMENTATION CHECKPOINT: Advanced optimization analyzer complete

# THINK PREDICTION: What will be the most impactful optimization for matrix operations?
# Your guess: _______

# MAGNIFY SYSTEMS INSIGHT: Comprehensive Kernel Optimization Analysis
def comprehensive_kernel_analysis():
    """Run complete kernel optimization analysis using the advanced analyzer."""
    try:
        print("ROCKET Comprehensive Kernel Optimization Analysis")
        print("=" * 60)

        # Initialize analyzer
        analyzer = KernelOptimizationAnalyzer()

        # 1. Cache efficiency analysis
        print("\n📊 Cache Efficiency Analysis:")
        cache_results = analyzer.analyze_cache_efficiency(
            matmul_baseline,
            data_sizes=[64, 128, 256, 512],
            access_patterns=['sequential', 'strided']
        )

        for size, analysis in cache_results['cache_efficiency'].items():
            print(f"  Size {size:3d}: {analysis['cache_level']} cache, {analysis['efficiency_estimate']:.1%} efficiency")

        print(f"  Recommendations: {'; '.join(cache_results['recommendations'])}")

        # 2. Vectorization potential analysis
        print("\nROCKET Vectorization Potential Analysis:")
        vec_results = analyzer.analyze_vectorization_potential(
            ['matmul', 'relu', 'add', 'multiply'],
            [(1000,), (1000, 1000)]
        )

        for op, potential in vec_results['simd_opportunities'].items():
            print(f"  {op}: {potential['potential']} potential, {potential['speedup']:.1f}x speedup")

        print(f"  Overall speedup estimate: {vec_results['speedup_estimates']['overall']:.1f}x")

        # 3. Parallel scaling analysis
        print("\n🔀 Parallel Scaling Analysis:")
        parallel_results = analyzer.analyze_parallel_scaling(
            parallel_relu,
            worker_counts=[1, 2, 4, 8],
            data_sizes=[10000, 50000]
        )

        for size, analysis in parallel_results['efficiency_analysis'].items():
            print(f"  Size {size:5d}: {analysis['max_speedup']:.1f}x max speedup, {analysis['scaling_efficiency']:.1%} efficiency")

        # 4. Quantization trade-offs analysis
        print("\n🗜️ Quantization Trade-offs Analysis:")
        quant_results = analyzer.analyze_quantization_trade_offs(
            [matmul_baseline, vectorized_relu],
            precision_levels=[32, 16, 8]
        )

        for precision, metrics in quant_results['deployment_recommendations'].items():
            if metrics['recommended_for_production']:
                print(f"  {precision}-bit: {metrics['avg_performance_gain']:.1f}x speedup, "
                      f"{metrics['avg_memory_reduction']:.1f}x memory reduction, "
                      f"{metrics['accuracy_success_rate']:.0%} accuracy success")

        # 5. Generate optimization roadmap
        print("\n🗺️ Optimization Roadmap:")
        roadmap = analyzer.generate_optimization_roadmap(
            target_hardware='cloud',
            priority_metrics=['performance', 'memory']
        )

        print(f"  Target: {roadmap['target_hardware']} deployment")
        print(f"  Expected outcomes: {roadmap['expected_outcomes']['performance_improvement']}, "
              f"{roadmap['expected_outcomes']['memory_efficiency']}")

        print("\n  Top optimization opportunities:")
        for i, opp in enumerate(roadmap['optimization_opportunities'][:3], 1):
            print(f"    {i}. {opp['type']}: {opp['description']}")
            print(f"       Impact: {opp['impact']}, Effort: {opp['implementation_effort']}")

        print("\n  Key recommendations:")
        for rec in roadmap['recommendations'][:3]:
            print(f"    • {rec}")

        # TIP WHY THIS MATTERS: Comprehensive analysis guides optimization decisions:
        # 1. Cache analysis reveals memory bottlenecks and optimal algorithms
        # 2. Vectorization analysis shows where SIMD can provide biggest gains
        # 3. Parallel analysis identifies when threading helps vs hurts
        # 4. Quantization analysis balances accuracy vs deployment efficiency
        # 5. Roadmap prioritizes efforts for maximum production impact

        return {
            'cache_analysis': cache_results,
            'vectorization_analysis': vec_results,
            'parallel_analysis': parallel_results,
            'quantization_analysis': quant_results,
            'optimization_roadmap': roadmap
        }

    except Exception as e:
        print(f"WARNING️ Comprehensive analysis error: {e}")
        return None

# Run the comprehensive analysis
comprehensive_analysis = comprehensive_kernel_analysis()

# %% [markdown]
"""
### TEST Unit Test: Advanced Optimization Analyzer
This test validates the comprehensive kernel optimization analyzer
"""

# %%
def test_unit_advanced_optimization_analyzer():
    """Test the advanced kernel optimization analyzer."""
    print("TEST Unit Test: Advanced Optimization Analyzer")

    # Test 1: Analyzer initialization
    analyzer = KernelOptimizationAnalyzer()

    assert hasattr(analyzer, 'hardware_config'), "Analyzer should have hardware config"
    assert analyzer.hardware_config['cpu_cores'] > 0, "Should detect CPU cores"
    print("PASS Initialization: Hardware configuration detected")

    # Test 2: Cache efficiency analysis
    cache_results = analyzer.analyze_cache_efficiency(matmul_baseline, [64, 128])

    assert 'cache_efficiency' in cache_results, "Should return cache efficiency results"
    assert 'bandwidth_utilization' in cache_results, "Should analyze bandwidth utilization"
    assert 'recommendations' in cache_results, "Should provide recommendations"
    print("PASS Cache analysis: Complete analysis with recommendations")

    # Test 3: Vectorization potential analysis
    vec_results = analyzer.analyze_vectorization_potential(['relu', 'add'])

    assert 'simd_opportunities' in vec_results, "Should identify SIMD opportunities"
    assert 'speedup_estimates' in vec_results, "Should estimate speedup potential"
    print("PASS Vectorization analysis: SIMD opportunities identified")

    # Test 4: Parallel scaling analysis
    parallel_results = analyzer.analyze_parallel_scaling(parallel_relu, [1, 2, 4])

    assert 'scaling_results' in parallel_results, "Should provide scaling results"
    assert 'efficiency_analysis' in parallel_results, "Should analyze efficiency"
    print("PASS Parallel analysis: Scaling efficiency measured")

    # Test 5: Quantization analysis
    quant_results = analyzer.analyze_quantization_trade_offs([vectorized_relu])

    assert 'deployment_recommendations' in quant_results, "Should provide deployment recommendations"
    assert 'accuracy_analysis' in quant_results, "Should analyze accuracy impact"
    print("PASS Quantization analysis: Trade-offs evaluated")

    # Test 6: Optimization roadmap
    roadmap = analyzer.generate_optimization_roadmap('cloud')

    assert 'optimization_opportunities' in roadmap, "Should identify opportunities"
    assert 'implementation_plan' in roadmap, "Should provide implementation plan"
    assert 'expected_outcomes' in roadmap, "Should estimate outcomes"
    assert 'recommendations' in roadmap, "Should give actionable recommendations"
    print("PASS Roadmap generation: Comprehensive optimization plan created")

    # Test 7: Integration across analyses
    assert len(analyzer.analysis_results) >= 4, "Should store all analysis results"
    print("PASS Integration: All analyses stored and accessible")

# Run the test
test_unit_advanced_optimization_analyzer()

# %% [markdown]
"""
## Integration - Bringing High-Performance Kernels Together

### Kernel Composition and Performance Pipeline
"""

# %%
def test_unit_all():
    """Run comprehensive kernel module validation."""
    print("TEST Running all kernel unit tests...")

    # Core infrastructure tests
    test_unit_timing_infrastructure()
    print()

    # Matrix operation tests
    test_unit_cache_friendly_matmul()
    print()

    # Vectorization tests
    test_unit_vectorized_operations()
    print()

    # Parallel processing tests
    test_unit_parallel_processing()
    print()

    # Quantization tests
    test_unit_quantization_kernels()
    print()

    # Advanced analyzer tests
    test_unit_advanced_optimization_analyzer()
    print()

    print("PASS All kernel unit tests passed! High-performance kernels ready for deployment.")

# %% [markdown]
"""
## Production Context - Real-World Kernel Usage

### How Production ML Systems Use Optimized Kernels

Modern ML frameworks achieve their performance through sophisticated kernel optimization:

**PyTorch Kernel Architecture:**
```python
# High-level PyTorch operation
result = torch.matmul(A, B)

# Dispatches to optimized kernels based on:
# - Hardware: CPU (Intel MKL) vs GPU (cuBLAS/cuDNN)
# - Data type: float32, float16, bfloat16, int8
# - Tensor properties: size, stride, memory layout
# - Available optimizations: Tensor Cores, quantization
```

**Performance Optimization Stack:**
```
Application Level:     model(input)
Framework Level:       torch.matmul(A, B)
Dispatcher Level:      select_optimal_kernel(A, B, device)
Kernel Level:          optimized_matmul_cuda/cpu(A, B)
Hardware Level:        CUDA cores, Tensor cores, SIMD units
```

**Real-World Impact:**
- **Training Acceleration**: Optimized kernels enable training larger models in reasonable time
- **Inference Speed**: Fast kernels reduce serving latency and costs
- **Edge Deployment**: Quantized kernels enable deployment on mobile/IoT devices
- **Energy Efficiency**: Efficient kernels reduce data center power consumption

### Framework Integration Patterns

**Automatic Kernel Selection:**
```python
# Framework chooses optimal implementation
if tensor.is_cuda and tensor.dtype == torch.float16:
    return tensor_core_matmul(A, B)
elif tensor.is_cpu and has_avx512():
    return vectorized_cpu_matmul(A, B)
else:
    return fallback_matmul(A, B)
```

**Performance Profiling Integration:**
```python
# Built-in profiling like our analyzer
with torch.profiler.profile() as prof:
    result = model(input)

# Reveals which kernels are bottlenecks
prof.export_chrome_trace("trace.json")
```
"""

# %%
if __name__ == "__main__":
    test_unit_all()

# %% [markdown]
"""
## THINK ML Systems Thinking: Interactive Questions

Now that you've implemented high-performance computational kernels, let's explore the systems implications through hands-on analysis.
"""

# %% [markdown]
"""
### Question 1: Cache Hierarchy Optimization Analysis

**Context**: Your `cache_friendly_matmul` function uses blocking to improve cache locality. You measured different block sizes and saw varying performance characteristics.

**Reflection Question**: Analyze the cache behavior patterns in your implementation. When you tested block sizes of 32, 64, and 128, how did performance scale with memory hierarchy levels (L1/L2/L3 cache)? Design an adaptive blocking strategy that automatically selects optimal block sizes based on runtime cache analysis. How would you extend your approach to handle matrices that don't fit entirely in any cache level?

**Think about**:
- Cache line sizes and prefetching behavior
- Multi-level cache optimization strategies
- Memory bandwidth vs cache capacity trade-offs
- Production deployment across different CPU architectures
"""

# %% [markdown]
"""
### Question 2: Vectorization and Parallelization Interaction Analysis

**Context**: You implemented both SIMD vectorization (`vectorized_relu`) and multi-threading parallelization (`parallel_relu`). Your performance analysis showed different scaling characteristics.

**Reflection Question**: Examine the interaction between vectorization and parallelization in your implementations. How does SIMD vectorization within each thread affect the optimal number of worker threads? Analyze the memory bandwidth contention when multiple threads are performing vectorized operations simultaneously. Design a hybrid optimization strategy that balances SIMD width, thread count, and memory bandwidth for maximum throughput.

**Think about**:
- Memory bandwidth limitations with multiple vectorized threads
- NUMA topology effects on parallel vectorized operations
- Thread affinity and cache sharing between cores
- Optimal work distribution strategies for vectorized workloads
"""

# %% [markdown]
"""
### Question 3: Production Deployment Optimization Strategy

**Context**: Your `KernelOptimizationAnalyzer` generated a comprehensive optimization roadmap with prioritized improvements for production deployment.

**Reflection Question**: Based on your optimization analysis results, design a production deployment strategy for a real-time ML inference service. How would you adapt your kernel optimizations for different deployment scenarios: cloud instances with 32+ cores, edge devices with 4 cores and limited memory, and mobile devices with thermal constraints? Create a decision framework that automatically selects optimal kernel implementations based on runtime hardware detection and performance requirements.

**Think about**:
- Runtime performance monitoring and adaptation
- Thermal management and performance throttling
- Memory pressure and kernel selection strategies
- Fallback mechanisms for unsupported optimizations
- Continuous performance optimization in production
"""

# %% [markdown]
"""
## TARGET MODULE SUMMARY: Kernels

Congratulations! You've successfully implemented high-performance computational kernels that power modern ML systems!

### What You've Accomplished
PASS **High-Performance Implementation**: 200+ lines of optimized kernel code with cache blocking, vectorization, and parallelization
PASS **Advanced Optimization Analysis**: Comprehensive `KernelOptimizationAnalyzer` with multi-dimensional performance evaluation
PASS **Production-Ready Kernels**: Matrix multiplication, activation functions, and quantization kernels optimized for real-world deployment
PASS **Systems Integration**: Complete optimization pipeline from profiling through deployment recommendations
PASS **Performance Engineering**: Deep understanding of cache hierarchy, SIMD vectorization, and parallel processing trade-offs

### Key Learning Outcomes
- **Cache Optimization**: Implementing cache-friendly algorithms that minimize memory access latency
- **Vectorization Mastery**: Leveraging SIMD instructions for 4-16x performance improvements
- **Parallel Processing**: Understanding when parallelization helps vs creates overhead
- **Quantization Engineering**: Balancing accuracy vs performance for efficient deployment
- **Production Optimization**: Systematic approach to kernel optimization for real-world ML systems

### Mathematical Foundations Mastered
- **Cache-Friendly Algorithms**: O(n³/B) cache complexity through blocking techniques
- **SIMD Vectorization**: Processing 4-16 elements simultaneously with vector instructions
- **Parallel Scaling**: Amdahl's law and parallel efficiency analysis across worker counts
- **Quantization Mathematics**: Precision reduction with controlled accuracy degradation

### Professional Skills Developed
- **Performance Engineering**: Systematic optimization methodology from profiling to deployment
- **Systems Architecture**: Understanding hardware-software interface for ML acceleration
- **Production Deployment**: Optimization strategies for cloud, edge, and mobile environments
- **Kernel Development**: Building high-performance computational primitives that power ML frameworks

### Ready for Advanced Applications
Your kernel implementations now enable:
- **Real-Time Inference**: Optimized kernels for low-latency ML serving
- **Large-Scale Training**: High-performance operations for training large models
- **Edge Deployment**: Memory-efficient kernels for resource-constrained devices
- **Framework Development**: Understanding how PyTorch and TensorFlow achieve high performance

### Connection to Real ML Systems
Your implementation mirrors production systems:
- **PyTorch**: ATen library with CUDA kernels, Intel MKL integration, and automatic kernel selection
- **TensorFlow**: XLA compiler with hardware-specific optimizations and kernel fusion
- **Industry Practice**: Cache blocking, vectorization, and quantization are fundamental to all modern ML frameworks

### Next Steps
1. **Export your module**: `tito module complete 13_kernels`
2. **Validate integration**: `tito test --module kernels`
3. **Explore advanced optimizations**: GPU kernels, custom CUDA implementations
4. **Ready for Module 14**: Performance analysis and benchmarking systems

**Performance Engineering Mastery**: Your high-performance kernel implementations demonstrate deep understanding of how to optimize ML operations for production deployment - the foundation for building scalable ML infrastructure!
"""