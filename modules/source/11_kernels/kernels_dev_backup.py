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
# Module 11: Kernels - Hardware-Aware Optimization

Welcome to the Kernels module! This is where you'll learn to optimize ML operations for real hardware performance.

## Learning Goals
- Understand hardware optimization principles for ML systems
- Implement vectorized operations using SIMD capabilities
- Build cache-friendly algorithms with memory hierarchy awareness
- Create parallel processing implementations for multi-core systems
- Connect basic algorithms to production-level performance optimization

## Build → Use → Understand
1. **Build**: Implement hardware-aware optimization techniques from scratch
2. **Use**: Compare performance differences between basic and optimized implementations
3. **Understand**: How hardware characteristics drive optimization strategies
"""

# %% nbgrader={"grade": false, "grade_id": "kernels-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.kernels

import numpy as np
import time
import multiprocessing as mp
import sys
import os
from typing import Callable, Dict, Any, Optional, List, Tuple

# Import the basic matrix multiplication from Module 03
# This is the triple-loop implementation students built earlier
try:
    from tinytorch.core.layers import matmul_naive as matmul_from_module03
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_layers'))
    try:
        from layers_dev import matmul as matmul_from_module03
    except ImportError:
        # Fallback if we can't import, define it directly
        def matmul_from_module03(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            rows_a, cols_a = A.shape
            rows_b, cols_b = B.shape
            
            if cols_a != rows_b:
                raise ValueError(f"Cannot multiply matrices with shapes {A.shape} and {B.shape}")
            
            result = np.zeros((rows_a, cols_b))
            
            for i in range(rows_a):
                for j in range(cols_b):
                    for k in range(cols_a):
                        result[i, j] += A[i, k] * B[k, j]
            
            return result

# Import shared profiling utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from profiler import SimpleProfiler, profile_function

print("🚀 TinyTorch Kernels Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version.split()[0]}")
print(f"CPU count: {mp.cpu_count()}")
print("Ready to optimize ML systems for hardware!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/11_kernels/kernels_dev.py`  
**Building Side:** Code exports to `tinytorch.core.kernels`

```python
# Final package structure:
from tinytorch.core.kernels import matmul_vectorized, matmul_cache_optimized
from tinytorch.core.kernels import matmul_parallel  # Hardware-optimized operations
```

**Why this matters:**
- **Learning:** Understand optimization from first principles
- **Production:** Real ML systems need hardware-aware implementations
- **Performance:** Bridge between theory and practical efficiency
- **Foundation:** Optimization skills transfer to all ML system components
"""

# %% [markdown]
"""
## Step 1: Understanding Our Baseline - The Module 03 Implementation

In Module 03, you implemented matrix multiplication using three nested loops. Let's use that **exact same function** as our baseline for optimization:

```python
# From Module 03 (your original implementation):
def matmul(A, B):
    rows_a, cols_a = A.shape
    rows_b, cols_b = B.shape
    
    result = np.zeros((rows_a, cols_b))
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i, j] += A[i, k] * B[k, j]
    
    return result
```

This is our **baseline** - the implementation we'll optimize in this module.

### Why This Baseline Matters
- **Your own code**: You built this in Module 03, so you understand it completely
- **Clear performance reference**: See exactly how much faster optimized versions are
- **Real improvement**: Measure actual performance gains of your optimizations
- **Educational value**: Connect basic concepts to hardware-aware programming

### Performance Characteristics of the Baseline
- **Time complexity**: O(n³) for n×n matrices
- **Memory access**: Not cache-friendly - jumps around memory
- **Parallelization**: No parallel execution
- **Vectorization**: No SIMD optimizations

**Goal**: Take this basic implementation and make it hardware-aware!
"""

# %%
#| export  
def matmul_custom(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Baseline matrix multiplication using the exact same implementation from Module 03.
    
    This directly calls the matmul function you built in Module 03,
    showing the clear connection between modules and avoiding code duplication.
    """
    return matmul_from_module03(A, B)

# %% [markdown]
"""
## Step 2: Vectorized Operations - Leveraging SIMD

### What is Vectorization?
**Vectorization** means using Single Instruction, Multiple Data (SIMD) operations to process multiple data elements simultaneously. Modern CPUs can perform the same operation on multiple numbers at once.

### Why Vectorization Matters
- **SIMD Instructions**: Modern CPUs have 128-bit, 256-bit, or 512-bit registers
- **Parallel Arithmetic**: One instruction can operate on 4-16 numbers simultaneously
- **Automatic Optimization**: NumPy uses highly optimized BLAS libraries
- **Massive Speedup**: Often 10-100x faster than basic loops

### The NumPy Advantage
NumPy operations like `@` (matrix multiplication) automatically use:
- **Intel MKL**: Math Kernel Library with hand-optimized assembly
- **OpenBLAS**: Open-source optimized BLAS implementation
- **BLAS/LAPACK**: Industry-standard linear algebra routines

### Learning Connection
This is why production ML frameworks (PyTorch, TensorFlow) are built on optimized libraries rather than pure Python loops.
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-vectorized", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Vectorized matrix multiplication using NumPy's optimized operations.
    
    TODO: Implement vectorized matrix multiplication using NumPy's @ operator.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Validate that the matrices can be multiplied (A.shape[1] == B.shape[0])
    2. Use NumPy's @ operator for optimized matrix multiplication
    3. Return the result directly
    
    EXAMPLE:
    A = [[1, 2],     B = [[5, 6],     Result = [[19, 22],
         [3, 4]]          [7, 8]]               [43, 50]]
    
    IMPLEMENTATION HINTS:
    - Check shape compatibility with A.shape[1] == B.shape[0]
    - Use A @ B for the actual multiplication
    - NumPy handles all the SIMD optimization automatically
    - This should be much faster than the triple-loop version
    
    LEARNING CONNECTIONS:
    - This uses the same optimized libraries as PyTorch and TensorFlow
    - SIMD operations process multiple numbers simultaneously
    - Why you should use library functions instead of writing loops
    """
    ### BEGIN SOLUTION
    # Validate matrix dimensions
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply matrices with shapes {A.shape} and {B.shape}")
    
    # Use NumPy's optimized matrix multiplication
    return A @ B
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Vectorized Implementation

Once you implement the `matmul_vectorized` function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-vectorized-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_vectorized_matrix_multiplication():
    """Test vectorized matrix multiplication implementation"""
    print("🔬 Unit Test: Vectorized Matrix Multiplication...")

    # Test simple 2x2 case
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul_vectorized(A, B)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Vectorized multiplication failed: expected {expected}, got {result}"
    
    # Compare with baseline
    baseline_result = matmul_custom(A, B)
    assert np.allclose(result, baseline_result), f"Doesn't match baseline: got {result}, expected {baseline_result}"

    # Test different shapes
    A2 = np.array([[1, 2, 3]], dtype=np.float32)  # 1x3
    B2 = np.array([[4], [5], [6]], dtype=np.float32)  # 3x1
    result2 = matmul_vectorized(A2, B2)
    expected2 = np.array([[32]], dtype=np.float32)  # 1*4 + 2*5 + 3*6 = 32
    
    assert np.allclose(result2, expected2), f"Different shapes failed: got {result2}, expected {expected2}"
    
    print("✅ Vectorized matrix multiplication works correctly!")

# Run the test
test_vectorized_matrix_multiplication()

# %% [markdown]
"""
## Step 3: Cache-Optimized Implementation - Memory Hierarchy Awareness

### What is Cache Optimization?
**Cache optimization** means organizing memory accesses to work efficiently with the CPU's memory hierarchy. Modern processors have multiple levels of cache that are much faster than main memory.

### Memory Hierarchy
- **L1 Cache**: ~1 cycle, ~32KB, per-core
- **L2 Cache**: ~10 cycles, ~256KB, per-core  
- **L3 Cache**: ~50 cycles, ~8MB, shared
- **Main Memory**: ~200 cycles, GB-scale

### Cache-Friendly Strategy: Blocking
**Blocking** (or tiling) divides large matrices into smaller blocks that fit in cache:
- Process one block at a time
- Reuse data while it's still in cache
- Reduce expensive memory fetches
- Better performance for large matrices

### Why This Matters
- **Cache misses are expensive**: 200x slower than cache hits
- **Locality of reference**: Access nearby data together
- **Real systems**: Production ML uses cache-aware algorithms
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-cache", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul_cache_optimized(A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Cache-optimized matrix multiplication using blocked algorithm.
    
    TODO: Implement cache-friendly matrix multiplication with blocking.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Validate matrix dimensions for multiplication
    2. Get matrix dimensions: m, n, p
    3. Initialize result matrix with zeros
    4. Use three nested loops over blocks (not elements):
       - i_block: iterate through row blocks of A
       - j_block: iterate through column blocks of B
       - k_block: iterate through shared dimension blocks
    5. For each block combination, extract submatrices and multiply
    6. Add the block result to the appropriate section of the output matrix
    
    EXAMPLE WALKTHROUGH:
    For 4x4 matrices with block_size=2:
    - Divide into 2x2 blocks
    - Multiply corresponding blocks
    - Accumulate results in output matrix
    
    IMPLEMENTATION HINTS:
    - Use range(0, dimension, block_size) for block iteration
    - Extract blocks: A_block = A[i:i_end, k:k_end]
    - Use @ operator for block multiplication
    - Accumulate: result[i:i_end, j:j_end] += A_block @ B_block
    - Handle edge cases where blocks don't divide evenly
    
    LEARNING CONNECTIONS:
    - This is how BLAS libraries optimize for cache hierarchy
    - Block size should match cache size for optimal performance
    - Cache-aware algorithms are essential for large-scale ML
    """
    ### BEGIN SOLUTION
    # Validate matrix dimensions
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply matrices with shapes {A.shape} and {B.shape}")
    
    m, n = A.shape
    n2, p = B.shape
    
    # Initialize result matrix
    result = np.zeros((m, p))
    
    # Blocked matrix multiplication
    for i in range(0, m, block_size):
        for j in range(0, p, block_size):
            for k in range(0, n, block_size):
                # Calculate block boundaries
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, p)
                k_end = min(k + block_size, n)
                
                # Extract blocks
                A_block = A[i:i_end, k:k_end]
                B_block = B[k:k_end, j:j_end]
                
                # Multiply blocks and accumulate
                result[i:i_end, j:j_end] += A_block @ B_block
    
    return result
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Cache-Optimized Implementation

Once you implement the `matmul_cache_optimized` function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-cache-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_cache_optimized_matrix_multiplication():
    """Test cache-optimized matrix multiplication implementation"""
    print("🔬 Unit Test: Cache-Optimized Matrix Multiplication...")

    # Test simple 2x2 case
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul_cache_optimized(A, B, block_size=2)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Cache-optimized multiplication failed: expected {expected}, got {result}"
    
    # Compare with baseline
    baseline_result = matmul_custom(A, B)
    assert np.allclose(result, baseline_result), f"Doesn't match baseline: got {result}, expected {baseline_result}"

    # Test larger matrix with different block sizes
    A2 = np.random.randn(8, 6).astype(np.float32)
    B2 = np.random.randn(6, 10).astype(np.float32)
    
    result_block2 = matmul_cache_optimized(A2, B2, block_size=2)
    result_block4 = matmul_cache_optimized(A2, B2, block_size=4)
    expected_large = A2 @ B2
    
    assert np.allclose(result_block2, expected_large), "Block size 2 failed on larger matrix"
    assert np.allclose(result_block4, expected_large), "Block size 4 failed on larger matrix"
    
    print("✅ Cache-optimized matrix multiplication works correctly!")

# Run the test
test_cache_optimized_matrix_multiplication()

# %% [markdown]
"""
## Step 4: Parallel Processing - Multi-Core Utilization

### What is Parallel Processing?
**Parallel processing** means distributing computation across multiple CPU cores to reduce overall execution time. Modern processors have multiple cores that can work simultaneously.

### Why Parallelization Matters
- **Multi-core CPUs**: Modern systems have 4-16+ cores
- **Independent operations**: Matrix rows can be computed independently  
- **Linear speedup potential**: 4 cores → ~4x faster (ideally)
- **Real-world necessity**: Production systems must use all available cores

### Parallelization Strategy: Row-wise Distribution
- Divide matrix rows among available cores
- Each core computes its assigned rows independently
- No communication needed between cores during computation
- Simple and effective for matrix multiplication

### Learning Connection
This demonstrates the foundations of distributed computing and parallel algorithms used in modern ML training.
"""

# %% nbgrader={"grade": false, "grade_id": "matmul-parallel", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def matmul_parallel(A: np.ndarray, B: np.ndarray, num_processes: Optional[int] = None) -> np.ndarray:
    """
    Parallel matrix multiplication using multiple CPU cores.
    
    TODO: Implement parallel matrix multiplication with row-wise distribution.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Validate matrix dimensions for multiplication
    2. Set default number of processes to CPU count if not specified
    3. For very small matrices, use single-threaded approach (overhead not worth it)
    4. Calculate chunk size: divide rows among processes
    5. Process chunks sequentially (simulating parallel processing)
    6. Combine results using np.vstack()
    
    EXAMPLE WALKTHROUGH:
    For 8x8 matrix with 4 cores:
    - Core 1: processes rows 0-1  
    - Core 2: processes rows 2-3
    - Core 3: processes rows 4-5
    - Core 4: processes rows 6-7
    
    IMPLEMENTATION HINTS:
    - Check if A.shape[0] < 20, use A @ B directly
    - Calculate chunk_size = max(1, A.shape[0] // num_processes)
    - Use range(0, A.shape[0], chunk_size) to iterate through chunks
    - For each chunk: A_chunk = A[i:end_i], result_chunk = A_chunk @ B
    - Collect all chunks in a list, then np.vstack(results)
    
    LEARNING CONNECTIONS:
    - This is how distributed training works across multiple GPUs/machines
    - Row-wise parallelization is embarrassingly parallel
    - Real parallel processing would use multiprocessing.Pool
    """
    ### BEGIN SOLUTION
    # Validate dimensions
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions incompatible: A is {A.shape}, B is {B.shape}")
    
    # Default to number of CPU cores
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # For very small matrices, use single-threaded approach
    if A.shape[0] < 20:
        return A @ B
    
    # Simple row-wise parallel processing simulation
    # (In real implementation, would use multiprocessing.Pool)
    chunk_size = max(1, A.shape[0] // num_processes)
    results = []
    
    for i in range(0, A.shape[0], chunk_size):
        end_i = min(i + chunk_size, A.shape[0])
        # Process this chunk of rows
        A_chunk = A[i:end_i]
        chunk_result = A_chunk @ B
        results.append(chunk_result)
    
    # Combine results
    return np.vstack(results)
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Parallel Implementation

Once you implement the `matmul_parallel` function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-parallel-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_parallel_matrix_multiplication():
    """Test parallel matrix multiplication implementation"""
    print("🔬 Unit Test: Parallel Matrix Multiplication...")

    # Test simple 2x2 case
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result = matmul_parallel(A, B, num_processes=2)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result, expected), f"Parallel multiplication failed: expected {expected}, got {result}"
    
    # Compare with baseline
    baseline_result = matmul_custom(A, B)
    assert np.allclose(result, baseline_result), f"Doesn't match baseline: got {result}, expected {baseline_result}"

    # Test larger matrix
    A2 = np.random.randn(24, 16).astype(np.float32)
    B2 = np.random.randn(16, 20).astype(np.float32)
    
    result_parallel = matmul_parallel(A2, B2, num_processes=4)
    expected_large = A2 @ B2
    
    assert np.allclose(result_parallel, expected_large), "Parallel processing failed on larger matrix"
    
    # Test different number of processes
    result_parallel2 = matmul_parallel(A2, B2, num_processes=2)
    assert np.allclose(result_parallel2, expected_large), "Different process count failed"
    
    print("✅ Parallel matrix multiplication works correctly!")

# Run the test
test_parallel_matrix_multiplication()

# %% [markdown]
"""
## 🧪 Unit Test: All Matrix Multiplication Implementations

**This is a unit test** - it tests all our matrix multiplication implementations for correctness.
"""

# %%
print("### 🧪 Unit Test: Matrix Multiplication Implementations")
print("**This is a unit test** - it tests our matrix multiplication implementations.")

# Test basic functionality
A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)

print("🔬 Unit Test: Matrix Multiplication...")

# Test our implementations
result_custom = matmul_custom(A, B)
result_vectorized = matmul_vectorized(A, B)
result_cache_optimized = matmul_cache_optimized(A, B)
result_parallel = matmul_parallel(A, B)

# Expected result
expected = np.array([[19, 22], [43, 50]], dtype=np.float32)

print(f"✅ Custom result correct: {np.allclose(result_custom, expected)}")
print(f"✅ Vectorized result correct: {np.allclose(result_vectorized, expected)}")
print(f"✅ Cache-optimized result correct: {np.allclose(result_cache_optimized, expected)}")
print(f"✅ Parallel result correct: {np.allclose(result_parallel, expected)}")

print("📈 Progress: All implementations work correctly ✓")

# %% [markdown]
"""
## Performance Comparison: Your Optimizations in Action

Now let's see how much faster your optimized implementations are compared to the Module 03 baseline:
"""

# %%
print("🔬 Testing performance comparison...")
print("Students can collect their own performance data:")

# Create profiler for measuring performance
profiler = SimpleProfiler(track_memory=True, track_cpu=True)

# Test matrices
A = np.random.randn(50, 50).astype(np.float32)
B = np.random.randn(50, 50).astype(np.float32)

# Profile each implementation
basic_result = profiler.profile(matmul_custom, A, B, name="Module 03 Baseline")
vectorized_result = profiler.profile(matmul_vectorized, A, B, name="Vectorized")
cache_result = profiler.profile(matmul_cache_optimized, A, B, name="Cache-Optimized")
parallel_result = profiler.profile(matmul_parallel, A, B, name="Parallel")

# Students analyze results themselves
print(f"✓ Module 03 Baseline: {basic_result['wall_time']:.4f}s")
print(f"✓ Vectorized: {vectorized_result['wall_time']:.4f}s")
print(f"✓ Cache-Optimized: {cache_result['wall_time']:.4f}s")
print(f"✓ Parallel: {parallel_result['wall_time']:.4f}s")

# Calculate speedups (students learn to do this themselves)
if basic_result['wall_time'] > 0:
    vec_speedup = basic_result['wall_time'] / vectorized_result['wall_time']
    cache_speedup = basic_result['wall_time'] / cache_result['wall_time'] 
    parallel_speedup = basic_result['wall_time'] / parallel_result['wall_time']
    
    print(f"\n📊 Performance Summary:")
    print(f"🏆 Speedups vs Module 03 Baseline:")
    print(f"   Vectorized: {vec_speedup:.1f}x faster")
    print(f"   Cache-Optimized: {cache_speedup:.1f}x faster")
    print(f"   Parallel: {parallel_speedup:.1f}x faster")

print("✅ Performance comparison works")
print("📈 Progress: Kernels Module ✓")

print("\n🎯 Module 11: Kernels Summary:")
print("- Built hardware-aware optimizations from scratch")
print("- Implemented vectorized, cache-optimized, and parallel algorithms")
print("- Learned to profile and compare performance systematically")
print("- Connected basic algorithms to production optimization strategies")
print("- Ready for comprehensive benchmarking and real-world optimization!")

print("\n🎉 Module 11: Kernels - Complete!")
print("Ready for Module 12: Benchmarking!")

# %% [markdown]
"""
## 🔍 Profiling and Analysis

You've learned to use the shared profiler utility to measure individual functions. Here are examples of how to collect data and make your own comparisons:

### Basic Performance Analysis
```python
from utils.profiler import SimpleProfiler, profile_function

# Create profiler
profiler = SimpleProfiler(track_memory=True, track_cpu=True)

# Profile individual functions
A = np.random.randn(100, 100)
B = np.random.randn(100, 100)

basic_result = profiler.profile(matmul_custom, A, B, name="Basic")
optimized_result = profiler.profile(matmul_vectorized, A, B, name="Optimized")

# Students analyze the results themselves
print(f"Basic: {basic_result['wall_time']:.4f}s")
print(f"Optimized: {optimized_result['wall_time']:.4f}s")
speedup = basic_result['wall_time'] / optimized_result['wall_time']
print(f"Speedup: {speedup:.1f}x")
```

### Memory and CPU Analysis
```python
# Profile with detailed output
result = profiler.profile(matmul_cache_optimized, A, B, name="Cache-Optimized")
profiler.print_result(result, show_details=True)

# Access specific metrics
wall_time = result['wall_time']
memory_used = result['memory_delta_mb']
cpu_efficiency = result['cpu_efficiency']
```

### Key Optimization Insights
- **Vectorization**: Massive speedups from SIMD operations
- **Cache optimization**: Better memory access patterns for large matrices
- **Parallelization**: Utilizing multiple CPU cores effectively
- **Hardware awareness**: Understanding system architecture drives optimization

### Educational Approach
Students learn to:
1. **Measure**: Profile individual functions with comprehensive metrics
2. **Collect**: Gather data from multiple implementations
3. **Compare**: Calculate speedups and efficiency differences themselves
4. **Analyze**: Understand what the metrics mean for optimization

This teaches proper benchmarking methodology and critical thinking about performance!
"""

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
## 🎯 Module Summary: Hardware-Aware Optimization Mastery!

Congratulations! You've successfully implemented hardware-aware optimization techniques for ML systems:

### ✅ What You've Built
- **Vectorized Operations**: Leveraging SIMD instructions for massive speedups
- **Cache-Optimized Algorithms**: Memory hierarchy-aware blocked implementations
- **Parallel Processing**: Multi-core utilization with row-wise distribution
- **Performance Profiling**: Systematic measurement and analysis techniques

### ✅ Key Learning Outcomes
- **Hardware Understanding**: How CPU architecture drives optimization strategies
- **Implementation Skills**: Built optimizations from scratch with detailed guidance
- **Performance Analysis**: Learned to measure and compare implementations systematically
- **Real-world Connection**: Connected basic algorithms to production optimization

### ✅ Optimization Mastery
- **SIMD Vectorization**: Using hardware parallel arithmetic units
- **Memory Hierarchy**: Organizing computation for cache efficiency
- **Parallel Computing**: Distributing work across multiple cores
- **Profiling Methodology**: Measuring performance correctly and fairly

### ✅ Professional Skills Developed
- **Systems thinking**: Understanding hardware-software interaction
- **Optimization mindset**: Identifying performance bottlenecks and solutions
- **Benchmarking skills**: Fair comparison and analysis techniques
- **Production awareness**: How real ML systems achieve high performance

### ✅ Ready for Next Steps
Your optimization skills are now ready for:
- **Module 12**: Comprehensive benchmarking and performance analysis
- **Production Systems**: Understanding how PyTorch, TensorFlow optimize operations
- **Custom Kernels**: Writing specialized operations for specific hardware
- **Distributed Computing**: Scaling optimizations across multiple machines

### 🔗 Connection to Real ML Systems
Your implementations demonstrate the foundations of:
- **BLAS Libraries**: Intel MKL, OpenBLAS, cuBLAS optimization strategies
- **Framework Internals**: How PyTorch and TensorFlow achieve performance
- **Hardware Acceleration**: GPU kernels, TPU operations, specialized chips
- **Production Deployment**: Optimizing ML inference for real-world constraints

### 🎯 The Power of Hardware-Aware Programming
You've learned the essential mindset for high-performance computing:
- **Know your hardware**: Understanding system architecture guides optimization
- **Profile everything**: Measurement drives optimization decisions  
- **Optimize systematically**: Vectorization → Memory → Parallelization
- **Think like production**: Real systems demand hardware-aware implementations

### 🧠 Optimization Insights
- **Why optimization matters**: Performance gaps can be 1000x+ between naive and optimized code
- **Hardware evolution**: Modern ML requires understanding of specialized accelerators
- **System design**: Optimization considerations influence software architecture
- **Continuous learning**: Hardware advances require constant learning and adaptation

**Next Module**: Benchmarking - Comprehensive performance analysis and systematic optimization!

You've built the optimizations. Now let's learn to analyze and benchmark them like production ML engineers!
""" 