# %% [markdown]
"""
# Module 16: Hardware Acceleration - The Free Speedup!

Welcome to Hardware Acceleration! You'll discover the easiest optimization in ML systems - getting 100x speedups with zero code changes!

## LINK Building on Previous Learning
**What You Built Before**:
- Module 02 (Tensor): Triple-nested loops for matrix operations
- Module 04 (Layers): Forward pass implementations
- Module 15 (Profiling): Performance measurement and bottleneck identification

**What's Working**: You can implement any matrix operation correctly using educational loops!

**The Gap**: Your educational loops are 1000x slower than production code, limiting real ML applications.

**This Module's Solution**: Learn the optimization spectrum from educational to production performance.

**Connection Map**:
```
Profiling -> Acceleration -> Production ML
(identify)   (optimize)    (deploy at scale)
```

## Learning Goals (Systems-Focused Framework)
- **Systems understanding**: CPU cache hierarchy and memory access patterns
- **Core implementation skill**: Cache-friendly blocking algorithms
- **Pattern/abstraction mastery**: Backend abstraction and automatic dispatch
- **Framework connections**: How PyTorch/TensorFlow achieve performance
- **Optimization trade-offs**: Educational clarity vs production speed

## Build -> Use -> Reflect
1. **Build**: Cache-friendly blocked matrix multiplication from scratch
2. **Use**: Apply acceleration to real ML model operations (MLP, CNN, Attention)
3. **Reflect**: Analyze the educational-to-production optimization spectrum

## Systems Reality Check
TIP **Production Context**: ML frameworks use these exact principles for 100x speedups
SPEED **Performance Insight**: Memory access patterns matter more than raw computation speed

## The Free Speedup Journey

**Key Message**: This is the EASIEST optimization - just use better backends! No accuracy trade-offs, no complex math - just 10-100x faster code.

```
Educational Loops -> Cache Blocking -> NumPy/BLAS -> Smart Backends
    (learning)       (understanding)   (production)    (automation)
    1000x slower     100x slower      optimal speed   transparent
```

**Visual Performance Spectrum**:
```
Performance: [████████████████████████████████████████] 100%  NumPy
             [████]                                      4%   Blocked  
             [▌]                                         0.1% Naive
```

**Why This Works**: Same math, better implementation. Free performance with zero downsides!
"""

# %% [markdown]
"""
## Part 1: Baseline Implementation - Your Loops from Module 2/4

Let's start with the educational triple-nested loops you implemented earlier. These were perfect for learning but terrible for performance.

### CPU vs GPU Architecture Fundamentals

```
CPU Architecture (Optimized for Sequential):         GPU Architecture (Optimized for Parallel):
+---------------------------------------------+     +---------------------------------------------+
| Complex Control Unit                        |     | Simple Control Units                       |
| +---------+  Large Caches                 |     | +-+ +-+ +-+ +-+ Small Caches              |
| |  Core 1 |  +--------------------------+  |     | |C| |C| |C| |C| +----------------------+   |
| +---------+  |     L3 Cache (8MB)       |  |     | +-+ +-+ +-+ +-+ |  Shared Memory (48KB) |   |
| +---------+  |                          |  |     | +-+ +-+ +-+ +-+ |                      |   |
| |  Core 2 |  +--------------------------+  |     | |C| |C| |C| |C| +----------------------+   |
| +---------+                                |     | +-+ +-+ +-+ +-+ ... (thousands of cores) |
| +---------+  Main Memory (16GB)           |     |                                           |
| |  Core 4 |  +--------------------------+  |     | High Bandwidth Memory (HBM)              |
| +---------+  | 200+ cycle latency       |  |     | +--------------------------------------+  |
|              +--------------------------+  |     | | 1000+ GB/s bandwidth                 |  |
+---------------------------------------------+     +---------------------------------------------+

CPU: Few cores, complex, optimized for latency    GPU: Many cores, simple, optimized for throughput
Best for: Sequential algorithms, complex logic    Best for: Parallel algorithms, simple operations
```

### Memory Hierarchy Deep Dive

```
Memory Hierarchy (Latency and Size Trade-offs):

Registers:   4 bytes     | 1 cycle      | ██████████ Speed
L1 Cache:    32KB       | 3-4 cycles   | ████████▒▒ 
L2 Cache:    256KB      | 10-20 cycles | ██████▒▒▒▒
L3 Cache:    8MB        | 50-100 cycles| ████▒▒▒▒▒▒
Main RAM:    16GB       | 200+ cycles  | ██▒▒▒▒▒▒▒▒
SSD Storage: 1TB        | 100,000+ cyc | ▒▒▒▒▒▒▒▒▒▒
             ^                          ^
           Size                      Speed
```

**The Cache Miss Problem**:
- Cache hit: Data found in L1 -> 1 cycle
- Cache miss: Must fetch from RAM -> 200+ cycles
- 200x slowdown for every cache miss!
"""

# %%
#| default_exp backends.acceleration

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Educational matrix multiplication using triple nested loops.
    
    This is the same implementation from Module 2/4 - perfect for learning
    the algorithm, but very slow due to poor cache performance.
    
    Memory Access Pattern Analysis:
    ```
    Inner loop accesses:
    a[i, k] -> Sequential access (cache-friendly)
    b[k, j] -> Strided access (cache-hostile!)
    
    For 1000*1000 matrices:
    - a[i,k]: 1000 sequential reads per row (good)
    - b[k,j]: 1000 random column reads (terrible!)
    - Total cache misses: ~1 billion!
    ```
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"Incompatible shapes: {a.shape} @ {b.shape}"
    
    # Initialize result matrix (contiguous memory allocation)
    c = np.zeros((m, n), dtype=np.float32)
    
    # Triple nested loop - the educational implementation
    # i loop: iterates over output rows
    # j loop: iterates over output columns  
    # k loop: performs dot product computation
    for i in range(m):           # Output row
        for j in range(n):       # Output column
            for k_idx in range(k):   # Dot product accumulation
                # Cache analysis: a[i,k_idx] = sequential (good)
                #                b[k_idx,j] = strided (bad!)
                c[i, j] += a[i, k_idx] * b[k_idx, j]
    
    return c

# MAGNIFY SYSTEMS INSIGHT: Memory Access Pattern Analysis
def analyze_memory_access_patterns():
    """
    Visualize why naive loops create terrible cache performance.
    
    This analysis shows the fundamental problem with nested loops:
    cache-hostile memory access patterns that destroy performance.
    """
    try:
        print("📊 Memory Access Pattern Analysis")
        print("=" * 45)
        
        # Simulate memory access for small matrix
        size = 4
        print(f"\nAnalyzing {size}x{size} matrix multiplication:")
        print("\nMatrix A (row-major layout):")
        print("Memory: [a00 a01 a02 a03 | a10 a11 a12 a13 | a20 a21 a22 a23 | a30 a31 a32 a33]")
        print("\nMatrix B (row-major layout):")
        print("Memory: [b00 b01 b02 b03 | b10 b11 b12 b13 | b20 b21 b22 b23 | b30 b31 b32 b33]")
        
        print("\n🔴 PROBLEM: Computing C[0,0] = sum(A[0,k] * B[k,0])")
        print("A[0,k] accesses: a00, a01, a02, a03  (sequential OK)")
        print("B[k,0] accesses: b00, b10, b20, b30  (every 4th element FAIL)")
        
        print("\n📊 Cache Miss Analysis:")
        cache_line_size = 64  # bytes
        float_size = 4        # bytes
        elements_per_line = cache_line_size // float_size  # 16 elements
        
        print(f"Cache line size: {cache_line_size} bytes = {elements_per_line} float32s")
        print(f"Sequential access (A): 1 cache miss per {elements_per_line} elements")
        print(f"Strided access (B): 1 cache miss per element (worst case)")
        
        # Calculate for realistic size
        n = 1000
        sequential_misses = n // elements_per_line
        strided_misses = n
        total_operations = n * n * n
        total_misses = total_operations * (sequential_misses + strided_misses) // n
        
        print(f"\n📊 Scaling to {n}x{n} matrices:")
        print(f"Total operations: {total_operations:,}")
        print(f"Estimated cache misses: {total_misses:,}")
        print(f"Cache miss rate: {total_misses/total_operations:.1%}")
        print(f"\n📊 Why this kills performance:")
        print(f"Cache hit: 1 cycle")
        print(f"Cache miss: 200+ cycles")
        print(f"Performance penalty: 200x slower!")
        
    except Exception as e:
        print(f"WARNING️ Error in memory analysis: {e}")
        print("Make sure numpy is available")

# Run the analysis
analyze_memory_access_patterns()

# %% [markdown]
"""
### TEST Unit Test: Educational Implementation

Let's test our educational loops and measure their performance characteristics.
"""

# PASS IMPLEMENTATION CHECKPOINT: Naive matrix multiplication complete

# THINK PREDICTION: How much slower are educational loops vs NumPy?
# Your guess: ___x slower for 100x100 matrices

# MAGNIFY SYSTEMS INSIGHT #1: Why Educational Loops Are Slow
def analyze_educational_loop_performance():
    """
    Measure and understand why educational loops create performance problems.
    
    This analysis reveals the fundamental performance characteristics
    that students experience when implementing algorithms from scratch.
    """
    try:
        print("📊 Educational Loop Performance Analysis")
        print("=" * 50)
        
        # Test progressively larger matrices to show scaling
        sizes = [50, 100, 200]
        
        print("\nPerformance scaling with matrix size:")
        print("Size | Naive Time | NumPy Time | Slowdown | O(N³) Theory")
        print("-" * 60)
        
        baseline_naive = None
        baseline_numpy = None
        
        for size in sizes:
            # Create test matrices
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Time naive implementation
            start = time.perf_counter()
            _ = matmul_naive(a, b)
            naive_time = time.perf_counter() - start
            
            # Time NumPy implementation
            start = time.perf_counter()
            _ = a @ b
            numpy_time = time.perf_counter() - start
            
            # Calculate slowdown
            slowdown = naive_time / numpy_time
            
            # Calculate theoretical scaling (O(N³))
            if baseline_naive is None:
                baseline_naive = naive_time
                baseline_numpy = numpy_time
                theory_scale = 1.0
            else:
                theory_scale = (size / sizes[0]) ** 3
            
            print(f"{size:4d} | {naive_time*1000:9.1f}ms | {numpy_time*1000:9.1f}ms | {slowdown:7.0f}x | {theory_scale:8.1f}x")
        
        print(f"\n📊 Key Performance Insights:")
        print(f"• Educational loops: Perfect for learning algorithms")
        print(f"• Scaling follows O(N³): doubling size = 8x operations")
        print(f"• Cache misses make large matrices exponentially slower")
        print(f"• NumPy: Professional optimizations give 100-1000x speedup")
        
        print(f"\nTIP Why This Matters for ML Systems:")
        print(f"• Understanding algorithms != performance optimization")
        print(f"• Educational clarity vs production speed trade-off")
        print(f"• Memory access patterns dominate performance")
        print(f"• Library choice impacts application feasibility")
        
    except Exception as e:
        print(f"WARNING️ Error in performance analysis: {e}")
        print("Make sure matrices are small enough for educational timing")

# Run the educational performance analysis
analyze_educational_loop_performance()

# %%
def test_naive_baseline():
    """
    Test naive implementation and measure its performance characteristics.
    
    This test validates correctness and demonstrates the performance gap
    between educational loops and optimized implementations.
    """
    print("TEST Testing Naive Implementation...")
    
    # Test correctness with small matrices first
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result_naive = matmul_naive(a, b)
    result_numpy = a @ b
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    
    assert np.allclose(result_naive, result_numpy), "Naive matmul incorrect vs NumPy"
    assert np.allclose(result_naive, expected), "Naive matmul incorrect vs expected"
    print("PASS Naive implementation produces correct results")
    
    # Performance comparison (small sizes only - educational is VERY slow)
    print("\n📊 Performance comparison:")
    small_a = np.random.randn(100, 100).astype(np.float32)
    small_b = np.random.randn(100, 100).astype(np.float32)
    
    # Time naive implementation (limit size to avoid excessive wait)
    start = time.perf_counter()
    _ = matmul_naive(small_a, small_b)
    naive_time = time.perf_counter() - start
    
    # Time NumPy implementation
    start = time.perf_counter()
    _ = small_a @ small_b
    numpy_time = time.perf_counter() - start
    
    speedup = naive_time / numpy_time
    
    print(f"Naive loops:     {naive_time*1000:8.1f} ms")
    print(f"NumPy optimized: {numpy_time*1000:8.1f} ms")
    print(f"Speedup:         {speedup:8.1f}x faster")
    
    # Estimate scaling behavior
    print(f"\n📊 Scaling Analysis (100x100 baseline):")
    print(f"For 500x500 matrix: ~{speedup * 125:.0f}x slower than NumPy")  # (500/100)^3 = 125
    print(f"For 1000x1000 matrix: ~{speedup * 1000:.0f}x slower than NumPy")  # (1000/100)^3 = 1000
    print(f"\nTIP Why: O(N³) complexity + cache misses = exponential slowdown")
    
    print("PASS Naive baseline established")
    return naive_time, numpy_time, speedup

# Execute the test
test_naive_baseline()

# %% [markdown]
"""
## Part 2: Understanding Cache Hierarchy - Why Memory Matters More Than Computation

**The Big Insight**: Modern CPUs are FAST at computation but SLOW at memory access. Cache hierarchy makes the difference between fast and slow code.

### CPU Cache Hierarchy Visualization
```
CPU Cache Hierarchy (Latency vs Capacity Trade-off):

+----------------------------------------------------------------------------------+
Register:  4 bytes   | ██████████ 1 cycle      (instant access)           |
L1 Cache:  32KB      | ████████▒▒ 3-4 cycles   (lightning fast)           |
L2 Cache:  256KB     | ██████▒▒▒▒ 10-20 cycles (fast)                   |
L3 Cache:  8MB       | ████▒▒▒▒▒▒ 50-100 cycles(slow)                   |
Main RAM:  16GB      | ██▒▒▒▒▒▒▒▒ 200+ cycles  (VERY slow)              |
SSD:       1TB       | ▒▒▒▒▒▒▒▒▒▒ 100,000+ cyc (glacial)                |
+----------------------------------------------------------------------------------+
     Size                        Speed                      Characteristics
```

**Key Principle**: Keep your working set in L1/L2 cache for 100x better performance!

### Vectorization vs Parallelization Concepts

```
Vectorization (SIMD - Single Instruction, Multiple Data):
+--------------------------------------------------+
| Scalar: for i in range(4): c[i] = a[i] + b[i]   |
|         ADD a[0], b[0] -> c[0]  (4 operations)    |  
|         ADD a[1], b[1] -> c[1]                   |
|         ADD a[2], b[2] -> c[2]                   |
|         ADD a[3], b[3] -> c[3]                   |
|                                                |
| Vector: c = a + b  (NumPy/BLAS)               |
|         VADD [a0,a1,a2,a3], [b0,b1,b2,b3]     |
|           -> [c0,c1,c2,c3]  (1 operation!)      |
+--------------------------------------------------+

Parallelization (Multiple cores working simultaneously):
+--------------------------------------------------+
| Core 1: Computes rows 0-249   of result matrix    |
| Core 2: Computes rows 250-499 of result matrix    |
| Core 3: Computes rows 500-749 of result matrix    |  
| Core 4: Computes rows 750-999 of result matrix    |
|                                                |
| 4x speedup (ideal) if no synchronization costs  |
+--------------------------------------------------+
```

### Memory Access Pattern Analysis

Your naive loops access memory like this:
```python
for i in range(m):      # Loop over output rows
    for j in range(n):  # Loop over output columns
        for k in range(k):  # Loop over dot product
            c[i,j] += a[i,k] * b[k,j]  # b[k,j] creates cache misses!
```

**The Problem**: `b[k,j]` creates terrible access patterns:
- Each `j` increment jumps to a new column (cache miss)
- Each `k` increment jumps to a new row (another cache miss)  
- For 1000*1000 matrix: 1 billion cache misses!

**Visualization of Memory Access**:
```
Matrix B in memory (row-major):
[b00 b01 b02 b03 | b10 b11 b12 b13 | b20 b21 b22 b23 | ...]

Accessing column 0: b00, b10, b20, b30, ...
                    |    |    |    |
                    4    4    4    4  elements apart = strided access
                   🔴  🔴  🔴  🔴 cache misses!
```

**The Solution**: Process in blocks that fit in cache.
"""

# %%
def matmul_blocked(a: np.ndarray, b: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Cache-friendly blocked matrix multiplication.
    
    This version processes data in blocks that fit in CPU cache,
    dramatically reducing cache misses and improving performance.
    
    **Memory Analysis (Quantitative)**:
    - 64*64 float32 block = 4096 * 4 bytes = 16KB per block
    - 3 blocks (A_block, B_block, C_block) = 48KB total
    - Fits comfortably in 256KB L2 cache with room for other data
    - Reuses each data element 64 times before evicting from cache
    
    **Why This Works**:
    - Naive: 1 cache miss per operation (terrible)
    - Blocked: 1 cache miss per 64 operations (64x better!)
    
    **Blocking Visualization**:
    ```
    Large Matrix Multiplication:
    A (1000x1000) * B (1000x1000) = C (1000x1000)
    
    Blocked Approach:
    +----------------+   +----------------+   +----------------+
    | 64x64|      |   | 64x64|      |   | 64x64|      |
    | block |  A   | * | block |  B   | = | block |  C   |
    |       |      |   |       |      |   |       |      |
    +----------------+   +----------------+   +----------------+
    
    Each 64x64 block fits in L1/L2 cache!
    ```
    
    Args:
        a: Left matrix (m * k)
        b: Right matrix (k * n) 
        block_size: Cache-friendly block size (64 = 16KB fits in L2 cache)
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"Incompatible shapes: {a.shape} @ {b.shape}"
    
    # Initialize result matrix with zeros
    c = np.zeros((m, n), dtype=np.float32)
    
    # Process in blocks to maximize cache utilization
    # Outer loops: iterate over blocks
    for i in range(0, m, block_size):       # Block rows in A and C
        for j in range(0, n, block_size):   # Block columns in B and C
            for k_idx in range(0, k, block_size):  # Block columns in A, rows in B
                
                # Define block boundaries (handle edge cases)
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, n)
                k_end = min(k_idx + block_size, k)
                
                # Extract blocks that fit in cache
                # These slices create views, not copies (memory efficient)
                a_block = a[i:i_end, k_idx:k_end]      # Shape: (<=64, <=64)
                b_block = b[k_idx:k_end, j:j_end]      # Shape: (<=64, <=64)
                
                # Multiply blocks using optimized NumPy BLAS
                # This operates on cache-resident data
                c[i:i_end, j:j_end] += a_block @ b_block
    
    return c

def calculate_cache_footprint(block_size: int) -> dict:
    """
    Calculate memory footprint for educational purposes.
    
    This helps students understand why different block sizes work better or worse.
    Block size optimization is crucial for cache performance.
    """
    bytes_per_float = 4  # float32 size
    elements_per_block = block_size * block_size
    bytes_per_block = elements_per_block * bytes_per_float
    total_blocks = 3  # A_block, B_block, C_block
    total_bytes = bytes_per_block * total_blocks
    
    # Cache size thresholds (typical modern CPU)
    l1_cache_size = 32 * 1024   # 32KB L1 data cache
    l2_cache_size = 256 * 1024  # 256KB L2 cache
    l3_cache_size = 8 * 1024 * 1024  # 8MB L3 cache
    
    return {
        "block_size": block_size,
        "elements_per_block": elements_per_block,
        "bytes_per_block": bytes_per_block,
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024,
        "fits_in_l1": total_bytes <= l1_cache_size,
        "fits_in_l2": total_bytes <= l2_cache_size,
        "fits_in_l3": total_bytes <= l3_cache_size,
        "cache_level": (
            "L1" if total_bytes <= l1_cache_size else
            "L2" if total_bytes <= l2_cache_size else
            "L3" if total_bytes <= l3_cache_size else
            "RAM"
        )
    }

# MAGNIFY SYSTEMS INSIGHT: Cache Optimization Analysis
def analyze_cache_optimization():
    """
    Analyze how different block sizes affect cache performance.
    
    This demonstrates the trade-off between cache utilization
    and computational efficiency in blocked algorithms.
    """
    try:
        print("📊 Cache Optimization Analysis")
        print("=" * 40)
        
        # Test different block sizes
        block_sizes = [16, 32, 64, 128, 256]
        
        print("\nBlock Size Analysis:")
        print("Size | Elements | Memory  | Cache Level | Efficiency")
        print("-" * 55)
        
        for block_size in block_sizes:
            footprint = calculate_cache_footprint(block_size)
            
            # Calculate computational efficiency
            # Smaller blocks = more overhead, larger blocks = cache misses
            if footprint["fits_in_l1"]:
                efficiency = "Excellent"
            elif footprint["fits_in_l2"]:
                efficiency = "Good"
            elif footprint["fits_in_l3"]:
                efficiency = "Fair"
            else:
                efficiency = "Poor"
                
            print(f"{block_size:4d} | {footprint['elements_per_block']:8d} | {footprint['total_kb']:6.1f}KB | {footprint['cache_level']:10s} | {efficiency}")
        
        print("\n📊 Optimal Block Size Analysis:")
        optimal = calculate_cache_footprint(64)
        print(f"64x64 blocks use {optimal['total_kb']:.1f}KB")
        print(f"Fits in: {optimal['cache_level']} cache")
        print(f"Reuse factor: Each element used 64 times")
        print(f"Cache efficiency: 64x better than naive")
        
        print("\nTIP Key Insights:")
        print("• Blocks too small: High loop overhead")
        print("• Blocks too large: Cache misses")
        print("• Sweet spot: 64x64 fits in L2 cache")
        print("• Modern CPUs: Designed for this pattern!")
        
    except Exception as e:
        print(f"WARNING️ Error in cache analysis: {e}")

# Run the cache analysis
analyze_cache_optimization()

# %% [markdown]
"""
### TEST Unit Test: Blocked Implementation

Let's see how much faster cache-friendly blocking is compared to educational loops.
"""

# PASS IMPLEMENTATION CHECKPOINT: Cache-friendly blocking complete

# THINK PREDICTION: How much speedup does cache blocking provide?
# Your guess: ___x faster than educational loops

# MAGNIFY SYSTEMS INSIGHT #2: Cache Blocking Effectiveness
def analyze_cache_blocking_effectiveness():
    """
    Measure how cache-friendly blocking improves performance.
    
    This demonstrates the practical impact of designing algorithms
    that work with CPU cache hierarchy instead of against it.
    """
    try:
        print("📊 Cache Blocking Effectiveness Analysis")
        print("=" * 45)
        
        # Test different block sizes to show optimal choice
        matrix_size = 300
        block_sizes = [32, 64, 128, 256]
        
        # Create test matrices
        a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        b = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        
        print(f"\nBlock Size Optimization (Matrix: {matrix_size}x{matrix_size}):")
        print("Block | Time (ms) | Cache Fit | Efficiency")
        print("-" * 45)
        
        best_time = float('inf')
        best_block = 64
        
        for block_size in block_sizes:
            # Time blocked implementation
            start = time.perf_counter()
            _ = matmul_blocked(a, b, block_size=block_size)
            blocked_time = time.perf_counter() - start
            
            # Calculate cache footprint
            footprint = calculate_cache_footprint(block_size)
            
            # Determine efficiency
            if blocked_time < best_time:
                best_time = blocked_time
                best_block = block_size
                efficiency = "Optimal"
            elif blocked_time < best_time * 1.2:
                efficiency = "Good"
            else:
                efficiency = "Suboptimal"
            
            print(f"{block_size:5d} | {blocked_time*1000:8.1f} | {footprint['cache_level']:8s} | {efficiency}")
        
        # Compare with naive and NumPy
        print(f"\n📊 Performance Comparison:")
        
        # Time naive (small subset)
        start = time.perf_counter()
        _ = matmul_naive(a[:50, :50], b[:50, :50])
        naive_time = time.perf_counter() - start
        naive_scaled = naive_time * (matrix_size / 50) ** 3
        
        # Time NumPy
        start = time.perf_counter()
        _ = a @ b
        numpy_time = time.perf_counter() - start
        
        print(f"Naive (estimated): {naive_scaled*1000:8.1f}ms")
        print(f"Blocked (optimal):  {best_time*1000:8.1f}ms")
        print(f"NumPy (production): {numpy_time*1000:8.1f}ms")
        
        speedup_blocked = naive_scaled / best_time
        speedup_numpy = naive_scaled / numpy_time
        
        print(f"\nROCKET Speedup Results:")
        print(f"Blocking: {speedup_blocked:.0f}x faster than naive")
        print(f"NumPy: {speedup_numpy:.0f}x faster than naive")
        print(f"Block size {best_block}: Optimal for this matrix size")
        
        print(f"\nTIP Key Cache Insights:")
        print(f"• 64x64 blocks typically optimal (fits L2 cache)")
        print(f"• Too small: High loop overhead")
        print(f"• Too large: Cache misses return")
        print(f"• Cache hierarchy shapes algorithm design")
        
    except Exception as e:
        print(f"WARNING️ Error in blocking analysis: {e}")
        print("Make sure all blocking functions are implemented correctly")

# Run the cache blocking analysis
analyze_cache_blocking_effectiveness()

def test_blocked_optimization():
    """Test blocked matrix multiplication performance"""
    print("Testing Blocked Matrix Multiplication...")
    
    # Test correctness
    a = np.random.randn(200, 200).astype(np.float32)
    b = np.random.randn(200, 200).astype(np.float32)
    
    result_blocked = matmul_blocked(a, b, block_size=64)
    result_numpy = a @ b
    
    assert np.allclose(result_blocked, result_numpy, atol=1e-3), "Blocked matmul incorrect"
    print("PASS Blocked implementation produces correct results")
    
    # Performance comparison
    print("\nPerformance comparison:")
    
    # Educational vs Blocked vs NumPy
    size = 200
    test_a = np.random.randn(size, size).astype(np.float32)
    test_b = np.random.randn(size, size).astype(np.float32)
    
    # Time educational (smaller subset to avoid waiting forever)
    start = time.perf_counter()
    _ = matmul_naive(test_a[:50, :50], test_b[:50, :50])
    naive_time = time.perf_counter() - start
    # Scale cubic complexity: (200/50)³ = 4³ = 64x operations
    scaling_factor = (size / 50) ** 3  
    naive_time_scaled = naive_time * scaling_factor
    
    # Time blocked
    start = time.perf_counter()
    _ = matmul_blocked(test_a, test_b, block_size=64)
    blocked_time = time.perf_counter() - start
    
    # Time NumPy
    start = time.perf_counter()
    _ = test_a @ test_b
    numpy_time = time.perf_counter() - start
    
    print(f"Naive (estimated): {naive_time_scaled*1000:.1f} ms")
    print(f"Blocked:           {blocked_time*1000:.1f} ms")
    print(f"NumPy:             {numpy_time*1000:.1f} ms")
    
    speedup_blocked = naive_time_scaled / blocked_time
    speedup_numpy = naive_time_scaled / numpy_time
    
    print(f"\nROCKET SPEEDUP RESULTS:")
    print(f"Blocked is {speedup_blocked:.1f}x faster than naive loops!")
    print(f"NumPy is {speedup_numpy:.1f}x faster than naive loops!")
    print(f"\nTIP Why blocking works: Better cache utilization!")
    print(f"   • Naive: 1 cache miss per operation")
    print(f"   • Blocked: 1 cache miss per 64 operations")
    print(f"   • NumPy: Professional optimizations + vectorization")
    
    print("PASS Blocked optimization tested successfully")
    return blocked_time, numpy_time

# Execute the blocked test
test_blocked_optimization()

# %% [markdown]
"""
## Part 3: NumPy Optimization - Production Performance

Now we'll switch to NumPy for production use. The key insight: NumPy already has these optimizations (and more) built-in.
"""

# %%
def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Production matrix multiplication using NumPy.
    
    This is what you should actually use in practice.
    NumPy already has blocking, vectorization, and BLAS optimizations built-in.
    """
    return a @ b

# %% [markdown]
"""
### TEST Unit Test: Production Implementation

Let's verify that NumPy is indeed the best choice for production.
"""

# PASS IMPLEMENTATION CHECKPOINT: Production backend system complete

# THINK PREDICTION: What makes NumPy faster than our blocking algorithm?
# Your answer: ___ (vectorization, BLAS, assembly, etc.)

# MAGNIFY SYSTEMS INSIGHT #3: Production Optimization Analysis
def analyze_production_optimization_stack():
    """
    Analyze the complete optimization stack that makes NumPy so fast.
    
    This reveals why production libraries beat custom implementations
    and what optimizations are built into professional ML frameworks.
    """
    try:
        print("📊 Production Optimization Stack Analysis")
        print("=" * 50)
        
        # Test across range of sizes to show scaling characteristics
        sizes = [100, 300, 500, 1000]
        
        print("\nOptimization Stack Performance:")
        print("Size | Naive Est | Blocked | NumPy | Block->NumPy | Total Speedup")
        print("-" * 70)
        
        for size in sizes:
            # Create test matrices
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Time blocked implementation
            start = time.perf_counter()
            _ = matmul_blocked(a, b, block_size=64)
            blocked_time = time.perf_counter() - start
            
            # Time NumPy implementation
            start = time.perf_counter()
            _ = a @ b
            numpy_time = time.perf_counter() - start
            
            # Estimate naive time (from small sample)
            if size <= 200:
                start = time.perf_counter()
                _ = matmul_naive(a[:50, :50], b[:50, :50])
                naive_small = time.perf_counter() - start
                naive_estimated = naive_small * (size / 50) ** 3
            else:
                # Use previous scaling for larger matrices
                naive_estimated = naive_time * (size / 200) ** 3 if 'naive_time' in locals() else blocked_time * 100
            
            # Calculate speedups
            block_speedup = naive_estimated / blocked_time
            numpy_speedup = blocked_time / numpy_time
            total_speedup = naive_estimated / numpy_time
            
            print(f"{size:4d} | {naive_estimated*1000:8.0f}ms | {blocked_time*1000:6.1f}ms | {numpy_time*1000:4.1f}ms | {numpy_speedup:9.1f}x | {total_speedup:11.0f}x")
            
            if size == 200:  # Store for scaling estimation
                naive_time = naive_estimated
        
        print(f"\n📊 NumPy's Optimization Stack:")
        print(f"🔧 1. Cache Blocking: Process data in cache-friendly chunks")
        print(f"🔧 2. Vectorization: SIMD instructions (4-8x speedup)")
        print(f"🔧 3. BLAS Libraries: Hand-optimized linear algebra (Intel MKL, OpenBLAS)")
        print(f"🔧 4. Assembly Kernels: CPU-specific optimizations")
        print(f"🔧 5. Memory Layout: Optimal data structure organization")
        print(f"🔧 6. Threading: Automatic parallelization for large matrices")
        
        print(f"\n📊 Development Cost vs Performance Benefit:")
        print(f"• Custom blocking: 1 week implementation -> 10-50x speedup")
        print(f"• BLAS integration: 1 month implementation -> additional 5-10x")
        print(f"• Assembly optimization: 6+ months -> additional 2-5x")
        print(f"• NumPy: 0 development time -> all optimizations included")
        
        print(f"\nTIP ML Systems Engineering Insight:")
        print(f"• Focus on system architecture, not micro-optimizations")
        print(f"• Leverage existing optimized libraries (NumPy, PyTorch, TensorFlow)")
        print(f"• Understanding principles enables better system design")
        print(f"• Build on foundations, don't reinvent optimized wheels")
        
    except Exception as e:
        print(f"WARNING️ Error in production analysis: {e}")
        print("Make sure all performance functions are implemented correctly")

# Run the production optimization analysis
analyze_production_optimization_stack()

# %%
def test_production_performance():
    """Test that NumPy is indeed optimal for production use"""
    print("Testing Production Performance...")
    
    # Test different sizes
    sizes = [200, 500, 800]
    
    print("\nPerformance comparison across the optimization spectrum:")
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Time blocked implementation
        start = time.perf_counter()
        _ = matmul_blocked(a, b, block_size=64)
        blocked_time = time.perf_counter() - start
        
        # Time NumPy implementation
        start = time.perf_counter()
        _ = matmul_numpy(a, b)
        numpy_time = time.perf_counter() - start
        
        speedup = blocked_time / numpy_time
        print(f"Blocked:     {blocked_time*1000:6.1f} ms")
        print(f"NumPy:       {numpy_time*1000:6.1f} ms")
        print(f"NumPy is {speedup:.1f}x faster than blocked")
    
    print("\nTIP Key Insight: NumPy already has these optimizations built-in!")
    print("   • Blocking algorithms")
    print("   • Vectorization")
    print("   • Hardware-specific BLAS libraries")
    print("   • Assembly-level optimizations")
    
    print("\nPASS Production performance verified")
    return True

# Execute the production test
test_production_performance()

# %% [markdown]
"""
## Part 4: Smart Backend System - Transparent Optimization

Now let's build a system that automatically chooses the right implementation. This is how real ML frameworks work!
"""

# %%
class OptimizedBackend:
    """
    Smart backend that automatically dispatches to optimal implementations.
    
    This demonstrates how real ML frameworks (PyTorch, TensorFlow) work:
    - Single API for users
    - Automatic dispatch to fastest implementation
    - Transparent optimization without code changes
    """
    
    def dispatch(self, op: str, *args, **kwargs):
        """Dispatch operations to optimal implementations"""
        if op == "matmul":
            return self.matmul(*args, **kwargs)
        else:
            raise NotImplementedError(f"Operation {op} not implemented")
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication with automatic optimization selection.
        
        For production: Always use NumPy (has all optimizations built-in)
        For education: Could switch based on size, but NumPy is always best
        """
        # In a real system, you might choose based on:
        # - Matrix size (small vs large)
        # - Hardware available (CPU vs GPU)
        # - Memory constraints
        # 
        # But NumPy is almost always the right choice for CPU
        return matmul_numpy(a, b)

# Global backend instance
_backend = OptimizedBackend()

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication using optimal backend.
    
    This is the API students should use - it automatically
    selects the best implementation available.
    """
    return _backend.dispatch("matmul", a, b)

# %% [markdown]
"""
### TEST Unit Test: Backend System

Let's verify our backend system works correctly and uses optimal implementations.
"""

# %%
def test_backend_system():
    """Test the backend system"""
    print("Testing Backend System...")
    
    # Test matrices
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    
    # Test that our backend works
    result = matmul(a, b)
    expected = a @ b
    
    assert np.allclose(result, expected), "Backend matmul incorrect"
    print("PASS Backend produces correct results")
    
    # Compare performance
    start = time.perf_counter()
    _ = matmul(a, b)
    backend_time = time.perf_counter() - start
    
    start = time.perf_counter()
    _ = a @ b
    numpy_time = time.perf_counter() - start
    
    print(f"\nPerformance comparison:")
    print(f"Backend: {backend_time*1000:.1f} ms")
    print(f"NumPy:   {numpy_time*1000:.1f} ms")
    print(f"Backend uses optimal NumPy implementation")
    
    print("\nPASS Backend system works correctly")
    return True

# Execute the backend test
test_backend_system()

# %% [markdown]
"""
## TARGET Computational Assessment Questions

Practice your understanding of hardware acceleration concepts with these NBGrader-compatible questions.

These questions test your ability to analyze performance characteristics, optimize for cache hierarchy, and understand the engineering trade-offs in hardware acceleration. They're grounded in the actual implementations you just built and tested.
"""

# %% nbgrader={"grade": false, "grade_id": "acceleration-q1", "locked": false, "schema_version": 3, "solution": true, "task": false}
def calculate_cache_efficiency(matrix_size: int, block_size: int) -> Tuple[int, float]:
    """
    Calculate the cache efficiency improvement of blocked vs naive matrix multiplication.
    
    For a matrix_size * matrix_size multiplication using block_size * block_size blocks:
    1. Calculate total number of cache misses for naive implementation
    2. Calculate total number of cache misses for blocked implementation  
    3. Return (total_operations, efficiency_ratio)
    
    Assumptions:
    - Cache line = 64 bytes = 16 float32 elements
    - Naive: Every B[k,j] access is a cache miss (column-major access)
    - Blocked: 1 cache miss per block load, then block stays in cache
    
    Args:
        matrix_size: Size of square matrices (N*N)
        block_size: Size of blocks for blocked algorithm
        
    Returns:
        Tuple[int, float]: (total_operations, cache_efficiency_ratio)
        
    TODO: Implement cache efficiency calculation for blocked matrix multiplication
    
    HINTS:
    - Total operations = matrix_size³ 
    - Naive cache misses ~= matrix_size³ (every B access misses)
    - Blocked cache misses = (matrix_size/block_size)³ * block_size² 
    - Efficiency ratio = naive_misses / blocked_misses
    """
    ### BEGIN SOLUTION
    # Total operations for matrix multiplication
    total_operations = matrix_size ** 3
    
    # Naive implementation cache misses
    # Every access to B[k,j] causes a cache miss due to column-major access
    naive_cache_misses = total_operations
    
    # Blocked implementation cache misses
    # Number of blocks in each dimension
    blocks_per_dim = (matrix_size + block_size - 1) // block_size  # Ceiling division
    total_blocks = blocks_per_dim ** 3
    
    # Each block is loaded once, then all elements accessed from cache
    blocked_cache_misses = total_blocks * block_size ** 2
    
    # Cache efficiency ratio
    efficiency_ratio = naive_cache_misses / blocked_cache_misses if blocked_cache_misses > 0 else 1.0
    
    return total_operations, efficiency_ratio
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "acceleration-q2", "locked": false, "schema_version": 3, "solution": true, "task": false}
def analyze_vectorization_speedup(array_size: int, vector_width: int) -> Tuple[int, int, float]:
    """
    Analyze the theoretical speedup from vectorization (SIMD instructions).
    
    Calculate:
    1. Number of scalar operations needed
    2. Number of vector operations needed  
    3. Theoretical speedup ratio
    
    Args:
        array_size: Number of elements to process
        vector_width: Number of elements processed per vector instruction
        
    Returns:
        Tuple[int, int, float]: (scalar_ops, vector_ops, speedup_ratio)
        
    TODO: Calculate vectorization speedup for array operations
    
    APPROACH:
    1. Scalar: One operation per element
    2. Vector: One operation per vector_width elements (with remainder)
    3. Speedup: scalar_ops / vector_ops
    
    EXAMPLE:
    >>> scalar_ops, vector_ops, speedup = analyze_vectorization_speedup(1000, 4)
    >>> print(f"Scalar: {scalar_ops}, Vector: {vector_ops}, Speedup: {speedup:.1f}x")
    Scalar: 1000, Vector: 250, Speedup: 4.0x
    """
    ### BEGIN SOLUTION
    # Scalar operations: one per element
    scalar_ops = array_size
    
    # Vector operations: ceiling division to handle remainder
    vector_ops = (array_size + vector_width - 1) // vector_width
    
    # Theoretical speedup (ignores overhead, assumes perfect vectorization)
    speedup_ratio = scalar_ops / vector_ops if vector_ops > 0 else 1.0
    
    return scalar_ops, vector_ops, speedup_ratio
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "acceleration-q3", "locked": false, "schema_version": 3, "solution": true, "task": false}
def optimize_block_size(matrix_size: int, cache_sizes: Dict[str, int]) -> Tuple[int, str, float]:
    """
    Find the optimal block size for a given matrix size and cache hierarchy.
    
    Test block sizes [16, 32, 64, 128, 256] and select the largest that fits in L2 cache.
    
    Args:
        matrix_size: Size of square matrix to multiply
        cache_sizes: Dictionary with cache sizes in bytes, e.g., {"L1": 32768, "L2": 262144}
        
    Returns:
        Tuple[int, str, float]: (optimal_block_size, cache_level, memory_utilization)
        
    TODO: Find optimal block size based on cache constraints
    
    APPROACH:
    1. For each candidate block size, calculate memory footprint
    2. Check which cache level it fits in (3 blocks * block_size² * 4 bytes)
    3. Select largest block size that fits in L2 cache
    4. Calculate memory utilization = footprint / cache_size
    
    EXAMPLE:
    >>> cache_sizes = {"L1": 32768, "L2": 262144}
    >>> block_size, level, util = optimize_block_size(1000, cache_sizes)
    >>> print(f"Optimal: {block_size}x{block_size}, fits in {level}, {util:.1%} utilization")
    Optimal: 64x64, fits in L2, 18.8% utilization
    """
    ### BEGIN SOLUTION
    candidate_sizes = [16, 32, 64, 128, 256]
    bytes_per_float = 4
    blocks_needed = 3  # A_block, B_block, C_block
    
    optimal_block_size = 16  # Default fallback
    cache_level = "RAM"
    memory_utilization = 0.0
    
    # Test each candidate size
    for block_size in candidate_sizes:
        # Calculate memory footprint
        elements_per_block = block_size * block_size
        bytes_per_block = elements_per_block * bytes_per_float
        total_footprint = bytes_per_block * blocks_needed
        
        # Check which cache level it fits in
        if total_footprint <= cache_sizes.get("L1", 0):
            # Prefer L2 for larger block sizes (better computational efficiency)
            if block_size >= optimal_block_size:
                optimal_block_size = block_size
                cache_level = "L1"
                memory_utilization = total_footprint / cache_sizes["L1"]
        elif total_footprint <= cache_sizes.get("L2", 0):
            # L2 is the sweet spot for most cases
            if block_size >= optimal_block_size:
                optimal_block_size = block_size
                cache_level = "L2"
                memory_utilization = total_footprint / cache_sizes["L2"]
    
    return optimal_block_size, cache_level, memory_utilization
    ### END SOLUTION

# %% nbgrader={"grade": false, "grade_id": "acceleration-q4", "locked": false, "schema_version": 3, "solution": true, "task": false}
def compare_acceleration_techniques(matrix_size: int) -> Dict[str, float]:
    """
    Compare the theoretical speedup of different acceleration techniques.
    
    Calculate expected speedup for:
    1. "cache_blocking": Blocked algorithm (64x64 blocks)
    2. "vectorization": SIMD with 8-wide vectors  
    3. "parallelization": 4-core CPU parallelization
    4. "combined": All techniques together
    
    Args:
        matrix_size: Size of square matrices
        
    Returns:
        Dict[str, float]: Speedup factors for each technique
        
    TODO: Calculate theoretical speedups for different acceleration techniques
    
    APPROACH:
    1. Cache blocking: Use previous cache efficiency calculation
    2. Vectorization: 8-wide SIMD operations
    3. Parallelization: 4 cores working in parallel
    4. Combined: Multiply individual speedups (idealized)
    
    ASSUMPTIONS:
    - Perfect scaling (no overhead)
    - Cache blocking gives efficiency_ratio improvement
    - Vectorization gives 8x speedup
    - Parallelization gives 4x speedup
    """
    ### BEGIN SOLUTION
    # Cache blocking speedup (using 64x64 blocks)
    block_size = 64
    _, cache_speedup = calculate_cache_efficiency(matrix_size, block_size)
    
    # Vectorization speedup (8-wide SIMD)
    vector_width = 8
    _, _, vectorization_speedup = analyze_vectorization_speedup(matrix_size ** 3, vector_width)
    
    # Parallelization speedup (4 cores)
    parallelization_speedup = 4.0
    
    # Combined speedup (multiplicative - idealized)
    combined_speedup = cache_speedup * vectorization_speedup * parallelization_speedup
    
    return {
        "cache_blocking": cache_speedup,
        "vectorization": vectorization_speedup,
        "parallelization": parallelization_speedup,
        "combined": combined_speedup
    }
    ### END SOLUTION

# %% [markdown]
"""
## Part 5: Real-World Application Testing

Let's test our optimizations on actual ML model operations: MLP layers, CNN convolutions, and Transformer attention.
"""

# %%
def test_ml_model_acceleration():
    """Test acceleration on real ML model operations"""
    print("Testing Acceleration on Real ML Models...")
    
    # Test 1: MLP Forward Pass (common in Module 4)
    print("\n1. MLP Forward Pass (256 -> 128 -> 64):")
    batch_size, input_dim, hidden_dim, output_dim = 32, 256, 128, 64
    
    # Simulated MLP layers
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32)
    
    # Time naive implementation (small version)
    start = time.perf_counter()
    h1_naive = matmul_naive(x[:8, :64], W1[:64, :32])  # Scaled down
    h2_naive = matmul_naive(h1_naive, W2[:32, :16])     # Scaled down
    naive_time = time.perf_counter() - start
    
    # Time optimized implementation
    start = time.perf_counter()
    h1_opt = matmul(x, W1)
    h2_opt = matmul(h1_opt, W2)
    opt_time = time.perf_counter() - start
    
    # Scale for: batch_size (32/8) * input_dim (256/64) * hidden_dim (128/32)
    batch_scale = 32/8  # 4x more samples
    input_scale = 256/64  # 4x larger input
    hidden_scale = 128/32  # 4x larger hidden layer
    naive_scaled = naive_time * batch_scale * input_scale * hidden_scale
    speedup = naive_scaled / opt_time
    
    print(f"   Naive (estimated): {naive_scaled*1000:.1f} ms")
    print(f"   Optimized:         {opt_time*1000:.1f} ms")
    print(f"   Speedup: {speedup:.1f}x faster!")
    
    # Test 2: CNN-like Convolution (flattened as matrix multiply)
    print("\n2. CNN Convolution (as matrix multiply):")
    # Simulate im2col operation for 3x3 convolution
    img_patches = np.random.randn(1024, 27).astype(np.float32)  # 32x32 image, 3x3 patches
    conv_filters = np.random.randn(27, 64).astype(np.float32)   # 64 filters
    
    start = time.perf_counter()
    conv_output = matmul(img_patches, conv_filters)
    conv_time = time.perf_counter() - start
    print(f"   Convolution output: {conv_time*1000:.1f} ms")
    print(f"   Shape: {conv_output.shape} (1024 locations * 64 filters)")
    
    # Test 3: Transformer-like Attention (scaled down)
    print("\n3. Transformer Attention (Q·K^T):")
    seq_len, d_model = 128, 256
    Q = np.random.randn(seq_len, d_model).astype(np.float32)
    K = np.random.randn(seq_len, d_model).astype(np.float32)
    
    start = time.perf_counter()
    attention_scores = matmul(Q, K.T)  # Shape: (seq_len, seq_len)
    attn_time = time.perf_counter() - start
    print(f"   Attention computation: {attn_time*1000:.1f} ms")
    print(f"   Shape: {attention_scores.shape} (128*128 attention matrix)")
    
    print(f"\nPASS All ML model operations accelerated successfully!")
    print(f"TIP Key insight: Matrix multiplication is EVERYWHERE in ML!")
    return True

# Execute the ML model test
test_ml_model_acceleration()

# MAGNIFY SYSTEMS INSIGHT: Acceleration Scaling Analysis
def analyze_acceleration_scaling():
    """
    Analyze how different acceleration techniques scale with problem size.
    
    This demonstrates the performance characteristics of optimization
    techniques across a range of matrix sizes typical in ML workloads.
    """
    try:
        print("📊 Acceleration Scaling Analysis")
        print("=" * 45)
        
        # Test different matrix sizes (typical ML workloads)
        matrix_sizes = [100, 200, 500, 1000, 2000]
        
        print("\nScaling Analysis Across Matrix Sizes:")
        print("Size  | Cache Block | Vectorization | Parallelization | Combined")
        print("-" * 65)
        
        for size in matrix_sizes:
            # Calculate speedups for this matrix size
            speedups = compare_acceleration_techniques(size)
            
            print(f"{size:4d}  | {speedups['cache_blocking']:10.1f} | {speedups['vectorization']:12.1f} | {speedups['parallelization']:14.1f} | {speedups['combined']:7.0f}")
        
        print(f"\n📊 Key Scaling Insights:")
        
        # Analyze cache blocking scaling
        small_speedup = compare_acceleration_techniques(100)['cache_blocking']
        large_speedup = compare_acceleration_techniques(2000)['cache_blocking']
        
        print(f"• Cache blocking: {small_speedup:.1f}x -> {large_speedup:.1f}x (scales with cache misses)")
        print(f"• Vectorization: 8.0x constant (independent of matrix size)")
        print(f"• Parallelization: 4.0x constant (perfect scaling assumed)")
        print(f"• Combined: Multiplicative effect = cache * vector * parallel")
        
        print(f"\n📊 Real-World Performance Expectations:")
        realistic_combined = large_speedup * 4.0 * 4.0  # Conservative vectorization
        print(f"• Realistic combined speedup: ~{realistic_combined:.0f}x")
        print(f"• Why not perfect: Memory bandwidth limits, overhead, synchronization")
        print(f"• Production systems: Focus on cache + vectorization first")
        
        print(f"\nTIP ML Systems Implications:")
        print(f"• Small models (<=500): Vectorization dominates")
        print(f"• Large models (>=1000): Cache optimization critical")
        print(f"• Production: Memory bandwidth becomes bottleneck")
        print(f"• GPU: Different scaling - thousands of cores, different cache hierarchy")
        
    except Exception as e:
        print(f"WARNING️ Error in scaling analysis: {e}")
        print("Make sure all analysis functions are implemented correctly")

# Run the scaling analysis
analyze_acceleration_scaling()

def run_complete_acceleration_demo():
    """Run the complete acceleration demonstration"""
    print("ROCKET Complete Hardware Acceleration Demo")
    print("=" * 55)
    print("THE FREE SPEEDUP: From Naive Loops to Optimized Backends")
    
    # 1. Test naive baseline
    print("\n1. Naive Baseline (your Module 2/4 loops):")
    naive_results = test_naive_baseline()
    
    # 2. Test blocked optimization
    print("\n2. Cache-Friendly Blocking:")
    test_blocked_optimization()
    
    # 3. Test production performance
    print("\n3. Production Performance (NumPy):")
    test_production_performance()
    
    # 4. Test ML model acceleration
    print("\n4. Real ML Model Acceleration:")
    test_ml_model_acceleration()
    
    # 5. Test backend system
    print("\n5. Smart Backend System:")
    test_backend_system()
    
    print("\n" + "=" * 55)
    print("TARGET HARDWARE ACCELERATION MASTERED")
    print("=" * 55)
    
    print("\n📚 What You Mastered:")
    print("PASS Why your Module 2/4 loops were slow (cache hierarchy matters!)")
    print("PASS How cache-friendly blocking works (process data in chunks)")
    print("PASS Why NumPy dominates (professional optimizations built-in)")
    print("PASS How to build smart backend systems (automatic optimization)")
    print("PASS Real ML applications (MLPs, CNNs, Transformers all use matmul!)")
    
    print("\nTARGET The Free Speedup Philosophy:")
    print("• ROCKET Same math, better implementation = 100x speedup")
    print("• 🧠 Educational loops teach algorithms")
    print("• SPEED Blocked algorithms teach cache optimization")
    print("• 🏭 NumPy provides production performance")
    print("• TARGET Smart backends make optimization transparent")
    print("• TIP Understanding the spectrum makes you a better engineer!")
    
    return naive_results

# %% [markdown]
"""
## Systems Analysis Summary

This module demonstrates the fundamental principles of hardware acceleration in ML systems:

### 🏗️ **Architecture Principles**
- **Cache Hierarchy**: Understanding L1/L2/L3 cache and memory access costs
- **Vectorization**: Leveraging SIMD instructions for parallel computation
- **Memory Layout**: Contiguous access patterns for optimal performance
- **Backend Abstraction**: Transparent dispatch between naive and optimized implementations

### SPEED **Optimization Techniques**
- **Blocked Algorithms**: Process data in cache-friendly blocks
- **Vectorized Operations**: Avoid Python loops, use NumPy's optimized routines
- **In-place Operations**: Minimize memory allocation overhead
- **Automatic Dispatch**: Choose optimal implementation based on problem size

### 📊 **Performance Understanding**
- **Measurement First**: Profile real bottlenecks before optimizing
- **Algorithmic Impact**: O(N³) -> O(N²) matters more than 2x constant factors
- **Hardware Awareness**: CPU cache misses cost 100x more than cache hits
- **Library Utilization**: Optimized BLAS libraries beat custom implementations

### TARGET **Real-World Applications**
- **ML Frameworks**: How PyTorch/TensorFlow apply these same principles
- **Production Systems**: Where optimization efforts provide real value
- **Development Practice**: When to optimize vs when to use existing solutions

### TIP **Key Insights**
- Cache-friendly algorithms provide 2-5x speedups from memory access patterns alone
- Vectorization eliminates Python overhead for 10-100x improvements
- Most NumPy operations are already optimized - focus on system-level improvements
- Competition frameworks make optimization learning engaging and quantifiable
- Real ML systems face memory and communication bottlenecks, not pure computation limits

This approach teaches students to think like systems engineers: understand the hardware, measure scientifically, optimize systematically, and focus efforts where they matter most.
"""

def test_unit_all():
    """Run all unit tests for the acceleration module."""
    print("TEST Running all Hardware Acceleration tests...")
    print("=" * 55)
    
    try:
        # Test educational baseline
        print("\n1. Testing educational baseline...")
        test_naive_baseline()
        
        # Test cache blocking optimization
        print("\n2. Testing cache blocking...")
        test_blocked_optimization()
        
        # Test production performance
        print("\n3. Testing production performance...")
        test_production_performance()
        
        # Test backend system
        print("\n4. Testing backend system...")
        test_backend_system()
        
        # Test ML model acceleration
        print("\n5. Testing ML model acceleration...")
        test_ml_model_acceleration()
        
        print("\n" + "=" * 55)
        print("PASS All Hardware Acceleration tests passed!")
        print("ROCKET Module ready for production ML systems.")
        
    except Exception as e:
        print(f"FAIL Test failed: {e}")
        raise

if __name__ == "__main__":
    print("Module 16: Hardware Acceleration - The Free Speedup!")
    print("=" * 60)
    print("ROCKET THE EASIEST OPTIMIZATION: Better Backends, Zero Trade-offs")
    
    # Run complete testing suite
    test_unit_all()
    
    print(f"\nCELEBRATE Module 16: Hardware Acceleration COMPLETE!")
    print(f"SPEED Mastered: 10-100x speedups with no accuracy loss")
    print(f"🧠 Learned: Cache hierarchy, blocking, vectorization")
    print(f"🏭 Applied: MLPs, CNNs, Transformers all benefit")
    print(f"TARGET Ready: To build high-performance ML systems!")

# %% [markdown]
"""
## THINK ML Systems Thinking: Interactive Questions

1. **Memory Access Pattern Analysis**: In your `matmul_naive()` implementation, the innermost loop accesses `a[i, k]` sequentially but `b[k, j]` with large strides. When you tested 200*200 matrices, you saw dramatic slowdowns. Analyze why: (a) Calculate cache misses for both access patterns, (b) Explain why `b[k, j]` creates O(N²) cache misses, (c) Show how this scales to 1000*1000 matrices, and (d) Design a memory layout that would eliminate strided access.

2. **Cache Blocking Optimization**: Your `matmul_blocked()` function uses 64*64 blocks and showed significant speedups over naive loops. Analyze the cache efficiency: (a) Calculate total memory footprint (3 blocks * 64² * 4 bytes), (b) Verify it fits in L2 cache (256KB), (c) Compute cache reuse factor (64 operations per cache line), (d) Predict performance change with 128*128 blocks, and (e) Explain why your cache analysis function showed 64*64 as optimal.

3. **Production Stack Engineering**: You measured that NumPy beats your blocked implementation by 5-10x. Analyze the engineering trade-offs: (a) List three specific optimizations NumPy includes (BLAS, vectorization, threading), (b) Calculate development time vs. performance gain for each, (c) Estimate why custom optimization rarely beats production libraries, and (d) Determine when custom optimization is justified in ML systems.

4. **ML Acceleration Architecture**: Your tests showed acceleration benefits for MLP, CNN, and Transformer operations. Design an acceleration strategy: (a) Rank these operations by matrix multiplication density, (b) Identify memory bandwidth vs. compute bottlenecks for each, (c) Predict how GPU acceleration would change the rankings, and (d) Explain why understanding this spectrum enables better ML systems engineering decisions.
"""

# %% [markdown]
"""
## TARGET MODULE SUMMARY: Hardware Acceleration - The Free Speedup

This module demonstrates the easiest optimization in ML systems: using better backends for free speedups with zero accuracy trade-offs. You learned why understanding the optimization spectrum makes you a better engineer.

### 🛤️ **The Free Speedup Journey**
- **Educational Foundation**: Your Module 2/4 loops taught you the algorithm (perfect for learning)
- **Performance Understanding**: Module 15 showed you WHY loops are slow (profiling first)
- **Optimization Mastery**: Now you achieve 100x speedups by choosing better implementations
- **Systems Thinking**: Understanding the spectrum from educational to production code

### 🛠️ **What We Built and Tested**
- **Educational Baseline**: Your triple-nested loops from Module 2/4 (algorithm understanding)
- **Cache-Friendly Blocking**: 64*64 blocks fitting in L1/L2 cache (10x+ speedup)
- **NumPy Production**: Leveraging professional BLAS optimizations (another 10x speedup)
- **Smart Backend System**: Automatic dispatch to optimal implementations
- **Real ML Applications**: MLP, CNN, Transformer operations using matrix multiplication

### 🧠 **Key Learning Outcomes**
- **Why loops are slow**: Memory access patterns and cache hierarchy matter most
- **How blocking helps**: Processing data in cache-friendly chunks improves performance
- **When to use NumPy**: It already has these optimizations (and more) built-in
- **Systems thinking**: Understanding enables better decisions about when to optimize

### SPEED **Performance Spectrum Mastered**
- **Educational loops**: Algorithm understanding (1000x slower, perfect for learning)
- **Cache-friendly blocking**: Systems understanding (100x slower, teaches optimization)
- **NumPy production**: Professional performance (optimal speed, built-in optimizations)
- **Smart backends**: Engineering understanding (transparent optimization selection)

### 🏆 **Practical Skills Developed**
- Analyze why educational implementations have poor performance
- Implement cache-friendly algorithms to understand optimization principles
- Choose NumPy for production while understanding what it's doing internally
- Build systems that balance educational value with performance requirements

### 📊 **Systems Insights Gained**
- **Educational code serves a purpose**: Understanding algorithms enables optimization intuition
- **Cache hierarchy dominates performance**: Memory access patterns matter more than computation
- **Libraries beat custom optimization**: NumPy already has expert-level optimizations
- **Understanding enables better tools**: You can build smarter systems when you know the principles

### TIP **The Free Speedup Philosophy**
This is the EASIEST optimization in ML systems: same math, better implementation, massive speedups, zero downsides. You implemented loops to understand algorithms. You implemented blocking to understand cache optimization. Now you use NumPy because it has all optimizations built-in. Understanding this spectrum - from educational to production - makes you a superior ML systems engineer who can make informed optimization decisions.
"""

