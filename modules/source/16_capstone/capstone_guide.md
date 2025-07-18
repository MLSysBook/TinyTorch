# 🎯 Capstone Project Guide: Performance Optimization Example

## **Example Project: Vectorized Matrix Operations**

This guide walks through a complete capstone project optimizing TinyTorch's matrix operations. Follow this example to understand the process, then apply it to your chosen optimization track.

---

## **Phase 1: Analysis & Profiling**

### **Step 1: Profile Your Current Implementation**

First, let's identify where TinyTorch spends most of its time:

```python
import cProfile
import pstats
import time
import numpy as np
from memory_profiler import profile

# Import your TinyTorch framework
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.networks import Sequential
from tinytorch.core.activations import ReLU

def profile_current_framework():
    """Profile a typical TinyTorch training scenario."""
    
    # Create a realistic model
    model = Sequential([
        Dense(784, 256),
        ReLU(),
        Dense(256, 128), 
        ReLU(),
        Dense(128, 10)
    ])
    
    # Generate realistic data (like MNIST)
    batch_size = 64
    X = Tensor(np.random.randn(batch_size, 784))
    
    # Profile forward pass
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple forward passes
    for _ in range(100):
        output = model.forward(X)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return stats

# Run profiling
print("🔍 Profiling Current TinyTorch Framework...")
profile_results = profile_current_framework()
```

### **Step 2: Analyze Bottlenecks**

Typical results show:
```
         1003 function calls in 2.450 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.001    0.000    2.449    0.024 networks.py:45(forward)
      300    0.002    0.000    2.448    0.008 layers.py:67(forward)
      300    2.440    0.008    2.446    0.008 layers.py:34(matmul_naive)  ← BOTTLENECK!
      200    0.004    0.000    0.004    0.000 activations.py:23(forward)
```

**Finding**: 99.6% of time spent in `matmul_naive`! This is our optimization target.

### **Step 3: Baseline Benchmarks**

```python
def benchmark_current_matmul():
    """Establish baseline performance metrics."""
    
    # Test various matrix sizes
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for m, n in sizes:
        A = np.random.randn(m, n)
        B = np.random.randn(n, m)
        
        # Time current implementation
        start = time.time()
        result = matmul_naive(A, B)  # Your current implementation
        current_time = time.time() - start
        
        # Time NumPy for comparison  
        start = time.time()
        numpy_result = np.dot(A, B)
        numpy_time = time.time() - start
        
        slowdown = current_time / numpy_time
        print(f"Size {m}x{n}: TinyTorch={current_time:.3f}s, NumPy={numpy_time:.3f}s, Slowdown={slowdown:.1f}x")

print("📊 Baseline Performance:")
benchmark_current_matmul()
```

**Typical Output:**
```
Size 100x100: TinyTorch=0.023s, NumPy=0.001s, Slowdown=23.0x
Size 500x500: TinyTorch=0.890s, NumPy=0.012s, Slowdown=74.2x  
Size 1000x1000: TinyTorch=7.234s, NumPy=0.089s, Slowdown=81.3x
```

**Goal**: Reduce this slowdown from 80x to under 5x.

---

## **Phase 2: Optimization Implementation**

### **Step 4: Implement Optimized Matrix Multiplication**

```python
def matmul_optimized_v1(A, B):
    """
    First optimization: Use NumPy's optimized dot product.
    
    This isn't cheating - NumPy is our computational backend,
    just like PyTorch uses BLAS/LAPACK under the hood.
    """
    # Validate inputs (keep your error checking)
    assert A.shape[1] == B.shape[0], f"Cannot multiply {A.shape} and {B.shape}"
    
    # Use NumPy's optimized implementation
    return np.dot(A, B)

def matmul_optimized_v2(A, B):
    """
    Second optimization: Block-based multiplication for large matrices.
    Better cache performance for very large operations.
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    
    # For small matrices, use simple NumPy
    if m * n * k < 1000000:  # Threshold tuned empirically
        return np.dot(A, B)
    
    # For large matrices, use block multiplication
    block_size = 256  # Optimized for L2 cache
    C = np.zeros((m, n))
    
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for l in range(0, k, block_size):
                # Extract blocks
                A_block = A[i:i+block_size, l:l+block_size]
                B_block = B[l:l+block_size, j:j+block_size]
                
                # Multiply blocks
                C[i:i+block_size, j:j+block_size] += np.dot(A_block, B_block)
    
    return C

def matmul_optimized_v3(A, B):
    """
    Third optimization: Memory layout optimization.
    Ensure contiguous memory for better performance.
    """
    # Ensure C-contiguous layout for better cache performance
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    if not B.flags['C_CONTIGUOUS']:
        B = np.ascontiguousarray(B)
    
    # Use the block approach with optimized memory layout
    return matmul_optimized_v2(A, B)
```

### **Step 5: Test and Benchmark Optimizations**

```python
def benchmark_optimizations():
    """Compare all optimization versions."""
    
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for m, n in sizes:
        A = np.random.randn(m, n)
        B = np.random.randn(n, m)
        
        # Test correctness first
        result_naive = matmul_naive(A, B)
        result_v1 = matmul_optimized_v1(A, B)
        result_v2 = matmul_optimized_v2(A, B)
        result_v3 = matmul_optimized_v3(A, B)
        
        # Verify all produce same results
        assert np.allclose(result_naive, result_v1, rtol=1e-10)
        assert np.allclose(result_naive, result_v2, rtol=1e-10)
        assert np.allclose(result_naive, result_v3, rtol=1e-10)
        
        # Benchmark performance
        times = {}
        for name, func in [
            ('naive', matmul_naive),
            ('v1_numpy', matmul_optimized_v1),
            ('v2_blocks', matmul_optimized_v2),
            ('v3_memory', matmul_optimized_v3)
        ]:
            start = time.time()
            _ = func(A, B)
            times[name] = time.time() - start
        
        print(f"\nSize {m}x{n}:")
        baseline = times['naive']
        for name, t in times.items():
            speedup = baseline / t
            print(f"  {name:12}: {t:.3f}s (speedup: {speedup:.1f}x)")

print("⚡ Optimization Results:")
benchmark_optimizations()
```

**Typical Results:**
```
Size 1000x1000:
  naive       : 7.234s (speedup: 1.0x)
  v1_numpy    : 0.089s (speedup: 81.3x)  ← Huge improvement!
  v2_blocks   : 0.091s (speedup: 79.5x)  ← Slight regression for this size
  v3_memory   : 0.087s (speedup: 83.1x)  ← Best overall
```

---

## **Phase 3: Integration & Testing**

### **Step 6: Update Your Dense Layer**

```python
class DenseOptimized:
    """Optimized Dense layer using improved matrix multiplication."""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights (same as before)
        self.weight = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
    
    def forward(self, x):
        """Forward pass using optimized matrix multiplication."""
        # Use our optimized matmul instead of naive version
        linear_output = matmul_optimized_v3(x, self.weight)
        return linear_output + self.bias
    
    def __call__(self, x):
        return self.forward(x)
```

### **Step 7: End-to-End Performance Test**

```python
def test_full_network_improvement():
    """Test the complete training pipeline with optimizations."""
    
    # Create identical networks with different matmul implementations
    print("🏗️ Creating test networks...")
    
    # Original network (using naive matmul)
    network_original = Sequential([
        Dense(784, 256),  # Uses matmul_naive
        ReLU(),
        Dense(256, 128),
        ReLU(), 
        Dense(128, 10)
    ])
    
    # Optimized network (using optimized matmul)
    network_optimized = Sequential([
        DenseOptimized(784, 256),  # Uses matmul_optimized_v3
        ReLU(),
        DenseOptimized(256, 128),
        ReLU(),
        DenseOptimized(128, 10)
    ])
    
    # Test data
    batch_size = 64
    X = np.random.randn(batch_size, 784)
    
    # Benchmark original network
    print("⏱️ Benchmarking original network...")
    start = time.time()
    for _ in range(100):
        output_orig = network_original.forward(X)
    time_original = time.time() - start
    
    # Benchmark optimized network  
    print("⚡ Benchmarking optimized network...")
    start = time.time()
    for _ in range(100):
        output_opt = network_optimized.forward(X)
    time_optimized = time.time() - start
    
    # Calculate improvement
    speedup = time_original / time_optimized
    time_saved = time_original - time_optimized
    
    print(f"\n🎉 Results:")
    print(f"  Original network: {time_original:.3f}s")
    print(f"  Optimized network: {time_optimized:.3f}s") 
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved: {time_saved:.3f}s ({time_saved/time_original*100:.1f}%)")
    
    # Verify outputs are identical (within numerical precision)
    assert np.allclose(output_orig, output_opt, rtol=1e-10), "Outputs don't match!"
    print(f"  ✅ Numerical correctness verified")

test_full_network_improvement()
```

**Expected Results:**
```
🎉 Results:
  Original network: 2.450s
  Optimized network: 0.035s
  Speedup: 70.0x
  Time saved: 2.415s (98.6%)
  ✅ Numerical correctness verified
```

---

## **Phase 4: Documentation & Analysis**

### **Step 8: Document Your Engineering Decisions**

Create `capstone_report.md`:

```markdown
# Performance Optimization Capstone Report

## Problem Analysis
TinyTorch's matrix multiplication was 80x slower than NumPy, making training 
impractically slow. Profiling showed 99.6% of computation time in `matmul_naive`.

## Technical Approach  
1. **Root Cause**: Triple-nested loops with poor cache locality
2. **Solution**: Leverage NumPy's optimized BLAS backend
3. **Enhancement**: Add block-based multiplication for huge matrices
4. **Polish**: Memory layout optimization for cache efficiency

## Engineering Trade-offs
- **Gained**: 70x speedup in real networks, maintained numerical precision
- **Lost**: Educational visibility into low-level matrix multiplication
- **Justified**: Students learn optimization thinking, not reinventing BLAS

## Performance Results
- Dense layer operations: 80x faster
- Full network training: 70x faster  
- Memory usage: Unchanged
- Numerical accuracy: Maintained (1e-10 relative tolerance)

## Future Optimizations
1. GPU acceleration using CuPy/JAX
2. Sparse matrix support for compressed models
3. Mixed-precision training for memory efficiency
```

### **Step 9: Create Demonstration**

Create `demo.py`:

```python
"""
TinyTorch Performance Optimization Demo

This demonstrates the 70x speedup achieved through matrix operation optimization.
Run this to see before/after performance on your machine.
"""

import time
import numpy as np
from tinytorch.core.networks import Sequential
from tinytorch.core.layers import Dense, DenseOptimized
from tinytorch.core.activations import ReLU

def main():
    print("🔥 TinyTorch Performance Optimization Demo")
    print("=" * 50)
    
    # Create test scenario: MNIST-like classification
    print("📊 Scenario: MNIST-like classification (784→256→128→10)")
    batch_size = 64
    X = np.random.randn(batch_size, 784)
    
    # Original network
    network_original = Sequential([
        Dense(784, 256), ReLU(),
        Dense(256, 128), ReLU(), 
        Dense(128, 10)
    ])
    
    # Optimized network
    network_optimized = Sequential([
        DenseOptimized(784, 256), ReLU(),
        DenseOptimized(256, 128), ReLU(),
        DenseOptimized(128, 10)
    ])
    
    # Benchmark
    print("\n⏱️ Running 1000 forward passes...")
    
    # Original
    start = time.time()
    for _ in range(1000):
        _ = network_original.forward(X)
    time_orig = time.time() - start
    
    # Optimized  
    start = time.time()
    for _ in range(1000):
        _ = network_optimized.forward(X)
    time_opt = time.time() - start
    
    # Results
    speedup = time_orig / time_opt
    print(f"\n🎉 Results:")
    print(f"  Original: {time_orig:.2f}s")
    print(f"  Optimized: {time_opt:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved: {time_orig - time_opt:.2f}s")
    
    if speedup > 50:
        print(f"  🚀 Excellent optimization!")
    elif speedup > 20:
        print(f"  ⚡ Great improvement!")
    else:
        print(f"  📈 Good progress, consider further optimization")

if __name__ == "__main__":
    main()
```

---

## **🎯 Your Turn: Apply This Process**

This example showed **Performance Engineering**. Now apply this same systematic approach to your chosen track:

### **For Algorithm Extensions:**
1. **Profile**: Which algorithms are missing from your framework?
2. **Plan**: What modern techniques would add most value?
3. **Implement**: Build new layers/optimizers using existing TinyTorch components
4. **Test**: Verify they work with your training pipeline
5. **Document**: Explain design decisions and integration patterns

### **For Systems Optimization:**
1. **Profile**: Where does memory usage spike? What limits parallelization?
2. **Plan**: Which systems improvements would have biggest impact?
3. **Implement**: Add memory profiling, gradient accumulation, checkpointing
4. **Test**: Verify improvements don't break existing functionality
5. **Document**: Analyze trade-offs between memory, speed, complexity

### **For Framework Analysis:**
1. **Profile**: How does TinyTorch compare to PyTorch on key operations?
2. **Plan**: What benchmarks would be most revealing?
3. **Implement**: Automated testing suites comparing both frameworks
4. **Test**: Run comprehensive performance analysis
5. **Document**: Identify specific optimization opportunities

### **For Developer Experience:**
1. **Profile**: What makes debugging TinyTorch difficult?
2. **Plan**: Which tools would help developers most?
3. **Implement**: Gradient visualization, error diagnosis, testing utilities
4. **Test**: Use tools on real debugging scenarios
5. **Document**: Show how tools improve development workflow

---

## **🚀 Success Criteria Reminder**

Your capstone succeeds when you can show:

1. **Measurable Impact**: 20%+ improvement in your chosen area
2. **Systems Integration**: Your improvements work with all TinyTorch modules
3. **Engineering Insight**: You understand and can explain the trade-offs
4. **Professional Documentation**: Clear problem, solution, and results

**Remember**: You're not just optimizing code—you're proving you understand ML systems engineering at the framework level.

**🔥 Start with profiling your current TinyTorch framework and identifying your biggest optimization opportunity!** 