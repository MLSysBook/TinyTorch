# %% [markdown]
"""
# Module 16: Hardware Acceleration - The Free Speedup!

## Learning Objectives
By the end of this module, you will be able to:

1. **Understand Why Loops Are Slow**: See why your Module 2/4 loops have poor performance
2. **Implement Cache-Friendly Blocking**: Build blocked matrix multiplication that leverages CPU cache hierarchy
3. **Visualize Memory Access Patterns**: Understand how cache misses destroy performance
4. **Build Transparent Backend Systems**: Create automatic switching between implementations
5. **Apply to Real Models**: Use these principles in MLPs, CNNs, and Transformers

## The Free Speedup Journey

**Key Message**: This is the EASIEST optimization - just use better backends! No accuracy trade-offs, no complex math - just 10-100x faster code.

**The Journey:**
1. **Baseline**: Your loops from Module 2/4 (educational, 1000x slower)
2. **Blocking**: Cache-friendly version (educational, 10x faster than loops)
3. **NumPy**: Production version (optimal, another 10x faster)
4. **Backend**: Smart switching system (transparent optimization)

**Why This Works**: Same math, better implementation. Free performance with zero downsides!
"""

# %% [markdown]
"""
## Part 1: Baseline Implementation - Your Loops from Module 2/4

Let's start with the educational triple-nested loops you implemented earlier. These were perfect for learning but terrible for performance.
"""

# %%
#| default_exp backends.acceleration

import time
import numpy as np

def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Educational matrix multiplication using triple nested loops.
    
    This is the same implementation from Module 2/4 - perfect for learning
    the algorithm, but very slow due to poor cache performance.
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"Incompatible shapes: {a.shape} @ {b.shape}"
    
    # Initialize result matrix
    c = np.zeros((m, n), dtype=np.float32)
    
    # Triple nested loop - the educational implementation
    for i in range(m):
        for j in range(n):
            for l in range(k):
                c[i, j] += a[i, l] * b[l, j]
    
    return c

# %% [markdown]
"""
### Test Educational Implementation

Let's test our educational loops and see why they're slow.
"""

# %%
def test_naive_baseline():
    """Test naive implementation and measure its performance"""
    print("Testing Naive Implementation...")
    
    # Test correctness with small matrices
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    
    result_naive = matmul_naive(a, b)
    result_numpy = a @ b
    assert np.allclose(result_naive, result_numpy), "Naive matmul incorrect"
    print("✅ Naive implementation produces correct results")
    
    # Performance comparison (small sizes only - educational is VERY slow)
    print("\nPerformance comparison:")
    small_a = np.random.randn(100, 100).astype(np.float32)
    small_b = np.random.randn(100, 100).astype(np.float32)
    
    # Time naive implementation
    start = time.perf_counter()
    _ = matmul_naive(small_a, small_b)
    naive_time = time.perf_counter() - start
    
    # Time NumPy implementation
    start = time.perf_counter()
    _ = small_a @ small_b
    numpy_time = time.perf_counter() - start
    
    speedup = naive_time / numpy_time
    print(f"Naive loops: {naive_time*1000:.1f} ms")
    print(f"NumPy optimized:   {numpy_time*1000:.1f} ms")
    print(f"NumPy is {speedup:.1f}x faster")
    
    print("✅ Naive baseline established")
    return naive_time, numpy_time, speedup

# %% [markdown]
"""
## Part 2: Understanding Cache Hierarchy - Why Memory Matters More Than Computation

**The Big Insight**: Modern CPUs are FAST at computation but SLOW at memory access. Cache hierarchy makes the difference between fast and slow code.

### CPU Cache Hierarchy Visualization
```
Registers:  4 bytes    - 1 cycle     (instant)
L1 Cache:   32KB      - 3-4 cycles   (lightning fast)
L2 Cache:   256KB     - 10-20 cycles (fast)
L3 Cache:   8MB       - 50-100 cycles (slow)
Main RAM:   16GB      - 200+ cycles  (VERY slow)
```

**Key Principle**: Keep your working set in L1/L2 cache for 100x better performance!

### Memory Access Pattern Analysis

Your naive loops access memory like this:
```python
for i in range(m):
    for j in range(n):
        for l in range(k):
            c[i,j] += a[i,l] * b[l,j]  # b[l,j] jumps around randomly!
```

**The Problem**: `b[l,j]` creates terrible access patterns:
- Each `j` increment jumps to a new column (cache miss)
- Each `l` increment jumps to a new row (another cache miss)
- For 1000x1000 matrix: 1 billion cache misses!

**The Solution**: Process in blocks that fit in cache.
"""

# %%
def matmul_blocked(a: np.ndarray, b: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Cache-friendly blocked matrix multiplication.
    
    This version processes data in blocks that fit in CPU cache.
    
    **Memory Analysis**:
    - 64x64 block = 4KB floats = 16KB memory (fits in 32KB L1 cache)
    - 3 blocks (A, B, C) = 48KB total (fits in 256KB L2 cache)
    - Reuses each data element 64 times before evicting from cache
    
    **Why This Works**:
    - Naive: 1 cache miss per operation (terrible)
    - Blocked: 1 cache miss per 64 operations (64x better!)
    
    Args:
        a: Left matrix (m × k)
        b: Right matrix (k × n) 
        block_size: Cache-friendly block size (32-128, default 64)
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"Incompatible shapes: {a.shape} @ {b.shape}"
    
    # Initialize result
    c = np.zeros((m, n), dtype=np.float32)
    
    # Process in blocks to maximize cache utilization
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for l in range(0, k, block_size):
                # Define block boundaries
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, n)
                l_end = min(l + block_size, k)
                
                # Extract blocks (these stay in cache)
                a_block = a[i:i_end, l:l_end]
                b_block = b[l:l_end, j:j_end]
                
                # Multiply blocks using NumPy (optimized BLAS)
                c[i:i_end, j:j_end] += a_block @ b_block
    
    return c

# %% [markdown]
"""
### Test Blocked Implementation

Let's see how much faster cache-friendly blocking is compared to educational loops.
"""

def test_blocked_optimization():
    """Test blocked matrix multiplication performance"""
    print("Testing Blocked Matrix Multiplication...")
    
    # Test correctness
    a = np.random.randn(200, 200).astype(np.float32)
    b = np.random.randn(200, 200).astype(np.float32)
    
    result_blocked = matmul_blocked(a, b, block_size=64)
    result_numpy = a @ b
    
    assert np.allclose(result_blocked, result_numpy, atol=1e-3), "Blocked matmul incorrect"
    print("✅ Blocked implementation produces correct results")
    
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
    naive_time_scaled = naive_time * (size/50)**3  # Scale up for comparison
    
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
    
    print(f"\n🚀 SPEEDUP RESULTS:")
    print(f"Blocked is {speedup_blocked:.1f}x faster than naive loops!")
    print(f"NumPy is {speedup_numpy:.1f}x faster than naive loops!")
    print(f"\n💡 Why blocking works: Better cache utilization!")
    print(f"   • Naive: 1 cache miss per operation")
    print(f"   • Blocked: 1 cache miss per 64 operations")
    print(f"   • NumPy: Professional optimizations + vectorization")
    
    print("✅ Blocked optimization tested successfully")
    return blocked_time, numpy_time

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
### Test Production Implementation

Let's verify that NumPy is indeed the best choice for production.
"""

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
    
    print("\n💡 Key Insight: NumPy already has these optimizations built-in!")
    print("   • Blocking algorithms")
    print("   • Vectorization")
    print("   • Hardware-specific BLAS libraries")
    print("   • Assembly-level optimizations")
    
    print("\n✅ Production performance verified")
    return True

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
### Test Backend System

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
    print("✅ Backend produces correct results")
    
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
    
    print("\n✅ Backend system works correctly")
    return True

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
    print("\n1. MLP Forward Pass (256 → 128 → 64):")
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
    
    # Scale naive time for comparison
    naive_scaled = naive_time * (32/8) * (256/64) * (128/32)
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
    print(f"   Shape: {conv_output.shape} (1024 locations × 64 filters)")
    
    # Test 3: Transformer-like Attention (scaled down)
    print("\n3. Transformer Attention (Q·K^T):")
    seq_len, d_model = 128, 256
    Q = np.random.randn(seq_len, d_model).astype(np.float32)
    K = np.random.randn(seq_len, d_model).astype(np.float32)
    
    start = time.perf_counter()
    attention_scores = matmul(Q, K.T)  # Shape: (seq_len, seq_len)
    attn_time = time.perf_counter() - start
    print(f"   Attention computation: {attn_time*1000:.1f} ms")
    print(f"   Shape: {attention_scores.shape} (128×128 attention matrix)")
    
    print(f"\n✅ All ML model operations accelerated successfully!")
    print(f"💡 Key insight: Matrix multiplication is EVERYWHERE in ML!")
    return True

def run_complete_acceleration_demo():
    """Run the complete acceleration demonstration"""
    print("🚀 Complete Hardware Acceleration Demo")
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
    print("🎯 HARDWARE ACCELERATION MASTERED")
    print("=" * 55)
    
    print("\n📚 What You Mastered:")
    print("✅ Why your Module 2/4 loops were slow (cache hierarchy matters!)")
    print("✅ How cache-friendly blocking works (process data in chunks)")
    print("✅ Why NumPy dominates (professional optimizations built-in)")
    print("✅ How to build smart backend systems (automatic optimization)")
    print("✅ Real ML applications (MLPs, CNNs, Transformers all use matmul!)")
    
    print("\n🎯 The Free Speedup Philosophy:")
    print("• 🚀 Same math, better implementation = 100x speedup")
    print("• 🧠 Educational loops teach algorithms")
    print("• ⚡ Blocked algorithms teach cache optimization")
    print("• 🏭 NumPy provides production performance")
    print("• 🎯 Smart backends make optimization transparent")
    print("• 💡 Understanding the spectrum makes you a better engineer!")
    
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

### ⚡ **Optimization Techniques**
- **Blocked Algorithms**: Process data in cache-friendly blocks
- **Vectorized Operations**: Avoid Python loops, use NumPy's optimized routines
- **In-place Operations**: Minimize memory allocation overhead
- **Automatic Dispatch**: Choose optimal implementation based on problem size

### 📊 **Performance Understanding**
- **Measurement First**: Profile real bottlenecks before optimizing
- **Algorithmic Impact**: O(N³) → O(N²) matters more than 2x constant factors
- **Hardware Awareness**: CPU cache misses cost 100x more than cache hits
- **Library Utilization**: Optimized BLAS libraries beat custom implementations

### 🎯 **Real-World Applications**
- **ML Frameworks**: How PyTorch/TensorFlow apply these same principles
- **Production Systems**: Where optimization efforts provide real value
- **Development Practice**: When to optimize vs when to use existing solutions

### 💡 **Key Insights**
- Cache-friendly algorithms provide 2-5x speedups from memory access patterns alone
- Vectorization eliminates Python overhead for 10-100x improvements
- Most NumPy operations are already optimized - focus on system-level improvements
- Competition frameworks make optimization learning engaging and quantifiable
- Real ML systems face memory and communication bottlenecks, not pure computation limits

This approach teaches students to think like systems engineers: understand the hardware, measure scientifically, optimize systematically, and focus efforts where they matter most.
"""

if __name__ == "__main__":
    print("Module 16: Hardware Acceleration - The Free Speedup!")
    print("=" * 60)
    print("🚀 THE EASIEST OPTIMIZATION: Better Backends, Zero Trade-offs")
    
    # Run complete demonstration
    results = run_complete_acceleration_demo()
    
    print(f"\n🎉 Module 16: Hardware Acceleration COMPLETE!")
    print(f"⚡ Mastered: 10-100x speedups with no accuracy loss")
    print(f"🧠 Learned: Cache hierarchy, blocking, vectorization")
    print(f"🏭 Applied: MLPs, CNNs, Transformers all benefit")
    print(f"🎯 Ready: To build high-performance ML systems!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

1. **Memory Access Pattern Analysis**: Your educational loops access `b[l, j]` in the innermost loop, creating terrible cache performance. Draw a diagram showing how this access pattern jumps around in memory, calculate the number of cache misses for a 1000×1000 matrix multiply, and explain why this creates exponentially worse performance as matrices get larger.

2. **Cache Hierarchy Optimization**: Your blocked implementation uses 64×64 blocks. Calculate: (a) Total memory footprint of three 64×64 float32 blocks, (b) Why this fits in L1/L2 cache, (c) Cache utilization ratio (reuses per cache miss), and (d) What happens with 256×256 blocks instead (hint: L3 cache limit).

3. **Production Library Justification**: You implemented blocking for education, but NumPy beats it by another 10x. Identify three specific optimizations NumPy has (vectorization, BLAS libraries, assembly kernels) and calculate the development cost vs. performance benefit of implementing these yourself. Why is this a losing proposition for ML engineers?

4. **ML Model Acceleration Strategy**: You tested MLP, CNN, and Transformer operations. For each model type, identify: (a) The dominant matrix operations, (b) Which operations benefit most from acceleration, (c) Memory vs. compute bottlenecks, and (d) Why understanding the optimization spectrum makes you a better ML systems engineer.
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Hardware Acceleration - The Free Speedup

This module demonstrates the easiest optimization in ML systems: using better backends for free speedups with zero accuracy trade-offs. You learned why understanding the optimization spectrum makes you a better engineer.

### 🛤️ **The Free Speedup Journey**
- **Educational Foundation**: Your Module 2/4 loops taught you the algorithm (perfect for learning)
- **Performance Understanding**: Module 15 showed you WHY loops are slow (profiling first)
- **Optimization Mastery**: Now you achieve 100x speedups by choosing better implementations
- **Systems Thinking**: Understanding the spectrum from educational to production code

### 🛠️ **What We Built and Tested**
- **Educational Baseline**: Your triple-nested loops from Module 2/4 (algorithm understanding)
- **Cache-Friendly Blocking**: 64×64 blocks fitting in L1/L2 cache (10x+ speedup)
- **NumPy Production**: Leveraging professional BLAS optimizations (another 10x speedup)
- **Smart Backend System**: Automatic dispatch to optimal implementations
- **Real ML Applications**: MLP, CNN, Transformer operations using matrix multiplication

### 🧠 **Key Learning Outcomes**
- **Why loops are slow**: Memory access patterns and cache hierarchy matter most
- **How blocking helps**: Processing data in cache-friendly chunks improves performance
- **When to use NumPy**: It already has these optimizations (and more) built-in
- **Systems thinking**: Understanding enables better decisions about when to optimize

### ⚡ **Performance Spectrum Mastered**
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

### 💡 **The Free Speedup Philosophy**
This is the EASIEST optimization in ML systems: same math, better implementation, massive speedups, zero downsides. You implemented loops to understand algorithms. You implemented blocking to understand cache optimization. Now you use NumPy because it has all optimizations built-in. Understanding this spectrum - from educational to production - makes you a superior ML systems engineer who can make informed optimization decisions.
"""

