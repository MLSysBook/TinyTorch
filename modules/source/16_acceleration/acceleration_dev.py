# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#| default_exp optimization.acceleration
#| export

# %% [markdown]
"""
# Module 16: Acceleration - Making Models Run Faster

Welcome to Module 16! You're about to master the art of neural network acceleration through vectorization, kernel fusion, and mixed precision training.

## 🔗 Prerequisites & Progress
**You've Built**: Complete training pipeline with profiling capabilities
**You'll Build**: Acceleration techniques including vectorization, operation fusion, and mixed precision
**You'll Enable**: Production-ready optimization for real-world deployment

**Connection Map**:
```
Profiling (Module 15) → Acceleration (Module 16) → Quantization (Module 17)
(measurement)         (optimization)             (precision reduction)
```

## Learning Objectives
By the end of this module, you will:
1. Implement vectorized operations for maximum throughput
2. Create fused operations to reduce memory bandwidth
3. Build mixed precision training for memory efficiency
4. Understand the relationship between compute and memory bandwidth
5. Analyze acceleration trade-offs in production systems

Let's optimize for speed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/16_acceleration/acceleration_dev.py`  
**Building Side:** Code exports to `tinytorch.optimization.acceleration`

```python
# How to use this module:
from tinytorch.optimization.acceleration import vectorized_matmul, fused_gelu, MixedPrecisionTrainer
```

**Why this matters:**
- **Learning:** Complete acceleration system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.amp and torch.jit with optimization components
- **Consistency:** All acceleration operations and mixed precision training in optimization.acceleration
- **Integration:** Works seamlessly with profiling for complete performance optimization
"""

# %%
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# %% [markdown]
"""
## 1. Introduction - The Performance Challenge

Modern neural networks face two fundamental bottlenecks that limit their speed:

### The Two Enemies of Performance

**1. Compute Bound Operations:**
```
CPU/GPU Cores: [====BUSY====] [====BUSY====] [====BUSY====]
Memory Bus:    [---idle---] [---idle---] [---idle---]

When: Matrix multiplication, convolutions
Solution: Vectorization, better algorithms
```

**2. Memory Bound Operations:**
```
CPU/GPU Cores: [--idle--] [--idle--] [--idle--]
Memory Bus:    [========SATURATED========]

When: Element-wise operations, small tensors
Solution: Kernel fusion, memory layout optimization
```

### The Roofline Model - Your Performance Compass

Every processor has fundamental limits:

```
Performance    │   Compute Bound Region
(GFLOPS)      │  ┌─────────────────────
              │  │ Peak Performance
              │  │
              │ ╱│ Memory Bound Region
              │╱ │
             ╱│  │
            ╱ │  │
           ╱  │  │
          ╱───│──│───────────────────────
         ╱    │  │
        ╱     │  │
       ╱──────│──│────────────────── Arithmetic Intensity
              │  │        (FLOPs/Byte)
           Low│  │High
```

**Key Insight**: Understand where your operations live on this graph to optimize effectively.

### Why This Module Matters

Real-world performance wins:
- **2-5× speedup** from vectorization
- **30-50% memory reduction** from mixed precision
- **2-3× throughput** from kernel fusion
- **10× scaling improvement** for large models
"""

# %% nbgrader={"grade": false, "grade_id": "tensor-import", "solution": true}
# Import required dependencies
### BEGIN SOLUTION
from tinytorch.core.tensor import Tensor
### END SOLUTION

# %% [markdown]
"""
## 2. Foundations - Vectorization: From Loops to Lightning

### The SIMD Revolution

Modern processors can execute **Single Instruction, Multiple Data** operations:

```
Traditional Loop (Scalar):               SIMD Vectorized:
for i in range(4):        ┌─────┐      ┌─────┬─────┬─────┬─────┐
    c[i] = a[i] + b[i]    │ ALU │  →   │ALU 0│ALU 1│ALU 2│ALU 3│
                          └─────┘      └─────┴─────┴─────┴─────┘
                          1 element     4 elements per cycle
                          per cycle
```

### Memory Access Patterns: The Hidden Performance Killer

```
Sequential Access (FAST):
Memory: [A][B][C][D][E][F][G][H]
Access:  ↓  ↓  ↓  ↓  → Cache friendly

Strided Access (SLOWER):
Memory: [A][ ][B][ ][C][ ][D][ ]
Access:  ↓     ↓     ↓     ↓   → Cache misses

Random Access (SLOWEST):
Memory: [A][B][C][D][E][F][G][H]
Access:  ↓     ↑  ↓     ↑       → Cache chaos
```

### Matrix Multiplication: The King of Vectorization

Matrix multiplication is **perfectly suited** for vectorization:

```
Matrix A (M×K) × Matrix B (K×N) = Matrix C (M×N)

Computation Pattern:
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ a₁₁ a₁₂ a₁₃ a₁₄│ × │ b₁₁ b₁₂ b₁₃ b₁₄│ = │ c₁₁ c₁₂ c₁₃ c₁₄│
│ a₂₁ a₂₂ a₂₃ a₂₄│   │ b₂₁ b₂₂ b₂₃ b₂₄│   │ c₂₁ c₂₂ c₂₃ c₂₄│
│ a₃₁ a₃₂ a₃₃ a₃₄│   │ b₃₁ b₃₂ b₃₃ b₃₄│   │ c₃₁ c₃₂ c₃₃ c₃₄│
│ a₄₁ a₄₂ a₄₃ a₄₄│   │ b₄₁ b₄₂ b₄₃ b₄₄│   │ c₄₁ c₄₂ c₄₃ c₄₄│
└─────────────────┘   └─────────────────┘   └─────────────────┘

For c₁₁: Row₁ · Column₁ = a₁₁×b₁₁ + a₁₂×b₂₁ + a₁₃×b₃₁ + a₁₄×b₄₁
                                    ↑
                              VECTORIZABLE!
```

**Why vectorization wins:**
- **High arithmetic intensity**: 2N³ FLOPs for N³ data
- **Predictable memory access**: Sequential row/column reads
- **Parallelizable**: Independent dot products
- **Cache-friendly**: Data reuse in inner loops
"""

# %% nbgrader={"grade": false, "grade_id": "vectorized-matmul", "solution": true}
def vectorized_matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    High-performance matrix multiplication using vectorized operations.

    This implementation leverages optimized BLAS libraries that use:
    - SIMD instructions for parallel computation
    - Cache-blocking for memory efficiency
    - Multi-threading for CPU parallelization

    TODO: Implement production-grade matrix multiplication

    APPROACH:
    1. Validate shapes are compatible for matrix multiplication
    2. Use NumPy's optimized dot product (calls BLAS GEMM)
    3. Return result wrapped in Tensor

    EXAMPLE:
    Matrix multiplication visualization:
    >>> a = Tensor([[1, 2], [3, 4]])  # 2×2
    >>> b = Tensor([[5, 6], [7, 8]])  # 2×2
    >>> result = vectorized_matmul(a, b)
    >>> print(result.data)
    [[19 22]    # [1×5+2×7, 1×6+2×8] = [19, 22]
     [43 50]]   # [3×5+4×7, 3×6+4×8] = [43, 50]

    PERFORMANCE CHARACTERISTICS:
    - Time Complexity: O(N³) but highly optimized
    - Space Complexity: O(N²) for result
    - Arithmetic Intensity: 2N³ FLOPs / 3N² bytes = 2N/3 (good for large N)

    HINTS:
    - Check a.shape[-1] == b.shape[-2] for inner dimension match
    - Use np.matmul() for batch support and optimization
    - Trust BLAS to handle the vectorization magic
    """
    ### BEGIN SOLUTION
    # Input validation for matrix multiplication
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"Matrix multiplication requires 2D+ tensors, got shapes {a.shape} and {b.shape}. "
            f"💡 HINT: Use reshape() to add dimensions if needed."
        )

    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Matrix multiplication shape mismatch: {a.shape} @ {b.shape}. "
            f"Inner dimensions must match: a.shape[-1]={a.shape[-1]} != b.shape[-2]={b.shape[-2]}. "
            f"💡 HINT: For A@B, A's columns must equal B's rows."
        )

    # Use NumPy's highly optimized matrix multiplication
    # This calls BLAS GEMM (General Matrix Multiply), which uses:
    # - SIMD vectorization for parallel arithmetic
    # - Cache blocking for memory efficiency
    # - Multi-threading on multi-core systems
    result_data = np.matmul(a.data, b.data)

    return Tensor(result_data)
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-vectorized-matmul", "locked": true, "points": 10}
def test_unit_vectorized_matmul():
    """🔬 Test vectorized matrix multiplication implementation."""
    print("🔬 Unit Test: Vectorized Matrix Multiplication...")

    # Test basic 2D multiplication
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    result = vectorized_matmul(a, b)

    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(result.data, expected), f"Basic matmul failed: expected {expected}, got {result.data}"

    # Test batch multiplication (3D tensors)
    batch_size, m, k, n = 2, 3, 4, 5
    a_batch = Tensor(np.random.randn(batch_size, m, k))
    b_batch = Tensor(np.random.randn(batch_size, k, n))
    result_batch = vectorized_matmul(a_batch, b_batch)

    assert result_batch.shape == (batch_size, m, n), f"Wrong batch shape: {result_batch.shape}"

    # Test broadcasting (different batch dimensions)
    a_single = Tensor(np.random.randn(m, k))
    b_batch = Tensor(np.random.randn(batch_size, k, n))
    result_broadcast = vectorized_matmul(a_single, b_batch)

    assert result_broadcast.shape == (batch_size, m, n), f"Broadcasting failed: {result_broadcast.shape}"

    # Test error cases
    try:
        vectorized_matmul(Tensor([1, 2, 3]), Tensor([4, 5]))  # 1D tensors
        assert False, "Should reject 1D tensors"
    except ValueError as e:
        assert "2D+" in str(e)

    try:
        vectorized_matmul(Tensor([[1, 2]]), Tensor([[1], [2], [3]]))  # Shape mismatch
        assert False, "Should reject incompatible shapes"
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()

    print("✅ vectorized_matmul works correctly!")

test_unit_vectorized_matmul()

# %% [markdown]
"""
## 3. Implementation - Kernel Fusion: Eliminating Memory Bottlenecks

### The Memory Bandwidth Crisis

Consider this innocent-looking computation: `y = gelu(x * weight + bias)`

**Naive Implementation (Memory Intensive):**
```
Step 1: temp1 = x * weight     → Write 4GB to memory
Step 2: temp2 = temp1 + bias   → Read 4GB, Write 4GB
Step 3: y = gelu(temp2)        → Read 4GB, Write 4GB
                                 Total: 20GB memory traffic!
```

**Fused Implementation (Memory Efficient):**
```
Single Step: y = gelu(x * weight + bias)  → Read 8GB, Write 4GB
                                            Total: 12GB memory traffic!
                                            60% memory bandwidth reduction!
```

### Understanding GELU: The Smooth Activation

GELU (Gaussian Error Linear Unit) is used in transformers because it's **smooth** (differentiable everywhere):

```
Activation Functions Compared:

ReLU:           GELU:           Sigmoid:
     |               |                 1 ┌─────
     |               |               ╱   │
     |           ╱───│───            ╱    │
─────┘       ╱───    │         ───╱     │
 Discontinuous   Smooth Curve    │ Smooth but saturates
 gradient at 0   everywhere      │
```

**GELU Formula**: `GELU(x) = x * Φ(x)` where Φ is the standard normal CDF

**Fast Approximation**: `GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

### Kernel Fusion Strategy

```
Unfused Operations:                    Fused Operation:
┌─────────────────┐                   ┌─────────────────┐
│ x³ computation  │ → temp1           │                 │
└─────────────────┘                   │                 │
┌─────────────────┐                   │                 │
│ polynomial part │ → temp2           │   All operations│
└─────────────────┘                   │   combined in   │
┌─────────────────┐                   │   single kernel │
│ tanh computation│ → temp3           │                 │
└─────────────────┘                   │                 │
┌─────────────────┐                   │                 │
│ final multiply  │ → result          │                 │
└─────────────────┘                   └─────────────────┘

5 memory round-trips                   1 memory round-trip
```
"""

# %% nbgrader={"grade": false, "grade_id": "fused-gelu", "solution": true}
def fused_gelu(x: Tensor) -> Tensor:
    """
    Fused GELU activation that combines all operations in a single kernel.

    GELU combines the benefits of ReLU and sigmoid:
    - Smooth everywhere (unlike ReLU's discontinuity at 0)
    - Non-saturating for positive values (unlike sigmoid)
    - Probabilistic interpretation: x * P(X ≤ x) where X ~ N(0,1)

    Mathematical Definition:
    GELU(x) = x * Φ(x) where Φ(x) is the standard normal CDF

    Fast Approximation (used here):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    TODO: Implement fused GELU to minimize memory bandwidth

    APPROACH:
    1. Compute all intermediate values in a single expression
    2. Avoid creating temporary arrays
    3. Let NumPy's broadcasting handle vectorization

    EXAMPLE:
    >>> x = Tensor([-2, -1, 0, 1, 2])
    >>> result = fused_gelu(x)
    >>> print(result.data)
    [-0.04550026 -0.15865526  0.          0.8413447   1.9544997 ]
    # Notice: smooth transition through 0, positive bias

    MEMORY EFFICIENCY:
    - Unfused: 5 temporary arrays × input_size × 4 bytes
    - Fused: 0 temporary arrays, direct computation
    - Bandwidth reduction: ~80% for memory-bound operations

    HINTS:
    - Use np.sqrt(2.0 / np.pi) for the constant
    - Keep entire expression in one line for maximum fusion
    - NumPy will optimize the expression tree automatically
    """
    ### BEGIN SOLUTION
    # Mathematical constant for GELU approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # Fused GELU computation - all operations in single expression
    # This minimizes memory bandwidth by avoiding intermediate arrays
    # NumPy's expression evaluator will optimize this into efficient machine code
    result_data = 0.5 * x.data * (
        1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3))
    )

    return Tensor(result_data)
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-fused-gelu", "locked": true, "points": 10}
def test_unit_fused_gelu():
    """🔬 Test fused GELU activation implementation."""
    print("🔬 Unit Test: Fused GELU...")

    # Test basic properties
    x = Tensor([-3, -1, 0, 1, 3])
    result = fused_gelu(x)

    # GELU(0) = 0 (exact property)
    assert abs(result.data[2]) < 1e-6, f"GELU(0) should be 0, got {result.data[2]}"

    # GELU is smooth and increasing
    assert result.data[4] > result.data[3] > result.data[2], "GELU should be increasing"

    # GELU has positive bias (unlike ReLU)
    assert result.data[3] > 0.8, "GELU(1) should be close to 1"
    assert result.data[1] > -0.2, "GELU(-1) should be slightly negative"

    # Test numerical stability with extreme values
    x_extreme = Tensor([-10, -5, 0, 5, 10])
    result_extreme = fused_gelu(x_extreme)

    assert not np.any(np.isnan(result_extreme.data)), "No NaN values allowed"
    assert not np.any(np.isinf(result_extreme.data)), "No infinite values allowed"

    # Test large tensor processing
    x_large = Tensor(np.random.randn(1000, 1000).astype(np.float32))
    result_large = fused_gelu(x_large)

    assert result_large.shape == x_large.shape, "Shape preservation failed"
    assert result_large.data.dtype == np.float32, "Data type preservation failed"

    # Test that positive inputs are mostly preserved (GELU ≈ x for large positive x)
    x_positive = Tensor([5.0])
    result_positive = fused_gelu(x_positive)
    assert result_positive.data[0] > 4.9, "Large positive values should be nearly preserved"

    print("✅ fused_gelu works correctly!")

test_unit_fused_gelu()

# %% [markdown]
"""
### 🔬 Performance Analysis: Measuring Fusion Benefits

Let's quantify the impact of kernel fusion by comparing fused vs unfused implementations.
"""

# %% nbgrader={"grade": false, "grade_id": "unfused-gelu", "solution": true}
def unfused_gelu(x: Tensor) -> Tensor:
    """
    Deliberately unfused GELU implementation for performance comparison.

    This version creates multiple intermediate tensors to simulate
    the memory bandwidth overhead of unfused operations.

    TODO: Implement GELU with explicit intermediate steps

    APPROACH:
    1. Break computation into individual steps
    2. Create temporary Tensor objects for each step
    3. This simulates real memory allocation overhead

    PERFORMANCE IMPACT:
    - Creates 7 temporary arrays
    - Each array allocation/deallocation has overhead
    - More memory bandwidth usage
    - Potential cache misses between operations
    """
    ### BEGIN SOLUTION
    # Unfused version - creates many intermediate arrays
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    # Each operation creates a temporary array (simulating kernel launches)
    temp1 = Tensor(x.data**3)  # x³
    temp2 = Tensor(0.044715 * temp1.data)  # 0.044715 * x³
    temp3 = Tensor(x.data + temp2.data)  # x + 0.044715 * x³
    temp4 = Tensor(sqrt_2_over_pi * temp3.data)  # √(2/π) * (...)
    temp5 = Tensor(np.tanh(temp4.data))  # tanh(...)
    temp6 = Tensor(1.0 + temp5.data)  # 1 + tanh(...)
    temp7 = Tensor(x.data * temp6.data)  # x * (1 + tanh(...))
    result = Tensor(0.5 * temp7.data)  # 0.5 * x * (...)

    return result
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-fusion-speedup", "locked": true, "points": 10}
def test_unit_fusion_speedup():
    """🔬 Measure the performance impact of kernel fusion."""
    print("🔬 Unit Test: Kernel Fusion Performance Impact...")

    # Create moderately large tensor for meaningful timing
    size = 2000
    x = Tensor(np.random.randn(size, size).astype(np.float32))
    warmup_iterations = 2
    timing_iterations = 5

    # Warmup both implementations
    for _ in range(warmup_iterations):
        _ = unfused_gelu(x)
        _ = fused_gelu(x)

    # Time unfused version
    start = time.time()
    for _ in range(timing_iterations):
        result_unfused = unfused_gelu(x)
    unfused_time = time.time() - start

    # Time fused version
    start = time.time()
    for _ in range(timing_iterations):
        result_fused = fused_gelu(x)
    fused_time = time.time() - start

    # Verify numerical correctness
    assert np.allclose(result_unfused.data, result_fused.data, atol=1e-6), \
        "Fused and unfused implementations must be numerically equivalent"

    # Calculate performance metrics
    speedup = unfused_time / fused_time if fused_time > 0 else 1.0
    unfused_per_elem = (unfused_time / timing_iterations) / (size * size) * 1e9  # ns per element
    fused_per_elem = (fused_time / timing_iterations) / (size * size) * 1e9

    print(f"📊 Kernel Fusion Performance Analysis:")
    print(f"   Tensor size: {size}×{size} = {size*size:,} elements")
    print(f"   Unfused time: {unfused_time/timing_iterations*1000:.2f} ms")
    print(f"   Fused time:   {fused_time/timing_iterations*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}× faster")
    print(f"   Per-element: {unfused_per_elem:.1f} ns → {fused_per_elem:.1f} ns")

    # Memory bandwidth estimate
    bytes_per_elem = 4  # float32
    unfused_memory_ops = 7  # 7 intermediate arrays
    fused_memory_ops = 2   # read input, write output

    unfused_bandwidth = (unfused_memory_ops * size * size * bytes_per_elem) / (unfused_time / timing_iterations) / 1e9
    fused_bandwidth = (fused_memory_ops * size * size * bytes_per_elem) / (fused_time / timing_iterations) / 1e9

    print(f"   Memory efficiency: {unfused_memory_ops}→{fused_memory_ops} memory ops")
    print(f"   Effective bandwidth: {unfused_bandwidth:.1f}→{fused_bandwidth:.1f} GB/s")

    # Interpret results
    if speedup > 1.5:
        print("🚀 Excellent! Kernel fusion providing significant speedup")
    elif speedup > 1.1:
        print("✅ Good! Kernel fusion providing measurable benefit")
    else:
        print("⚠️  Limited speedup - may be compute-bound or small tensor size")

    print("✅ Fusion performance analysis completed!")

test_unit_fusion_speedup()

# %% [markdown]
"""
## 4. Integration - Mixed Precision Training: Memory and Speed

### The Mixed Precision Revolution

Modern GPUs (like V100, A100) have specialized **Tensor Cores** that can perform FP16 operations much faster than FP32:

```
Performance Comparison (Theoretical Peak):
┌─────────────────┬────────────────┬────────────────┐
│   Precision     │   V100 TFLOPS  │   A100 TFLOPS  │
├─────────────────┼────────────────┼────────────────┤
│   FP32 (float)  │      15.7      │      19.5      │
│   FP16 (half)   │     125.0      │     312.0      │
│   Speedup       │      8×        │      16×       │
└─────────────────┴────────────────┴────────────────┘
```

### The Challenge: FP16 Precision Limitations

FP16 has a much smaller range than FP32:

```
FP32 (32-bit):                    FP16 (16-bit):
┌─────────────────────────────┐   ┌───────────────┐
│ Sign │ 8-bit │   23-bit     │   │Sign│5-bit│10-bit│
│  bit │ Exp   │  Mantissa    │   │bit │ Exp │Mant. │
└─────────────────────────────┘   └───────────────┘
Range: ±3.4 × 10³⁸              Range: ±6.5 × 10⁴
Precision: ~7 decimal digits     Precision: ~3 decimal digits

Problem: Small gradients (< 6e-5) become ZERO in FP16!
```

### The Solution: Automatic Loss Scaling

```
Training Step Without Scaling:       Training Step With Scaling:

Loss = 0.0001                       Loss = 0.0001
    ↓                                   ↓
Gradients = 0.00001                 Scale × 1024
    ↓                                   ↓
Convert to FP16                     Loss = 0.1024
    ↓                                   ↓
Gradients = 0.0 (UNDERFLOW!)        Gradients = 0.01024
    ↓                                   ↓
No learning!                        Convert to FP16: 0.01024 ✓
                                        ↓
                                    Unscale: 0.01024 / 1024 = 0.00001
                                        ↓
                                    Successful learning!
```

### Mixed Precision Memory Benefits

```
Model Component Breakdown:
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│   Component     │ FP32 Memory │ FP16 Memory │   Savings   │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Parameters      │    4N       │     4N      │     0%      │
│ Gradients       │    4N       │     2N      │    50%      │
│ Activations     │    4A       │     2A      │    50%      │
│ Optimizer State │    8N       │     8N      │     0%      │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Total Typical   │   ~20N      │    ~16N     │    20%      │
│ Activation-Heavy│   ~40N      │    ~24N     │    40%      │
└─────────────────┴─────────────┴─────────────┴─────────────┘

N = parameter count, A = activation memory
```
"""

# %% nbgrader={"grade": false, "grade_id": "mixed-precision-trainer", "solution": true}
#| export
class MixedPrecisionTrainer:
    """
    Mixed precision trainer with automatic loss scaling.

    Implements the same pattern as PyTorch's Automatic Mixed Precision (AMP):
    1. Forward pass in FP16 for speed and memory efficiency
    2. Loss scaling to prevent gradient underflow
    3. Gradient computation and unscaling
    4. Parameter updates in FP32 for numerical stability

    The key insight: keep different parts of training in optimal precision.
    """

    def __init__(self, model, optimizer, loss_scale: float = 1024.0, max_loss_scale: float = 65536.0):
        """
        Initialize mixed precision training infrastructure.

        TODO: Set up automatic loss scaling and overflow detection

        APPROACH:
        1. Store model and optimizer references
        2. Initialize dynamic loss scaling parameters
        3. Set up overflow detection and scale adjustment logic

        Args:
            model: Neural network model
            optimizer: Parameter optimizer (SGD, Adam, etc.)
            loss_scale: Initial scaling factor for gradients
            max_loss_scale: Maximum allowed loss scale

        LOSS SCALING STRATEGY:
        - Start with reasonable scale (1024)
        - Increase gradually if no overflow (better precision)
        - Decrease immediately on overflow (stability)
        - This balances numerical precision with training stability

        HINTS:
        - Track consecutive successful steps for scale increases
        - Use exponential backoff on overflow detection
        - Keep scale within reasonable bounds [1, 65536]
        """
        ### BEGIN SOLUTION
        self.model = model
        self.optimizer = optimizer

        # Loss scaling parameters
        self.loss_scale = loss_scale
        self.max_loss_scale = max_loss_scale
        self.min_loss_scale = 1.0

        # Dynamic scaling parameters
        self.scale_growth_factor = 2.0      # Multiply by 2 when increasing
        self.scale_backoff_factor = 0.5     # Divide by 2 when decreasing
        self.growth_interval = 2000         # Steps between scale increases
        self.steps_since_last_scale_update = 0

        # Overflow tracking
        self.overflow_detected = False
        ### END SOLUTION

    def scale_loss(self, loss: Tensor) -> Tensor:
        """
        Scale loss to prevent gradient underflow in FP16.

        The fundamental challenge: FP16 can only represent values ≥ 6e-5.
        Small gradients (common in deep networks) become zero without scaling.

        TODO: Apply loss scaling for mixed precision stability

        APPROACH:
        1. Multiply loss by current scale factor
        2. This amplifies gradients proportionally
        3. Return scaled loss for backward pass

        MATHEMATICAL INSIGHT:
        If loss = 1e-6 and scale = 1024:
        scaled_loss = 1e-6 × 1024 = 1.024e-3

        After backward pass:
        scaled_gradients = 1.024e-3 × dloss/dparam = 1024 × gradients

        These larger gradients survive FP16 conversion!

        EXAMPLE:
        >>> trainer = MixedPrecisionTrainer(model, optimizer)
        >>> loss = Tensor([0.0001])  # Small loss
        >>> scaled = trainer.scale_loss(loss)
        >>> print(scaled.data)  # [0.1024] (0.0001 × 1024)
        """
        ### BEGIN SOLUTION
        # Scale the loss to amplify gradients
        # This prevents gradient underflow in FP16 arithmetic
        scaled_data = loss.data * self.loss_scale
        return Tensor(scaled_data)
        ### END SOLUTION

    def unscale_gradients(self, parameters: List[Tensor]) -> bool:
        """
        Unscale gradients and detect overflow from FP16 conversion.

        After backward pass on scaled loss, gradients are scaled too.
        We must unscale them AND check for overflow/underflow.

        TODO: Implement gradient unscaling with overflow detection

        APPROACH:
        1. Divide all gradients by loss scale (restore original magnitude)
        2. Check for inf/nan values (indicates FP16 overflow)
        3. Return True if gradients are valid, False if overflow detected

        OVERFLOW DETECTION:
        inf/nan in gradients indicates:
        - Gradient magnitude too large for FP16
        - Numerical instability in computation
        - Loss scale too aggressive

        When overflow occurs:
        - Skip parameter update (unstable gradients)
        - Reduce loss scale for next iteration
        - Continue training with lower scale

        HINTS:
        - Use np.isfinite() to detect inf/nan efficiently
        - Process all parameters even if overflow found
        - Set self.overflow_detected flag for scale adjustment
        """
        ### BEGIN SOLUTION
        self.overflow_detected = False

        # Unscale all gradients and check for overflow
        for param in parameters:
            if param.grad is not None:
                # Unscale gradients to original magnitude
                param.grad.data = param.grad.data / self.loss_scale

                # Check for overflow/underflow (inf/nan values)
                if not np.all(np.isfinite(param.grad.data)):
                    self.overflow_detected = True
                    # Continue processing to unscale all gradients

        return not self.overflow_detected
        ### END SOLUTION

    def update_loss_scale(self):
        """
        Dynamically adjust loss scale based on training stability.

        Implements the "Goldilocks" principle for loss scaling:
        - Too low: precision loss from small gradients
        - Too high: overflow and instability
        - Just right: maximum precision without overflow

        TODO: Implement adaptive loss scale adjustment

        APPROACH:
        1. If overflow detected: reduce scale immediately (stability)
        2. If no overflow for many steps: increase scale (precision)
        3. Keep scale within reasonable bounds

        SCALING STRATEGY:
        - Aggressive reduction on overflow (×0.5)
        - Conservative growth during stability (×2 every 2000 steps)
        - This favors stability over maximum precision

        WHY THIS WORKS:
        - Most training is stable (gradual scale increase)
        - Occasional instability (rapid scale decrease)
        - Converges to optimal scale for current training phase
        """
        ### BEGIN SOLUTION
        if self.overflow_detected:
            # Immediately reduce scale on overflow
            self.loss_scale = max(
                self.min_loss_scale,
                self.loss_scale * self.scale_backoff_factor
            )
            self.steps_since_last_scale_update = 0
        else:
            # Gradually increase scale if stable
            self.steps_since_last_scale_update += 1
            if self.steps_since_last_scale_update >= self.growth_interval:
                self.loss_scale = min(
                    self.max_loss_scale,
                    self.loss_scale * self.scale_growth_factor
                )
                self.steps_since_last_scale_update = 0
        ### END SOLUTION

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """
        Execute complete mixed precision training step.

        Orchestrates the entire mixed precision training process:
        1. Forward pass (FP16 in real implementation)
        2. Loss computation and scaling
        3. Backward pass on scaled loss
        4. Gradient unscaling and overflow detection
        5. Conditional parameter update
        6. Loss scale adjustment

        TODO: Implement end-to-end mixed precision training step

        APPROACH:
        1. Clear gradients from previous step
        2. Forward pass through model
        3. Compute and scale loss
        4. Backward pass to compute scaled gradients
        5. Unscale gradients and check for overflow
        6. Update parameters only if no overflow
        7. Adjust loss scale based on stability

        CRITICAL INSIGHT:
        Skip parameter updates on overflow! Unstable gradients
        would move parameters in wrong direction.

        RETURN FORMAT:
        Dictionary with training metrics:
        - loss: unscaled loss value
        - loss_scale: current scaling factor
        - overflow: whether overflow occurred
        - gradients_valid: whether update was applied

        HINTS:
        - Use self.optimizer.zero_grad() to clear gradients
        - Get parameters with gradients for unscaling
        - Only call optimizer.step() if gradients are valid
        """
        ### BEGIN SOLUTION
        inputs, targets = batch

        # Clear gradients from previous step
        self.optimizer.zero_grad()

        # Forward pass (would use FP16 autocast in real implementation)
        # For simulation, we work in FP32 but apply scaling principles
        outputs = self.model(inputs)

        # Compute loss (unscaled)
        loss = self._compute_loss(outputs, targets)

        # Scale loss for mixed precision
        scaled_loss = self.scale_loss(loss)

        # Backward pass on scaled loss
        scaled_loss.backward()

        # Get all parameters with gradients
        parameters = [p for p in self.model.parameters() if p.grad is not None]

        # Unscale gradients and detect overflow
        gradients_valid = self.unscale_gradients(parameters)

        # Update parameters only if no overflow
        if gradients_valid:
            self.optimizer.step()

        # Adjust loss scale based on stability
        self.update_loss_scale()

        # Return training metrics
        return {
            'loss': loss.data.item() if hasattr(loss.data, 'item') else float(loss.data),
            'loss_scale': self.loss_scale,
            'overflow': self.overflow_detected,
            'gradients_valid': gradients_valid
        }
        ### END SOLUTION

    def _compute_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Simple MSE loss for demonstration purposes."""
        diff = Tensor(outputs.data - targets.data)
        return Tensor(np.mean(diff.data**2))

# %% nbgrader={"grade": true, "grade_id": "test-mixed-precision", "locked": true, "points": 15}
def test_unit_mixed_precision():
    """🔬 Test mixed precision training components comprehensively."""
    print("🔬 Unit Test: Mixed Precision Training...")

    # Create mock model and optimizer for testing
    class MockModel:
        def __init__(self):
            self.weight = Tensor(np.random.randn(10, 5).astype(np.float32))
            self.weight.grad = None

        def __call__(self, x):
            return x.matmul(self.weight)

        def parameters(self):
            return [self.weight]

    class MockOptimizer:
        def __init__(self, params):
            self.params = params
            self.updates_applied = 0

        def zero_grad(self):
            for p in self.params:
                p.grad = None

        def step(self):
            for p in self.params:
                if p.grad is not None:
                    p.data = p.data - 0.01 * p.grad.data
                    self.updates_applied += 1

    # Initialize mixed precision trainer
    model = MockModel()
    optimizer = MockOptimizer(model.parameters())
    trainer = MixedPrecisionTrainer(model, optimizer, loss_scale=1024.0)

    # Test 1: Loss scaling
    print("   Testing loss scaling...")
    loss = Tensor([0.001])
    scaled_loss = trainer.scale_loss(loss)
    expected_scaled = 0.001 * 1024.0
    assert np.isclose(scaled_loss.data[0], expected_scaled), \
        f"Loss scaling failed: expected {expected_scaled}, got {scaled_loss.data[0]}"

    # Test 2: Gradient unscaling (normal case)
    print("   Testing gradient unscaling...")
    model.weight.grad = Tensor(np.full((10, 5), 1024.0))  # Simulate scaled gradients
    valid = trainer.unscale_gradients([model.weight])
    assert valid, "Should detect valid gradients"
    assert np.allclose(model.weight.grad.data, 1.0), "Gradient unscaling failed"

    # Test 3: Overflow detection
    print("   Testing overflow detection...")
    model.weight.grad = Tensor(np.full((10, 5), np.inf))  # Simulate overflow
    valid = trainer.unscale_gradients([model.weight])
    assert not valid, "Should detect overflow"
    assert trainer.overflow_detected, "Overflow flag not set"

    # Test 4: Loss scale adjustment after overflow
    print("   Testing loss scale adjustment...")
    initial_scale = trainer.loss_scale
    trainer.update_loss_scale()  # Should reduce scale due to overflow
    assert trainer.loss_scale < initial_scale, \
        f"Scale should decrease after overflow: {initial_scale} → {trainer.loss_scale}"

    # Test 5: Loss scale increase during stability
    print("   Testing loss scale increase...")
    trainer.overflow_detected = False
    trainer.steps_since_last_scale_update = 2000  # Simulate stable training
    scale_before = trainer.loss_scale
    trainer.update_loss_scale()
    assert trainer.loss_scale > scale_before, "Scale should increase during stability"

    # Test 6: End-to-end training step
    print("   Testing complete training step...")
    inputs = Tensor(np.random.randn(8, 10).astype(np.float32))
    targets = Tensor(np.random.randn(8, 5).astype(np.float32))

    initial_updates = optimizer.updates_applied
    metrics = trainer.train_step((inputs, targets))

    # Verify metrics structure
    required_keys = ['loss', 'loss_scale', 'overflow', 'gradients_valid']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

    # Verify loss is reasonable
    assert isinstance(metrics['loss'], (int, float)), "Loss should be numeric"
    assert metrics['loss'] >= 0, "Loss should be non-negative"

    # Verify loss scale is positive
    assert metrics['loss_scale'] > 0, "Loss scale should be positive"

    print("✅ Mixed precision training works correctly!")

test_unit_mixed_precision()

# %% [markdown]
"""
## 5. Systems Analysis - Performance Scaling Patterns

Let's analyze how our acceleration techniques perform across different scenarios and understand their scaling characteristics.
"""

# %% nbgrader={"grade": false, "grade_id": "analyze-vectorization", "solution": true}
def analyze_vectorization_scaling():
    """📊 Analyze vectorization performance across different tensor sizes."""
    print("📊 Analyzing vectorization scaling behavior...")

    # Test sizes spanning different cache regimes
    sizes = [64, 128, 256, 512, 1024, 2048]

    print("\n🔍 Vectorization Scaling Analysis:")
    print("┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│  Size   │ Time (ms)   │ GFLOPS      │ Bandwidth   │ Efficiency  │")
    print("│         │             │             │ (GB/s)      │ (% of peak) │")
    print("├─────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for size in sizes:
        # Create test matrices
        a = Tensor(np.random.randn(size, size).astype(np.float32))
        b = Tensor(np.random.randn(size, size).astype(np.float32))

        # Warm up
        for _ in range(2):
            _ = vectorized_matmul(a, b)

        # Time vectorized implementation
        iterations = max(1, 100 // (size // 64))  # Fewer iterations for larger sizes
        start = time.time()
        for _ in range(iterations):
            result = vectorized_matmul(a, b)
        elapsed = (time.time() - start) / iterations

        # Calculate performance metrics
        flops = 2 * size**3  # 2N³ FLOPs for matrix multiplication
        gflops = flops / (elapsed * 1e9)

        bytes_accessed = 3 * size * size * 4  # 3 matrices × size² × 4 bytes
        bandwidth = bytes_accessed / (elapsed * 1e9)

        # Estimate efficiency (rough baseline: modern CPU ~100-500 GFLOPS peak)
        estimated_peak_gflops = 200  # Conservative estimate
        efficiency = min(100, gflops / estimated_peak_gflops * 100)

        print(f"│ {size:6d}  │ {elapsed*1000:9.2f}   │ {gflops:9.1f}   │ {bandwidth:9.1f}   │ {efficiency:9.1f}   │")

    print("└─────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Vectorization insights:")
    print(f"   • Small matrices: Limited by overhead and cache effects")
    print(f"   • Medium matrices: Sweet spot for cache reuse")
    print(f"   • Large matrices: Memory bandwidth becomes limiting factor")
    print(f"   • BLAS libraries automatically optimize for each size regime")
    print("🚀 Vectorization effectiveness depends on problem size and hardware")

analyze_vectorization_scaling()

# %% nbgrader={"grade": false, "grade_id": "analyze-arithmetic-intensity", "solution": true}
def analyze_arithmetic_intensity():
    """📊 Demonstrate the roofline model with different operations."""
    print("📊 Analyzing arithmetic intensity patterns...")

    size = 1024
    iterations = 10

    operations = []

    # Create test data
    x = Tensor(np.random.randn(size, size).astype(np.float32))
    y = Tensor(np.random.randn(size, size).astype(np.float32))

    print("\n🎯 Arithmetic Intensity Analysis:")
    print("┌─────────────────────┬─────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Operation           │ AI      │ Time (ms)   │ GFLOPS      │ GB/s        │")
    print("│                     │(FLOPs/B)│             │             │             │")
    print("├─────────────────────┼─────────┼─────────────┼─────────────┼─────────────┤")

    # 1. Element-wise addition (very low arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = Tensor(x.data + y.data)
    add_time = (time.time() - start) / iterations

    add_flops = size * size  # One addition per element
    add_bytes = 3 * size * size * 4  # Read x, read y, write result
    add_ai = add_flops / add_bytes
    add_gflops = add_flops / (add_time * 1e9)
    add_bandwidth = add_bytes / (add_time * 1e9)

    print(f"│ Element-wise Add    │ {add_ai:6.3f}  │ {add_time*1000:9.2f}   │ {add_gflops:9.1f}   │ {add_bandwidth:9.1f}   │")

    # 2. Element-wise multiply (still low, but slightly higher)
    start = time.time()
    for _ in range(iterations):
        _ = Tensor(x.data * y.data)
    mul_time = (time.time() - start) / iterations

    mul_flops = size * size
    mul_bytes = 3 * size * size * 4
    mul_ai = mul_flops / mul_bytes
    mul_gflops = mul_flops / (mul_time * 1e9)
    mul_bandwidth = mul_bytes / (mul_time * 1e9)

    print(f"│ Element-wise Mult   │ {mul_ai:6.3f}  │ {mul_time*1000:9.2f}   │ {mul_gflops:9.1f}   │ {mul_bandwidth:9.1f}   │")

    # 3. GELU (medium arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = fused_gelu(x)
    gelu_time = (time.time() - start) / iterations

    gelu_flops = size * size * 8  # Approximate: x³, add, mul, tanh, etc.
    gelu_bytes = 2 * size * size * 4  # Read x, write result
    gelu_ai = gelu_flops / gelu_bytes
    gelu_gflops = gelu_flops / (gelu_time * 1e9)
    gelu_bandwidth = gelu_bytes / (gelu_time * 1e9)

    print(f"│ Fused GELU          │ {gelu_ai:6.3f}  │ {gelu_time*1000:9.2f}   │ {gelu_gflops:9.1f}   │ {gelu_bandwidth:9.1f}   │")

    # 4. Matrix multiplication (high arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = vectorized_matmul(x, y)
    matmul_time = (time.time() - start) / iterations

    matmul_flops = 2 * size**3  # 2N³ FLOPs
    matmul_bytes = 3 * size * size * 4  # 3 matrices
    matmul_ai = matmul_flops / matmul_bytes
    matmul_gflops = matmul_flops / (matmul_time * 1e9)
    matmul_bandwidth = matmul_bytes / (matmul_time * 1e9)

    print(f"│ Matrix Multiply     │ {matmul_ai:6.3f}  │ {matmul_time*1000:9.2f}   │ {matmul_gflops:9.1f}   │ {matmul_bandwidth:9.1f}   │")

    print("└─────────────────────┴─────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Roofline Model Insights:")
    print(f"   📊 Low AI (< 1): Memory bound - limited by bandwidth")
    print(f"   📊 Med AI (1-10): Transitional - depends on implementation")
    print(f"   📊 High AI (> 10): Compute bound - limited by ALU throughput")
    print(f"   🎯 Matrix multiplication ({matmul_ai:.1f} AI) is ideal for GPUs/TPUs")
    print(f"   ⚡ Element-wise ops ({add_ai:.3f} AI) need memory optimization")
    print("🚀 Design algorithms with high arithmetic intensity for performance")

analyze_arithmetic_intensity()

# %% nbgrader={"grade": false, "grade_id": "analyze-mixed-precision-benefits", "solution": true}
def analyze_mixed_precision_benefits():
    """📊 Quantify mixed precision memory and performance benefits."""
    print("📊 Analyzing mixed precision benefits across model sizes...")

    # Define representative model configurations
    model_configs = [
        ("Tiny CNN", {"params": 50_000, "activations": 100_000}),
        ("Small BERT", {"params": 10_000_000, "activations": 5_000_000}),
        ("Medium GPT", {"params": 100_000_000, "activations": 50_000_000}),
        ("Large Transformer", {"params": 1_000_000_000, "activations": 500_000_000}),
    ]

    print("\n🧮 Mixed Precision Memory Analysis:")
    print("┌─────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Model Type      │ Parameters  │ FP32 Memory │ FP16 Memory │ Savings     │")
    print("│                 │             │ (GB)        │ (GB)        │ (%)         │")
    print("├─────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for name, config in model_configs:
        param_count = config["params"]
        activation_count = config["activations"]

        # Memory calculation (bytes)
        # Parameters: always FP32 for stability
        param_memory = param_count * 4

        # FP32 training memory
        fp32_activations = activation_count * 4
        fp32_gradients = param_count * 4
        fp32_optimizer = param_count * 8  # Adam: momentum + velocity
        fp32_total = param_memory + fp32_activations + fp32_gradients + fp32_optimizer

        # Mixed precision memory
        fp16_activations = activation_count * 2  # FP16 activations
        fp16_gradients = param_count * 2  # FP16 gradients during backward
        mixed_total = param_memory + fp16_activations + fp16_gradients + fp32_optimizer

        # Calculate savings
        savings_gb = (fp32_total - mixed_total) / 1e9
        savings_pct = (fp32_total - mixed_total) / fp32_total * 100

        print(f"│ {name:14s}  │ {param_count:10,d}  │ {fp32_total/1e9:9.1f}   │ {mixed_total/1e9:9.1f}   │ {savings_pct:9.1f}   │")

    print("└─────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    # Performance simulation
    print(f"\n⚡ Mixed Precision Performance Simulation:")

    # Simulate different batch sizes to show memory pressure
    batch_sizes = [8, 16, 32, 64]
    hidden_size = 1024
    seq_length = 512

    print("┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Batch Size  │ FP32 Mem    │ FP16 Mem    │ Throughput  │ Efficiency  │")
    print("│             │ (GB)        │ (GB)        │ Gain        │ Gain        │")
    print("├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for batch_size in batch_sizes:
        # Memory for activations (dominant for large models)
        elements = batch_size * seq_length * hidden_size

        fp32_mem = elements * 4 / 1e9  # 4 bytes per FP32
        fp16_mem = elements * 2 / 1e9  # 2 bytes per FP16

        # Simulate throughput gains (based on Tensor Core speedups)
        # Real speedups depend on hardware and operation mix
        throughput_gain = 1.4  # Conservative estimate for mixed workloads

        # Memory efficiency enables larger batch sizes
        max_fp32_batch = 32  # Assume memory limit
        max_fp16_batch = 64   # Double capacity with FP16

        efficiency_gain = max_fp16_batch / max_fp32_batch if batch_size <= max_fp32_batch else "OOM"
        efficiency_str = f"{efficiency_gain:.1f}×" if isinstance(efficiency_gain, float) else efficiency_gain

        print(f"│ {batch_size:10d}  │ {fp32_mem:9.2f}   │ {fp16_mem:9.2f}   │ {throughput_gain:9.1f}×  │ {efficiency_str:9s}   │")

    print("└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n💡 Mixed Precision Key Benefits:")
    print(f"   🎯 Memory: 20-40% reduction enables larger models/batches")
    print(f"   ⚡ Speed: 1.3-2× throughput on modern hardware (V100+)")
    print(f"   📈 Scale: Essential for billion-parameter models")
    print(f"   ⚠️  Complexity: Requires careful loss scaling and overflow handling")
    print("🚀 Mixed precision is crucial for competitive ML training")

analyze_mixed_precision_benefits()

# %% [markdown]
"""
## 6. Optimization Insights - Production Acceleration Strategy

Understanding when and how to apply different acceleration techniques in real-world scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "acceleration-decision-framework", "solution": true}
def analyze_acceleration_decision_framework():
    """📊 Decision framework for choosing acceleration techniques."""
    print("📊 Acceleration Technique Decision Framework...")

    # Define workload characteristics
    workloads = [
        ("Research Training", {
            "memory_pressure": "medium",
            "latency_sensitive": False,
            "stability_critical": False,
            "development_speed": "high",
            "hardware_variety": "high"
        }),
        ("Production Training", {
            "memory_pressure": "high",
            "latency_sensitive": False,
            "stability_critical": True,
            "development_speed": "medium",
            "hardware_variety": "low"
        }),
        ("Real-time Inference", {
            "memory_pressure": "medium",
            "latency_sensitive": True,
            "stability_critical": True,
            "development_speed": "low",
            "hardware_variety": "medium"
        }),
        ("Edge Deployment", {
            "memory_pressure": "very_high",
            "latency_sensitive": True,
            "stability_critical": True,
            "development_speed": "low",
            "hardware_variety": "very_high"
        }),
        ("Batch Inference", {
            "memory_pressure": "low",
            "latency_sensitive": False,
            "stability_critical": True,
            "development_speed": "medium",
            "hardware_variety": "low"
        })
    ]

    # Define technique characteristics
    techniques = {
        "Vectorization": {
            "implementation_cost": "low",
            "memory_benefit": "none",
            "latency_benefit": "high",
            "stability_risk": "none",
            "hardware_dependency": "low"
        },
        "Kernel Fusion": {
            "implementation_cost": "medium",
            "memory_benefit": "medium",
            "latency_benefit": "medium",
            "stability_risk": "low",
            "hardware_dependency": "medium"
        },
        "Mixed Precision": {
            "implementation_cost": "high",
            "memory_benefit": "high",
            "latency_benefit": "high",
            "stability_risk": "medium",
            "hardware_dependency": "high"
        },
        "Graph Optimization": {
            "implementation_cost": "very_high",
            "memory_benefit": "medium",
            "latency_benefit": "very_high",
            "stability_risk": "low",
            "hardware_dependency": "very_high"
        }
    }

    print("\n🎯 Acceleration Technique Recommendations:")
    print("┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Workload            │ Vectorize   │ Fuse Kernels│ Mixed Prec  │ Graph Opt   │")
    print("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for workload_name, workload_chars in workloads:
        recommendations = []

        for technique_name in ["Vectorization", "Kernel Fusion", "Mixed Precision", "Graph Optimization"]:
            tech_chars = techniques[technique_name]
            score = 0

            # Benefit vs requirement matching
            if workload_chars["memory_pressure"] in ["high", "very_high"]:
                if tech_chars["memory_benefit"] in ["medium", "high"]:
                    score += 2

            if workload_chars["latency_sensitive"]:
                if tech_chars["latency_benefit"] in ["medium", "high", "very_high"]:
                    score += 2

            # Risk vs tolerance matching
            if workload_chars["stability_critical"]:
                if tech_chars["stability_risk"] in ["none", "low"]:
                    score += 1
                elif tech_chars["stability_risk"] == "medium":
                    score -= 1

            # Implementation cost vs development speed
            if workload_chars["development_speed"] == "high":
                if tech_chars["implementation_cost"] in ["low", "medium"]:
                    score += 1
                elif tech_chars["implementation_cost"] in ["high", "very_high"]:
                    score -= 1

            # Hardware dependency vs variety
            if workload_chars["hardware_variety"] in ["high", "very_high"]:
                if tech_chars["hardware_dependency"] in ["low", "medium"]:
                    score += 1
                elif tech_chars["hardware_dependency"] in ["high", "very_high"]:
                    score -= 2

            # Convert score to recommendation
            if score >= 3:
                rec = "✅ High"
            elif score >= 1:
                rec = "⚡ Medium"
            elif score >= 0:
                rec = "⚠️  Low"
            else:
                rec = "❌ Skip"

            recommendations.append(rec)

        rec_line = " │ ".join(f"{rec:10s}" for rec in recommendations)
        print(f"│ {workload_name:18s}  │ {rec_line} │")

    print("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    # Implementation priority framework
    print(f"\n🛠️  Implementation Priority Framework:")
    print(f"   📊 Phase 1 (Always): Vectorization")
    print(f"      • Low risk, high reward")
    print(f"      • Works on any hardware")
    print(f"      • Foundation for other optimizations")
    print(f"   ")
    print(f"   📊 Phase 2 (Memory constrained): Kernel Fusion")
    print(f"      • Targets memory-bound operations")
    print(f"      • Moderate complexity")
    print(f"      • Significant wins on element-wise ops")
    print(f"   ")
    print(f"   📊 Phase 3 (Large models): Mixed Precision")
    print(f"      • Essential for large model training")
    print(f"      • Requires careful validation")
    print(f"      • Hardware-dependent benefits")
    print(f"   ")
    print(f"   📊 Phase 4 (Production): Graph Optimization")
    print(f"      • Maximum performance extraction")
    print(f"      • High implementation cost")
    print(f"      • Deployment-specific tuning")

    print(f"\n💡 Key Decision Factors:")
    print(f"   🎯 Start simple: Vectorization first, always")
    print(f"   📈 Scale up: Add complexity only when needed")
    print(f"   ⚡ Measure impact: Profile before and after each optimization")
    print(f"   🔄 Iterate: Optimization is an ongoing process, not one-time")
    print("🚀 Systematic acceleration beats random optimization")

analyze_acceleration_decision_framework()

# %% [markdown]
"""
## 7. Module Integration Test

Final validation that all acceleration components work together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 20}
def test_module():
    """
    Comprehensive test of entire acceleration module functionality.

    This final test ensures:
    - All acceleration techniques work correctly
    - Performance improvements are measurable
    - Mixed precision training is stable
    - Components integrate seamlessly
    - Module is ready for production use
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_vectorized_matmul()
    test_unit_fused_gelu()
    test_unit_fusion_speedup()
    test_unit_mixed_precision()

    print("\nRunning integration scenarios...")

    # Test realistic acceleration pipeline
    print("🔬 Integration Test: Complete acceleration pipeline...")

    # Create realistic model scenario
    batch_size, seq_len, hidden_dim = 16, 64, 256
    print(f"   Model config: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}")

    # Test data
    x = Tensor(np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32))
    weight = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
    print(f"   Input tensor: {x.shape}, Weight tensor: {weight.shape}")

    # Test complete pipeline: reshape → matmul → activation → mixed precision
    print("   Testing vectorized operations...")

    # Reshape for matrix multiplication (flatten batch and sequence)
    x_reshaped = Tensor(x.data.reshape(-1, hidden_dim))
    assert x_reshaped.shape == (batch_size * seq_len, hidden_dim)

    # Vectorized matrix multiplication
    linear_output = vectorized_matmul(x_reshaped, weight)
    assert linear_output.shape == (batch_size * seq_len, hidden_dim)
    print(f"   ✅ Matrix multiplication: {x_reshaped.shape} @ {weight.shape} → {linear_output.shape}")

    # Fused activation
    activated = fused_gelu(linear_output)
    assert activated.shape == linear_output.shape
    print(f"   ✅ Fused GELU activation: {linear_output.shape} → {activated.shape}")

    # Reshape back to original structure
    final_output = Tensor(activated.data.reshape(batch_size, seq_len, hidden_dim))
    assert final_output.shape == x.shape
    print(f"   ✅ Output reshape: {activated.shape} → {final_output.shape}")

    print("   Testing mixed precision training integration...")

    # Create complete model for mixed precision testing
    class TransformerBlock:
        def __init__(self, hidden_dim):
            self.hidden_dim = hidden_dim
            self.weight1 = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
            self.weight2 = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
            self.weight1.grad = None
            self.weight2.grad = None

        def __call__(self, x):
            # Simulate transformer block: linear → activation → linear
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = Tensor(x.data.reshape(-1, hidden_dim))

            # First linear layer
            h1 = vectorized_matmul(x_flat, self.weight1)
            h1_activated = fused_gelu(h1)

            # Second linear layer
            h2 = vectorized_matmul(h1_activated, self.weight2)

            # Reshape back
            output = Tensor(h2.data.reshape(batch_size, seq_len, hidden_dim))
            return output

        def parameters(self):
            return [self.weight1, self.weight2]

    class SimpleOptimizer:
        def __init__(self, params):
            self.params = params

        def zero_grad(self):
            for p in self.params:
                p.grad = None

        def step(self):
            for p in self.params:
                if p.grad is not None:
                    p.data = p.data - 0.001 * p.grad.data

    # Initialize model and optimizer
    model = TransformerBlock(hidden_dim)
    optimizer = SimpleOptimizer(model.parameters())
    trainer = MixedPrecisionTrainer(model, optimizer, loss_scale=512.0)

    print(f"   Model parameters: {len(model.parameters())}")
    print(f"   Initial loss scale: {trainer.loss_scale}")

    # Simulate training steps
    print("   Running training steps...")
    targets = Tensor(np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32))

    training_metrics = []
    for step in range(5):
        metrics = trainer.train_step((x, targets))
        training_metrics.append(metrics)

        # Verify metrics are reasonable
        assert isinstance(metrics['loss'], (int, float))
        assert metrics['loss'] >= 0
        assert metrics['loss_scale'] > 0
        assert isinstance(metrics['overflow'], bool)
        assert isinstance(metrics['gradients_valid'], bool)

    print(f"   ✅ Completed {len(training_metrics)} training steps")

    # Analyze training stability
    losses = [m['loss'] for m in training_metrics]
    overflows = [m['overflow'] for m in training_metrics]

    print(f"   Loss range: {min(losses):.6f} - {max(losses):.6f}")
    print(f"   Overflow rate: {sum(overflows)}/{len(overflows)} steps")

    print("   Testing performance characteristics...")

    # Verify acceleration provides measurable benefits
    test_sizes = [128, 256]
    for size in test_sizes:
        test_x = Tensor(np.random.randn(size, size).astype(np.float32))
        test_y = Tensor(np.random.randn(size, size).astype(np.float32))

        # Time operations and verify reasonable performance
        start = time.time()
        _ = vectorized_matmul(test_x, test_y)
        matmul_time = time.time() - start

        start = time.time()
        _ = fused_gelu(test_x)
        gelu_time = time.time() - start

        # Verify operations complete in reasonable time
        assert matmul_time < 1.0, f"Matrix multiplication too slow: {matmul_time:.3f}s"
        assert gelu_time < 0.1, f"GELU activation too slow: {gelu_time:.3f}s"

        print(f"   ✅ Size {size}: matmul={matmul_time*1000:.1f}ms, gelu={gelu_time*1000:.1f}ms")

    print("   Testing memory efficiency...")

    # Verify mixed precision reduces memory usage conceptually
    param_count = sum(p.data.size for p in model.parameters())
    activation_count = batch_size * seq_len * hidden_dim

    fp32_memory = (param_count + activation_count) * 4  # 4 bytes per FP32
    mixed_memory = param_count * 4 + activation_count * 2  # FP32 params + FP16 activations
    memory_savings = (fp32_memory - mixed_memory) / fp32_memory * 100

    print(f"   Memory analysis: {memory_savings:.1f}% savings from mixed precision")
    assert memory_savings > 0, "Mixed precision should reduce memory usage"

    print("✅ End-to-end acceleration pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 16")

# Call the module test
test_module()

# %% nbgrader={"grade": false, "grade_id": "main-execution", "solution": false}
# Main execution block
if __name__ == "__main__":
    print("🚀 Running Acceleration module...")
    test_module()
    print("✅ Module validation complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Acceleration and Performance

### Question 1: Arithmetic Intensity Analysis
You implemented vectorized matrix multiplication and fused GELU.
- Matrix multiplication (1024×1024): Performs ~2.1 billion FLOPs, reads ~12 MB data
- Arithmetic intensity: _____ FLOPs/byte
- Compared to element-wise addition (0.33 FLOPs/byte): _____× higher intensity
- Why does this make matrix multiplication ideal for GPUs? _____

### Question 2: Kernel Fusion Memory Benefits
Your fused_gelu combines 7 operations into a single expression.
- Unfused version memory accesses: 7 reads + 7 writes = _____ per element
- Fused version memory accesses: 1 read + 1 write = _____ per element
- Memory bandwidth reduction: _____%
- Why is this critical for transformer inference? _____

### Question 3: Mixed Precision Memory Calculation
Your MixedPrecisionTrainer uses FP16 activations, FP32 parameters.
For a 100M parameter model with 50M activation elements:
- FP32 memory: (100M + 50M) × 4 bytes = _____ MB
- Mixed precision memory: 100M × 4 + 50M × 2 = _____ MB
- Memory reduction: _____%

### Question 4: Loss Scaling Strategy
Your trainer starts with loss_scale=1024, grows by 2×, shrinks by 0.5×.
- Minimum FP16 representable value: ~6e-5
- Without scaling, gradients < _____ become zero
- With 1024× scaling, gradients down to _____ are preserved
- Why increase scale gradually but decrease immediately? _____

### Question 5: Production Optimization Strategy
Based on your decision framework analysis:
For edge deployment (memory critical, stability required, hardware diverse):
- Priority 1 technique: _____ (low risk, universal)
- Priority 2 technique: _____ (memory benefits)
- Skip technique: _____ (why: _____)
- What's the primary constraint: memory, compute, or power? _____
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Acceleration

Congratulations! You've mastered the fundamental techniques for accelerating neural networks!

### Key Accomplishments
- Built **vectorized operations** leveraging SIMD and optimized BLAS for 2-5× speedups
- Implemented **kernel fusion** reducing memory bandwidth by 60-80% for element-wise operations
- Created **mixed precision training** with automatic loss scaling for 20-40% memory savings
- Analyzed **arithmetic intensity patterns** and their impact on the roofline model
- Developed **production decision framework** for systematic optimization
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Discovered
- **Roofline Model**: Operations with high arithmetic intensity (FLOPs/byte) scale better
- **Memory Bandwidth**: Often the limiting factor for modern accelerators
- **Kernel Fusion**: Critical for memory-bound workloads, reduces intermediate storage overhead
- **Mixed Precision**: Essential for large model training, requires careful gradient scaling
- **Optimization Strategy**: Start simple (vectorization), add complexity as needed

### Production Impact
Your acceleration techniques enable:
- **Training larger models** within memory constraints
- **Faster iteration cycles** during research and development
- **Better hardware utilization** across different deployment targets
- **Cost reduction** through improved efficiency

### Ready for Next Steps
Your acceleration implementations provide the foundation for quantization techniques in Module 17.
The performance analysis skills transfer directly to production optimization workflows.

Export with: `tito module complete 16`

**Next**: Module 17 will add quantization to further reduce memory and increase throughput while maintaining accuracy!
"""