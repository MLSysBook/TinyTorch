# TinyTorch Optimization Modules Implementation Plan
## Modules 15-20: Clean, Minimal, Production-Ready

Based on PyTorch expert review - focusing on MUST HAVE features only.

---

## Module 15: Acceleration ✅ 
**Status**: Already well-structured  
**Focus**: Backend optimization with clear pedagogical progression

### MUST HAVE Implementation
```python
# 1. Educational baseline (show the journey)
def matmul_naive(A, B):  # From Module 2
def matmul_blocked(A, B):  # Cache-friendly
def matmul_numpy(A, B):  # Library backend

# 2. OptimizedBackend class
class OptimizedBackend:
    def dispatch(self, op, *args):
        # Smart operation routing
        
# 3. Performance comparison
# Show 10-100x differences between implementations
```

### Key Learning
- Why cache-friendly matters (memory hierarchy)
- When to use optimized libraries vs custom code
- Backend dispatch patterns (like PyTorch)

---

## Module 16: Quantization 🔧
**Status**: Needs content migration from Module 17  
**Focus**: INT8 post-training quantization for CNNs

### MUST HAVE Implementation
```python
# 1. Simple INT8 quantization
class INT8Quantizer:
    def quantize_weights(self, weights, calibration_data):
        # Compute scale and zero point
        # Convert FP32 → INT8
        
# 2. Calibration approach
def calibrate(model, calibration_dataset):
    # Run representative data
    # Collect statistics
    # Compute optimal quantization params

# 3. Quantized operations
class QuantizedConv2d:
    # INT8 convolution implementation
    
# 4. Accuracy comparison
# Show <1% accuracy loss with 4x speedup
```

### Key Learning
- Numerical precision trade-offs
- Why INT8 works for inference
- Calibration vs training-time quantization

---

## Module 17: Compression (Pruning) 🔧
**Status**: Needs new implementation  
**Focus**: Magnitude-based pruning for all architectures

### MUST HAVE Implementation
```python
# 1. Magnitude-based pruning
class MagnitudePruner:
    def prune(self, weights, sparsity=0.7):
        # Remove 70% smallest weights
        
# 2. Structured pruning for CNNs
def prune_conv_filters(conv_layer, sparsity=0.5):
    # Remove entire filters
    # Maintain conv structure

# 3. Sparse operations
class SparseLinear:
    # Efficient sparse matrix multiply
    
# 4. Accuracy tracking
# Show 70% sparsity with <2% accuracy loss
```

### Key Learning
- Neural network redundancy
- Structured vs unstructured pruning
- When pruning fails (critical connections)

---

## Module 18: Caching (KV Cache) ✅
**Status**: Well-scoped  
**Focus**: KV caching for transformer autoregressive generation

### MUST HAVE Implementation
```python
# 1. KV Cache implementation
class KVCache:
    def __init__(self, max_seq_len, n_heads, head_dim):
        self.cache = {}
    
    def update(self, layer, key, value, position):
        # Store computed K,V
        
    def get(self, layer, positions):
        # Retrieve cached K,V

# 2. Modified attention with cache
class CachedAttention:
    def forward(self, x, past_kv=None):
        # Use cached values for past positions
        # Only compute new position
        
# 3. Performance demonstration
# Show O(N²) → O(N) speedup for generation
```

### Key Learning
- Memory-compute trade-offs
- Incremental computation patterns
- Why caching matters for production inference

### CRITICAL: Module 14 Transformer must be updated
```python
# Module 14 needs this change:
class TransformerBlock:
    def forward(self, x, past_kv=None):  # ADD THIS PARAMETER
        # Support for KV caching
```

---

## Module 19: Profiling 🔧
**Status**: Needs complete rewrite (currently autotuning)  
**Focus**: Build measurement infrastructure for Module 20

### MUST HAVE Implementation
```python
# 1. Timer with statistical rigor
class Timer:
    def measure(self, func, warmup=3, runs=100):
        # Warmup runs
        # Statistical sampling
        # Return percentiles (p50, p95, p99)

# 2. Memory profiler
class MemoryProfiler:
    def profile(self, func):
        # Track allocations
        # Measure peak usage
        # Identify leaks

# 3. FLOP counter
class FLOPCounter:
    def count_ops(self, model, input):
        # Count arithmetic operations
        # Identify compute bottlenecks

# 4. Profiler context manager
class ProfilerContext:
    def __enter__(self):
        # Start profiling
    def __exit__(self):
        # Generate report
```

### Key Learning
- Importance of warmup and statistics
- Memory vs compute bottlenecks
- How to measure, not guess

---

## Module 20: Benchmarking (Competition) 🎯
**Status**: Needs focus on competition, not infrastructure  
**Focus**: TinyMLPerf Olympics using Module 19 profiler

### MUST HAVE Implementation
```python
# 1. Standard benchmark models
class TinyMLPerf:
    MLP_SPRINT = load_model('benchmarks/mlp.pkl')
    CNN_MARATHON = load_model('benchmarks/cnn.pkl')
    TRANSFORMER_DECATHLON = load_model('benchmarks/transformer.pkl')

# 2. Benchmark harness using Module 19
def benchmark_model(model, profiler):
    with profiler:
        # Measure inference speed
        # Measure training speed
        # Measure memory usage
    return profiler.get_results()

# 3. Relative scoring (hardware-independent)
def compute_speedup(baseline, optimized):
    # Compare against vanilla TinyTorch
    # Return improvement ratios

# 4. Competition submission
class CompetitionSubmission:
    def validate(self):
        # Check all optimizations work
    def compute_score(self):
        # Weight different metrics
    def submit_to_leaderboard(self):
        # Update rankings
```

### Key Learning
- Fair benchmarking methodology
- Reproducible performance measurement
- Real-world optimization strategies

---

## Implementation Priority & Dependencies

### Must Complete First
1. **Module 14 Update**: Add `past_kv` parameter to transformers
2. **Module 16 Fix**: Move quantization content from Module 17
3. **Module 19 Rewrite**: Replace autotuning with profiling

### Development Order
1. Module 15 (Acceleration) - Already good, minor polish
2. Module 16 (Quantization) - Move content, implement INT8
3. Module 17 (Compression) - New pruning implementation
4. Module 18 (Caching) - KV cache implementation
5. Module 19 (Profiling) - Complete rewrite needed
6. Module 20 (Benchmarking) - Use Module 19 profiler

### Critical Cross-Module Dependencies
- Module 14 → 18: Transformer must support KV caching
- Module 19 → 20: Profiler used in benchmarking
- Module 15-18 → 20: All optimizations tested in competition

---

## Success Metrics

Each module is successful when students can:

1. **Module 15**: Achieve 10-100x speedup with backend optimization
2. **Module 16**: Quantize CNN to INT8 with <1% accuracy loss
3. **Module 17**: Prune 70% of parameters with <2% accuracy loss
4. **Module 18**: Speed up transformer generation by 5-10x with KV cache
5. **Module 19**: Profile and identify bottlenecks in any model
6. **Module 20**: Submit competition entry showing cumulative speedup

---

## Common Pitfalls to Avoid

❌ **Don't**: Try to cover every optimization technique  
✅ **Do**: Focus on 3-4 techniques done well

❌ **Don't**: Hide implementation details  
✅ **Do**: Show clear before/after performance

❌ **Don't**: Make competition about absolute performance  
✅ **Do**: Focus on relative improvement and learning

❌ **Don't**: Mix concepts (e.g., quantization with memory optimization)  
✅ **Do**: One clear concept per module

---

## Next Steps

1. Fix Module 14 transformer to support KV caching
2. Move quantization content to Module 16
3. Launch parallel development of Modules 15-19
4. Module 20 development after Module 19 is complete