---
title: "Profiling - Performance Analysis and Optimization"
description: "Build profilers to identify bottlenecks and guide optimization decisions"
difficulty: 3
time_estimate: "5-6 hours"
prerequisites: ["All modules 01-14"]
next_steps: ["Acceleration"]
learning_objectives:
  - "Implement timing profilers with statistical rigor for accurate measurements"
  - "Design memory profilers to track allocation patterns and identify leaks"
  - "Build FLOP counters to measure computational complexity"
  - "Understand performance bottlenecks across different architectures"
  - "Apply data-driven analysis to guide optimization priorities"
---

# 15. Profiling

**⚡ OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 5-6 hours

## Overview

Build comprehensive profiling tools to measure where time and memory go in your ML systems. This module implements timing profilers, memory trackers, and FLOP counters that reveal bottlenecks and guide optimization decisions.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement timing profilers** with statistical rigor (multiple runs, confidence intervals) for accurate measurements
2. **Design memory profilers** to track allocation patterns, peak usage, and identify memory leaks
3. **Build FLOP counters** to measure theoretical computational complexity of different operations
4. **Understand performance bottlenecks** by comparing MLPs, CNNs, and Transformers systematically
5. **Apply data-driven analysis** to prioritize optimization efforts based on actual impact

## Why This Matters

### Production Context

Profiling is mandatory for production ML systems:

- **Google TPU teams** profile every operation to optimize hardware utilization
- **OpenAI** profiles GPT training to identify $millions in compute savings
- **Meta** profiles inference to serve billions of requests per day efficiently  
- **NVIDIA** uses profiling to optimize cuDNN kernels for peak performance

### Historical Context

Profiling evolved with ML scale:

- **Early ML (pre-2012)**: Ad-hoc timing with `time.time()`; no systematic profiling
- **Deep Learning Era (2012-2017)**: NVIDIA profiler, TensorBoard timing; focus on GPU utilization
- **Production Scale (2018+)**: Comprehensive profiling (compute, memory, I/O, network); optimization critical for economics
- **Modern Systems (2020+)**: Automated profiling and optimization; ML compilers use profiling data

Without profiling, you're optimizing blind—profiling shows you where to focus.

## Pedagogical Pattern: Build → Use → Optimize

### 1. Build

Implement from first principles:
- High-precision timing with multiple runs
- Statistical analysis (mean, std, confidence intervals)
- Memory profiler tracking allocations and deallocations
- FLOP counter for theoretical complexity
- Comparative profiler across architectures

### 2. Use

Apply to real problems:
- Profile attention vs feedforward in transformers
- Compare MLP vs CNN vs Transformer efficiency
- Identify memory bottlenecks in training loops
- Measure impact of batch size on throughput
- Analyze scaling behavior with model size

### 3. Optimize

Production insights:
- Prioritize optimizations by impact (80/20 rule)
- Measure before/after optimization
- Understand hardware utilization (CPU vs GPU)
- Identify memory bandwidth vs compute bottlenecks
- Build optimization roadmap based on data

## Implementation Guide

### Core Components

**High-Precision Timer**
```python
class Timer:
    """High-precision timing with statistical analysis.
    
    Performs multiple runs to account for variance and noise.
    Reports mean, std, and confidence intervals.
    
    Example:
        timer = Timer()
        with timer:
            model.forward(x)
        print(f"Time: {timer.mean:.3f}ms ± {timer.std:.3f}ms")
    """
    def __init__(self, num_runs=10, warmup_runs=3):
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.times = []
    
    def __enter__(self):
        # Warmup runs (not counted)
        for _ in range(self.warmup_runs):
            start = time.perf_counter()
            # Operation happens in with block
        
        # Timed runs
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed * 1000)  # Convert to ms
    
    @property
    def mean(self):
        return np.mean(self.times)
    
    @property
    def std(self):
        return np.std(self.times)
    
    @property
    def confidence_interval(self, confidence=0.95):
        """95% confidence interval using t-distribution."""
        from scipy import stats
        ci = stats.t.interval(confidence, len(self.times)-1,
                              loc=self.mean, scale=stats.sem(self.times))
        return ci
    
    def report(self):
        ci = self.confidence_interval()
        return f"{self.mean:.3f}ms ± {self.std:.3f}ms (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"
```

**Memory Profiler**
```python
class MemoryProfiler:
    """Track memory allocations and peak usage.
    
    Monitors memory throughout execution to identify:
    - Peak memory usage
    - Memory leaks
    - Allocation patterns
    - Memory bandwidth bottlenecks
    """
    def __init__(self):
        self.snapshots = []
        self.peak_memory = 0
    
    def snapshot(self, label=""):
        """Take memory snapshot at current point."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        snapshot = {
            'label': label,
            'rss': mem_info.rss / 1024**2,  # MB
            'vms': mem_info.vms / 1024**2,  # MB
            'timestamp': time.time()
        }
        self.snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, snapshot['rss'])
        
        return snapshot
    
    def report(self):
        """Generate memory usage report."""
        print(f"Peak Memory: {self.peak_memory:.2f} MB")
        print("\nMemory Timeline:")
        for snap in self.snapshots:
            print(f"  {snap['label']:30s}: {snap['rss']:8.2f} MB")
        
        # Calculate memory growth
        if len(self.snapshots) >= 2:
            growth = self.snapshots[-1]['rss'] - self.snapshots[0]['rss']
            print(f"\nTotal Growth: {growth:+.2f} MB")
            
            # Check for potential memory leak
            if growth > 100:  # Arbitrary threshold
                print("⚠️  Potential memory leak detected!")
```

**FLOP Counter**
```python
class FLOPCounter:
    """Count floating-point operations for complexity analysis.
    
    Provides theoretical computational complexity independent of hardware.
    Useful for comparing different architectural choices.
    """
    def __init__(self):
        self.total_flops = 0
        self.op_counts = {}
    
    def count_matmul(self, A_shape, B_shape):
        """Count FLOPs for matrix multiplication.
        
        C = A @ B where A is (m, k) and B is (k, n)
        FLOPs = 2*m*k*n (multiply-add for each output element)
        """
        m, k = A_shape
        k2, n = B_shape
        assert k == k2, "Invalid matmul dimensions"
        
        flops = 2 * m * k * n
        self.total_flops += flops
        self.op_counts['matmul'] = self.op_counts.get('matmul', 0) + flops
        return flops
    
    def count_attention(self, batch, seq_len, d_model, num_heads):
        """Count FLOPs for multi-head attention.
        
        Components:
        - Q,K,V projections: 3 * (batch * seq_len * d_model * d_model)
        - Attention scores: batch * heads * seq_len * seq_len * d_k
        - Attention weighting: batch * heads * seq_len * seq_len * d_v
        - Output projection: batch * seq_len * d_model * d_model
        """
        d_k = d_model // num_heads
        
        # QKV projections
        qkv_flops = 3 * self.count_matmul((batch * seq_len, d_model), (d_model, d_model))
        
        # Attention computation
        scores_flops = batch * num_heads * seq_len * seq_len * d_k * 2
        weights_flops = batch * num_heads * seq_len * seq_len * d_k * 2
        attention_flops = scores_flops + weights_flops
        
        # Output projection
        output_flops = self.count_matmul((batch * seq_len, d_model), (d_model, d_model))
        
        total = qkv_flops + attention_flops + output_flops
        self.op_counts['attention'] = self.op_counts.get('attention', 0) + total
        return total
    
    def report(self):
        """Generate FLOP report with breakdown."""
        print(f"Total FLOPs: {self.total_flops / 1e9:.2f} GFLOPs")
        print("\nBreakdown by operation:")
        for op, flops in sorted(self.op_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (flops / self.total_flops) * 100
            print(f"  {op:20s}: {flops/1e9:8.2f} GFLOPs ({percentage:5.1f}%)")
```

**Architecture Profiler - Comparative Analysis**
```python
class ArchitectureProfiler:
    """Compare performance across different architectures.
    
    Profiles MLP, CNN, and Transformer on same task to understand
    compute/memory trade-offs.
    """
    def __init__(self):
        self.results = {}
    
    def profile_model(self, model, input_data, model_name):
        """Profile a model comprehensively."""
        result = {
            'model_name': model_name,
            'parameters': count_parameters(model),
            'timing': {},
            'memory': {},
            'flops': {}
        }
        
        # Timing profile
        timer = Timer(num_runs=10)
        for _ in range(timer.num_runs + timer.warmup_runs):
            with timer:
                output = model.forward(input_data)
        result['timing']['forward'] = timer.mean
        
        # Memory profile
        mem = MemoryProfiler()
        mem.snapshot("Before forward")
        output = model.forward(input_data)
        mem.snapshot("After forward")
        result['memory']['peak'] = mem.peak_memory
        
        # FLOP count
        flop_counter = FLOPCounter()
        # Count FLOPs based on model architecture
        result['flops']['total'] = flop_counter.total_flops
        
        self.results[model_name] = result
        return result
    
    def compare(self):
        """Generate comparative report."""
        print("\nArchitecture Comparison")
        print("=" * 80)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Parameters: {result['parameters']/1e6:.2f}M")
            print(f"  Forward time: {result['timing']['forward']:.3f}ms")
            print(f"  Peak memory: {result['memory']['peak']:.2f}MB")
            print(f"  FLOPs: {result['flops']['total']/1e9:.2f}GFLOPs")
```

### Step-by-Step Implementation

1. **Build High-Precision Timer**
   - Use `time.perf_counter()` for nanosecond precision
   - Implement multiple runs with warmup
   - Calculate mean, std, confidence intervals
   - Test with known delays

2. **Implement Memory Profiler**
   - Track memory at key points (before/after operations)
   - Calculate peak memory usage
   - Identify memory growth patterns
   - Detect potential leaks

3. **Create FLOP Counter**
   - Count operations for matmul, convolution, attention
   - Build hierarchical counting (operation → layer → model)
   - Compare theoretical vs actual performance
   - Identify compute-bound vs memory-bound operations

4. **Build Architecture Profiler**
   - Profile MLP on MNIST/CIFAR
   - Profile CNN on CIFAR
   - Profile Transformer on text
   - Generate comparative reports

5. **Analyze Results**
   - Identify bottleneck operations (Pareto principle)
   - Compare efficiency across architectures
   - Understand scaling behavior
   - Prioritize optimization opportunities

## Testing

### Inline Tests

Run inline tests while building:
```bash
cd modules/source/15_profiling
python profiling_dev.py
```

Expected output:
```
Unit Test: Timer with statistical analysis...
✅ Multiple runs produce consistent results
✅ Confidence intervals computed correctly
✅ Warmup runs excluded from statistics
Progress: Timing Profiler ✓

Unit Test: Memory profiler...
✅ Snapshots capture memory correctly
✅ Peak memory tracked accurately
✅ Memory growth detected
Progress: Memory Profiler ✓

Unit Test: FLOP counter...
✅ Matmul FLOPs: 2*m*k*n verified
✅ Attention FLOPs match theoretical
✅ Operation breakdown correct
Progress: FLOP Counter ✓
```

### Export and Validate

```bash
tito export 15_profiling
tito test 15_profiling
```

## Where This Code Lives

```
tinytorch/
├── profiler/
│   └── profiling.py            # Your implementation goes here
└── __init__.py                 # Exposes Timer, MemoryProfiler, etc.

Usage:
>>> from tinytorch.profiler import Timer, MemoryProfiler, FLOPCounter
>>> timer = Timer()
>>> with timer:
>>>     model.forward(x)
>>> print(timer.report())
```

## Systems Thinking Questions

1. **Amdahl's Law**: If attention is 70% of compute and you optimize it 2×, what's the overall speedup? Why can't you get 2× end-to-end speedup?

2. **Memory vs Compute Bottlenecks**: Your GPU can do 100 TFLOPs/s but memory bandwidth is 900 GB/s. For FP32 operations needing 4 bytes/FLOP, what's the bottleneck? When?

3. **Batch Size Impact**: Doubling batch size doesn't double throughput. Why? What's the relationship between batch size, memory, and throughput?

4. **Profiling Overhead**: Your profiler adds 5% overhead. Is this acceptable? When would you use sampling profilers vs instrumentation profilers?

5. **Hardware Differences**: Your code runs 10× slower on CPU than GPU for large matrices, but only 2× slower for small ones. Why? What's the crossover point?

## Real-World Connections

### Industry Applications

**Google TPU Optimization**
- Profile every kernel to maximize TPU utilization
- Optimize for both FLOPs and memory bandwidth
- Use profiling to guide hardware design decisions
- Achieve 40-50% utilization (very high for accelerators)

**OpenAI Training Optimization**
- Profile GPT training to find $millions in savings
- Identify gradient checkpointing opportunities
- Optimize data loading pipelines
- Achieve 50%+ MFU (model FLOPs utilization)

**Meta Inference Serving**
- Profile PyTorch models for production deployment
- Identify operator fusion opportunities
- Optimize for latency (p50, p99) not just throughput
- Serve billions of requests per day efficiently

### Research Impact

This module implements patterns from:
- TensorBoard Profiler (Google, 2019): Visual profiling for TensorFlow
- PyTorch Profiler (Meta, 2020): Comprehensive profiling for PyTorch
- NVIDIA Nsight (2021): GPU-specific profiling and optimization
- MLPerf (2022): Standardized benchmarking and profiling

## What's Next?

In **Module 16: Acceleration**, you'll use your profiling data to actually optimize:

- Implement operator fusion based on profiling insights
- Optimize memory access patterns
- Apply algorithmic improvements to bottlenecks
- Measure impact of each optimization

Profiling shows you *what* to optimize—acceleration shows you *how* to optimize it!

---

**Ready to become a performance detective?** Open `modules/source/15_profiling/profiling_dev.py` and start implementing.
