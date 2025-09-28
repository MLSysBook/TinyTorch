# Module 15: Profiling - Performance Detective Work

## Overview
Become a performance detective! You just built MLPs, CNNs, and Transformers - but why is your transformer 100x slower than PyTorch? Build professional profiling infrastructure to reveal bottlenecks and guide optimization decisions.

## What You'll Build
- **Timer Class**: Statistical timing with warmup runs and percentile reporting
- **Memory Profiler**: Track allocations, peak usage, and memory patterns
- **FLOP Counter**: Count operations and analyze computational complexity
- **Profiler Context**: Comprehensive profiling manager combining all tools
- **Performance Analysis**: Complete bottleneck detection and optimization guidance

## Learning Objectives
1. **Statistical Timing**: Build robust timing infrastructure with confidence intervals
2. **Memory Analysis**: Track allocations and identify memory bottlenecks
3. **Computational Complexity**: Count FLOPs and understand scaling behavior
4. **Bottleneck Detection**: Use Amdahl's Law to identify optimization targets
5. **Systems Thinking**: Connect profiling insights to production decisions

## Prerequisites
- Module 14: Transformers (need models to profile)
- Understanding of basic complexity analysis (O(n), O(n²))

## Key Concepts

### Professional Timing Infrastructure
```python
timer = Timer()
stats = timer.measure(model.forward, warmup=3, runs=100)
# Returns: mean, std, p50, p95, p99 with confidence intervals
```

### Memory Profiling with tracemalloc
```python
profiler = MemoryProfiler()
stats = profiler.profile(expensive_operation)
# Tracks: baseline, peak, allocated, memory patterns
```

### FLOP Analysis for Architecture Comparison
```python
counter = FLOPCounter()
flops = counter.count_attention(seq_len=128, d_model=512)
# Reveals: O(n²) scaling, computational bottlenecks
```

### Comprehensive Profiling Context
```python
with ProfilerContext("MyModel") as profiler:
    result = profiler.profile_function(model.forward, args=(input,))
# Automatic report: timing + memory + FLOPs + insights
```

## Performance Insights
- **MLPs**: Linear scaling, memory efficient, excellent for classification
- **CNNs**: Moderate speed, vectorizable, great for spatial data
- **Transformers**: O(n²) attention scaling, memory hungry, powerful but expensive

## Real-World Applications
- **Bottleneck Identification**: Find the 20% of code using 80% of time
- **Hardware Selection**: Use profiling data to choose CPU vs GPU
- **Cost Prediction**: Estimate infrastructure costs from FLOP counts  
- **Optimization ROI**: Amdahl's Law guides where to optimize first

## Module Structure
1. **Timer Class**: Statistical timing with warmup and confidence intervals
2. **Memory Profiler**: Allocation tracking and peak usage analysis
3. **FLOP Counter**: Operation counting for different layer types
4. **Profiler Context**: Integrated profiling with automatic reporting
5. **Architecture Comparison**: MLP vs CNN vs Transformer analysis
6. **Bottleneck Detection**: Complete model profiling and optimization guidance
7. **Systems Analysis**: Connect profiling insights to production decisions

## Hands-On Detective Work
```python
# Reveal the transformer bottleneck
with ProfilerContext("Transformer Analysis") as profiler:
    output = profiler.profile_function(transformer.forward, args=(tokens,))
    
# Result: Attention consumes 73% of compute time!
# Next: Optimize attention in Module 16 (Acceleration)
```

## Success Criteria
- ✅ Build timer with statistical rigor (warmup, percentiles, confidence intervals)
- ✅ Implement memory profiler tracking allocations and peak usage
- ✅ Create FLOP counter analyzing computational complexity
- ✅ Develop integrated profiling context for comprehensive analysis
- ✅ Identify bottlenecks using data-driven analysis

## Systems Insights
- **Attention is O(n²)**: 2x sequence length = 4x computation
- **Memory bandwidth matters**: Large models are memory-bound, not compute-bound
- **Amdahl's Law rules**: Optimize the bottleneck first for maximum impact
- **Profiling drives decisions**: Every major ML optimization started with profiling

## ML Systems Focus
This module teaches performance analysis as the foundation of all optimization work. You'll build the same profiling tools used to optimize GPT, BERT, and every production ML system. Understanding performance through measurement is the first step toward building efficient ML systems.

The detective work you do here reveals the bottlenecks that Module 16 (Acceleration) will fix!