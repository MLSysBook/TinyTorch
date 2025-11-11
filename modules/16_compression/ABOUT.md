---
title: "Compression - Pruning and Model Compression"
description: "Prune unnecessary weights and compress models for deployment"
difficulty: 3
time_estimate: "5-6 hours"
prerequisites: ["Quantization"]
next_steps: ["Acceleration"]
learning_objectives:
  - "Implement magnitude-based pruning to remove unimportant weights"
  - "Design structured pruning strategies (channel, layer-wise)"
  - "Apply iterative pruning with fine-tuning for accuracy preservation"
  - "Combine pruning with quantization for maximum compression"
  - "Measure compression ratios and inference speedups"
---

# 17. Compression

**⚡ OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 5-6 hours

## Overview

Compress neural networks through pruning (removing weights) and combining with quantization. This module implements techniques to achieve 10-50× compression with minimal accuracy loss, enabling deployment on resource-constrained devices.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement magnitude-based pruning** to identify and remove unimportant weights
2. **Design structured pruning strategies** (channel pruning, layer-wise) for actual speedups
3. **Apply iterative pruning** with fine-tuning to maintain model accuracy
4. **Combine pruning with quantization** for maximum compression (50-100× possible)
5. **Measure compression ratios** and verify inference speedup vs accuracy trade-offs

## Why This Matters

### Production Context

Compression enables practical deployment:

- **BERT Distillation (DistilBERT)**: 40% smaller, 60% faster, 97% accuracy retention
- **MobileNet**: Structured pruning + quantization for mobile deployment
- **Lottery Ticket Hypothesis**: Sparse networks train as well as dense ones
- **GPT-3 Distillation**: Smaller models approaching GPT-3 performance

### Historical Context

- **Pre-2015**: Limited compression work; models small enough for hardware
- **2015-2017**: Magnitude pruning (Han et al.); Lottery Ticket Hypothesis
- **2018-2020**: Structured pruning; distillation; BERT compression
- **2020+**: Extreme compression (100×); sparse transformers; efficient architectures

Compression is now standard for deployment, not optional.

## Implementation Guide

### Core Techniques

**Magnitude Pruning**
- Sort weights by absolute value
- Remove smallest X% (typically 50-90%)
- Fine-tune remaining weights
- Can achieve 10× compression with <1% accuracy loss

**Structured Pruning**
- Remove entire channels/neurons
- Achieves actual speedup (vs unstructured sparsity)
- Typically 2-5× compression
- More aggressive accuracy impact

**Iterative Pruning**
- Prune gradually (10% at a time)
- Fine-tune after each pruning step
- Better accuracy than one-shot pruning
- More training cost

**Pruning + Quantization**
- Prune 90% of weights → 10× reduction
- Quantize FP32 → INT8 → 4× reduction
- Combined: 40× compression

## Testing

```bash
tito export 18_compression
tito test 18_compression
```

## Where This Code Lives

```
tinytorch/
├── compression/
│   └── prune.py
└── __init__.py
```

## Systems Thinking Questions

1. **Lottery Ticket Hypothesis**: Why can pruned networks retrain to full accuracy? What does this say about overparameterization?

2. **Structured vs Unstructured**: Unstructured pruning achieves better compression but no speedup. Why? When is sparse computation actually faster?

3. **Distillation vs Pruning**: Both compress models. When would you use each? Can you combine them?

## Real-World Connections

**DistilBERT**: 40% smaller BERT with 97% performance
**MobileNetV2**: Efficient architectures + pruning for mobile
**NVIDIA TensorRT**: Automatic pruning + quantization for deployment

## What's Next?

In **Module 19: Benchmarking**, you'll measure everything you've built:
- Fair comparison across optimizations
- Statistical significance testing
- MLPerf-style benchmarking protocols
- Comprehensive performance reports

---

**Ready to compress models?** Open `modules/18_compression/compression_dev.py` and start implementing.
