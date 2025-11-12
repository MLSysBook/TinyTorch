# Milestone 06: MLPerf - The Optimization Era (2018)

## Historical Context

As ML models grew larger and deployment became critical, the community needed **systematic optimization methodologies**. MLCommons' MLPerf (2018) established standardized benchmarking and optimization workflows, shifting the focus from "can we build it?" to "can we deploy it efficiently?"

This milestone teaches **production optimization** - the systematic process of profiling, compressing, and accelerating models for real-world deployment.

## What You're Building

A complete MLPerf-style optimization pipeline that takes a trained transformer and systematically optimizes it for production deployment. You'll learn to:

1. **Profile** to find bottlenecks
2. **Compress** to reduce model size
3. **Accelerate** to speed up inference

## Required Modules

**Run after Module 18** (Full optimization suite)

**Note:** This milestone builds on a working transformer from Milestone 05 (Modules 01-13). The table below shows the ADDITIONAL optimization modules required.

| Module | Component | What It Provides |
|--------|-----------|------------------|
| Module 13 | Transformers | YOUR base model to optimize |
| Module 14 | Profiling | YOUR tools to measure performance |
| Module 15 | Quantization | YOUR INT8/FP16 implementations |
| Module 16 | Compression | YOUR pruning techniques |
| Module 17 | Memoization | YOUR KV-cache for generation |
| Module 18 | Acceleration | YOUR batching strategies |

## Milestone Structure

This milestone uses **progressive optimization** with 3 scripts:

### 01_baseline_profile.py
**Purpose:** Establish baseline metrics

- Profile model size, FLOPs, latency
- Measure generation speed (tokens/sec)
- Identify bottlenecks (attention, embeddings, etc.)
- **Output:** Baseline report showing what to optimize

**Historical Anchor:** MLPerf Inference v0.5 (2018) - First standardized profiling

### 02_compression.py
**Purpose:** Reduce model size

- Apply INT8 quantization (4× compression)
- Apply magnitude pruning (2-4× compression)
- Combine techniques (8-16× total)
- **Output:** Accuracy vs. size tradeoff curves

**Historical Anchor:** Han et al. "Deep Compression" (2015) + MLPerf Mobile (2019)

### 03_generation_opts.py
**Purpose:** Speed up inference

- Implement KV-caching (6-10× speedup)
- Add batched generation (2-4× speedup)
- **Output:** 12-40× faster generation overall

**Historical Anchor:** Production transformers (2019-2020) - GPT-2/GPT-3 deployment

## Expected Results

| Optimization Stage | Accuracy | Size | Speed | Notes |
|-------------------|----------|------|-------|-------|
| Baseline | 100% | 100% | 1× | Unoptimized model |
| + Quantization | 98-99% | 25% | 1× | INT8 inference |
| + Pruning | 95-98% | 12.5% | 1× | 50-75% weights removed |
| + KV-Cache | 95-98% | 12.5% | 6-10× | Generation speedup |
| + Batching | 95-98% | 12.5% | 12-40× | **Production ready!** |

## Key Learning: Optimization is Iterative

Unlike earlier milestones where you "build and run," optimization requires:
1. **Measure** (profile to find bottlenecks)
2. **Optimize** (apply targeted techniques)
3. **Validate** (check accuracy didn't degrade)
4. **Repeat** (iterate until deployment targets met)

This is the **systems thinking** that makes TinyTorch unique - you're not just learning ML, you're learning **ML systems engineering**.

## Running the Milestone

```bash
cd milestones/06_2018_mlperf

# Step 1: Profile and establish baseline
python 01_baseline_profile.py

# Step 2: Apply compression (quantization + pruning)
python 02_compression.py

# Step 3: Optimize generation (KV-cache + batching)
python 03_generation_opts.py
```

## Further Reading

- **MLPerf**: https://mlcommons.org/en/inference-edge-11/
- **Deep Compression** (Han et al., 2015): https://arxiv.org/abs/1510.00149
- **MobileNets** (Howard et al., 2017): https://arxiv.org/abs/1704.04861
- **Efficient Transformers Survey**: https://arxiv.org/abs/2009.06732

## Achievement Unlocked

After completing this milestone, you'll understand:
- How to profile ML models systematically
- Quantization and pruning tradeoffs
- Why generation is slow and how to fix it
- The iterative nature of production optimization

**You've learned ML Systems Engineering - the skill that ships products!**
