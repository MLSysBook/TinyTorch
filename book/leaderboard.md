# TinyMLPerf Leaderboard

```{admonition} TinyMLPerf Competition
:class: tip
**Coming Spring 2025** - Be among the first to set baseline performance scores with your TinyTorch implementations!
```

## Competition Overview

TinyMLPerf is TinyTorch's performance benchmarking competition where students optimize their implementations and compete across multiple categories. Test your ML systems engineering skills against peers worldwide!

### Competition Categories

| Event | Challenge | Metric | Status |
|-------|-----------|--------|--------|
| **MLP Sprint** | Train MNIST MLP to 95%+ accuracy | Training time + memory | 🚧 Coming Soon |
| **CNN Marathon** | CIFAR-10 CNN to 75%+ accuracy | Inference speed + memory | 🚧 Coming Soon |
| **Transformer Decathlon** | TinyGPT generation quality | Tokens/second + BLEU score | 🚧 Coming Soon |
| **Memory Master** | Minimize memory footprint | Peak memory usage | 🚧 Coming Soon |
| **Innovation Track** | Novel optimization techniques | Judged submissions | 🚧 Coming Soon |

### Leaderboard Structure

#### Speed Champions
*Fastest inference times across all benchmarks*

#### Memory Masters  
*Lowest memory footprint while maintaining accuracy*

#### Efficiency Leaders
*Best accuracy-per-resource ratio*

#### Innovation Awards
*Most creative optimization techniques*

## How to Participate

### 1. Complete Your TinyTorch Implementation
Finish modules 1-20 to unlock all benchmarking capabilities:

```bash
# Check your progress
tito checkpoint status

# Ensure all modules are complete
tito module status --all
```

### 2. Run Local Benchmarks
Test your optimizations locally:

```bash
# Quick performance check
tito benchmark run --event mlp_sprint

# Full competition suite
tito benchmark run --all-events

# Get detailed profiling
tito benchmark profile --event cnn_marathon
```

### 3. Submit to Leaderboard
*Coming Spring 2025*

```bash
# Future submission command
tito benchmark submit --event cnn_marathon --team "YourTeamName"
```

## Current Status

### 🚧 Development Phase

- **Baseline Implementation**: Setting reference scores with optimized TinyTorch
- **Benchmark Infrastructure**: Building fair comparison framework  
- **Submission System**: Creating automated evaluation pipeline
- **Leaderboard Website**: Developing at `tinytorch.org/leaderboard`

### Launch Timeline

| Milestone | Date | Status |
|-----------|------|--------|
| Baseline Scores | January 2025 | In Progress |
| Beta Testing | February 2025 | Planned |
| Public Launch | March 2025 | Planned |
| First Competition | April 2025 | Planned |

## Benchmark Specifications

### MLP Sprint
- **Dataset**: MNIST (60K training, 10K test)
- **Target**: 95%+ accuracy
- **Measurement**: Wall-clock training time + peak memory
- **Hardware**: Standardized CPU benchmarking environment

### CNN Marathon  
- **Dataset**: CIFAR-10 (50K training, 10K test)
- **Target**: 75%+ accuracy  
- **Measurement**: Inference time for 1K samples + memory usage
- **Optimization**: Any technique (quantization, pruning, etc.)

### Transformer Decathlon
- **Task**: Text generation with TinyGPT
- **Evaluation**: Perplexity + generation speed
- **Context**: 512 token sequences
- **Quality Threshold**: Coherent multi-sentence generation

## Fair Competition Guidelines

### Allowed Optimizations
✅ **Mathematical optimizations** (vectorization, cache-friendly algorithms)  
✅ **Memory management** (gradient checkpointing, activation recomputation)  
✅ **Quantization** (INT8, FP16 precision)  
✅ **Model compression** (pruning, knowledge distillation)  
✅ **Hardware-specific** (SIMD, parallel processing)  

### Prohibited Actions
❌ **External libraries** beyond NumPy (no PyTorch, TensorFlow, etc.)  
❌ **Pre-trained models** or weights  
❌ **Dataset modifications** or augmentations  
❌ **Accuracy sacrifices** below minimum thresholds  

## Recognition & Rewards

### Digital Badges
- **Speed Demon**: Top 10% in any speed category
- **Memory Wizard**: Top 10% in memory efficiency  
- **Innovation Pioneer**: Novel optimization technique
- **Triple Crown**: Top 10% in three different events

### Academic Recognition
- Certificate of achievement for course completion
- Special recognition for leaderboard leaders
- Potential research collaboration opportunities

## Get Ready

### Optimization Strategies to Explore

**Memory Optimization:**
- Gradient accumulation instead of large batches
- Activation checkpointing for memory-compute trade-offs
- Efficient attention implementations (O(N√N) variants)

**Speed Optimization:**
- Vectorized operations and cache-friendly memory access
- Batch processing and parallel computation
- Custom kernel implementations for bottlenecks

**Quality Optimization:**
- Learning rate scheduling and optimizer tuning
- Architecture search within parameter constraints
- Regularization techniques for better generalization

---

## Join the Competition

**Ready to compete?** Start building your TinyTorch implementation:

1. **[Begin Module 01](chapters/01-setup.md)** - Set up your development environment
2. **[Track Progress](checkpoint-system.md)** - Use checkpoints to monitor your journey
3. **[Join Community](https://github.com/mlsysbook/TinyTorch/discussions)** - Connect with other competitors

**Stay Updated:**
- 📧 **Email**: Subscribe to TinyTorch announcements
- 🌐 **Website**: `tinytorch.org/leaderboard` (launching soon)
- 💬 **Discord**: TinyTorch community server (coming soon)

---

> 🏆 **"The best way to understand ML systems is to optimize them yourself"** - Be ready to prove your mastery when the leaderboard launches!