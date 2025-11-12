# Track Your Progress

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Monitor Your Learning Journey</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Track your capability development through 20 modules and 6 historical milestones</p>
</div>

**Purpose**: Monitor your progress as you build a complete ML framework from scratch. Track module completion and milestone achievements.

## The Core Workflow

TinyTorch follows a simple three-step cycle:

```
1. Edit modules → 2. Export to package → 3. Validate with milestones
```

**📖 See [Student Workflow](student-workflow.html)** for the complete development cycle.

## Understanding Modules vs Checkpoints vs Milestones

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">

**Modules (18 total)**: What you're building - the actual code implementations

- Located in `modules/source/`
- You implement each component from scratch
- Export with `tito module complete N`

**Milestones (6 total)**: How you validate - historical proof scripts

- Located in `milestones/`
- Run scripts that use YOUR implementations
- Recreate ML history (1957 Perceptron → 2018 MLPerf)

**Checkpoints (21 total)**: Optional progress tracking

- Use `tito checkpoint status` to view
- Tracks capability mastery
- Not required for the core workflow

**📖 See [Journey Through ML History](chapters/milestones.html)** for milestone details.

</div>

## Your Learning Path Overview

TinyTorch organizes learning through **three pedagogically-motivated tiers**, each building essential ML systems capabilities:

**📖 See [Three-Tier Learning Structure](chapters/00-introduction.html#three-tier-learning-pathway-build-complete-ml-systems)** for detailed tier breakdown, time estimates, and learning outcomes.

## Student Learning Journey

### Typical Student Progression by Tier
- **🏗️ Foundation Tier (6-8 weeks)**: Build mathematical infrastructure - tensors, autograd, optimizers, training loops
- **🏛️ Architecture Tier (4-6 weeks)**: Implement modern AI architectures - CNNs for vision, transformers for language
- **⚡ Optimization Tier (4-6 weeks)**: Deploy production systems - profiling, quantization, acceleration

### Study Approaches
- **Complete Builder** (14-18 weeks): Implement all three tiers from scratch
- **Focused Explorer** (4-8 weeks): Pick specific tiers based on your goals
- **Guided Learner** (8-12 weeks): Study implementations with hands-on exercises

**📖 See [Quick Start Guide](quickstart-guide.html)** for immediate hands-on experience with your first module.

## Module Progression

Your journey through 20 modules organized in three tiers:

### 🏗️ Foundation Tier (Modules 01-07)

Build the mathematical infrastructure:

| Module | Component | What You Build |
|--------|-----------|----------------|
| 01 | Tensor | N-dimensional arrays with operations |
| 02 | Activations | ReLU, Softmax, nonlinear functions |
| 03 | Layers | Linear layers, forward/backward |
| 04 | Losses | CrossEntropyLoss, MSELoss |
| 05 | Autograd | Automatic differentiation engine |
| 06 | Optimizers | SGD, Adam, parameter updates |
| 07 | Training | Complete training loops |

**Milestone unlocked**: M01 Perceptron (1957), M02 XOR (1969)

### 🏛️ Architecture Tier (Modules 08-13)

Implement modern architectures:

| Module | Component | What You Build |
|--------|-----------|----------------|
| 08 | DataLoader | Batching and data pipelines |
| 09 | Spatial | Conv2d, MaxPool2d for vision |
| 10 | Tokenization | Character-level tokenizers |
| 11 | Embeddings | Token and positional embeddings |
| 12 | Attention | Multi-head self-attention |
| 13 | Transformers | LayerNorm, TransformerBlock, GPT |

**Milestones unlocked**: M03 MLP (1986), M04 CNN (1998), M05 Transformers (2017)

### ⚡ Optimization Tier (Modules 14-20)

Optimize for production:

| Module | Component | What You Build |
|--------|-----------|----------------|
| 14 | Profiling | Performance measurement tools |
| 15 | Quantization | INT8/FP16 implementations |
| 16 | Compression | Pruning techniques |
| 17 | Memoization | KV-cache for generation |
| 18 | Acceleration | Batching strategies |
| 19 | Benchmarking | MLPerf-style fair comparison |
| 20 | Competition | Capstone optimization challenge |

**Milestone unlocked**: M06 MLPerf (2018)

## Optional: Checkpoint System

Track capability mastery with the optional checkpoint system:

```bash
tito checkpoint status  # View your progress
```

This provides 21 capability checkpoints corresponding to modules and validates your understanding. Helpful for self-assessment but **not required** for the core workflow.

**📖 See [Essential Commands](tito-essentials.html)** for checkpoint commands.

---

## Capability Development Approach

### Foundation Building (Checkpoints 0-3)
**Capability Focus**: Core computational infrastructure
- Environment configuration and dependency management
- Mathematical foundations with tensor operations
- Neural intelligence through nonlinear activation functions
- Network component abstractions and forward propagation

### Learning Systems (Checkpoints 4-7)
**Capability Focus**: Training and optimization
- Loss measurement and error quantification
- Automatic differentiation for gradient computation
- Parameter optimization with advanced algorithms
- Complete training loop implementation

### Advanced Architectures (Checkpoints 8-13)
**Capability Focus**: Specialized neural networks
- Spatial processing for computer vision systems
- Efficient data loading and preprocessing pipelines
- Natural language processing and tokenization
- Representation learning with embeddings
- Attention mechanisms for sequence understanding
- Complete transformer architecture mastery

### Production Systems (Checkpoints 14-15)
**Capability Focus**: Performance and deployment
- Profiling, optimization, and bottleneck analysis
- End-to-end ML systems engineering
- Production-ready deployment and monitoring

---

## Start Building Capabilities

Begin developing ML systems competencies immediately:

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Begin Capability Development</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Start with foundational capabilities and progress systematically</p>
<a href="quickstart-guide.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">15-Minute Start →</a>
<a href="chapters/01-setup.html" style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Begin Setup →</a>
</div>

## How to Track Your Progress

The essential workflow:

```bash
# 1. Work on a module
cd modules/source/03_layers
jupyter lab 03_layers_dev.py

# 2. Export when ready
tito module complete 03

# 3. Validate with milestones
cd ../../milestones/01_1957_perceptron
python 01_rosenblatt_forward.py  # Uses YOUR implementation!
```

**Optional**: Use `tito checkpoint status` to see capability tracking

**📖 See [Student Workflow](student-workflow.html)** for the complete development cycle.

**Approach**: You're building ML systems engineering capabilities through hands-on implementation. Each module adds new functionality to your framework, and milestones prove it works.