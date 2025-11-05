# TinyTorch

**Build ML Systems From First Principles**

<!-- Core Badges -->
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Documentation](https://img.shields.io/badge/docs-jupyter_book-orange.svg)](https://mlsysbook.github.io/TinyTorch/)
![Status](https://img.shields.io/badge/status-active-success.svg)

<!-- Activity Badges -->
![Last Commit](https://img.shields.io/github/last-commit/MLSysBook/TinyTorch)
![Commit Activity](https://img.shields.io/github/commit-activity/m/MLSysBook/TinyTorch)
![GitHub Stars](https://img.shields.io/github/stars/MLSysBook/TinyTorch?style=social)
![Contributors](https://img.shields.io/github/contributors/MLSysBook/TinyTorch)

> 🚧 **Work in Progress** - Actively developing TinyTorch for Spring 2025! All 20 core modules (01-20) are implemented but still being debugged and tested. Core foundation modules (01-09) are stable. Transformer and optimization modules (10-20) are functional but undergoing refinement. Join us in building the future of ML systems education.

## 📖 Table of Contents
- [Why TinyTorch?](#why-tinytorch)
- [What You'll Build](#what-youll-build) - Including the **CIFAR-10 North Star Goal**
- [Quick Start](#quick-start) - Get running in 5 minutes
- [Learning Journey](#learning-journey) - 20 progressive modules
- [Learning Progression & Checkpoints](#learning-progression--checkpoints) - 21 capability checkpoints
- [Key Features](#key-features) - Essential-only design
- [Milestone Examples](#milestone-examples) - Real achievements
- [Documentation & Resources](#-documentation--resources) - For students, instructors, developers
- [Ready to Start Building?](#-ready-to-start-building) - Your path forward

## Why TinyTorch?

**"Most ML education teaches you to _use_ frameworks. TinyTorch teaches you to _build_ them."**

In an era where AI is reshaping every industry, the difference between ML users and ML engineers determines who drives innovation versus who merely consumes it. TinyTorch bridges this critical gap by teaching you to build every component of modern AI systems from scratch—from tensors to transformers.

A Harvard University course that transforms you from framework user to systems engineer, giving you the deep understanding needed to optimize, debug, and innovate at the foundation of AI.

## What You'll Build

A **complete ML framework** capable of:

🎯 **North Star Achievement**: Train CNNs on CIFAR-10 to **75%+ accuracy**
- Real computer vision with 50,000 training images
- Built entirely from scratch using only NumPy
- Competitive performance with modern frameworks

**Additional Capabilities**:
- Building GPT-style language models with attention mechanisms
- Modern optimizers (Adam, SGD) with learning rate scheduling
- Performance profiling, optimization, and competitive benchmarking
- Complete ML systems pipeline from tensors to deployment

**No dependencies on PyTorch or TensorFlow - everything is YOUR code!**

## Repository Structure

```
TinyTorch/
├── modules/           # 🏗️ YOUR workspace - implement ML systems here
│   ├── source/
│   │   ├── 01_tensor/        # Module 01: Tensor operations from scratch
│   │   ├── 02_activations/   # Module 02: ReLU, Softmax activations
│   │   ├── 03_layers/        # Module 03: Linear layers, Module system
│   │   ├── 04_losses/        # Module 04: MSE, CrossEntropy losses
│   │   ├── 05_autograd/      # Module 05: Automatic differentiation
│   │   ├── 06_optimizers/    # Module 06: SGD, Adam optimizers
│   │   ├── 07_training/      # Module 07: Complete training loops
│   │   ├── 08_dataloader/    # Module 08: Efficient data pipelines
│   │   ├── 09_spatial/       # Module 09: Conv2d, MaxPool2d, CNNs
│   │   ├── 10_tokenization/  # Module 10: Text processing
│   │   ├── 11_embeddings/    # Module 11: Token & positional embeddings
│   │   ├── 12_attention/     # Module 12: Multi-head attention
│   │   ├── 13_transformers/  # Module 13: Complete transformer blocks
│   │   ├── 14_kvcaching/     # Module 14: KV-cache optimization
│   │   ├── 15_profiling/     # Module 15: Performance analysis
│   │   ├── 16_acceleration/  # Module 16: Hardware optimization
│   │   ├── 17_quantization/  # Module 17: Model compression
│   │   ├── 18_compression/   # Module 18: Pruning & distillation
│   │   ├── 19_benchmarking/  # Module 19: Performance measurement
│   │   └── 20_capstone/      # Module 20: Complete ML systems
│
├── milestones/        # 🏆 Historical ML evolution - prove what you built!
│   ├── 01_1957_perceptron/   # Rosenblatt's first trainable network
│   ├── 02_1969_xor_crisis/   # Minsky's challenge & multi-layer solution
│   ├── 03_1986_mlp_revival/  # Backpropagation & MNIST digits
│   ├── 04_1998_cnn_revolution/ # LeCun's CNNs & CIFAR-10
│   ├── 05_2017_transformer_era/ # Attention mechanisms & language
│   └── 06_2024_systems_age/  # Modern optimization & profiling
│
├── tinytorch/         # 📦 Generated package (auto-built from your work)
│   ├── core/          # Your tensor, autograd implementations
│   ├── nn/            # Your neural network components
│   └── optim/         # Your optimizers
│
├── tests/             # 🧪 Comprehensive validation system
│   ├── 01_tensor/     # Per-module integration tests
│   ├── 02_activations/
│   └── ...            # Tests mirror module structure
│
└── book/              # 📚 Complete course documentation (Jupyter Book)
    ├── chapters/      # Learning guides for each module
    └── resources/     # Additional learning materials
```

**🚨 CRITICAL: Work in `modules/`, Import from `tinytorch/`**
- ✅ **Edit code**: Always in `modules/XX_name/name_dev.py` files
- ✅ **Import & use**: Your built components from `tinytorch.core.component`
- ❌ **Never edit**: Files in `tinytorch/` directly (auto-generated from modules)
- 🔄 **Sync changes**: Use `tito module complete XX_name` to update package

**Why this structure?** Learn by building (modules) → Use what you built (tinytorch) → Validate mastery (tests)

## Quick Start

```bash
# Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# One-command setup (handles Apple Silicon, dependencies, everything)
./setup-environment.sh

# Activate environment
source activate.sh

# Verify setup
tito system doctor

# Start building
tito module view 01_tensor
```

**That's it!** The setup script handles:
- ✅ Virtual environment creation (arm64 on Apple Silicon)
- ✅ All dependencies (NumPy, Jupyter, Rich, etc.)
- ✅ TinyTorch package installation in development mode
- ✅ Architecture detection and optimization

## Learning Journey

### 20 Progressive Modules

#### Part I: Neural Network Foundations (Modules 1-7)
Build and train neural networks from scratch

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 01 | Tensor | N-dimensional arrays + operations | **Memory layout, cache efficiency**, broadcasting semantics |
| 02 | Activations | ReLU + Softmax (essential functions) | **Numerical stability**, gradient flow, function properties |
| 03 | Layers | Linear layers + Module abstraction | **Parameter management**, weight initialization, forward/backward |
| 04 | Losses | MSE + CrossEntropy (essential losses) | **Numerical precision**, loss landscapes, training objectives |
| 05 | Autograd | Automatic differentiation engine | **Computational graphs**, memory management, gradient flow |
| 06 | Optimizers | SGD + Adam (essential optimizers) | **Memory efficiency** (Adam uses 3x memory), convergence |
| 07 | Training | Complete training loops + evaluation | **Training dynamics**, checkpoints, monitoring systems |

**Milestone Achievement**: Train XOR solver and MNIST classifier after Module 7

---

#### Part II: Computer Vision (Modules 8-9)
Build CNNs that classify real images

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 08 | DataLoader | Efficient data pipelines + CIFAR-10 | **Batch processing**, memory-mapped I/O, data pipeline bottlenecks |
| 09 | Spatial | Conv2d + MaxPool2d + CNN operations | **Parameter scaling**, spatial locality, convolution efficiency |

**Milestone Achievement**: CIFAR-10 CNN with 75%+ accuracy

---

#### Part III: Language Models (Modules 10-14)
Build transformers that generate text

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 10 | Tokenization | Text processing + vocabulary | **Vocabulary scaling**, tokenization bottlenecks, sequence processing |
| 11 | Embeddings | Token embeddings + positional encoding | **Embedding tables** (vocab × dim parameters), lookup performance |
| 12 | Attention | Multi-head attention mechanisms | **O(N²) scaling**, memory bottlenecks, attention optimization |
| 13 | Transformers | Complete transformer blocks | **Layer scaling**, memory requirements, architectural trade-offs |
| 14 | KV-Caching | Inference optimization for transformers | **Memory vs compute trade-offs**, cache management, generation efficiency |

**Milestone Achievement**: TinyGPT language generation with optimized inference

---

#### Part IV: System Optimization (Modules 15-20)
Profile, optimize, and benchmark ML systems

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 15 | Profiling | Performance analysis + bottleneck detection | **Memory profiling**, FLOP counting, **Amdahl's Law**, performance measurement |
| 16 | Acceleration | Hardware optimization + cache-friendly algorithms | **Cache hierarchies**, memory access patterns, **vectorization vs loops** |
| 17 | Quantization | Model compression + precision reduction | **Precision trade-offs** (FP32→INT8), memory reduction, accuracy preservation |
| 18 | Compression | Pruning + knowledge distillation | **Sparsity patterns**, parameter reduction, **compression ratios** |
| 19 | Benchmarking | Performance measurement + TinyMLPerf competition | **Competitive optimization**, relative performance metrics, innovation scoring |
| 20 | Capstone | Complete end-to-end ML systems project | **Integration**, production deployment, **real-world ML engineering** |

**Milestone Achievement**: TinyMLPerf optimization competition & portfolio capstone project

---


## Learning Philosophy

**Most courses teach you to USE frameworks. TinyTorch teaches you to UNDERSTAND them.**

```python
# Traditional Course:
import torch
model.fit(X, y)  # Magic happens

# TinyTorch:
# You implement every component
# You measure memory usage
# You optimize performance
# You understand the systems
```

### Why Build Your Own Framework?

- **Deep Understanding** - Know exactly what `loss.backward()` does
- **Systems Thinking** - Understand memory, compute, and scaling
- **Debugging Skills** - Fix problems at any level of the stack
- **Production Ready** - Learn patterns used in real ML systems

## Learning Progression & Checkpoints

### Capability-Based Learning System

Track your progress through **capability-based checkpoints** that validate your ML systems knowledge:

```bash
# Check your current progress
tito checkpoint status

# See your capability development timeline
tito checkpoint timeline
```

**Module Progression:**
- **01-02**: Foundation (Tensor, Activations)
- **03-07**: Core Networks (Layers, Losses, Autograd, Optimizers, Training)
- **08-09**: Computer Vision (DataLoader, Spatial ops - unlocks CIFAR-10 @ 75%+)
- **10-14**: Language Models (Tokenization, Embeddings, Attention, Transformers, KV-Caching)
- **15-19**: System Optimization (Profiling, Acceleration, Quantization, Compression, Benchmarking)
- **20**: Capstone (Complete end-to-end ML systems)

Each module asks: **"Can I build this capability from scratch?"** with hands-on validation.

### Module Completion Workflow

```bash
# Complete a module (automatic export + testing)
tito module complete 01_tensor

# This automatically:
# 1. Exports your implementation to the tinytorch package
# 2. Runs the corresponding capability checkpoint test
# 3. Shows your achievement and suggests next steps
```  

## Key Features

### Essential-Only Design
- **Focus on What Matters**: ReLU + Softmax (not 20 activation functions)
- **Production Relevance**: Adam + SGD (the optimizers you actually use)
- **Core ML Systems**: Memory profiling, performance analysis, scaling insights
- **Real Applications**: CIFAR-10 CNNs, not toy examples

### For Students
- **Interactive Demos**: Rich CLI visualizations for every concept
- **Checkpoint System**: Track your learning progress through 16 capabilities
- **Immediate Testing**: Validate your implementations instantly
- **Systems Focus**: Learn ML engineering, not just algorithms

### For Instructors
- **NBGrader Integration**: Automated grading workflow
- **Progress Tracking**: Monitor student achievements
- **Jupyter Book**: Professional course website
- **Complete Solutions**: Reference implementations included

## 🏆 Milestone Examples - Journey Through ML History

As you complete modules, unlock historical ML milestones demonstrating YOUR implementations:

### 🧠 01. Perceptron (1957) - After Module 03
```bash
cd milestones/01_1957_perceptron
python perceptron_trained.py
# Rosenblatt's first trainable neural network
# YOUR Linear layer + Sigmoid recreates history!
```
**Requirements**: Modules 01-03 (Tensor, Activations, Layers)  
**Achievement**: Binary classification with gradient descent

---

### ⚡ 02. XOR Crisis (1969) - After Module 05
```bash
cd milestones/02_1969_xor_crisis
python xor_solved.py
# Solve Minsky's XOR challenge with hidden layers
# YOUR autograd enables multi-layer learning!
```
**Requirements**: Modules 01-05 (+ Autograd)  
**Achievement**: Non-linear problem solving

---

### 🔢 03. MLP Revival (1986) - After Module 07
```bash
cd milestones/03_1986_mlp_revival
python mlp_digits.py     # 8x8 digit classification
python mlp_mnist.py      # Full MNIST dataset
# Backpropagation revolution on real vision!
# YOUR training loops achieve 95%+ accuracy
```
**Requirements**: Modules 01-07 (+ Optimizers, Training)  
**Achievement**: Real computer vision with MLPs

---

### 🖼️ 04. CNN Revolution (1998) - After Module 09
```bash
cd milestones/04_1998_cnn_revolution
python cnn_digits.py     # Spatial features on digits
python lecun_cifar10.py  # Natural images (CIFAR-10)
# LeCun's CNNs achieve 75%+ on CIFAR-10!
# YOUR Conv2d + MaxPool2d unlock spatial intelligence
```
**Requirements**: Modules 01-09 (+ DataLoader, Spatial)  
**Achievement**: **🎯 North Star - CIFAR-10 @ 75%+ accuracy**

---

### 🤖 05. Transformer Era (2017) - After Module 13
```bash
cd milestones/05_2017_transformer_era
python vaswani_shakespeare.py
# Attention mechanisms for language modeling
# YOUR attention implementation generates text!
```
**Requirements**: Modules 01-13 (+ Tokenization, Embeddings, Attention, Transformers)  
**Achievement**: Language generation with self-attention

---

### ⚡ 06. Systems Age (2024) - After Module 19
```bash
cd milestones/06_2024_systems_age
python optimize_models.py
# Profile, optimize, and benchmark YOUR framework
# Compete on TinyMLPerf leaderboard!
```
**Requirements**: Modules 01-19 (Full optimization suite)  
**Achievement**: Production-grade ML systems engineering

---

**Why Milestones Matter:**
- 🎓 **Educational**: Experience the actual evolution of AI (1957→2024)
- 🔧 **Systems Thinking**: Understand why each innovation mattered
- 🏆 **Proof of Mastery**: Real achievements with YOUR implementations
- 📈 **Progressive**: Each milestone builds on previous foundations

**These aren't toy demos** - they're historically significant ML achievements rebuilt with YOUR framework!

## Testing & Validation

All demos and modules are thoroughly tested:

```bash
# Check your learning progress
tito checkpoint status

# Test specific capabilities
tito checkpoint test 01  # Foundation checkpoint
tito checkpoint test 05  # Autograd checkpoint

# Complete and test modules
tito module complete 01_tensor  # Exports and tests

# Run comprehensive validation
pytest tests/
```

**Current Status**:
- ✅ **20 modules implemented** (01 Tensor → 20 Capstone) - all code exists
- ✅ **6 historical milestones** (1957 Perceptron → 2024 Systems Age)
- ✅ **Foundation modules stable** (01-09): Tensor through Spatial operations
- 🚧 **Transformer modules functional** (10-14): Tokenization through KV-Caching - undergoing testing
- 🚧 **Optimization modules functional** (15-20): Profiling through Capstone - undergoing testing
- ✅ **KISS principle design** for clear, maintainable code
- ✅ **Essential-only features**: Focus on what's used in production ML systems
- 🎯 **Target: Spring 2025** - Active debugging and refinement in progress  

## 📚 Documentation & Resources

### 🎓 For Students
- **[Interactive Course Website](https://mlsysbook.github.io/TinyTorch/)** - Complete learning platform
- **[Getting Started Guide](docs/README.md)** - Installation and first steps
- **[CIFAR-10 Training Guide](docs/cifar10-training-guide.md)** - Achieving the north star goal
- **[Module READMEs](/modules/)** - Individual module documentation

### 👨‍🏫 For Instructors
- **[Instructor Guide](instructor/README.md)** - Complete teaching resources
- **[NBGrader Workflow](book/instructor-guide.md)** - Automated grading setup
- **[System Architecture](book/system-architecture.md)** - Technical overview

### 🛠️ For Developers
- **[Agent Coordination](.claude/guidelines/AGENT_COORDINATION.md)** - Development workflow
- **[Module Development](.claude/guidelines/MODULE_DEVELOPMENT.md)** - Creating new modules
- **[Testing Standards](.claude/guidelines/TESTING_STANDARDS.md)** - Quality assurance

## TinyMLPerf Competition & Leaderboard

### Compete and Compare Your Optimizations

TinyMLPerf is our performance benchmarking competition where you optimize your TinyTorch implementations and compete on the leaderboard:

```bash
# Run benchmarks locally
tito benchmark run --event mlp_sprint      # Quick MLP benchmark
tito benchmark run --event cnn_marathon    # CNN optimization challenge
tito benchmark run --event transformer_decathlon  # Ultimate transformer test

# Submit to leaderboard (coming soon)
tito benchmark submit --event cnn_marathon
```

**Leaderboard Categories:**
- **Speed**: Fastest inference time
- **Memory**: Lowest memory footprint  
- **Efficiency**: Best accuracy/resource ratio
- **Innovation**: Novel optimization techniques

📊 **View Leaderboard**: [TinyMLPerf Competition](https://mlsysbook.github.io/TinyTorch/leaderboard.html) | Future: `tinytorch.org/leaderboard`

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

We acknowledge several excellent educational ML framework projects with similar names:
- [tinygrad](https://github.com/tinygrad/tinygrad) - George Hotz's minimalist deep learning framework
- [micrograd](https://github.com/karpathy/micrograd) - Andrej Karpathy's tiny autograd engine
- [MiniTorch](https://minitorch.github.io/) - Cornell's educational framework
- Other TinyTorch implementations - Various educational implementations on GitHub

**Our TinyTorch** focuses specifically on ML systems engineering with a complete curriculum, NBGrader integration, and production deployment—designed as a comprehensive university course rather than a standalone library.

## Acknowledgments

Created by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) at Harvard University.

Special thanks to students and contributors who helped refine this educational framework.

---

## 🚀 Ready to Start Building?

**TinyTorch transforms you from ML framework user to ML systems engineer.**

### What Makes TinyTorch Different?
- ✅ **Essential-only features** - Focus on what's actually used in production
- 🚧 **Complete implementation** - Build every component from scratch (20 modules in development)
- 🎯 **Real achievements** - Train CNNs on CIFAR-10 to 75%+ accuracy (target)
- ✅ **Systems thinking** - Understand memory, performance, and scaling
- ✅ **Production relevance** - Learn patterns from PyTorch and TensorFlow
- ✅ **Progressive learning** - 20 modules from tensors to transformers to optimization

### Your Learning Journey
1. **Week 1-2**: Foundation (Tensors, Activations, Layers)
2. **Week 3-4**: Training Pipeline (Losses, Autograd, Optimizers, Training)
3. **Week 5-6**: Computer Vision (Spatial ops, DataLoaders, CIFAR-10)
4. **Week 7-8**: Language Models (Tokenization, Attention, Transformers)
5. **Week 9-10**: Optimization (Profiling, Acceleration, Benchmarking)

### Getting Started
```bash
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch && source setup.sh
cd modules/source/01_tensor && jupyter lab tensor_dev.py
```

---

**Start Small. Go Deep. Build ML Systems.**
