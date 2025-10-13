# TinyTorch

**Build ML Systems From First Principles**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Documentation](https://img.shields.io/badge/docs-jupyter_book-orange.svg)](https://mlsysbook.github.io/TinyTorch/)
![Status](https://img.shields.io/badge/status-active-success.svg)

---
> 🚧 **This Project is Actively Under Development**
>
> TinyTorch is not yet complete. Modules, docs, and examples are being added and refined weekly.  
> A stable release is planned for **end of this year**.  
> Expect rapid updates, occasional breaks, and lots of new content.
> You are welcome to skim this web
---

## 📖 Table of Contents
- [Why TinyTorch?](#why-tinytorch)
- [What You'll Build](#what-youll-build) - Including several north star goals
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
│   ├── 01_tensor/     # Start: Build tensor operations from scratch
│   ├── 02_activations/# Add: Neural network intelligence (ReLU, Softmax)
│   ├── 03_layers/     # Build: Network components (Linear, Module system)
│   └── ...            # Progress through 20 learning modules
│
├── tinytorch/         # 📦 Generated package (auto-built from your work)
│   ├── core/          # Your implementations exported for use
│   ├── nn/            # Neural network components you built
│   └── optim/         # Optimizers you implemented
│
├── tests/             # 🧪 Comprehensive validation system
│   ├── checkpoints/   # 16 capability tests tracking your progress
│   └── integration/   # Full system validation tests
│
├── book/              # 📚 Complete course documentation (Jupyter Book)
│   ├── chapters/      # Learning guides for each module
│   └── resources/     # Additional learning materials
│
└── examples/          # 🎯 Milestone demonstrations (unlock as you progress)
    ├── mnist_training.py    # Train neural networks on real data
    └── cifar10_cnn.py       # Achieve 75%+ accuracy on CIFAR-10
```

**🚨 CRITICAL: Work in `modules/`, Import from `tinytorch/`**
- ✅ **Edit code**: Always in `modules/XX_name/name_dev.py` files
- ✅ **Import & use**: Your built components from `tinytorch.core.component`
- ❌ **Never edit**: Files in `tinytorch/` directly (auto-generated from modules)
- 🔄 **Sync changes**: Use `tito module complete XX_name` to update package

**Why this structure?** Learn by building (modules) → Use what you built (tinytorch) → Validate mastery (tests)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Start learning
cd modules/01_tensor
jupyter lab tensor_dev.py

# Track progress
tito checkpoint status
```

## Learning Journey

### 20 Progressive Modules

#### Part I: Neural Network Foundations (Modules 1-8)
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
| 08 | Spatial | Conv2d + MaxPool2d + CNN operations | **Parameter scaling**, spatial locality, convolution efficiency |

**Milestone Achievement**: Train XOR solver and MNIST classifier after Module 8

---

#### Part II: Computer Vision (Modules 9-10)
Build CNNs that classify real images

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 09 | DataLoader | Efficient data pipelines + CIFAR-10 | **Batch processing**, memory-mapped I/O, data pipeline bottlenecks |
| 10 | Tokenization | Text processing + vocabulary | **Vocabulary scaling**, tokenization bottlenecks, sequence processing |

**Milestone Achievement**: CIFAR-10 CNN with 75%+ accuracy

---

#### Part III: Language Models (Modules 11-14)
Build transformers that generate text

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 11 | Tokenization | Text processing + vocabulary | **Vocabulary scaling** (memory vs sequence length), tokenization bottlenecks |
| 12 | Embeddings | Token embeddings + positional encoding | **Embedding tables** (vocab × dim parameters), lookup performance |
| 13 | Attention | Multi-head attention mechanisms | **O(N²) scaling**, memory bottlenecks, attention optimization |
| 14 | Transformers | Complete transformer blocks | **Layer scaling**, memory requirements, architectural trade-offs |

**Milestone Achievement**: TinyGPT language generation

---

#### Part IV: System Optimization (Modules 15-20)
Profile, optimize, and benchmark ML systems

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 15 | Profiling | Performance analysis + bottleneck detection | **Memory profiling**, FLOP counting, **Amdahl's Law**, performance measurement |
| 16 | Acceleration | Hardware optimization + cache-friendly algorithms | **Cache hierarchies**, memory access patterns, **vectorization vs loops** |
| 17 | Quantization | Model compression + precision reduction | **Precision trade-offs** (FP32→INT8), memory reduction, accuracy preservation |
| 18 | Compression | Pruning + knowledge distillation | **Sparsity patterns**, parameter reduction, **compression ratios** |
| 19 | Caching | Memory optimization + KV caching | **Memory vs compute trade-offs**, cache management, generation efficiency |
| 20 | Benchmarking | **TinyMLPerf competition framework** | **Competitive optimization**, relative performance metrics, innovation scoring |

**Milestone Achievement**: TinyMLPerf optimization competition

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

## Key Features

### Essential-Only Design
- **Focus on What Matters**: ReLU + Softmax (not 20 activation functions)
- **Production Relevance**: Adam + SGD (the optimizers you actually use)
- **Core ML Systems**: Memory profiling, performance analysis, scaling insights

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

## Milestone Examples

As you complete modules, exciting examples unlock to show your framework in action:

### After Module 04: First Neural Network
```bash
cd examples/perceptron_1957
python rosenblatt_perceptron.py
# Build the first trainable neural network (1957)
```

### After Module 06: Multi-Layer Networks
```bash
cd examples/xor_1969  
python minsky_xor_problem.py
# Solve the XOR problem with multi-layer networks (1969)
```

### After Module 08: Real Computer Vision
```bash
cd examples/mnist_mlp_1986
python train_mlp.py
# Achieve 95%+ accuracy on MNIST (1986)
```

### After Module 10: Modern CNNs  
```bash
cd examples/cifar_cnn_modern
python train_cnn.py
# Achieve 75%+ accuracy on CIFAR-10
```

### After Module 14: Language Models
```bash
cd examples/gpt_2018
python train_gpt.py
# Generate text with your transformer implementation
```

### After Module 20: TinyMLPerf Competition
```bash
# Use TinyMLPerf to benchmark your optimizations
tito benchmark run --event mlp_sprint
tito benchmark run --event cnn_marathon  
tito benchmark run --event transformer_decathlon
# Compete in ML systems optimization benchmarks
```

### After Module 20: Complete Optimization Suite
```bash
# Use TinyMLPerf to benchmark and optimize your complete framework
tito benchmark run --comprehensive
python examples/optimization_showcase.py
# Professional ML systems optimization
```

**These aren't toy demos** - they're real ML applications achieving solid results with YOUR framework built from scratch and optimized for performance!

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
python tests/run_all_modules.py
```

- **20 modules** passing all tests with 100% health status
- **21 capability checkpoints** tracking learning progress
- **Complete optimization pipeline** from profiling to benchmarking
- **TinyMLPerf competition framework** for performance excellence
- **KISS principle design** for clear, maintainable code
- **Streamlined development**: 7-agent workflow for efficient coordination
- **Essential-only features**: Focus on what's used in production ML systems  

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
- ✅ **Complete implementation** - Build every component from scratch
- ✅ **Real achievements** - Train CNNs on CIFAR-10 to 75%+ accuracy
- ✅ **Systems thinking** - Understand memory, performance, and scaling
- ✅ **Production relevance** - Learn patterns from PyTorch and TensorFlow
- ✅ **Immediate validation** - 21 capability checkpoints track progress

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
cd modules/01_tensor && jupyter lab tensor_dev.py
```

---

**Start Small. Go Deep. Build ML Systems.**
