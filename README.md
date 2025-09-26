# TinyTorch

**Build ML Systems From First Principles**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Documentation](https://img.shields.io/badge/docs-jupyter_book-orange.svg)](https://mlsysbook.github.io/TinyTorch/)
![Status](https://img.shields.io/badge/status-active-success.svg)

> 🚧 **Work in Progress** - We're actively developing TinyTorch for Spring 2025! All core modules are complete and tested. Join us in building the future of ML systems education.

## Why TinyTorch?

**"Most ML education teaches you to _use_ frameworks. TinyTorch teaches you to _build_ them."**

In an era where AI is reshaping every industry, the difference between ML users and ML engineers determines who drives innovation versus who merely consumes it. TinyTorch bridges this critical gap by teaching you to build every component of modern AI systems from scratch—from tensors to transformers.

A Harvard University course that transforms you from framework user to systems engineer, giving you the deep understanding needed to optimize, debug, and innovate at the foundation of AI.

## What You'll Build

A **complete ML framework** capable of:
- Training neural networks on CIFAR-10 to 75%+ accuracy (reliably achievable!)
- Building GPT-style language models  
- Implementing modern optimizers (Adam, learning rate scheduling)
- Production deployment with monitoring and MLOps

All built from scratch using only NumPy - no PyTorch, no TensorFlow!

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
cd modules/source/01_setup
jupyter lab setup_dev.py

# Track progress
tito checkpoint status
```

## Learning Journey

### 20 Progressive Modules

#### Part I: Neural Network Foundations (Modules 1-8)
Build and train neural networks from scratch

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 01 | Setup | Development environment | CLI tools, dependency management, testing frameworks |
| 02 | Tensor | N-dimensional arrays + gradients | **Memory layout, cache efficiency**, broadcasting semantics |
| 03 | Activations | ReLU + Softmax + derivatives | **Numerical stability**, saturation analysis, gradient flow |
| 04 | Layers | Linear + Module + parameter management | **Parameter counting**, weight initialization, modularity patterns |
| 05 | Loss | MSE + CrossEntropy + gradient computation | **Numerical precision**, loss landscape analysis, convergence metrics |
| 06 | Autograd | Automatic differentiation engine | **Computational graphs**, memory management, gradient accumulation |
| 07 | Optimizers | SGD + Adam + learning schedules | **Memory efficiency** (Adam uses 3x SGD), convergence dynamics |
| 08 | Training | Complete training loops + evaluation | **Training dynamics**, checkpoint systems, performance monitoring |

**Milestone Achievement**: Train XOR solver and MNIST classifier after Module 8

---

#### Part II: Computer Vision (Modules 9-10)
Build CNNs that classify real images

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 09 | Spatial | Conv2d + MaxPool2d + CNN operations | **Parameter scaling** (filters × channels), spatial locality, convolution efficiency |
| 10 | DataLoader | Efficient data pipelines + CIFAR-10 | **Batch processing**, memory-mapped I/O, data pipeline bottlenecks |

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

### For Students
- **Interactive Demos**: Rich CLI visualizations for every concept
- **Checkpoint System**: Track your learning progress
- **Immediate Testing**: Validate your implementations instantly
- **Real Datasets**: Train on CIFAR-10, not toy examples

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
# Run comprehensive test suite (recommended)
tito test --comprehensive

# Run checkpoint tests
tito checkpoint test 01

# Test specific modules
tito test --module tensor

# Run all module tests
python tests/run_all_modules.py
```

- **20 modules** passing all tests with 100% health status  
- **16 capability checkpoints** tracking learning progress  
- **Complete optimization pipeline** from profiling to benchmarking  
- **TinyMLPerf competition framework** for performance excellence  
- **KISS principle design** for clear, maintainable code  

## Documentation

- **[Course Website](https://mlsysbook.github.io/TinyTorch/)** - Complete interactive course
- **[Instructor Guide](docs/INSTRUCTOR_GUIDE.md)** - Teaching resources  
- **[Student Quickstart](docs/STUDENT_QUICKSTART.md)** - Getting started guide
- **[CIFAR-10 Training Guide](docs/cifar10-training-guide.md)** - Detailed training walkthrough

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

📊 **View Leaderboard**: [Coming Spring 2025] - Be among the first to set baselines!

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

**Start Small. Go Deep. Build ML Systems.**