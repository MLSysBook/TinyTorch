# Tiny🔥Torch: Build a Machine Learning System from Scratch

TinyTorch is a pedagogical project designed to accompany the [*Machine Learning Systems*](https://mlsysbook.ai) textbook. Inspired by OS and compiler courses where students build entire systems from first principles, TinyTorch guides you through building a complete ML training and inference runtime — from autograd to data pipelines, optimizers to profilers — **entirely from scratch**.

This is not a PyTorch tutorial. In TinyTorch, you'll **write the components that frameworks like PyTorch are built on.**

---

## 🎯 What You'll Build

By the end of this project, you'll have implemented a fully functional ML system capable of:

- **Training neural networks** (MLPs, CNNs) on real datasets 10)
- **Automatic differentiation** with a custom autograd engine
- **Memory-efficient data loading** with custom DataLoader implementations
- **Multiple optimization algorithms** (SGD, Adam, RMSprop)
- **Performance profiling** and bottleneck identification
- **Model compression** through pruning and quantization
- **Custom compute kernels** for matrix operations
- **Production monitoring** with MLOps infrastructure
- **Reproducible experiments** with checkpointing and logging

**End Goal**: Train a CNN on CIFAR-10 achieving >85% accuracy using only your implementation.

---

## 🧠 Project Goals & Learning Objectives

### Core Learning Objectives
- **Systems Understanding**: Learn how modern ML systems are constructed, not just how to use them
- **Full-Stack ML Infrastructure**: Build core components from tensor operations to training orchestration
- **Performance Engineering**: Understand computational and memory bottlenecks in ML workloads
- **Software Architecture**: Design modular, extensible systems with clean abstractions
- **Infrastructure Thinking**: Make design decisions that impact performance, reproducibility, and maintainability

### Technical Skills Gained
- **Low-level ML Implementation**: Tensor operations, gradient computation, optimization algorithms
- **Memory Management**: Efficient data structures, gradient accumulation, batch processing
- **Performance Optimization**: Profiling, kernel optimization, memory access patterns
- **System Design**: Modular architecture, clean APIs, extensible frameworks
- **Testing & Validation**: Numerical stability, gradient checking, performance regression testing

---

## 🏗️ System Architecture

### Core Components Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TinyTorch System                         │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface (bin/tito.py)                               │
├─────────────────────────────────────────────────────────────┤
│  Training Orchestration (trainer.py)                       │
├─────────────────────────────────────────────────────────────┤
│  Model Definition     │  Data Pipeline    │  Optimization   │
│  (modules.py)         │  (dataloader.py)  │  (optimizer.py) │
├─────────────────────────────────────────────────────────────┤
│  Automatic Differentiation Engine (autograd)               │
├─────────────────────────────────────────────────────────────┤
│  Tensor Operations & Storage (tensor.py)                   │
├─────────────────────────────────────────────────────────────┤
│  Profiling & MLOps (profiler.py, mlops.py)                 │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles
1. **Modularity**: Each component has a single responsibility and clean interfaces
2. **Composability**: Components can be mixed and matched for different use cases
3. **Performance**: Designed for efficiency without sacrificing readability
4. **Extensibility**: Easy to add new layers, optimizers, and functionality
5. **Debuggability**: Built-in profiling and logging for understanding system behavior

---

## 📚 Curriculum Integration & Roadmap

TinyTorch aligns with **Chapters 1–13** of the [*Machine Learning Systems*](https://mlsysbook.ai) textbook. Each project builds progressively toward a complete ML infrastructure.

### 📚 Course Details & Learning Objectives

Each project is inspired by key themes from the [*Machine Learning Systems*](https://mlsysbook.ai) textbook:

### 📖 Part I: The Essentials (Chapters 1-4)
*Core principles, components, and architectures*

| Project | Chapter | Core Learning | Key Deliverable |
|---------|---------|---------------|-----------------|
| Setup | 1 (Introduction) | Environment setup, tool familiarity | Working dev environment + CLI |
| Tensor | 2 (ML Systems) | Basic tensor operations, NumPy-style | Simple Tensor class with math ops |
| MLP | 3 (DL Primer) | Forward pass, manual backprop | Train MLP on MNIST (manual gradients) |
| CNN | 4 (DNN Architectures) | Convolution concepts, architectures | Basic conv implementation |

### 🏗️ Part II: Engineering Principles (Chapters 5-13)
*Workflows, data engineering, optimization strategies, and operational challenges*

| Project | Chapter | Core Learning | Key Deliverable |
|---------|---------|---------------|-----------------|
| Data | 6 (Data Engineering) | Efficient data loading, batching | Custom DataLoader with transformations |
| Training | 8 (AI Training) | **Autograd engine**, optimization algorithms | Complete training system with autodiff |
| Profiling | 9 (Efficient AI) | Performance measurement, debugging | Memory/compute profiler with visualizations |
| Compression | 10 (Model Optimizations) | Pruning, quantization techniques | Compress model while maintaining accuracy |
| Kernels | 11 (AI Acceleration) | Low-level optimization, vectorization | Optimized matrix multiplication kernels |
| Benchmarking | 12 (Benchmarking AI) | Performance testing, comparison | Comprehensive benchmarking suite |
| MLOps | 13 (ML Operations) | Production monitoring, deployment | MLOps pipeline with drift detection |

**Note**: Chapters 5 (AI Workflow) and 7 (AI Frameworks) provide conceptual frameworks that inform the systems projects. Part III (AI Best Practice, Chapters 14-18) and Part IV (Closing Perspectives, Chapters 19-20) focus on deployment considerations and emerging trends covered through readings and discussions.

### Milestone Targets

- **Week 1**: Environment setup (`setup`) and basic command familiarity
- **Week 3**: Basic tensor operations (`tensor`) working
- **Week 5**: Train MLP on MNIST (`mlp`) with manual backprop achieving >90% accuracy  
- **Week 7**: Train CNN on CIFAR-10 (`cnn`) basic implementation achieving >70% accuracy
- **Week 9**: Data pipeline (`data`) operational with efficient loading
- **Week 11**: Complete autograd engine and training framework (`training`) working
- **Week 13**: Optimized system with profiling tools (`profiling`)
- **Final**: Complete production system with MLOps monitoring (`mlops`)

---

## 📦 Course Repository Structure

```
TinyTorch/
├── bin/                           # Command-line interfaces
│   └── tito.py                    # Main TinyTorch CLI (tito)
├── tinytorch/                     # Core ML system package
│   ├── core/                      # Core ML components
│   │   ├── __init__.py
│   │   ├── tensor.py              # Tensor class with autograd support
│   │   ├── autograd.py            # Automatic differentiation engine
│   │   ├── modules.py             # Neural network layers and models
│   │   ├── functional.py          # Core operations (conv2d, relu, etc.)
│   │   ├── dataloader.py          # Data loading and preprocessing
│   │   ├── optimizer.py           # Optimization algorithms
│   │   ├── trainer.py             # Training loop orchestration
│   │   ├── profiler.py            # Performance measurement tools
│   │   ├── benchmark.py           # Benchmarking and evaluation
│   │   ├── mlops.py               # MLOps and production monitoring
│   │   └── utils.py               # Utility functions
│   ├── configs/                   # Configuration files
│   │   ├── default.yaml           # Default training configuration
│   │   ├── models/                # Model-specific configs
│   │   └── datasets/              # Dataset-specific configs
│   └── datasets/                  # Dataset implementations
│       ├── __init__.py
│       ├── mnist.py
│       ├── cifar10.py
│       └── transforms.py
├── modules/                      # 🧩 System Modules
│   ├── 01_setup/                # Environment setup & onboarding
│   ├── 02_tensor/               # Basic tensor implementation
│   ├── 03_mlp/                  # Multi-layer perceptron (manual backprop)
│   ├── 04_cnn/                  # Convolutional neural networks (basic)
│   ├── 05_data/                 # Data pipeline & loading
│   ├── 06_training/             # Autograd engine & training optimization
│   ├── 07_profiling/            # Performance profiling tools
│   ├── 08_compression/          # Model compression techniques
│   ├── 09_kernels/              # Custom compute kernels
│   ├── 10_benchmarking/         # Performance benchmarking
│   └── 11_mlops/                # MLOps & production monitoring
├── docs/                         # Course documentation
│   ├── tutorials/                # Step-by-step tutorials
│   ├── api/                      # API documentation
│   └── lectures/                 # Lecture materials
├── notebooks/                    # Jupyter tutorials and demos
├── examples/                     # Working examples and demos
│   ├── train_mnist_mlp.py
│   ├── train_cifar_cnn.py
│   └── benchmark_ops.py
├── tests/                        # Comprehensive test suite
│   ├── test_tensor.py
│   ├── test_autograd.py
│   ├── test_modules.py
│   └── test_training.py
├── grading/                      # Course grading materials
│   ├── rubrics/                  # Assignment rubrics
│   ├── autograders/              # Automated grading scripts
│   └── solutions/               # Reference solutions
├── resources/                    # Course resources
│   ├── datasets/                 # Course datasets
│   ├── pretrained/              # Pre-trained models
│   └── references/              # Reference materials
├── logs/                         # Training logs and artifacts
│   └── runs/
├── checkpoints/                  # Model checkpoints
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎯 Course Navigation & Getting Started

**New to TinyTorch?** Start here: [`modules/01_setup/README.md`](modules/01_setup/README.md)

### 📋 Module Sequence
Each module builds on the previous ones. Click the links to jump to specific instructions:

| Order | Module | Status | Description | Instructions |
|-------|--------|--------|-------------|--------------|
| 0 | **Setup** | 🚀 **START HERE** | Environment & CLI setup | [`modules/01_setup/README.md`](modules/01_setup/README.md) |
| 1 | **Tensor** | ⏳ Coming Next | Basic tensor operations | [`modules/02_tensor/README.md`](modules/02_tensor/README.md) |
| 2 | **MLP** | ⏳ Future | Multi-layer perceptron (manual backprop) | [`modules/03_mlp/README.md`](modules/03_mlp/README.md) |
| 3 | **CNN** | ⏳ Future | Convolutional networks (basic) | [`modules/04_cnn/README.md`](modules/04_cnn/README.md) |
| 4 | **Data** | ⏳ Future | Data loading pipeline | [`modules/05_data/README.md`](modules/05_data/README.md) |
| 5 | **Training** | ⏳ Future | Autograd engine & optimization | [`modules/06_training/README.md`](modules/06_training/README.md) |
| 6 | **Profiling** | ⏳ Future | Performance profiling | [`modules/07_profiling/README.md`](modules/07_profiling/README.md) |
| 7 | **Compression** | ⏳ Future | Model compression | [`modules/08_compression/README.md`](modules/08_compression/README.md) |
| 8 | **Kernels** | ⏳ Future | Custom compute kernels | [`modules/09_kernels/README.md`](modules/09_kernels/README.md) |
| 9 | **Benchmarking** | ⏳ Future | Performance benchmarking | [`modules/10_benchmarking/README.md`](modules/10_benchmarking/README.md) |
| 10 | **MLOps** | ⏳ Future | Production monitoring | [`modules/11_mlops/README.md`](modules/11_mlops/README.md) |

### 🚀 Quick Start Guide
**First time?** Follow this exact sequence:

1. **📖 Read the overview** (you're here!)
2. **🎯 Detailed guidance**: [`COURSE_GUIDE.md`](COURSE_GUIDE.md) (comprehensive walkthrough)
3. **🚀 One-command setup**: `python3 bin/tito.py setup` (sets up everything!)
4. **✅ Run activation**: Follow the command it gives you (usually `bin/activate-tinytorch.sh`)
5. **📖 Start building**: [`projects/setup/README.md`](projects/setup/README.md)

### Prerequisites
- **Python 3.8+** (type hints and modern features required)
- **NumPy** (numerical computations)
- **Optional**: Numba (JIT compilation for performance)
- **Development**: pytest, black, mypy (for testing and code quality)

### Environment Setup
```bash
# Clone and setup environment
git clone <repository-url>
cd TinyTorch

# ONE SMART COMMAND - shows clear commands to copy
python3 bin/tito.py setup
# ↳ First time: Creates environment, installs dependencies, shows activation command
# ↳ Already exists: Shows activation command to copy
# ↳ Already active: Can show deactivation command

# Copy and run the highlighted command, then:
tito info  # Check system status
```

### For Instructors
```bash
# Set up the complete course environment
pip install -r requirements.txt

# Generate course materials
python3 bin/tito.py generate-projects
python3 bin/tito.py setup-autograders

# View course progress
python3 bin/tito.py course-status
```

### For Students
```bash
# ONE SMART COMMAND - handles everything (IMPORTANT!)
python3 bin/tito.py setup
# ↳ Shows clear commands to copy and paste!

# Start with Project 0: Setup  
cd projects/setup/
cat README.md  # Read instructions
python3 -m pytest test_setup.py -v  # Run tests

# Then move through the sequence (Part I: The Essentials)
cd ../tensor/           # Project 1: Basic tensors
cd ../mlp/              # Project 2: Multi-layer perceptron (manual backprop)
cd ../cnn/              # Project 3: Convolutional networks (basic)

# Part II: Engineering Principles  
cd ../data/             # Project 4: Data pipeline
cd ../training/         # Project 5: Autograd engine & training

# Always run tests before submitting
python3 -m pytest projects/tensor/test_tensor.py -v
python3 bin/tito.py submit --project tensor

python3 -m pytest projects/mlp/test_mlp.py -v
python3 bin/tito.py submit --project mlp
```

---

## 🎯 Implementation Guidelines

### Code Quality Standards
- **Type Hints**: All public APIs must have complete type annotations
- **Documentation**: Docstrings for all classes and public methods
- **Testing**: >90% code coverage with unit and integration tests
- **Performance**: Profile-guided optimization with benchmarking
- **Style**: Black code formatting, consistent naming conventions

### API Design Principles
- **Familiar Interface**: Similar to PyTorch where it makes sense (for learning transfer)
- **Explicit Over Implicit**: Clear parameter names and behavior
- **Composable**: Small, focused components that work together
- **Debuggable**: Rich error messages and debugging hooks

### Performance Targets
- **MNIST MLP**: <5 seconds per epoch on modern laptop
- **CIFAR-10 CNN**: <30 seconds per epoch on modern laptop
- **Memory Usage**: <2GB RAM for standard training runs
- **Numerical Stability**: Gradient checking passes for all operations

---

## 🧪 Testing & Validation Strategy

### Test Categories
1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Component interaction and data flow
3. **Numerical Tests**: Gradient checking and mathematical correctness
4. **Performance Tests**: Regression testing for speed and memory
5. **End-to-End Tests**: Complete training runs with known results

### Validation Methodology
- **Gradient Checking**: Numerical verification of all autodiff operations
- **Reference Comparisons**: Output validation against NumPy/PyTorch (where applicable)
- **Convergence Testing**: Training curves must match expected behavior
- **Ablation Studies**: Systematic testing of individual components

---

## 💡 Educational Philosophy

> "You don't really understand a system until you've built it."

### Learning Through Building
TinyTorch emphasizes **active construction** over passive consumption. Students don't just learn about autograd—they implement it. They don't just use optimizers—they write them from scratch.

### Systems Thinking
By building a complete system, students understand:
- **Abstraction Boundaries**: What belongs where in the system hierarchy
- **Performance Trade-offs**: How design decisions impact speed and memory
- **Debugging Strategies**: How to trace problems through complex systems
- **Integration Challenges**: How components interact and depend on each other

### Real-World Relevance
Every component in TinyTorch has a direct analog in production ML systems. The skills learned here transfer directly to understanding and contributing to frameworks like PyTorch, TensorFlow, and JAX.

---

## 🔧 Advanced Features & Extensions

### Chapter 13: MLOps Deep Dive

**Core MLOps Components** (Chapter 13 will implement):
- **Data Drift Detection**: Statistical tests for distribution shifts in input features
- **Model Performance Monitoring**: Track accuracy, latency, and throughput in production  
- **Automatic Retraining Triggers**: When to retrain based on performance degradation
- **A/B Testing Framework**: Compare model versions safely in production
- **Model Registry**: Version control and metadata tracking for deployed models
- **Alert Systems**: Notifications for model failures or performance drops
- **Rollback Mechanisms**: Safe deployment and quick rollback strategies

**Production Integration**:
- **REST API Serving**: Deploy models as web services
- **Batch Inference Pipelines**: Large-scale offline predictions
- **Feature Store Integration**: Consistent feature engineering across training/serving
- **Monitoring Dashboards**: Real-time system health visualization

### Optional Advanced Components
- **Mixed Precision Training**: FP16/FP32 mixed precision implementation
- **Distributed Training**: Multi-GPU and multi-node training support
- **Dynamic Graphs**: Support for variable computation graphs
- **Custom Operators**: Framework for implementing new operations
- **JIT Compilation**: Integration with Numba or custom compilation

### Research Extensions
- **Novel Optimizers**: Implement cutting-edge optimization algorithms
- **Architecture Search**: Automated neural architecture search
- **Compression Techniques**: Advanced pruning and quantization methods
- **Hardware Acceleration**: GPU kernels and specialized hardware support

---

## 📊 Success Metrics

### Technical Milestones
- [ ] Train MLP on MNIST achieving >95% accuracy
- [ ] Implement working autograd engine with gradient checking
- [ ] Train CNN on CIFAR-10 achieving >85% accuracy
- [ ] Profile and optimize for 2x performance improvement
- [ ] Complete all core project implementations

### Learning Outcomes Assessment
- **Code Reviews**: Peer and instructor evaluation of implementations
- **Design Document**: Architecture decisions and trade-off analysis
- **Performance Analysis**: Profiling report and optimization strategy
- **Presentation**: Explain system design and key insights learned

---

## 🤝 Course Management

### For Instructors

**Project Management**:
- Each chapter has structured projects in `projects/XX-name/`
- Rubrics and grading criteria in `grading/rubrics/`
- Automated testing and grading in `grading/autograders/`
- Reference solutions in `grading/solutions/`

**Progress Tracking**:
- Student progress dashboards
- Automated testing and feedback
- Performance benchmarking
- Code quality metrics

### For Students

**Development Process**:
1. **Start Each Project**: Read project description in `projects/XX-name/README.md`
2. **Implement Features**: Follow step-by-step guided implementation
3. **Run Tests**: Use automated tests to validate implementation
4. **Submit Work**: Automated submission and grading system
5. **Get Feedback**: Detailed feedback on implementation and performance

### Getting Help
- **Documentation**: Comprehensive docs in `docs/`
- **Tutorials**: Step-by-step tutorials in `notebooks/`
- **Office Hours**: Regular sessions for questions and debugging
- **Peer Discussion**: Collaborative learning encouraged
- **Issue Tracking**: GitHub issues for bugs and feature requests

---

## 📬 License and Attribution

TinyTorch is part of the *Machine Learning Systems* course and textbook by Vijay Janapa Reddi et al. Inspired by systems-style pedagogical projects like xv6 (OS), PintOS (OS), and cs231n assignments (ML).

**License**: MIT  
**Citation**: Please cite the Machine Learning Systems textbook when using this educational material.

---

## 🔗 Additional Resources

- **Textbook**: [*Machine Learning Systems*](https://mlsysbook.ai) (Chapters 1-13) | [PDF](https://mlsysbook.ai/Machine-Learning-Systems.pdf)
- **Course Website**: Coming soon
- **Video Lectures**: Coming soon
- **External Reading**: Coming soon
- **Community Forum**: [GitHub Discussions](../../discussions)
- **Office Hours**: Coming soon
