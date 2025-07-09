# Tiny🔥Torch: Build a Machine Learning System from Scratch

TinyTorch is a pedagogical project designed to accompany the *Machine Learning Systems* textbook. Inspired by OS and compiler courses where students build entire systems from first principles, TinyTorch guides you through building a complete ML training and inference runtime — from autograd to data pipelines, optimizers to profilers — **entirely from scratch**.

This is not a PyTorch tutorial. In TinyTorch, you'll **write the components that frameworks like PyTorch are built on.**

---

## 🎯 What You'll Build

By the end of this project, you'll have implemented a fully functional ML system capable of:

- **Training neural networks** (MLPs, CNNs) on real datasets (MNIST, CIFAR-10)
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

TinyTorch aligns with **Chapters 1–13** of the *Machine Learning Systems* textbook. Each project builds progressively toward a complete ML infrastructure.

### 📚 Course Details & Learning Objectives

Each project corresponds to specific chapters in the *Machine Learning Systems* textbook:

| Project | Chapters | Core Learning | Key Deliverable |
|---------|----------|---------------|-----------------|
| Setup | - | Environment setup, tool familiarity | Working dev environment + CLI |
| Tensor | 1-2 | Tensor operations, memory management | Working Tensor class with basic ops |
| MLP | 3 | Forward/backward pass, gradient computation | Train simple MLP on MNIST |
| CNN | 4 | Convolution, pooling operations | Conv2D and MaxPool implementations |
| Autograd | 5 | Computational graphs, autodiff | Complete autograd engine |
| Data | 6 | Efficient data loading, batching | Custom DataLoader with transformations |
| Training | 7-8 | Optimization algorithms, metrics | SGD, Adam optimizers + training harness |
| Config | 9 | Experiment management, logging | YAML configs + structured logging |
| Profiling | 10 | Performance measurement, debugging | Memory/compute profiler with visualizations |
| Compression | 11 | Pruning, quantization techniques | Compress model while maintaining accuracy |
| Kernels | 12 | Low-level optimization, vectorization | Optimized matrix multiplication kernels |
| Benchmarking | 13 | Performance testing, comparison | Comprehensive benchmarking suite |
| MLOps | 14 | Data drift detection, continuous updates | Production monitoring and auto-retraining system |

### Milestone Targets
- **Week 1**: Environment setup (`setup`) and basic command familiarity
- **Week 3**: Core tensor operations (`tensor`) working
- **Week 5**: Train MLP on MNIST (`mlp`) achieving >95% accuracy  
- **Week 8**: Train CNN on CIFAR-10 (`cnn`) achieving >80% accuracy
- **Week 10**: Complete autograd engine (`autograd`) with gradient checking
- **Week 12**: Optimized system with profiling tools (`profiling`)
- **Final**: Complete system with MLOps monitoring (`mlops`)

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
├── projects/                      # Component-specific projects
│   ├── setup/                    # Environment setup & onboarding
│   ├── tensor/                   # Core tensor implementation
│   ├── mlp/                      # Multi-layer perceptron
│   ├── cnn/                      # Convolutional neural networks
│   ├── config/                   # Configuration system
│   ├── data/                     # Data pipeline & loading
│   ├── autograd/                 # Automatic differentiation
│   ├── training/                 # Training loop & optimization
│   ├── profiling/                # Performance profiling tools
│   ├── compression/              # Model compression techniques
│   ├── kernels/                  # Custom compute kernels
│   ├── benchmarking/             # Performance benchmarking
│   └── mlops/                    # MLOps & production monitoring
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

**New to TinyTorch?** Start here: [`projects/setup/README.md`](projects/setup/README.md)

### 📋 Project Sequence
Each project builds on the previous ones. Click the links to jump to specific instructions:

| Order | Project | Status | Description | Instructions |
|-------|---------|--------|-------------|--------------|
| 0 | **Setup** | 🚀 **START HERE** | Environment & CLI setup | [`projects/setup/README.md`](projects/setup/README.md) |
| 1 | **Tensor** | ⏳ Coming Next | Core tensor operations | [`projects/tensor/README.md`](projects/tensor/README.md) |
| 2 | **MLP** | ⏳ Future | Multi-layer perceptron | [`projects/mlp/README.md`](projects/mlp/README.md) |
| 3 | **CNN** | ⏳ Future | Convolutional networks | [`projects/cnn/README.md`](projects/cnn/README.md) |
| 4 | **Autograd** | ⏳ Future | Automatic differentiation | [`projects/autograd/README.md`](projects/autograd/README.md) |
| 5 | **Data** | ⏳ Future | Data loading pipeline | [`projects/data/README.md`](projects/data/README.md) |
| 6 | **Training** | ⏳ Future | Training loop & optimization | [`projects/training/README.md`](projects/training/README.md) |
| 7 | **Config** | ⏳ Future | Configuration system | [`projects/config/README.md`](projects/config/README.md) |
| 8 | **Profiling** | ⏳ Future | Performance profiling | [`projects/profiling/README.md`](projects/profiling/README.md) |
| 9 | **Compression** | ⏳ Future | Model compression | [`projects/compression/README.md`](projects/compression/README.md) |
| 10 | **Kernels** | ⏳ Future | Custom compute kernels | [`projects/kernels/README.md`](projects/kernels/README.md) |
| 11 | **Benchmarking** | ⏳ Future | Performance benchmarking | [`projects/benchmarking/README.md`](projects/benchmarking/README.md) |
| 12 | **MLOps** | ⏳ Future | Production monitoring | [`projects/mlops/README.md`](projects/mlops/README.md) |

### 🚀 Quick Start Guide
**First time?** Follow this exact sequence:

1. **📖 Read the overview** (you're here!)
2. **🎯 Detailed guidance**: [`COURSE_GUIDE.md`](COURSE_GUIDE.md) (comprehensive walkthrough)
3. **🔧 Environment setup**: [`projects/setup/README.md`](projects/setup/README.md)
4. **✅ Verify setup**: Run `python3 projects/setup/check_setup.py`
5. **🎯 Start Project 1**: [`projects/tensor/README.md`](projects/tensor/README.md)

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

# Automated setup (creates virtual environment + installs dependencies)
python3 projects/setup/create_env.py

# Activate environment (do this every time you work)
source tinytorch-env/bin/activate  # macOS/Linux
# OR: tinytorch-env\Scripts\activate  # Windows

# Verify setup
python3 projects/setup/check_setup.py

# Check system status
python3 bin/tito.py info --show-architecture
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
# Start with environment setup (IMPORTANT!)
python3 projects/setup/create_env.py
source tinytorch-env/bin/activate  # Always activate first!

# Start with Project 0: Setup  
cd projects/setup/
cat README.md  # Read instructions
python3 -m pytest test_setup.py -v  # Run tests

# Then move through the sequence
cd ../tensor/           # Project 1: Core tensors
cd ../mlp/              # Project 2: Multi-layer perceptron  
cd ../autograd/         # Project 3: Automatic differentiation

# Always run tests before submitting
python3 -m pytest projects/tensor/test_tensor.py -v
python3 bin/tito.py submit --project tensor
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

- **Textbook**: *Machine Learning Systems* (Chapters 1-13)
- **Course Website**: [Course URL]
- **Video Lectures**: Complementary video content for each chapter
- **External Reading**: Curated list of papers and blog posts
- **Community Forum**: Discussion and Q&A platform
- **Office Hours**: [Schedule and locations]
