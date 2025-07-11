# 🔥 TinyTorch Project Guide

**Building Machine Learning Systems from Scratch**

This guide helps you navigate through the complete TinyTorch course. Each module builds progressively toward a complete ML system using a notebook-first development approach with nbdev.

## 🎯 Module Progress Tracker

Track your progress through the course:

- [ ] **Module 0: Setup** - Environment & CLI setup  
- [ ] **Module 1: Tensor** - Core tensor operations
- [ ] **Module 2: MLP** - Multi-layer perceptron  
- [ ] **Module 3: CNN** - Convolutional networks
- [ ] **Module 4: Autograd** - Automatic differentiation
- [ ] **Module 5: Data** - Data loading pipeline
- [ ] **Module 6: Training** - Training loop & optimization
- [ ] **Module 7: Config** - Configuration system
- [ ] **Module 8: Profiling** - Performance profiling
- [ ] **Module 9: Compression** - Model compression
- [ ] **Module 10: Kernels** - Custom compute kernels
- [ ] **Module 11: Benchmarking** - Performance benchmarking
- [ ] **Module 12: MLOps** - Production monitoring

## 🚀 Getting Started

### First Time Setup
1. **Clone the repository**
2. **Go to**: [`modules/setup/README.md`](modules/setup/README.md)
3. **Follow all setup instructions**
4. **Verify with**: `python modules/setup/check_setup.py`

### Daily Workflow
```bash
cd TinyTorch
source .venv/bin/activate  # Always activate first!
python bin/tito.py info            # Check system status
```

## 📋 Module Development Workflow

Each module follows this pattern:
1. **Read overview**: `modules/[name]/README.md`
2. **Work in notebook**: `modules/[name]/[name]_dev.ipynb`
3. **Export code**: `python bin/tito.py sync`
4. **Run tests**: `python bin/tito.py test --module [name]`
5. **Move to next module when tests pass**

## 📚 Module Details

### 🔧 Module 0: Setup
**Goal**: Get your development environment ready
**Time**: 30 minutes
**Location**: [`modules/setup/`](modules/setup/)

**Key Tasks**:
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Implement `hello_tinytorch()` function
- [ ] Pass all setup tests
- [ ] Learn the `tito` CLI

**Verification**:
```bash
python modules/setup/check_setup.py  # Should show all ✅
python bin/tito.py test --module setup
```

---

### 🔢 Module 1: Tensor
**Goal**: Build the core tensor system
**Prerequisites**: Module 0 complete
**Location**: [`modules/tensor/`](modules/tensor/)

**Key Tasks**:
- [ ] Implement `Tensor` class in notebook
- [ ] Basic operations (add, mul, reshape)
- [ ] Memory management
- [ ] Shape validation
- [ ] Broadcasting support

**Verification**:
```bash
python bin/tito.py test --module tensor
```

---

### 🧠 Module 2: MLP
**Goal**: Build multi-layer perceptron
**Prerequisites**: Module 1 complete
**Location**: [`modules/mlp/`](modules/mlp/)

**Key Tasks**:
- [ ] Implement `Linear` layer
- [ ] Activation functions (ReLU, Sigmoid)
- [ ] Forward pass
- [ ] Basic backward pass
- [ ] Train on MNIST

**Target**: >95% accuracy on MNIST

---

### 🖼️ Module 3: CNN
**Goal**: Build convolutional neural networks
**Prerequisites**: Module 2 complete
**Location**: [`modules/cnn/`](modules/cnn/)

**Key Tasks**:
- [ ] Implement `Conv2d` layer
- [ ] `MaxPool2d` layer
- [ ] Padding and stride support
- [ ] Train CNN on CIFAR-10

**Target**: >80% accuracy on CIFAR-10

---

### ⚡ Module 4: Autograd
**Goal**: Automatic differentiation engine
**Prerequisites**: Module 3 complete
**Location**: [`modules/autograd/`](modules/autograd/)

**Key Tasks**:
- [ ] Computational graph construction
- [ ] Backward pass automation
- [ ] Gradient checking
- [ ] Memory efficient gradients

**Verification**: All gradient checks pass

---

### 📊 Module 5: Data
**Goal**: Efficient data loading
**Prerequisites**: Module 4 complete
**Location**: [`modules/data/`](modules/data/)

**Key Tasks**:
- [ ] Custom `DataLoader` implementation
- [ ] Batch processing
- [ ] Data transformations
- [ ] Multi-threaded loading

---

### 🎯 Module 6: Training
**Goal**: Complete training system
**Prerequisites**: Module 5 complete
**Location**: [`modules/training/`](modules/training/)

**Key Tasks**:
- [ ] Training loop implementation
- [ ] SGD optimizer
- [ ] Adam optimizer
- [ ] Learning rate scheduling
- [ ] Metric tracking

---

### ⚙️ Module 7: Config
**Goal**: Configuration management
**Prerequisites**: Module 6 complete
**Location**: [`modules/config/`](modules/config/)

**Key Tasks**:
- [ ] YAML configuration system
- [ ] Experiment logging
- [ ] Reproducible training
- [ ] Hyperparameter management

---

### 📊 Module 8: Profiling
**Goal**: Performance measurement
**Prerequisites**: Module 7 complete
**Location**: [`modules/profiling/`](modules/profiling/)

**Key Tasks**:
- [ ] Memory profiler
- [ ] Compute profiler
- [ ] Bottleneck identification
- [ ] Performance visualizations

---

### 🗜️ Module 9: Compression
**Goal**: Model compression techniques
**Prerequisites**: Module 8 complete
**Location**: [`modules/compression/`](modules/compression/)

**Key Tasks**:
- [ ] Pruning implementation
- [ ] Quantization
- [ ] Knowledge distillation
- [ ] Compression benchmarks

---

### 🔥 Module 10: Kernels
**Goal**: Custom compute kernels
**Prerequisites**: Module 9 complete
**Location**: [`modules/kernels/`](modules/kernels/)

**Key Tasks**:
- [ ] Optimized matrix multiplication
- [ ] Vectorized operations
- [ ] CPU optimization

---

### 📈 Module 11: Benchmarking
**Goal**: Performance benchmarking system
**Prerequisites**: Module 10 complete
**Location**: [`modules/benchmarking/`](modules/benchmarking/)

**Key Tasks**:
- [ ] Benchmark suite implementation
- [ ] Performance regression testing
- [ ] Comparative analysis
- [ ] Automated reporting

---

### 🚀 Module 12: MLOps
**Goal**: Production monitoring and deployment
**Prerequisites**: Module 11 complete
**Location**: [`modules/mlops/`](modules/mlops/)

**Key Tasks**:
- [ ] Model monitoring
- [ ] Production deployment
- [ ] A/B testing framework
- [ ] Performance dashboards

## 🧪 Testing Strategy

### Module-Level Testing
```bash
# Test specific module
python bin/tito.py test --module tensor

# Test all modules
python bin/tito.py test

# Check overall status
python bin/tito.py info
```

### Integration Testing
Each module integrates into the main `tinytorch` package through nbdev:
- Notebooks automatically export to `tinytorch/core/`
- Integration tests verify cross-module compatibility
- Final package assembly validates the complete system

## 🏆 Completion Criteria

A module is complete when:
- [ ] All notebook cells run without errors
- [ ] `tito sync` exports code successfully
- [ ] `tito test --module [name]` passes all tests
- [ ] Integration with previous modules works
- [ ] Ready to proceed to next module

## 🎓 Learning Philosophy

This course teaches **systems thinking** by building one cohesive ML system rather than isolated components. Each module contributes essential functionality that later modules depend upon, creating a complete, production-ready machine learning framework.

## 💡 Need Help?

- **Quick Start**: See [`modules/setup/QUICKSTART.md`](modules/setup/QUICKSTART.md)
- **Development Workflow**: Each module's README.md
- **CLI Reference**: `python bin/tito.py --help`
- **Integration Issues**: Check `python bin/tito.py info` 