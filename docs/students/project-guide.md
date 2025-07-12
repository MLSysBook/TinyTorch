# 🔥 TinyTorch Project Guide

**Building Machine Learning Systems from Scratch**

This guide helps you navigate through the complete TinyTorch course. Each module builds progressively toward a complete ML system using a notebook-first development approach with nbdev.

## 🎯 Module Progress Tracker

Track your progress through the course:

- [ ] **Module 0: Setup** - Environment & CLI setup  
- [ ] **Module 1: Tensor** - Core tensor operations
- [ ] **Module 2: Layers** - Neural network layers
- [ ] **Module 3: Networks** - Complete model architectures
- [ ] **Module 4: Autograd** - Automatic differentiation
- [ ] **Module 5: DataLoader** - Data loading pipeline
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
2. **Go to**: [`modules/setup/README.md`](../../modules/setup/README.md)
3. **Follow all setup instructions**
4. **Verify with**: `tito system doctor`

### Daily Workflow
```bash
cd TinyTorch
source .venv/bin/activate  # Always activate first!
tito system info            # Check system status
```

## 📋 Module Development Workflow

Each module follows this pattern:
1. **Read overview**: `modules/[name]/README.md`
2. **Work in Python file**: `modules/[name]/[name]_dev.py`
3. **Export code**: `tito package sync`
4. **Run tests**: `tito module test --module [name]`
5. **Move to next module when tests pass**

## 📚 Module Details

### 🔧 Module 0: Setup
**Goal**: Get your development environment ready
**Time**: 30 minutes
**Location**: [`modules/setup/`](../../modules/setup/)

**Key Tasks**:
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Implement `hello_tinytorch()` function
- [ ] Pass all setup tests
- [ ] Learn the `tito` CLI

**Verification**:
```bash
tito system doctor           # Should show all ✅
tito module test --module setup
```

---

### 🔢 Module 1: Tensor
**Goal**: Build the core tensor system
**Prerequisites**: Module 0 complete
**Location**: [`modules/tensor/`](../../modules/tensor/)

**Key Tasks**:
- [ ] Implement `Tensor` class
- [ ] Basic operations (add, mul, reshape)
- [ ] Memory management
- [ ] Shape validation
- [ ] Broadcasting support

**Verification**:
```bash
tito module test --module tensor
```

---

### 🧠 Module 2: Layers
**Goal**: Build neural network layers
**Prerequisites**: Module 1 complete
**Location**: [`modules/layers/`](../../modules/layers/)

**Key Tasks**:
- [ ] Implement `Linear` layer
- [ ] Activation functions (ReLU, Sigmoid)
- [ ] Forward pass implementation
- [ ] Parameter management
- [ ] Layer composition

**Verification**:
```bash
tito module test --module layers
```

---

### 🖼️ Module 3: Networks
**Goal**: Build complete neural networks
**Prerequisites**: Module 2 complete
**Location**: [`modules/networks/`](../../modules/networks/)

**Key Tasks**:
- [ ] Implement `Sequential` container
- [ ] CNN architectures
- [ ] Model saving/loading
- [ ] Train on CIFAR-10

**Target**: >80% accuracy on CIFAR-10

---

### ⚡ Module 4: Autograd
**Goal**: Automatic differentiation engine
**Prerequisites**: Module 3 complete
**Location**: [`modules/autograd/`](../../modules/autograd/)

**Key Tasks**:
- [ ] Computational graph construction
- [ ] Backward pass automation
- [ ] Gradient checking
- [ ] Memory efficient gradients

**Verification**: All gradient checks pass

---

### 📊 Module 5: DataLoader
**Goal**: Efficient data loading
**Prerequisites**: Module 4 complete
**Location**: [`modules/dataloader/`](../../modules/dataloader/)

**Key Tasks**:
- [ ] Custom `DataLoader` implementation
- [ ] Batch processing
- [ ] Data transformations
- [ ] Multi-threaded loading

---

### 🎯 Module 6: Training
**Goal**: Complete training system
**Prerequisites**: Module 5 complete
**Location**: [`modules/training/`](../../modules/training/)

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
**Location**: [`modules/config/`](../../modules/config/)

**Key Tasks**:
- [ ] YAML configuration system
- [ ] Experiment logging
- [ ] Reproducible training
- [ ] Hyperparameter management

---

### 📊 Module 8: Profiling
**Goal**: Performance measurement
**Prerequisites**: Module 7 complete
**Location**: [`modules/profiling/`](../../modules/profiling/)

**Key Tasks**:
- [ ] Memory profiler
- [ ] Compute profiler
- [ ] Bottleneck identification
- [ ] Performance visualizations

---

### 🗜️ Module 9: Compression
**Goal**: Model compression techniques
**Prerequisites**: Module 8 complete
**Location**: [`modules/compression/`](../../modules/compression/)

**Key Tasks**:
- [ ] Pruning implementation
- [ ] Quantization
- [ ] Knowledge distillation
- [ ] Compression benchmarks

---

### ⚡ Module 10: Kernels
**Goal**: Custom compute kernels
**Prerequisites**: Module 9 complete
**Location**: [`modules/kernels/`](../../modules/kernels/)

**Key Tasks**:
- [ ] CUDA kernel implementation
- [ ] Performance optimization
- [ ] Memory coalescing
- [ ] Kernel benchmarking

---

### 📈 Module 11: Benchmarking
**Goal**: Performance benchmarking
**Prerequisites**: Module 10 complete
**Location**: [`modules/benchmarking/`](../../modules/benchmarking/)

**Key Tasks**:
- [ ] Benchmarking framework
- [ ] Performance comparisons
- [ ] Scaling analysis
- [ ] Optimization recommendations

---

### 🚀 Module 12: MLOps
**Goal**: Production monitoring
**Prerequisites**: Module 11 complete
**Location**: [`modules/mlops/`](../../modules/mlops/)

**Key Tasks**:
- [ ] Model monitoring
- [ ] Performance tracking
- [ ] Alert systems
- [ ] Production deployment

## 🛠️ Essential Commands

### **System Commands**
```bash
tito system info              # System information and course navigation
tito system doctor            # Environment diagnosis
tito system jupyter           # Start Jupyter Lab
```

### **Module Development**
```bash
tito module status            # Check all module status
tito module test --module X   # Test specific module
tito module test --all        # Test all modules
tito module notebooks --module X  # Convert Python to notebook
```

### **Package Management**
```bash
tito package sync            # Export all notebooks to package
tito package sync --module X # Export specific module
tito package reset           # Reset package to clean state
```

## 🎯 **Success Criteria**

Each module is complete when:
- [ ] **All tests pass**: `tito module test --module [name]`
- [ ] **Code exports**: `tito package sync --module [name]`
- [ ] **Understanding verified**: Can explain key concepts and trade-offs
- [ ] **Ready for next**: Prerequisites met for following modules

## 🆘 **Getting Help**

### **Troubleshooting**
- **Environment Issues**: `tito system doctor`
- **Module Status**: `tito module status --details`
- **Integration Issues**: Check `tito system info`

### **Resources**
- **Course Overview**: [Main README](../../README.md)
- **Development Guide**: [Module Development](../development/module-development-guide.md)
- **Quick Reference**: [Commands and Patterns](../development/quick-module-reference.md)

---

**💡 Pro Tip**: Use `tito module status` regularly to track your progress and see which modules are ready to work on next! 