# 🔥 TinyTorch Course Guide

**Building Machine Learning Systems from Scratch**

This guide helps you navigate through the complete TinyTorch course. Each project builds progressively toward a complete ML system.

## 🎯 Current Status

Track your progress through the course:

- [ ] **Project 0: Setup** - Environment & CLI setup
- [ ] **Project 1: Tensor** - Core tensor operations  
- [ ] **Project 2: MLP** - Multi-layer perceptron
- [ ] **Project 3: CNN** - Convolutional networks
- [ ] **Project 4: Autograd** - Automatic differentiation
- [ ] **Project 5: Data** - Data loading pipeline
- [ ] **Project 6: Training** - Training loop & optimization
- [ ] **Project 7: Config** - Configuration system
- [ ] **Project 8: Profiling** - Performance profiling
- [ ] **Project 9: Compression** - Model compression
- [ ] **Project 10: Kernels** - Custom compute kernels
- [ ] **Project 11: Benchmarking** - Performance benchmarking
- [ ] **Project 12: MLOps** - Production monitoring

## 🚀 Getting Started

### First Time Setup
1. **Clone the repository**
2. **Go to**: [`projects/setup/README.md`](projects/setup/README.md)
3. **Follow all setup instructions**
4. **Verify with**: `python3 projects/setup/check_setup.py`

### Daily Workflow
```bash
cd TinyTorch
source .venv/bin/activate  # Always activate first!
python3 bin/tito.py info           # Check system status
```

## 📋 Project Details

### 🔧 Project 0: Setup
**Goal**: Get your development environment ready
**Time**: 30 minutes
**Instructions**: [`projects/setup/README.md`](projects/setup/README.md)

**Key Tasks**:
- [ ] Create virtual environment
- [ ] Install dependencies 
- [ ] Implement `hello_tinytorch()` function
- [ ] Pass all setup tests
- [ ] Learn the `tito` CLI

**Verification**:
```bash
python3 projects/setup/check_setup.py  # Should show all ✅
```

---

### 🔢 Project 1: Tensor
**Goal**: Build the core tensor system
**Prerequisites**: Project 0 complete
**Instructions**: [`projects/tensor/README.md`](projects/tensor/README.md)

**Key Tasks**:
- [ ] Implement `Tensor` class
- [ ] Basic operations (add, mul, reshape)
- [ ] Memory management
- [ ] Shape validation
- [ ] Broadcasting support

**Verification**:
```bash
python3 -m pytest projects/tensor/test_tensor.py -v
```

---

### 🧠 Project 2: MLP  
**Goal**: Build multi-layer perceptron
**Prerequisites**: Project 1 complete
**Instructions**: [`projects/mlp/README.md`](projects/mlp/README.md)

**Key Tasks**:
- [ ] Implement `Linear` layer
- [ ] Activation functions (ReLU, Sigmoid)
- [ ] Forward pass
- [ ] Basic backward pass
- [ ] Train on MNIST

**Target**: >95% accuracy on MNIST

---

### 🖼️ Project 3: CNN
**Goal**: Build convolutional neural networks
**Prerequisites**: Project 2 complete  
**Instructions**: [`projects/cnn/README.md`](projects/cnn/README.md)

**Key Tasks**:
- [ ] Implement `Conv2d` layer
- [ ] `MaxPool2d` layer
- [ ] Padding and stride support
- [ ] Train CNN on CIFAR-10

**Target**: >80% accuracy on CIFAR-10

---

### ⚡ Project 4: Autograd
**Goal**: Automatic differentiation engine
**Prerequisites**: Project 3 complete
**Instructions**: [`projects/autograd/README.md`](projects/autograd/README.md)

**Key Tasks**:
- [ ] Computational graph construction
- [ ] Backward pass automation
- [ ] Gradient checking
- [ ] Memory efficient gradients

**Verification**: All gradient checks pass

---

### 📊 Project 5: Data
**Goal**: Efficient data loading
**Prerequisites**: Project 4 complete
**Instructions**: [`projects/data/README.md`](projects/data/README.md)

**Key Tasks**:
- [ ] Custom `DataLoader` implementation
- [ ] Batch processing
- [ ] Data transformations
- [ ] Multi-threaded loading

---

### 🎯 Project 6: Training
**Goal**: Complete training system
**Prerequisites**: Project 5 complete
**Instructions**: [`projects/training/README.md`](projects/training/README.md)

**Key Tasks**:
- [ ] Training loop implementation
- [ ] SGD optimizer
- [ ] Adam optimizer
- [ ] Learning rate scheduling
- [ ] Metric tracking

---

### ⚙️ Project 7: Config
**Goal**: Configuration management
**Prerequisites**: Project 6 complete
**Instructions**: [`projects/config/README.md`](projects/config/README.md)

**Key Tasks**:
- [ ] YAML configuration system
- [ ] Experiment logging
- [ ] Reproducible training
- [ ] Hyperparameter management

---

### 📊 Project 8: Profiling
**Goal**: Performance measurement
**Prerequisites**: Project 7 complete
**Instructions**: [`projects/profiling/README.md`](projects/profiling/README.md)

**Key Tasks**:
- [ ] Memory profiler
- [ ] Compute profiler  
- [ ] Bottleneck identification
- [ ] Performance visualizations

---

### 🗜️ Project 9: Compression
**Goal**: Model compression techniques
**Prerequisites**: Project 8 complete
**Instructions**: [`projects/compression/README.md`](projects/compression/README.md)

**Key Tasks**:
- [ ] Pruning implementation
- [ ] Quantization
- [ ] Knowledge distillation
- [ ] Compression benchmarks

---

### 🔥 Project 10: Kernels
**Goal**: Custom compute kernels
**Prerequisites**: Project 9 complete
**Instructions**: [`projects/kernels/README.md`](projects/kernels/README.md)

**Key Tasks**:
- [ ] Optimized matrix multiplication
- [ ] Vectorized operations
- [ ] CPU optimization
- [ ] Performance comparisons

---

### 📈 Project 11: Benchmarking
**Goal**: Performance evaluation
**Prerequisites**: Project 10 complete
**Instructions**: [`projects/benchmarking/README.md`](projects/benchmarking/README.md)

**Key Tasks**:
- [ ] Comprehensive benchmarking suite
- [ ] Performance regression testing
- [ ] Comparison with other frameworks
- [ ] Performance reporting

---

### 🚀 Project 12: MLOps
**Goal**: Production monitoring
**Prerequisites**: Project 11 complete
**Instructions**: [`projects/mlops/README.md`](projects/mlops/README.md)

**Key Tasks**:
- [ ] Data drift detection
- [ ] Model performance monitoring
- [ ] Auto-retraining system
- [ ] Production deployment

## 🏆 Final Achievement

By the end of the course, you'll have:
- ✅ Complete ML system built from scratch
- ✅ CNN trained on CIFAR-10 achieving >85% accuracy
- ✅ Production-ready MLOps pipeline
- ✅ Deep understanding of ML system internals

## 🆘 Getting Help

**Stuck on a project?**
1. Read the project's README thoroughly
2. Run the verification commands
3. Check the troubleshooting section
4. Ask in office hours
5. Review previous projects for patterns

**Common Commands**:
```bash
python3 bin/tito.py info                    # System status
python3 bin/tito.py test --project setup    # Test specific project
python3 projects/setup/check_setup.py       # Comprehensive verification
```

---

**Ready to start?** Go to [`projects/setup/README.md`](projects/setup/README.md) 🚀 