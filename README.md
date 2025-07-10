# 🔥 TinyTorch: Build ML Systems from Scratch

> A hands-on systems course where you implement every component of a modern ML system

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![nbdev](https://img.shields.io/badge/built%20with-nbdev-orange.svg)](https://nbdev.fast.ai/)

TinyTorch is a comprehensive machine learning systems course where you'll build everything from tensors to production monitoring systems. Using a **module-first** approach, students work in self-contained modules while building a complete ML framework.

## 🎯 What You'll Build

By the end of this course, you will have implemented:

- ✅ **Core tensor operations** with automatic differentiation
- ✅ **Neural network layers** (Linear, CNN, RNN, Transformer)
- ✅ **Training algorithms** (SGD, Adam, distributed training)
- ✅ **Data pipelines** with efficient loading and preprocessing
- ✅ **Model compression** (pruning, quantization, distillation)
- ✅ **Performance optimization** (profiling, kernel fusion)
- ✅ **Production systems** (deployment, monitoring, MLOps)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/tinytorch/TinyTorch.git
cd TinyTorch

# Setup development environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Start with Setup Module

```bash
# Navigate to the setup module
cd modules/setup/

# Read the module overview
cat README.md

# Open the development notebook
jupyter lab setup.ipynb
```

### 3. Development Workflow

1. **Work in module notebooks** (`modules/[module]/[module].ipynb`)
2. **Mark code for export** with `#| export` directives
3. **Export to package** with `python bin/tito.py sync`
4. **Test your code** with `python bin/tito.py test --module [module]`
5. **Move to next module** when tests pass

## 📚 Course Structure

TinyTorch follows a progressive module structure. Each module builds on the previous ones:

| Module | Location | Topic | Exports To |
|--------|----------|-------|-----------|
| setup | `modules/setup/` | Environment & Hello World | `tinytorch.core.utils` |
| tensor | `modules/tensor/` | Core Tensor Implementation | `tinytorch.core.tensor` |
| autograd | `modules/autograd/` | Automatic Differentiation | `tinytorch.core.autograd` |
| mlp | `modules/mlp/` | Neural Network Layers | `tinytorch.core.modules` |
| cnn | `modules/cnn/` | Convolutional Networks | `tinytorch.models.cnn` |
| training | `modules/training/` | Training Loops | `tinytorch.training` |
| data | `modules/data/` | Data Loading Pipeline | `tinytorch.data` |
| kernels | `modules/kernels/` | Custom CUDA Kernels | `tinytorch.kernels` |
| compression | `modules/compression/` | Model Compression | `tinytorch.compression` |
| profiling | `modules/profiling/` | Performance Profiling | `tinytorch.profiling` |
| benchmarking | `modules/benchmarking/` | Performance Benchmarks | `tinytorch.benchmarking` |
| config | `modules/config/` | Configuration Management | `tinytorch.config` |
| mlops | `modules/mlops/` | Production Monitoring | `tinytorch.mlops` |

## 🔧 Key Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `python bin/tito.py sync` | Export notebook code to package | Export all modules |
| `python bin/tito.py test --module [name]` | Test specific module | Test tensor module |
| `python bin/tito.py test --all` | Run all tests | Test everything |
| `python bin/tito.py info` | Check implementation status | Show progress |
| `jupyter lab [module].ipynb` | Start module development | Open tensor notebook |

## 📦 Package Structure

The final TinyTorch package structure (auto-generated from modules):

```
tinytorch/                  # Auto-generated from modules/
├── __init__.py            # Main package
├── core/                  # Core ML components
│   ├── tensor.py         # From modules/tensor/
│   ├── autograd.py       # From modules/autograd/
│   ├── modules.py        # From modules/mlp/
│   └── utils.py          # From modules/setup/
├── data/                 # From modules/data/
├── training/             # From modules/training/
├── models/               # Model architectures
│   └── cnn.py           # From modules/cnn/
├── kernels/              # From modules/kernels/
├── compression/          # From modules/compression/
├── profiling/            # From modules/profiling/
├── benchmarking/         # From modules/benchmarking/
├── config/               # From modules/config/
└── mlops/                # From modules/mlops/
```

## 🎓 Learning Approach

### Module-First Development

TinyTorch uses a **module-first** approach where each module is self-contained:

- ✅ **Self-contained**: Each module has its own notebook, tests, and documentation
- ✅ **Progressive**: Modules build on each other in a logical sequence
- ✅ **Interactive**: Work in Jupyter notebooks with immediate feedback
- ✅ **Tested**: Comprehensive tests verify your implementation
- ✅ **Integrated**: nbdev automatically exports to the main package

### Development Workflow per Module

```bash
# 1. Navigate to module
cd modules/[module-name]/

# 2. Read the overview
cat README.md

# 3. Open development notebook
jupyter lab [module-name].ipynb

# 4. Implement functions with #| export
# 5. Test interactively in notebook

# 6. Export to package
python bin/tito.py sync

# 7. Run automated tests
python bin/tito.py test --module [module-name]

# 8. Move to next module when tests pass
```

### Progressive Complexity

1. **Foundation** (setup, tensor): Basic building blocks
2. **Core ML** (autograd, mlp): Neural network fundamentals  
3. **Advanced Architectures** (cnn): Specialized network types
4. **Training Systems** (training, data): Complete learning pipelines
5. **Optimization** (kernels, compression, profiling): Performance
6. **Production** (benchmarking, config, mlops): Real-world deployment

## 🛠️ Module Structure

Each module follows a consistent structure:

```
modules/[module-name]/
├── README.md              # 📖 Module overview and instructions
├── [module-name].ipynb    # 📓 Main development notebook
├── test_[module].py       # 🧪 Automated tests
└── check_[module].py      # ✅ Manual verification (optional)
```

### Module Development Process

1. **Read README.md** - Understand learning objectives and requirements
2. **Open notebook** - Work through guided implementation
3. **Mark exports** - Use `#| export` for package code
4. **Test locally** - Verify functionality in notebook
5. **Export code** - Run `tito sync` to update package
6. **Run tests** - Ensure implementation meets requirements
7. **Iterate** - Fix issues and repeat until tests pass

## 📋 Requirements

- **Python 3.8+**
- **Jupyter Lab/Notebook**
- **nbdev** (for notebook development)
- **pytest** (for testing)
- **NumPy, Matplotlib** (for ML operations)

See `requirements.txt` for complete dependency list.

## 🎯 Goals & Philosophy

### Educational Goals

- **Deep Understanding**: Implement every component from first principles
- **Systems Thinking**: Understand how components interact
- **Performance Awareness**: Learn to optimize real systems
- **Production Skills**: Build systems that work in practice

### Design Philosophy

- **Module-First**: Self-contained learning units
- **Notebook-Driven**: Interactive development with immediate feedback
- **Test-Driven**: Comprehensive testing for reliability
- **Incremental**: Build understanding step by step
- **Real-World**: Techniques used in production systems

## 🚀 Getting Started

### For New Students

1. **Setup Environment**: `pip install -r requirements.txt`
2. **Start with Setup**: `cd modules/setup/ && cat README.md`
3. **Follow the sequence**: Complete modules in order
4. **Test frequently**: Use `tito test` to verify progress
5. **Build incrementally**: Each module prepares for the next

### For Instructors

- **Add modules**: Copy existing module structure
- **Update tests**: Add to `modules/[name]/test_[name].py`
- **Document well**: Clear READMEs and notebook explanations
- **Test integration**: Ensure modules work together

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [PyTorch](https://pytorch.org/), [fastai](https://fast.ai/), and [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- Built with [nbdev](https://nbdev.fast.ai/) for seamless notebook development
- Course structure inspired by modern ML systems courses
