# Tiny🔥Torch: Build ML Systems from Scratch

> A hands-on systems course where you implement every component of a modern ML system

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![nbdev](https://img.shields.io/badge/built%20with-nbdev-orange.svg)](https://nbdev.fast.ai/)

> **Disclaimer**: TinyTorch is an educational framework developed independently and is not affiliated with or endorsed by Meta or the PyTorch project.

**Tiny🔥Torch** is a hands-on companion to [*Machine Learning Systems*](https://mlsysbook.ai), providing practical coding exercises that complement the book's theoretical foundations. Rather than just learning *about* ML systems, you'll build one from scratch—implementing everything from tensors and autograd to hardware-aware optimization and deployment systems.

## 🎯 What You'll Build

By completing this course, you will have implemented a complete ML system:

**Core Framework** → **Training Pipeline** → **Production System**
- ✅ Tensors with automatic differentiation
- ✅ Neural network layers (MLP, CNN, Transformer)
- ✅ Training loops with optimizers (SGD, Adam)
- ✅ Data loading and preprocessing pipelines
- ✅ Model compression (pruning, quantization)
- ✅ Performance profiling and optimization
- ✅ Production deployment and monitoring

## 🚀 Quick Start

**Ready to build? Choose your path:**

### 🏃‍♂️ I want to start building now
→ **[QUICKSTART.md](QUICKSTART.md)** - Get coding in 10 minutes

### 📚 I want to understand the full course structure  
→ **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Complete learning roadmap

### 🔍 I want to see the course in action
→ **[modules/setup/](modules/setup/)** - Browse the first module

## 🎓 Learning Approach

**Module-First Development**: Each module is self-contained with its own notebook, tests, and learning objectives. You'll work in Jupyter notebooks using the [nbdev](https://nbdev.fast.ai/) workflow to build a real Python package.

**The Cycle**: `Write Code → Export → Test → Next Module`

```bash
# The rhythm you'll use for every module
jupyter lab tensor_dev.ipynb    # Write & test interactively  
python bin/tito.py sync         # Export to Python package
python bin/tito.py test         # Verify implementation
```

## 📚 Course Structure

| Phase | Modules | What You'll Build |
|-------|---------|-------------------|
| **Foundation** | Setup, Tensor, Autograd | Core mathematical engine |
| **Neural Networks** | MLP, CNN | Learning algorithms |
| **Training Systems** | Data, Training, Config | End-to-end pipelines |
| **Production** | Profiling, Compression, MLOps | Real-world deployment |

**Total Time**: 40-80 hours over several weeks • **Prerequisites**: Python basics

## 🛠️ Key Commands

```bash
python bin/tito.py info               # Check progress
python bin/tito.py sync               # Export notebooks  
python bin/tito.py test --module [name]  # Test implementation
```

## 🌟 Why Tiny🔥Torch?

**Systems Engineering Principles**: Learn to design ML systems from first principles
**Hardware-Software Co-design**: Understand how algorithms map to computational resources  
**Performance-Aware Development**: Build systems optimized for real-world constraints
**End-to-End Systems**: From mathematical foundations to production deployment

## 📖 Educational Approach

**Companion to [Machine Learning Systems](https://mlsysbook.ai)**: This course provides hands-on implementation exercises that bring the book's concepts to life through code.

**Learning by Building**: Following the educational philosophy of [Karpathy's micrograd](https://github.com/karpathy/micrograd), we learn complex systems by implementing them from scratch.

**Real-World Systems**: Drawing from production [PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io/) architectures to understand industry-proven design patterns.

## 🤝 Contributing

We welcome contributions! Whether you're a student who found a bug or an instructor wanting to add modules, see our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Ready to start building?** → [**QUICKSTART.md**](QUICKSTART.md) 🚀
