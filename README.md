# TinyTorch 🔥

**Build ML Systems From First Principles**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Documentation](https://img.shields.io/badge/docs-jupyter_book-orange.svg)](https://mlsysbook.github.io/TinyTorch/)
![Status](https://img.shields.io/badge/status-active-success.svg)

A Harvard University course that teaches ML systems engineering by building a complete deep learning framework from scratch. From tensors to transformers, understand every line of code powering modern AI.

## 🎯 What You'll Build

A **complete ML framework** capable of:
- Training neural networks on CIFAR-10 to 55%+ accuracy (reliably achievable!)
- Building GPT-style language models  
- Implementing modern optimizers (Adam, learning rate scheduling)
- Production deployment with monitoring and MLOps

All built from scratch using only NumPy - no PyTorch, no TensorFlow!

## 🚀 Quick Start

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

## 📚 Streamlined Learning Journey - No Forward Dependencies!

### **21 Progressive Modules** - Build Complete ML Systems Step by Step!

#### **Part I: Neural Network Foundations** (Modules 1-8)
**"I can train neural networks from scratch!"**

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

**✅ Capstone**: XOR + MNIST - Train real neural networks after just 8 modules!

---

#### **Part II: Computer Vision** (Modules 9-10)
**"I can build CNNs that classify real images!"**

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 09 | Spatial | Conv2d + MaxPool2d + CNN operations | **Parameter scaling** (filters × channels), spatial locality, convolution efficiency |
| 10 | DataLoader | Efficient data pipelines + CIFAR-10 | **Batch processing**, memory-mapped I/O, data pipeline bottlenecks |

**✅ Capstone**: CIFAR-10 CNN - 55%+ accuracy on real images

---

#### **Part III: Language Models** (Modules 11-14)
**"I can build transformers that generate text!"**

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 11 | Tokenization | Text processing + vocabulary | **Vocabulary scaling** (memory vs sequence length), tokenization bottlenecks |
| 12 | Embeddings | Token embeddings + positional encoding | **Embedding tables** (vocab × dim parameters), lookup performance |
| 13 | Attention | Multi-head attention mechanisms | **O(N²) scaling**, memory bottlenecks, attention optimization |
| 14 | Transformers | Complete transformer blocks | **Layer scaling**, memory requirements, architectural trade-offs |

**✅ Capstone**: TinyGPT - Generate text with transformers

---

#### **Part IV: System Optimization** (Modules 15-20)
**"I can profile, optimize, and benchmark ML systems!"**

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 15 | Profiling | Performance analysis + bottleneck detection | **Memory profiling**, FLOP counting, **Amdahl's Law**, performance measurement |
| 16 | Acceleration | Hardware optimization + cache-friendly algorithms | **Cache hierarchies**, memory access patterns, **vectorization vs loops** |
| 17 | Quantization | Model compression + precision reduction | **Precision trade-offs** (FP32→INT8), memory reduction, accuracy preservation |
| 18 | Compression | Pruning + knowledge distillation | **Sparsity patterns**, parameter reduction, **compression ratios** |
| 19 | Caching | Memory optimization + KV caching | **Memory vs compute trade-offs**, cache management, generation efficiency |
| 20 | Benchmarking | **TinyMLPerf competition framework** | **Competitive optimization**, relative performance metrics, innovation scoring |

**✅ Capstone**: TinyMLPerf Competition - Optimize models for speed and efficiency

---

#### **Part V: Production Systems** (Module 21)
**"I can deploy and monitor ML systems in production!"**

| Module | Topic | What You Build | ML Systems Learning |
|--------|-------|----------------|-------------------|
| 21 | MLOps | Model monitoring + drift detection + automated retraining | **Production monitoring**, model lifecycle management, **drift detection**, automated response systems |

**✅ Capstone**: Production ML Pipeline - Complete end-to-end system

## 🎓 Learning Philosophy

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

✅ **Deep Understanding** - Know exactly what `loss.backward()` does  
✅ **Systems Thinking** - Understand memory, compute, and scaling  
✅ **Debugging Skills** - Fix problems at any level of the stack  
✅ **Production Ready** - Learn patterns used in real ML systems  

## 🛠️ Key Features

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

## 🔥 Examples You Can Run

As you complete modules, exciting examples unlock to show your framework in action:

### **After Module 08** → Neural Network Foundations Complete! 🔥
```bash
cd examples/perceptron_1957
python rosenblatt_perceptron.py
# 🎯 Classic perceptron implementation!

cd examples/xor_1969  
python minsky_xor_problem.py
# 🧠 Solve the famous XOR problem!

cd examples/lenet_1998
python train_mlp.py
# 🏆 95%+ accuracy on MNIST handwritten digits!
```

### **After Module 10** → Computer Vision Complete! 🎯  
```bash
cd examples/alexnet_2012
python train_cnn.py
# 🏆 55%+ accuracy on CIFAR-10 real images!
```

### **After Module 14** → Language Models Complete! 🚀
```bash
cd examples/gpt_2018
python train_gpt.py
# 🔥 Generate text with transformers you built!
```

### **After Module 20** → System Optimization Complete! ⚡
```bash
# Use TinyMLPerf to benchmark your optimizations
tito benchmark run --event mlp_sprint
tito benchmark run --event cnn_marathon  
tito benchmark run --event transformer_decathlon
# 🏆 Compete in the Olympics of ML Systems Optimization!
```

### **After Module 21** → Production Systems Complete! 🌟
```bash
# Deploy complete production ML pipeline
python examples/production_pipeline.py
# 🚀 Monitor, deploy, and scale ML systems like a pro!
```

**These aren't toy demos** - they're real ML applications achieving solid results with YOUR framework built from scratch, optimized for performance, and deployed at production scale!

## 🧪 Testing & Validation

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

✅ **21 modules passing all tests** with 100% health status  
✅ **16 capability checkpoints** tracking learning progress  
✅ **Complete optimization pipeline** from profiling to competition benchmarking  
✅ **Production-ready MLOps** with monitoring and automated retraining  
✅ **KISS principle design** for clear, maintainable code  

## 📖 Documentation

- **[Course Website](https://mlsysbook.github.io/TinyTorch/)** - Complete interactive course
- **[Instructor Guide](docs/INSTRUCTOR_GUIDE.md)** - Teaching resources  
- **[Student Quickstart](docs/STUDENT_QUICKSTART.md)** - Getting started guide
- **[CIFAR-10 Training Guide](docs/cifar10-training-guide.md)** - Detailed training walkthrough

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Created by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) at Harvard University.

Special thanks to students and contributors who helped refine this educational framework.

---

**Start Small. Go Deep. Build ML Systems.**