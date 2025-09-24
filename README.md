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

### **12 Progressive Modules** - Build Complete ML Systems Step by Step!

#### **Part I: Neural Network Foundations** (Modules 1-7)
**"I can train neural networks from scratch!"**

| Module | Topic | What You Build | Key Innovation |
|--------|-------|----------------|----------------|
| 01 | Setup | Development environment | CLI tools, testing framework |
| 02 | Tensor | N-dimensional arrays + **Basic Autograd** | Gradients from the start! |
| 03 | Activations | **ReLU + Softmax ONLY** | Focus on what matters most |
| 04 | Layers | Linear + Module + **Flatten** | Complete building blocks |
| 05 | Loss | **MSE + CrossEntropy** | Define learning objectives |
| 06 | Optimizers | **SGD + Adam** | How we learn |
| 07 | Training | **Complete training loops** | Put it all together |

**✅ Capstone**: XOR + MNIST - Train real neural networks after just 7 modules!

---

#### **Part II: Computer Vision** (Modules 8-9)
**"I can build CNNs that classify real images!"**

| Module | Topic | What You Build |
|--------|-------|----------------|
| 08 | CNN Ops | Conv2d + MaxPool2d |
| 09 | DataLoader | Efficient data pipelines |

**✅ Capstone**: CIFAR-10 CNN - 55%+ accuracy on real images

---

#### **Part III: Language Models** (Modules 10-12)
**"I can build transformers that generate text!"**

| Module | Topic | What You Build |
|--------|-------|----------------|
| 10 | Embeddings | Token embeddings, positional encoding |
| 11 | Attention | Multi-head attention |
| 12 | Transformers | Transformer blocks |

**✅ Capstone**: TinyGPT - Generate text with transformers

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

### **After Module 07** → `examples/xornet/` + `examples/mnist/` 🔥
```bash
cd examples/xornet
python train_xor.py
# 🎯 100% accuracy on XOR problem!

cd examples/mnist
python train_mlp.py
# 🏆 95%+ accuracy on handwritten digits!
```

### **After Module 09** → `examples/cifar10/` 🎯  
```bash
cd examples/cifar10
python train_cnn.py
# 🏆 55%+ accuracy on real images!
```

### **After Module 12** → `examples/tinygpt/` 🚀
```bash
cd examples/tinygpt
python train_gpt.py
# 🔥 Generate text with transformers!
```

**These aren't toy demos** - they're real ML applications achieving solid results with YOUR framework built from scratch following KISS principles!

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

✅ **16 modules passing all tests** with 100% health status  
✅ **16 capability checkpoints** tracking learning progress  
✅ **Comprehensive testing framework** with module and integration tests  
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