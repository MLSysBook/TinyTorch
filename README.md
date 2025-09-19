# TinyTorch 🔥

**Build ML Systems From First Principles**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Documentation](https://img.shields.io/badge/docs-jupyter_book-orange.svg)](https://mlsysbook.github.io/TinyTorch/)
![Status](https://img.shields.io/badge/status-active-success.svg)

A Harvard University course that teaches ML systems engineering by building a complete deep learning framework from scratch. From tensors to transformers, understand every line of code powering modern AI.

## 🎯 What You'll Build

A **complete ML framework** capable of:
- Training CNNs on CIFAR-10 to 75%+ accuracy
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

## 📚 Course Structure

### **16 Progressive Modules**

| Module | Topic | What You Build |
|--------|-------|----------------|
| **Foundations** | | |
| 01 | Setup | Development environment |
| 02 | Tensors | N-dimensional arrays |
| 03 | Activations | ReLU, Sigmoid, Softmax |
| 04 | Layers | Dense layers |
| 05 | Networks | Sequential models |
| **Deep Learning** | | |
| 06 | Spatial | CNNs for vision |
| 07 | Attention | Transformers |
| 08 | DataLoader | Efficient data pipelines |
| 09 | Autograd | Automatic differentiation |
| 10 | Optimizers | SGD, Adam |
| **Production** | | |
| 11 | Training | Complete training loops |
| 12 | Compression | Model optimization |
| 13 | Kernels | Performance optimization |
| 14 | Benchmarking | Profiling tools |
| 15 | MLOps | Production deployment |
| **Language Models** | | |
| 16 | TinyGPT | Complete GPT implementation |

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

## 📊 Example: Train a CNN on CIFAR-10

```python
from tinytorch.core.networks import Sequential
from tinytorch.core.spatial import Conv2D
from tinytorch.core.activations import ReLU
from tinytorch.core.dataloader import CIFAR10Dataset
from tinytorch.core.training import Trainer
from tinytorch.core.optimizers import Adam

# Load real data
dataset = CIFAR10Dataset(download=True)
train_loader = DataLoader(dataset.train_data, batch_size=32)

# Build CNN
model = Sequential([
    Conv2D(3, 32, kernel_size=3),
    ReLU(),
    Conv2D(32, 64, kernel_size=3),
    ReLU(),
    Dense(64*28*28, 10)
])

# Train
trainer = Trainer(model, loss=CrossEntropyLoss(), optimizer=Adam())
trainer.fit(train_loader, epochs=30)
# Achieves 75%+ accuracy!
```

## 🧪 Testing & Validation

All demos and modules are thoroughly tested:

```bash
# Test all demos
python test_all_demos.py

# Validate implementations
python validate_demos.py

# Run checkpoint tests
tito checkpoint test 01
```

✅ **100% test coverage** across 8 interactive demos  
✅ **48 validation checks** ensuring correctness  
✅ **16 capability checkpoints** tracking progress  

## 📖 Documentation

- **[Course Website](https://mlsysbook.github.io/TinyTorch/)** - Complete interactive course
- **[Instructor Guide](docs/INSTRUCTOR_GUIDE.md)** - Teaching resources
- **[API Reference](https://mlsysbook.github.io/TinyTorch/api)** - Framework documentation

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Created by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) at Harvard University.

Special thanks to students and contributors who helped refine this educational framework.

---

**Start Small. Go Deep. Build ML Systems.**