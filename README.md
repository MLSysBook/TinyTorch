# Tiny🔥Torch 

**Build your own ML framework. Start small. Go deep.**

![Work in Progress](https://img.shields.io/badge/status-work--in--progress-yellow)
![Educational Project](https://img.shields.io/badge/purpose-educational-informational)
[![GitHub](https://img.shields.io/badge/github-mlsysbook/TinyTorch-blue.svg)](https://github.com/MLSysBook/TinyTorch)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![Jupyter Book](https://img.shields.io/badge/docs-Jupyter_Book-orange.svg)](https://mlsysbook.github.io/TinyTorch/)

A hands-on [Machine Learning Systems](https://mlsysbook.ai) course companion where students don’t just learn ML — they build it. 

Tiny🔥Torch is the minimalist, code-first companion to any machine learning systems course. It embraces a “start small, go deep” philosophy—starting with tensors and layers, and guiding learners through each system component, all the way to a complete MLOps pipelines built from scratch in their own codebase—albeit within a deliberately small-scale educational framework.

📚 **[Read the Interactive Course →](https://mlsysbook.github.io/TinyTorch/)**

---

## 🎯 What You'll Build

* **Complete ML Framework** — Your own PyTorch-style toolkit, from tensors to MLOps
* **Real Applications** — Train neural networks on real datasets using your code
* **Production Skills** — Full ML system lifecycle: training, deployment, monitoring
* **Deep Understanding** — Build every component, understand every decision

---

## 🚀 Quick Start (2 minutes)

### 🧑‍🎓 **Students**

```bash
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
pip install -e .
tito system doctor                         # Verify your setup
cd modules/source/00_setup
jupyter lab setup_dev.py                  # Launch your first module
```

### 👩‍🏫 **Instructors**

```bash
# System check
tito system info
tito system doctor

# Module workflow
tito export 00_setup
tito test 00_setup
tito nbdev build                          # Update package
```

---

## 📚 Complete Course: 14 Modules

### **🏗️ Foundations** (Modules 00-04)
* **00_setup**: Development environment and CLI tools
* **01_tensor**: N-dimensional arrays and tensor operations  
* **02_activations**: ReLU, Sigmoid, Tanh, Softmax functions
* **03_layers**: Dense layers and matrix operations
* **04_networks**: Sequential networks and MLPs

### **🧠 Deep Learning** (Modules 05-08)
* **05_cnn**: Convolutional neural networks and image processing
* **06_dataloader**: Data loading, batching, and preprocessing
* **07_autograd**: Automatic differentiation and backpropagation  
* **08_optimizers**: SGD, Adam, and learning rate scheduling

### **⚡ Systems & Production** (Modules 09-13)
* **09_training**: Training loops, metrics, and validation
* **10_compression**: Model pruning, quantization, and distillation
* **11_kernels**: Performance optimization and custom operations
* **12_benchmarking**: Profiling, testing, and performance analysis
* **13_mlops**: Monitoring, deployment, and production systems

**Status**: All 14 modules complete with inline tests and educational content

---

## 📖 Documentation

### **Interactive Jupyter Book**
- **Live Site**: https://mlsysbook.github.io/TinyTorch/
- **Auto-updated** from source code on every release
- **Complete course content** with executable examples
- **Real implementation details** with solution code

### **Development Workflow**
- **`dev` branch**: Active development and experiments  
- **`main` branch**: Stable releases that trigger documentation deployment
- **Inline testing**: Tests embedded directly in source modules
- **Continuous integration**: Automatic building and deployment

---

## 🛠️ Development Workflow

### **Module Development**
```bash
# Work on dev branch
git checkout dev

# Edit source modules  
cd modules/source/01_tensor
jupyter lab tensor_dev.py

# Export to package
tito export 01_tensor

# Test your implementation
tito test 01_tensor

# Build complete package
tito nbdev build
```

### **Release Process**
```bash
# Ready for release
git checkout main
git merge dev
git push origin main        # Triggers documentation deployment
```

---

## 🧠 Pedagogical Framework: Build → Use → Reflect

### **Real Engineering, Real Understanding**

1. **Build** — Implement `ReLU()` activation function
2. **Use** — Apply it via `tinytorch.core.activations.ReLU()`  
3. **Reflect** — Understand its role in neural network design
4. **Iterate** — Extend knowledge with each module

### **Example Learning Cycle**

```python
# Step 1: You implement this in tensor_dev.py
class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
    
    def __add__(self, other):
        return Tensor(self.data + other.data)

# Step 2: Export and use in your framework
from tinytorch.core.tensor import Tensor
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
result = a + b  # Your implementation at work!

# Step 3: Apply to real problems
model = Sequential([Dense(784, 128), ReLU(), Dense(128, 10)])
```

---

## 🎓 Teaching Philosophy

### **No Black Boxes**
* Build every component from scratch
* Understand performance trade-offs  
* See how engineering decisions impact ML outcomes

### **Production-Ready Thinking**
* Use real datasets (CIFAR-10, MNIST)
* Implement proper testing and benchmarking
* Learn MLOps and system design principles

### **Iterative Mastery**
* Each module builds on previous work
* Immediate feedback through inline testing
* Progressive complexity with solid foundations

---

## 📁 Project Structure

```
TinyTorch/
├── modules/source/XX/               # 14 source modules with inline tests
├── tinytorch/core/                  # Your exported ML framework
├── tito/                           # CLI and course management tools
├── book/                           # Jupyter Book source and config
├── tests/                          # Integration tests
└── docs/                           # Development guides and workflows
```

---

## 🧪 Tech Stack

* **Python 3.8+** — Modern Python with type hints
* **NumPy** — Numerical foundations  
* **Jupyter Lab** — Interactive development
* **Rich** — Beautiful CLI output
* **NBDev** — Literate programming and packaging
* **Jupyter Book** — Interactive documentation
* **GitHub Actions** — Continuous integration and deployment

---

## ✅ Verified Learning Outcomes

Students who complete TinyTorch can:

✅ **Build complete neural networks** from tensors to training loops  
✅ **Implement modern ML algorithms** (Adam, dropout, batch norm)  
✅ **Optimize performance** with profiling and custom kernels  
✅ **Deploy production systems** with monitoring and MLOps  
✅ **Debug and test** ML systems with proper engineering practices  
✅ **Understand trade-offs** between accuracy, speed, and resources  

---

## 🏃‍♀️ Getting Started

### **Option 1: Interactive Course**
👉 **[Start Learning Now](https://mlsysbook.github.io/TinyTorch/)** — Complete course in your browser

### **Option 2: Local Development**
```bash
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
pip install -e .
tito system doctor
cd modules/source/00_setup
jupyter lab setup_dev.py
```

### **Option 3: Instructor Setup**
```bash
# Clone and verify system
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
tito system info

# Test module workflow
tito export 00_setup && tito test 00_setup
```

---

**🔥 Ready to build your own ML framework? Start with TinyTorch and understand every layer. _Start Small. Go Deep._**
