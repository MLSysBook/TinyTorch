# Tiny🔥Torch: Build your own Machine Learning framework from scratch. 

**Most ML education teaches you to _use_ frameworks. TinyTorch teaches you to _understand_ them.**

```{admonition} 🎯 What You'll Build
:class: tip
**A complete ML framework from scratch** — your own PyTorch-style toolkit that can:
- ✅ Train neural networks on CIFAR-10 (real dataset!)
- ✅ Implement automatic differentiation (the "magic" behind PyTorch)  
- ✅ Deploy production systems with 75% model compression
- ✅ Handle complete ML pipeline from data to monitoring

**Result:** You become the expert others ask about "how PyTorch actually works."
```

---

## 🎯 **The Learning Opportunity**

Most ML education focuses on using existing frameworks — import PyTorch, call `nn.Linear()`, run `optimizer.step()`. This approach gets you building models quickly, but leaves a gap: **how do these frameworks actually work under the hood?**

TinyTorch fills that gap. Instead of treating frameworks as black boxes, you'll build every component yourself. The result? Deep understanding that makes you the person others come to with questions about "how PyTorch really works."

---

## 💡 **The Core Difference**

```python
Traditional ML Course:          TinyTorch Approach:
├── import torch               ├── class Tensor:
├── model = nn.Linear(10, 1)   │     def __add__(self, other): ...
├── loss = nn.MSELoss()        │     def backward(self): ...
└── optimizer.step()           ├── class Linear:
                               │     def forward(self, x):
                               │       return x @ self.weight + self.bias
                               ├── def mse_loss(pred, target):
                               │     return ((pred - target) ** 2).mean()
                               ├── class SGD:
                               │     def step(self):
                               └──     param.data -= lr * param.grad

Go from "How does this work?" 🤷 to "I implemented every line!" 💪
```

This isn't just about learning — it's about developing the deep systems thinking that distinguishes ML engineers from ML users.

---

## 🌟 **What Makes TinyTorch Different**

### **🔬 Build-First Philosophy**
You don't just learn about tensors — you implement the `Tensor` class from scratch. You don't just use ReLU — you write the activation function yourself. Every component you build becomes part of your personal ML framework that actually works on real data.

### **🚀 Production-Ready Skills**
From day one, you'll use professional development practices: the `tito` CLI for project management, automated testing for quality assurance, real datasets like CIFAR-10 for training, and MLOps patterns for deployment. This isn't toy code — it's the foundation for production ML systems.

### **⚡ Instant Results**
Your code works immediately. Implement a `ReLU` function in Module 3, and by Module 5 you're watching it power real neural networks. Visual progress indicators and comprehensive testing ensure you always know your implementations are correct.

### **🎯 Progressive Mastery**
Start simple with a `hello_world()` function, build systematically through tensors and layers, and end with production MLOps systems. Each module builds on previous work, creating a complete learning journey from foundations to advanced systems.

---

## 🎓 **Learning Philosophy: Build → Use → Master**

Every component follows the same powerful learning cycle:

### **Example: Activation Functions**

**🔧 Build:** Implement ReLU from scratch
```python
def relu(x):
    # YOU implement this function
    return np.maximum(0, x)  # Your solution
```

**🚀 Use:** Immediately use your own code
```python
from tinytorch.core.activations import ReLU  # YOUR implementation!
layer = ReLU()
output = layer.forward(input_tensor)  # Your code working!
```

**💡 Master:** See it working in real networks
```python
# Your ReLU is now part of a real neural network
model = Sequential([
    Dense(784, 128),
    ReLU(),           # <-- Your implementation
    Dense(128, 10)
])
```

This pattern repeats for every component — tensors, layers, optimizers, even MLOps systems. You build it, use it immediately, then see how it fits into larger systems.

---

## 📚 **Course Journey: 14 Modules**

```{admonition} 🏗️ Foundation (Modules 1-5)
:class: note
**Weeks 1-6: Core Infrastructure**
- **Setup**: Professional development workflow with `tito` CLI and testing
- **Tensors**: Multi-dimensional arrays with operations (like NumPy, but yours!)
- **Activations**: ReLU, Sigmoid, Tanh — the mathematical functions that enable learning
- **Layers**: Dense layers with matrix multiplication and weight management
- **Networks**: Sequential architecture — chain layers into complete models
```

```{admonition} 🧠 Deep Learning (Modules 6-10)
:class: note
**Weeks 7-12: Complete Training Systems**
- **CNNs**: Convolutional operations for computer vision applications
- **DataLoader**: CIFAR-10 loading, batching, and preprocessing pipelines
- **Autograd**: Automatic differentiation engine (the "magic" behind PyTorch)
- **Optimizers**: SGD with momentum, Adam with adaptive learning rates
- **Training**: Loss functions, metrics, and complete training orchestration
```

```{admonition} ⚡ Production (Modules 11-14)
:class: note
**Weeks 13-16: Real-World Deployment**
- **Compression**: Model pruning and quantization for 75% size reduction
- **Kernels**: High-performance custom operations and optimization
- **Benchmarking**: Systematic evaluation and performance measurement
- **MLOps**: Production monitoring, continuous learning, complete pipeline
```

---

## 🚀 **Choose Your Learning Path**

```{admonition} Three Ways to Engage with TinyTorch
:class: important

### **🔬 [Quick Exploration](usage-paths/quick-exploration.md)** *(5 minutes)*
*"I want to see what this is about"*
- Click and run code immediately in your browser (Binder)
- No installation or setup required
- Implement ReLU, tensors, neural networks interactively
- Perfect for getting a feel for the course

### **🏗️ [Serious Development](usage-paths/serious-development.md)** *(8+ weeks)*
*"I want to build this myself"*
- Fork the repo and work locally with full development environment
- Build complete ML framework from scratch with `tito` CLI
- 14 progressive assignments from setup to production MLOps
- Professional development workflow with automated testing

### **👨‍🏫 [Classroom Use](usage-paths/classroom-use.md)** *(Instructors)*
*"I want to teach this course"*
- Complete course infrastructure with NBGrader integration
- Automated grading for comprehensive testing
- Flexible pacing (8-16 weeks) with proven pedagogical structure
- Turn-key solution for ML systems education
```

---

## 🚀 **Ready to Start?**

### **Quick Taste: Try Module 1 Right Now**
Want to see what TinyTorch feels like? **[Launch the Setup chapter](chapters/01-setup.md)** in Binder and implement your first TinyTorch function in 2 minutes!

---

## 📚 **Academic Foundation**

TinyTorch grew out of CS249r: Tiny Machine Learning Systems at Harvard University. While the [Machine Learning Systems book](https://mlsysbook.ai) covers broad principles, TinyTorch gives you hands-on implementation experience.
