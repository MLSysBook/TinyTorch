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

## 🚀 **Start Building Now**

```{admonition} Choose Your Adventure
:class: important

### **🔬 [Quick Try (5 min)](usage-paths/quick-exploration.md)** 
*Just want to see what this is?* → Click and code in your browser. No setup needed.

### **🏗️ [Build for Real (8+ weeks)](usage-paths/serious-development.md)** 
*Ready to build your own ML framework?* → Fork the repo and start your journey.

### **👨‍🏫 [Teach This Course](usage-paths/classroom-use.md)** 
*Want complete course infrastructure?* → Turn-key solution for educators.
```

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

---

## 🌟 **Why TinyTorch Works**

### **🔬 Build-First Learning**
No black boxes. Implement every component from scratch. Use YOUR code in real neural networks.

### **🚀 Production-Ready Skills**
Professional workflow with `tito` CLI. Real datasets like CIFAR-10. MLOps patterns from day one.

### **⚡ Instant Results**
Code works immediately. Visual progress indicators. Watch your `ReLU` power real networks.

### **🎯 Progressive Mastery**
Start simple (`hello_world()`). Build systematically. End powerful (production MLOps).

---

## 📚 **Course Journey: 14 Modules**

```{admonition} 🏗️ Foundation (Modules 1-5)
:class: note
**Weeks 1-6: Core Infrastructure**
- **Setup**: Professional development workflow  
- **Tensors**: Multi-dimensional arrays (like NumPy, but yours!)
- **Activations**: ReLU, Sigmoid, Tanh - the math that enables learning
- **Layers**: Dense layers with matrix multiplication
- **Networks**: Sequential architecture - chain layers into complete models
```

```{admonition} 🧠 Deep Learning (Modules 6-10)
:class: note
**Weeks 7-12: Complete Training Systems**
- **CNNs**: Convolutional operations for computer vision
- **DataLoader**: CIFAR-10 loading, batching, preprocessing  
- **Autograd**: Automatic differentiation engine (PyTorch's "magic")
- **Optimizers**: SGD, Adam, learning rate scheduling
- **Training**: Loss functions, metrics, complete orchestration
```

```{admonition} ⚡ Production (Modules 11-14)
:class: note
**Weeks 13-16: Real-World Deployment**
- **Compression**: Model pruning and quantization (75% size reduction)
- **Kernels**: High-performance custom operations
- **Benchmarking**: Systematic evaluation and performance measurement
- **MLOps**: Production monitoring, continuous learning, complete pipeline
```

---

## 🎓 **Learning Philosophy: Build → Use → Master**

### **Example: Activation Functions**

**🔧 Build:** Implement ReLU from scratch
```python
def relu(x):
    # YOU implement this function
    return ???  # What should this be?
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

This pattern repeats for every component - you build it, use it immediately, then see how it fits into larger systems.

---

## 📚 **Academic Foundation**

TinyTorch grew out of CS249r: Tiny Machine Learning Systems at Harvard University. While the [Machine Learning Systems book](https://mlsysbook.ai) covers broad principles, TinyTorch gives you hands-on implementation experience.

---

## 🚀 **Ready to Start?**

### **Quick Taste: Try Module 1 Right Now**
Want to see what TinyTorch feels like? **[Launch the Setup chapter](chapters/01-setup.md)** in Binder and implement your first TinyTorch function in 2 minutes!
