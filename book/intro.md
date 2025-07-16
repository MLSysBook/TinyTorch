# Tiny🔥Torch: Build your own Machine Learning framework from scratch. 

**Learn ML by building your own PyTorch-style framework from the ground up. _Start small. Go deep._**

```{admonition} 🎯 What You'll Achieve
:class: tip
By the end of this course, you'll have **built your own complete ML framework** that can:
- ✅ Train neural networks on CIFAR-10 images (real dataset!)
- ✅ Implement automatic differentiation (the "magic" behind PyTorch)  
- ✅ Optimize models for production deployment (75% size reduction)
- ✅ Handle the complete ML pipeline from data loading to monitoring

**Most importantly:** You'll understand how modern ML frameworks *actually* work under the hood.
```

---

## 🏗️ **The Big Picture: Why Build from Scratch?**

**Most ML education teaches you to _use_ frameworks.** TinyTorch teaches you to _understand_ them.

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

**Result:** You become the person others come to when they need to understand "how PyTorch actually works under the hood."

---

## 🌟 **What Makes TinyTorch Different**

### **🔬 Build-First Philosophy**
- **No black boxes**: Implement every component from scratch
- **Immediate ownership**: Use YOUR code in real neural networks
- **Deep understanding**: Know exactly how each piece works

### **🚀 Real Production Skills**
- **Professional workflow**: Development with `tito` CLI, automated testing
- **Real datasets**: Train on CIFAR-10, not toy data
- **Production patterns**: MLOps, monitoring, optimization from day one

### **🎯 Progressive Mastery** 
- **Start simple**: Implement `hello_world()` function
- **Build systematically**: Each module enables the next
- **End powerful**: Deploy production ML systems with monitoring

### **⚡ Instant Feedback**
- **Code works immediately**: No waiting to see results
- **Visual progress**: Success indicators and system integration
- **"Aha moments"**: Watch your `ReLU` power real neural networks

---

## 📚 Educational Foundation

TinyTorch grew out of the CS249r: Tiny Machine Learning Systems course at Harvard University. While the [Machine Learning Systems book](https://mlsysbook.ai) covers the broad principles and practices of engineering ML systems, TinyTorch gives you hands-on experience building the systems yourself.

---

## 🚀 Choose Your Learning Path

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
- Automated grading for 200+ tests across all assignments
- Flexible pacing (8-16 weeks) with proven pedagogical outcomes
- Turn-key solution for ML systems education
```

---

## 🎯 What You'll Build

### **Progressive Complexity - Each Module Builds on Prior Work**

```{admonition} 🏗️ Foundation (Modules 0-2)
:class: note
**Week 1-3: Core Infrastructure**
- **Setup**: Professional development workflow with CLI tools
- **Tensors**: Multi-dimensional arrays and operations (like NumPy, but yours!)
- **Activations**: ReLU, Sigmoid, Tanh - the math that makes learning possible
```

```{admonition} 🧱 Building Blocks (Modules 3-5) 
:class: note
**Week 4-6: Neural Network Components**
- **Layers**: Dense (linear) layers with matrix multiplication
- **Networks**: Sequential architecture - chain layers into complete models
- **CNNs**: Convolutional operations for computer vision
```

```{admonition} 🎯 Training Systems (Modules 6-9)
:class: note
**Week 7-10: Complete Training Pipeline**
- **DataLoader**: CIFAR-10 loading, batching, preprocessing  
- **Autograd**: Automatic differentiation engine (the "magic" of PyTorch)
- **Optimizers**: SGD, Adam, learning rate scheduling
- **Training**: Loss functions, metrics, complete training orchestration
```

```{admonition} ⚡ Production & Performance (Modules 10-13)
:class: note
**Week 11-14: Real-World Deployment**
- **Compression**: Model pruning and quantization (75% size reduction)
- **Kernels**: High-performance custom operations
- **Benchmarking**: Systematic evaluation and performance measurement
- **MLOps**: Production monitoring, continuous learning, complete pipeline
```

---

## 🎓 Learning Philosophy: Build → Use → Understand

### **Example: How You'll Learn Activation Functions**

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

**💡 Understand:** See it working in real networks
```python
# Your ReLU is now part of a real neural network
model = Sequential([
    Dense(784, 128),
    ReLU(),           # <-- Your implementation
    Dense(128, 10)
])
```

**This pattern repeats for every component** - you build it, use it immediately, then see how it fits into larger systems.

---

## 🚀 Ready to Start?

```{admonition} Choose Your Adventure
:class: tip
**Just exploring?** → **[🔬 Quick Exploration](usage-paths/quick-exploration.md)** *(Click and code in 30 seconds)*

**Ready to build?** → **[🏗️ Serious Development](usage-paths/serious-development.md)** *(Fork repo and build your ML framework)*

**Teaching a class?** → **[👨‍🏫 Classroom Use](usage-paths/classroom-use.md)** *(Complete course infrastructure)*
```

### **Quick Taste: Try Chapter 1 Right Now**

Want to see what TinyTorch feels like? **[Launch the Setup chapter](chapters/01-setup.md)** in Binder and implement your first TinyTorch function in 2 minutes!
