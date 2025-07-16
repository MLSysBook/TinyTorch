---
html_meta:
  "property=og:title": "Tiny🔥Torch: Build your own ML framework from scratch"
  "property=og:description": "Learn ML systems by building them. Implement tensors, autograd, optimizers from scratch. Build the rocket ship, don't just be the astronaut."
  "property=og:url": "https://mlsysbook.github.io/TinyTorch/"
  "property=og:type": "website"
  "property=og:image": "https://mlsysbook.github.io/TinyTorch/logo.png"
  "property=og:site_name": "Tiny🔥Torch Course"
  "name=twitter:card": "summary_large_image"
  "name=twitter:title": "Tiny🔥Torch: Build your own ML framework"
  "name=twitter:description": "Tiny🔥Torch is a minimalist framework for building machine learning systems from scratch—from tensors to systems."
  "name=twitter:image": "https://mlsysbook.github.io/TinyTorch/logo.png"
---

# Tiny🔥Torch: Build your own Machine Learning framework from scratch. 

**Most ML education teaches you to _use_ frameworks. TinyTorch teaches you to _build_ them.**

TinyTorch is a minimalist educational framework designed for learning by doing. Instead of relying on PyTorch or TensorFlow, you implement everything from scratch—tensors, autograd, optimizers, even MLOps tooling. This hands-on approach builds the deep systems intuition that sets ML engineers apart from ML users.

```{admonition} 🎯 What You'll Build
:class: tip
**A complete ML framework from scratch**: your own PyTorch style toolkit that can:
- ✅ Train neural networks on CIFAR-10 (real dataset!)
- ✅ Implement automatic differentiation (the "magic" behind PyTorch)  
- ✅ Deploy production systems with 75% model compression
- ✅ Handle complete ML pipeline from data to monitoring

**Result:** You become the expert others ask about "how PyTorch actually works."
```

_Everyone wants to be an astronaut._ 🧑‍🚀 _TinyTorch teaches you how to build the rocket ship._ 🚀

```{admonition} 📖 Complementary Learning
:class: note
For comprehensive ML systems knowledge, we recommend [**Machine Learning Systems**](https://mlsysbook.ai) by [Prof. Vijay Janapa Reddi](https://profvjreddi.github.io/website/). While TinyTorch teaches you to **build** ML systems from scratch, that book provides the broader **systems context** and engineering principles for production AI.
```

---

## 💡 **The Core Difference**

Most ML courses focus on algorithms and theory. You learn *what* neural networks do and *why* they work, but you import everything:

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

TinyTorch focuses on implementation and systems thinking. You learn *how* to build working systems with progressive scaffolding, production ready practices, and comprehensive course infrastructure that bridges the gap between learning and building.

---

## 🎓 **Learning Philosophy: Build, Use, Reflect**

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

**💡 Reflect:** See it working in real networks
```python
# Your ReLU is now part of a real neural network
model = Sequential([
    Dense(784, 128),
    ReLU(),           # <-- Your implementation
    Dense(128, 10)
])
```

This pattern repeats for every component: tensors, layers, optimizers, even MLOps systems. You build it, use it immediately, then reflect on how it fits into larger systems.

---

## 📚 **Course Journey: 15 Modules**

```{admonition} 🏗️ Foundation
:class: note
**1. Setup** • **2. Tensors** • **3. Activations**

Understanding workflow, multi-dimensional arrays, and the mathematical functions that enable learning.
```

```{admonition} 🧱 Building Blocks
:class: note
**4. Layers** • **5. Networks** • **6. CNNs**

Dense layers, sequential architecture, and convolutional operations for computer vision.
```

```{admonition} 🎯 Training Systems
:class: note
**7. DataLoader** • **8. Autograd** • **9. Optimizers** • **10. Training**

CIFAR-10 loading, automatic differentiation, SGD/Adam optimizers, and complete training orchestration.
```

```{admonition} ⚡ Production & Performance
:class: note
**11. Compression** • **12. Kernels** • **13. Benchmarking** • **14. MLOps**

Model optimization, high-performance operations, systematic evaluation, and production monitoring.
```

```{admonition} 🎓 Capstone Project
:class: note
**15. Capstone Project**

Choose your focus: performance engineering, algorithm extensions, systems optimization, framework analysis, or developer tools.
```

---

## 🔗 **Complete System Integration**

**This isn't 14 separate exercises.** Every component you build integrates into one fully functional ML framework:

```{admonition} 🎯 How It All Connects
:class: important
**Module 2: Your Tensor class** → **Module 3: Powers your activation functions** → **Module 4: Enables your layers** → **Module 5: Forms your networks** → **Module 8: Drives your autograd system** → **Module 9: Optimizes with your SGD/Adam** → **Module 10: Trains on real CIFAR-10 data**

**Result:** A complete, working ML framework that you built from scratch, capable of training real neural networks on real datasets.
```

### **🎯 Capstone: Optimize Your Framework**

After completing the 14 core modules, you have a **complete ML framework**. Now make it better through systems engineering:

**Choose Your Focus:**
- ⚡ **Performance Optimization**: GPU kernels, vectorization, memory-efficient operations
- 🧠 **Algorithm Extensions**: Transformer layers, BatchNorm, Dropout, advanced optimizers
- 🔧 **Systems Engineering**: Multi-GPU training, distributed computing, memory profiling
- 📊 **Benchmarking Deep Dive**: Compare your framework to PyTorch, identify bottlenecks
- 🛠️ **Developer Experience**: Better debugging tools, visualization, error messages

**The Challenge:** Use **only your TinyTorch implementation** as the base. No copying from PyTorch. This proves you understand the engineering trade-offs and can optimize real ML systems.

---

## 🛤️ **Choose Your Learning Path**

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

## ⚡ **Ready to Start?**

### **Quick Taste: Try Module 1 Right Now**
Want to see what TinyTorch feels like? **[Launch the Setup chapter](chapters/01-setup.md)** in Binder and implement your first TinyTorch function in 2 minutes!

---

## ❓ **Common Questions**

<details>
<summary><strong>🧮 "Do I need to know advanced math to succeed?"</strong></summary>

**You need high school math + willingness to learn.** We explain the math as we go:

- **Linear algebra**: We explain matrix multiplication when we build Dense layers
- **Calculus**: We cover derivatives when implementing backpropagation  
- **Statistics**: We introduce concepts like gradients in context of optimization

**Key insight:** You learn the math by implementing it, not the other way around. Most students find this more intuitive than traditional math courses.
</details>

<details>
<summary><strong>✅ "How do I know if I'm implementing things correctly?"</strong></summary>

**Built-in validation at every step:**
- **200+ automated tests** that check your implementations
- **Progressive testing**: Start simple, then add complexity
- **Clear error messages**: "Your ReLU should return 0 for negative inputs"
- **Expected outputs**: Know exactly what your code should produce

**Example:** Your tensor multiplication either passes the test or gets specific feedback about what went wrong.
</details>

<details>
<summary><strong>🔗 "Can I skip around or must I do modules in order?"</strong></summary>

**Designed for flexibility with recommended paths:**
- **Core foundation**: Modules 1-4 build on each other (do in order)
- **Parallel tracks**: Modules 5-8 can be done in different sequences
- **Advanced modules**: Pick what interests you most

**Smart approach:** Do Setup → Tensor → Activations in order, then choose your adventure.
</details>

<details>
<summary><strong>🚀 "Will this actually work with real data and real problems?"</strong></summary>

**Yes - we use production datasets from day one:**
- **CIFAR-10**: Train on 60,000 real images, not toy data
- **Standard benchmarks**: Compare your results to published papers
- **Real performance**: Your CNN will achieve 85%+ accuracy on image classification
- **Production patterns**: CLI tools, testing, packaging like professional frameworks

**This isn't a toy - it's a real framework that handles real problems.**
</details>

---

## 🙏 **Acknowledgments**

TinyTorch originated from CS249r: Tiny Machine Learning Systems at Harvard University. We're inspired by projects like [tinygrad](https://github.com/geohot/tinygrad), [micrograd](https://github.com/karpathy/micrograd), and [MiniTorch](https://minitorch.github.io/) that demonstrate the power of minimal implementations.


