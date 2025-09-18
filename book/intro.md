---
html_meta:
  "property=og:title": "TinyTorch: Build your own ML framework from scratch"
  "property=og:description": "Learn ML systems by building them. From computer vision to language models. Comprehensive educational framework for understanding ML systems engineering."
  "property=og:url": "https://mlsysbook.github.io/TinyTorch/"
  "property=og:type": "website"
  "property=og:image": "https://mlsysbook.github.io/TinyTorch/logo.png"
  "property=og:site_name": "TinyTorch Course"
  "name=twitter:card": "summary_large_image"
  "name=twitter:title": "TinyTorch: Build your own ML framework"
  "name=twitter:description": "TinyTorch is a minimalist framework for building machine learning systems from scratch—from vision to language."
  "name=twitter:image": "https://mlsysbook.github.io/TinyTorch/logo.png"
---

# TinyTorch: Build Your Own ML Framework from First Principles

**Most ML education teaches you to _use_ frameworks. TinyTorch teaches you to _build_ them.**

Tiny🔥Torch is a minimalist framework for building machine learning systems from scratch—from tensors to systems. Instead of relying on PyTorch or TensorFlow, you implement everything yourself—tensors, autograd, optimizers, even MLOps tooling.

**The Vision: Train ML Systems Engineers, Not Just ML Users**

This hands-on approach builds the deep systems intuition that separates ML engineers from ML users. You'll understand not just *what* neural networks do, but *how* they work under the hood, *why* certain design choices matter in production, and *when* to make trade-offs between memory, speed, and accuracy.

```{admonition} What You'll Build
:class: tip
**A complete ML framework from scratch**: your own production-ready toolkit that can:
- Train neural networks on CIFAR-10 (real dataset)
- Implement automatic differentiation from first principles
- Deploy production systems with 75% model compression
- Handle complete ML pipeline from data to monitoring
- **Build GPT-style language models with 95% component reuse**

**Result:** You become the expert others ask about "how ML frameworks actually work" and "why neural architectures share universal foundations."
```

_Understanding how to build ML systems makes you a more effective ML engineer._

```{admonition} The Perfect Learning Combination
:class: note
TinyTorch was designed as the hands-on lab companion to [**Machine Learning Systems**](https://mlsysbook.ai) by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) (Harvard). The book teaches you ML systems **theory and principles** - TinyTorch lets you **implement and experience** those concepts firsthand. Together, they provide complete ML systems mastery.
```

---

## The Core Difference

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

Transform from "How does this work?" to "I implemented every line!"
```

TinyTorch focuses on implementation and systems thinking. You learn *how* to build working systems with progressive scaffolding, production ready practices, and comprehensive course infrastructure that bridges the gap between learning and building.

**What Makes This Different: Systems-First Thinking**

Traditional ML courses teach algorithms. TinyTorch teaches **ML systems engineering**:
- **Memory Management**: Why Adam uses 3× more memory than SGD and when that matters
- **Performance Analysis**: How attention mechanisms scale O(N²) and limit context length  
- **Production Trade-offs**: When to use gradient accumulation vs larger GPUs
- **Hardware Awareness**: How cache misses make naive convolution 100× slower
- **System Design**: How autograd graphs consume memory and enable gradient checkpointing

**Result**: You become the engineer who designs ML systems, not just uses them.

---

## Learning Philosophy: Build, Use, Reflect

Every component follows the same powerful learning cycle:

### Example: Activation Functions

**Build:** Implement ReLU from scratch
```python
def relu(x):
    # YOU implement this function
    return np.maximum(0, x)  # Your solution
```

**Use:** Immediately use your own code
```python
from tinytorch.core.activations import ReLU  # YOUR implementation!
layer = ReLU()
output = layer.forward(input_tensor)  # Your code working!
```

**Reflect:** See it working in real networks
```python
# Your ReLU is now part of a real neural network
model = Sequential([
    Dense(784, 128),
    ReLU(),           # <-- Your implementation
    Dense(128, 10)
])
```

This pattern repeats for every component: tensors, layers, optimizers, even MLOps systems. You build it, use it immediately, then reflect on how it fits into larger systems.

**🎯 Track Your Capabilities**

TinyTorch uses a [checkpoint system](checkpoint-system.md) to track your progress through **ML systems engineering capabilities**:
- **Foundation** → Core ML primitives and setup
- **Architecture** → Neural network building  
- **Training** → Model training pipeline
- **Inference** → Deployment and optimization
- **Serving** → Complete system integration

Use `tito checkpoint status` to see your progress anytime!

**🎯 Beyond Code: Systems Intuition**

Each module includes **ML Systems Thinking** sections that connect your implementations to production reality:
- *"How does your tensor implementation compare to PyTorch's memory management?"*
- *"When would you choose SGD over Adam in production training?"* 
- *"How do frameworks handle the quadratic memory scaling of attention?"*
- *"What happens to your autograd implementation under distributed training?"*

These aren't just academic questions - they're the system-level challenges that ML engineers solve every day.

---

## 👥 Who This Is For

### 🎯 Perfect For:
- **CS students** who want to understand ML systems beyond high-level APIs
- **Software engineers** transitioning to ML engineering roles
- **ML practitioners** who want to optimize and debug production systems
- **Researchers** who need to implement custom operations and architectures
- **Anyone curious** about how PyTorch/TensorFlow actually work under the hood

### 📚 Prerequisites:
- **Python programming** (comfortable with classes, functions, basic NumPy)
- **Linear algebra basics** (matrix multiplication, gradients)
- **Learning mindset** - we'll teach you everything else!

### 🚀 Career Impact:
After TinyTorch, you'll be the person your team asks:
- *"Why is this training so slow?"* (You'll know how to profile and optimize)
- *"Can we fit this model in GPU memory?"* (You'll understand memory trade-offs)  
- *"What's the best optimizer for this problem?"* (You'll know the system implications)

---

## 📚 Course Journey: 16 Modules - Foundation to Framework

```{admonition} Foundation
:class: note
**0. Setup** • **1. Tensors** • **2. Activations**

Development workflow, multi-dimensional arrays, and mathematical functions that enable learning.
```

```{admonition} Building Blocks
:class: note
**3. Layers** • **4. Dense** • **5. Spatial** • **6. Attention**

Dense layers, sequential networks, convolutional operations, and self-attention mechanisms with memory analysis.
```

```{admonition} Training Systems
:class: note
**7. DataLoader** • **8. Autograd** • **9. Optimizers** • **10. Training**

CIFAR-10 loading, automatic differentiation with graph management, SGD/Adam with memory profiling, and complete training orchestration.
```

```{admonition} Production Systems
:class: note
**11. Compression** • **12. Kernels** • **13. Benchmarking** • **14. MLOps**

Model optimization, high-performance operations, systematic evaluation, and production monitoring with real deployment patterns.
```

```{admonition} Framework Generalization
:class: note
**15. TinyGPT**

Demonstrate framework universality: GPT-style transformers, character tokenization, autoregressive generation with 95% component reuse from your ML systems foundation.
```

---

## 🔗 Complete System Integration

**This isn't 16 separate exercises.** Every component you build integrates into one fully functional ML framework with universal foundations:

```{admonition} 🎯 How It All Connects
:class: important

```{mermaid}
flowchart TD
    Z[00_introduction<br/>🎯 System Overview] --> A[01_setup<br/>Setup & Environment] 
    A --> B[02_tensor<br/>Core Tensor Operations]
    B --> C[03_activations<br/>ReLU, Sigmoid, Tanh]
    B --> I[09_autograd<br/>Automatic Differentiation]
    
    C --> D[04_layers<br/>Dense Layers]
    D --> E[05_dense<br/>Sequential Networks]
    
    E --> F[06_spatial<br/>Convolutional Networks]
    E --> G[07_attention<br/>Self-Attention]
    
    B --> H[08_dataloader<br/>Data Loading]
    
    I --> J[10_optimizers<br/>SGD & Adam]
    
    H --> K[11_training<br/>Training Loops]
    E --> K
    F --> K
    G --> K
    J --> K
    
    K --> L[12_compression<br/>Model Optimization]
    K --> M[13_kernels<br/>High-Performance Ops]
    K --> N[14_benchmarking<br/>Performance Analysis]
    K --> O[15_mlops<br/>Production Monitoring]
    
    L --> P[16_tinygpt<br/>🔥 Language Models]
    G --> P
    J --> P
    K --> P
```

**Result:** Every component you build converges into TinyGPT - proving your framework is complete and production-ready.
```

### 🔥 TinyGPT: The Complete Framework in Action

After building all the components, TinyGPT is your **capstone demonstration** - showing how everything clicks together into a working system.

**What TinyGPT Proves:**
- 🧩 **Component Integration**: Your tensors, layers, autograd, and optimizers work together seamlessly  
- 🔄 **Universal Foundations**: The same mathematical primitives power any neural architecture
- ⚡ **Framework Completeness**: You built a production-ready ML framework from scratch
- 🎯 **Systems Mastery**: You understand how every piece fits together under the hood

**The Achievement:** Build a complete GPT-style language model using **only components you implemented**. This proves your framework is real, complete, and ready for any ML task.

---

## Choose Your Learning Path

```{admonition} Three Ways to Engage with TinyTorch
:class: important

### [Quick Exploration](usage-paths/quick-exploration.md) *(5 minutes)*
*"I want to see what this is about"*
- Click and run code immediately in your browser (Binder)
- No installation or setup required
- Implement ReLU, tensors, neural networks interactively
- Perfect for getting a feel for the course

### [Serious Development](usage-paths/serious-development.md) *(8+ weeks)*
*"I want to build this myself"*
- Fork the repo and work locally with full development environment
- Build complete ML framework from scratch with `tito` CLI
- 16 progressive assignments from setup to language models
- Professional development workflow with automated testing

### [Classroom Use](usage-paths/classroom-use.md) *(Instructors)*
*"I want to teach this course"*
- Complete course infrastructure with NBGrader integration
- Automated grading for comprehensive testing
- Flexible pacing (8-16 weeks) with proven pedagogical structure
- Turn-key solution for ML systems education
```

---

## Ready to Start?

### Quick Taste: Try Module 1 Right Now
Want to see what TinyTorch feels like? **[Launch the Setup chapter](chapters/01-setup.md)** in Binder and implement your first TinyTorch function in 2 minutes!

---

## Acknowledgments

TinyTorch originated from CS249r: Tiny Machine Learning Systems at Harvard University. We're inspired by projects like [tinygrad](https://github.com/geohot/tinygrad), [micrograd](https://github.com/karpathy/micrograd), and [MiniTorch](https://minitorch.github.io/) that demonstrate the power of minimal implementations.


