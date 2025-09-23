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

```{admonition} What You'll Build: The Complete ML Evolution Story
:class: tip
**A complete ML framework from scratch** that recreates the history of ML breakthroughs:

**🧠 MLP Era (1980s): The Foundation**
- **Train MLPs to 52.7% accuracy on CIFAR-10** (the baseline everyone tried to beat)
- Implement automatic differentiation from first principles
- Master gradient-based optimization with SGD and Adam

**📡 CNN Revolution (1989-1998): Spatial Intelligence**
- **LeNet-1 (1989)**: Build the first successful CNN architecture (39.4% accuracy)
- **LeNet-5 (1998)**: Implement the classic CNN that established the standard (47.5% accuracy)
- **Modern CNNs**: Push beyond MLPs with optimized architectures (55%+ achievable)

**🔥 Transformer Era (2017-present): Language & Beyond**
- **TinyGPT**: Complete language models using your vision framework
- **Universal Architecture**: 95% component reuse from vision to language
- **Modern ML Systems**: Full pipeline from data loading to deployment

**Result:** You experience firsthand how ML evolved from simple perceptrons to modern AI systems, implementing every breakthrough yourself. All 16 modules pass comprehensive tests with 100% health status.
```

_Understanding how to build ML systems makes you a more effective ML engineer._

```{admonition} The Perfect Learning Combination
:class: note
TinyTorch was designed as the hands-on lab companion to [**Machine Learning Systems**](https://mlsysbook.ai) by [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu) (Harvard). The book teaches you ML systems **theory and principles** - TinyTorch lets you **implement and experience** those concepts firsthand. Together, they provide complete ML systems mastery.
```

---

## The Historic Journey: From MLPs to Modern AI

TinyTorch recreates the actual progression of machine learning breakthroughs. You don't just learn modern AI - you **experience the evolution that created it**:

```python
🧠 MLP Era (1980s):           📡 CNN Revolution (1989):      🔥 Transformer Era (2017):
├── class MLP:                ├── class LeNet1:               ├── class TinyGPT:
│   def forward(self, x):     │   def forward(self, x):       │   def forward(self, x):
│   h = x.reshape(batch,-1)   │   h = self.conv1(x)          │   h = self.embed(x)
│   h = self.fc1(h)          │   h = self.pool(h)           │   h = self.attention(h)
│   return self.fc2(h)        │   h = self.conv2(h)          │   return self.lm_head(h)
│                             │   return self.fc(h.flat())    │
│ Result: 52.7% CIFAR-10     │ Result: 47.5% CIFAR-10       │ Result: Language generation
└── "Good, but can we do     └── "Spatial features help!"   └── "Universal intelligence!"
    better with images?"

The SAME tensor operations power all three eras - you build them once, use everywhere.
```

**The ML Evolution Story:**
- **1980s**: MLPs could learn, but struggled with complex patterns
- **1989**: LeNet-1 proved convolutions extract spatial features  
- **1998**: LeNet-5 established CNNs as the vision standard
- **2012**: AlexNet showed deep CNNs dominate computer vision
- **2017**: Transformers unified vision AND language processing
- **Today**: Same mathematical foundations power all AI systems

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

## 📚 Course Journey: Recreating ML History in 16 Modules

```{admonition} 🧠 MLP Era Foundation (Modules 1-4)
:class: note
**1. Setup** • **2. Tensors** • **3. Activations** • **4. Layers**

Build the mathematical foundation that powered 1980s neural networks: tensor operations, nonlinear functions, and dense layers.
```

```{admonition} 🧠 MLP Intelligence (Modules 5-6) 
:class: note
**5. Dense Networks** • **6. Training Loops**

Complete the MLP era: sequential networks and training systems that achieve **52.7% CIFAR-10 accuracy** - the baseline everyone tried to beat.
```

```{admonition} 📡 CNN Revolution (Modules 7-8)
:class: note
**7. Spatial Operations** • **8. DataLoader**

Enter the 1989 CNN breakthrough: convolutional layers and real data loading. Build **LeNet-1** (39.4%) and **LeNet-5** (47.5%) - witness the spatial intelligence revolution.
```

```{admonition} 🔥 Modern Training Systems (Modules 9-12)
:class: note
**9. Autograd** • **10. Optimizers** • **11. Training** • **12. Attention**

Master the systems that power modern AI: automatic differentiation, advanced optimizers, and attention mechanisms. Push CNNs beyond MLP baselines.
```

```{admonition} 🚀 Production Systems (Modules 13-15)
:class: note
**13. Compression** • **14. Kernels** • **15. MLOps**

Scale to production: model optimization, high-performance computing, and deployment monitoring with real-world patterns.
```

```{admonition} 🤖 Universal Intelligence (Module 16)
:class: note
**16. TinyGPT**

The culmination: GPT-style transformers for language generation using **95% of your vision components**. Prove your framework is universal - the same foundations power vision AND language.
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

### 🔥 TinyGPT: Proving Framework Universality

TinyGPT is your **capstone achievement** - demonstrating that the same foundations power all modern AI:

**The Historical Proof:**
- **1980s MLP components** → **1989 CNN revolution** → **2017 Transformer era**
- **95% component reuse**: Your tensors, layers, and training systems work across all three eras
- **Universal mathematics**: The same operations that power MLPs (52.7%) and CNNs (LeNet-5: 47.5%) also power language models

**What TinyGPT Proves:**
- **Framework Universality**: Vision and language use identical mathematical foundations  
- **Component Integration**: All 16 modules work together seamlessly across domains
- **Systems Mastery**: You understand how modern AI builds on historical breakthroughs
- **Career Readiness**: You can implement any architecture from any era

**The Achievement:** Build GPT using components you designed for computer vision. This proves you didn't just learn isolated techniques - you built a complete, universal ML framework capable of any task.

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


