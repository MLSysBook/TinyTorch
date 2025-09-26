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

**MLP Era (1980s): The Foundation**
- **Train MLPs to 95%+ accuracy on MNIST** (handwritten digit recognition)
- Implement automatic differentiation from first principles
- Master gradient-based optimization with SGD and Adam

**CNN Revolution (2012): Spatial Intelligence**
- **Modern CNNs**: Build convolutional networks from scratch
- **CIFAR-10 Achievement**: 75%+ accuracy on real images (our north star goal)
- **Spatial Operations**: Conv2d, MaxPool2d, and efficient data pipelines

**Transformer Era (2017-present): Language & Beyond**
- **TinyGPT**: Complete language models using your vision framework
- **Universal Architecture**: 95% component reuse from vision to language
- **Modern ML Systems**: Full pipeline from data loading to deployment

**Result:** You experience firsthand how ML evolved from simple perceptrons to modern AI systems, implementing every breakthrough yourself. All 20 modules pass comprehensive tests with 100% health status.
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
MLP Era (1980s):              CNN Revolution (2012):         Transformer Era (2017):
├── class MLP:                ├── class CNN:                 ├── class TinyGPT:
│   def forward(self, x):     │   def forward(self, x):      │   def forward(self, x):
│   h = x.reshape(batch,-1)   │   h = self.conv1(x)          │   h = self.embed(x)
│   h = self.fc1(h)           │   h = self.pool(h)           │   h = self.attention(h)
│   return self.fc2(h)        │   h = self.conv2(h)          │   return self.lm_head(h)
│                             │   return self.fc(h.flat())   │
│ Result: 95%+ MNIST          │ Result: 75%+ CIFAR-10        │ Result: Language generation
└── "Foundation for all       └── "Spatial features excel"  └── "Universal architecture"
    neural networks"

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

### Track Your Capabilities

TinyTorch uses a [checkpoint system](checkpoint-system.md) to track your progress through **ML systems engineering capabilities**:
- **Foundation** → Core ML primitives and setup
- **Architecture** → Neural network building  
- **Training** → Model training pipeline
- **Inference** → Deployment and optimization
- **Serving** → Complete system integration

Use `tito checkpoint status` to see your progress anytime!

### Beyond Code: Systems Intuition

Each module includes **ML Systems Thinking** sections that connect your implementations to production reality:
- *"How does your tensor implementation compare to PyTorch's memory management?"*
- *"When would you choose SGD over Adam in production training?"* 
- *"How do frameworks handle the quadratic memory scaling of attention?"*
- *"What happens to your autograd implementation under distributed training?"*

These aren't just academic questions - they're the system-level challenges that ML engineers solve every day.

---

## Who This Is For

### Perfect For:
- **CS students** who want to understand ML systems beyond high-level APIs
- **Software engineers** transitioning to ML engineering roles
- **ML practitioners** who want to optimize and debug production systems
- **Researchers** who need to implement custom operations and architectures
- **Anyone curious** about how PyTorch/TensorFlow actually work under the hood

### Prerequisites:
- **Python programming** (comfortable with classes, functions, basic NumPy)
- **Linear algebra basics** (matrix multiplication, gradients)
- **Learning mindset** - we'll teach you everything else!

### Career Impact:
After TinyTorch, you'll be the person your team asks:
- *"Why is this training so slow?"* (You'll know how to profile and optimize)
- *"Can we fit this model in GPU memory?"* (You'll understand memory trade-offs)  
- *"What's the best optimizer for this problem?"* (You'll know the system implications)

---

## Progressive Learning Journey: 20 Modules

```{admonition} ML Systems Engineering Focus 
:class: important
**20 modules** that build complete ML systems understanding through implementation:
From tensors to transformers, profiling to production deployment!
```

```{admonition} Neural Network Foundations (Modules 1-8)
:class: note
**1. Setup** • **2. Tensor** • **3. Activations** • **4. Layers** • **5. Losses** • **6. Autograd** • **7. Optimizers** • **8. Training**

**Foundation**: Complete neural network training capability with gradient computation!
- **Module 2**: Tensor operations and autograd from the start
- **Module 6**: Computational graphs and automatic differentiation
- **Module 8**: **Train XOR and MNIST with complete framework!**
```

```{admonition} Computer Vision (Modules 9-10)
:class: note
**9. Spatial** • **10. DataLoader**

Add convolutional intelligence: Conv2d, MaxPool2d, and efficient data loading.
**Result**: Train CNNs on CIFAR-10 to 75%+ accuracy!
```

```{admonition} Language Models (Modules 11-14)
:class: note
**11. Tokenization** • **12. Embeddings** • **13. Attention** • **14. Transformers**

Universal intelligence: Build GPT-style language models using your vision infrastructure.
**Result**: Complete TinyGPT using 95% of your vision components!
```

```{admonition} Systems Optimization (Modules 15-20)
:class: note
**15. Profiling** • **16. Acceleration** • **17. Quantization** • **18. Compression** • **19. Caching** • **20. Benchmarking**

Production systems: Optimize, profile, and compete in ML systems performance.
**Result**: TinyMLPerf competition framework for optimization excellence!
```

---

## Complete System Integration

**This isn't 20 separate exercises.** Every component you build integrates into one fully functional ML framework with universal foundations:

```{admonition} How It All Connects
:class: important

```{mermaid}
flowchart TD
    A[01_setup<br/>Environment & CLI] --> B[02_tensor<br/>Tensor + Autograd<br/>GRADIENTS FROM START]
    
    B --> C[03_activations<br/>ReLU + Softmax<br/>Nonlinearity Functions]
    
    C --> D[04_layers<br/>Linear + Module<br/>Building Blocks]
    
    D --> E[05_losses<br/>MSE + CrossEntropy<br/>Loss Functions]
    
    E --> F[06_autograd<br/>Computation Graphs<br/>Automatic Differentiation]
    
    F --> G[07_optimizers<br/>SGD + Adam<br/>Gradient Descent]
    
    G --> H[08_training<br/>Complete Training<br/>TRAIN NETWORKS NOW]
    
    H --> I[09_spatial<br/>Conv2d + MaxPool2d<br/>Spatial Operations]
    
    H --> J[10_dataloader<br/>Generic DataLoader<br/>Efficient Batching]
    
    I --> J
    J --> K[Vision Applications<br/>CIFAR-10: 75%+ Accuracy]
    
    H --> L[11_tokenization<br/>Text Processing]
    L --> M[12_embeddings<br/>Token Embeddings]
    M --> N[13_attention<br/>Multi-Head Attention]
    N --> O[14_transformers<br/>TinyGPT<br/>Language Models]
    
    K --> P[15_profiling<br/>Performance Analysis]
    O --> P
    P --> Q[16_acceleration<br/>Hardware Optimization]
    Q --> R[17_quantization<br/>Model Compression]
    R --> S[18_compression<br/>Pruning & Distillation]
    S --> T[19_caching<br/>Memory Optimization]
    T --> U[20_benchmarking<br/>TinyMLPerf Competition]
    
    style H fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff
    style K fill:#4ecdc4,stroke:#333,stroke-width:3px,color:#fff
    style O fill:#45b7d1,stroke:#333,stroke-width:3px,color:#fff
    style U fill:#ffd93d,stroke:#333,stroke-width:3px,color:#333
```

**Result:** Every component you build converges into TinyGPT - proving your framework is complete and production-ready.
```

### TinyGPT: Proving Framework Universality

TinyGPT is your **capstone achievement** - demonstrating that the same foundations power all modern AI:

**The Historical Proof:**
- **1980s MLP components** → **1989 CNN revolution** → **2017 Transformer era**
- **95% component reuse**: Your tensors, layers, and training systems work across all three eras
- **Universal mathematics**: The same operations that power MLPs (52.7%) and CNNs (LeNet-5: 47.5%) also power language models

**What TinyGPT Proves:**
- **Framework Universality**: Vision and language use identical mathematical foundations  
- **Component Integration**: All 20 modules work together seamlessly across domains
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
- 20 progressive assignments from setup to system optimization
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


