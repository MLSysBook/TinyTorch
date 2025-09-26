# The TinyTorch Vision

**Training ML Systems Engineers: From Computer Vision to Language Models**

---

## The Problem We're Solving

The ML field has a critical gap: **most education teaches you to use frameworks, not build them.**

### Traditional ML Education:
```python
import torch
import torch.nn as nn
model = nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters())
```

**Questions students can't answer:**
- Why does Adam use 3× more memory than SGD?
- How does `loss.backward()` actually compute gradients?
- When should you use gradient accumulation vs larger batch sizes?
- Why do attention mechanisms limit context length?

### The TinyTorch Difference:
```python
class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight + self.bias  # YOU implemented @
    
    def backward(self, grad_output):
        # YOU understand exactly how gradients flow
        self.weight.grad = x.T @ grad_output
        return grad_output @ self.weight.T
```

**Questions students CAN answer:**
- Exactly how automatic differentiation works
- Why certain optimizers use more memory
- How to debug training instability
- When to make performance vs accuracy trade-offs

---

## What We Teach: Systems Thinking

### Beyond Algorithms: System-Level Understanding

**Memory Management:**
- Why Adam needs 3× parameter memory (parameters + momentum + variance)
- How attention matrices scale O(N²) with sequence length
- When gradient accumulation saves memory vs compute trade-offs

**Performance Analysis:**
- Why naive convolution is 100× slower than optimized versions
- How cache misses destroy performance in matrix operations
- When vectorization provides 10-100× speedups

**Production Trade-offs:**
- SGD vs Adam: convergence speed vs memory constraints
- Gradient checkpointing: trading compute for memory
- Mixed precision: 2× memory savings with accuracy considerations

**Hardware Awareness:**
- How memory bandwidth limits ML performance
- Why GPU utilization matters more than peak FLOPS
- When distributed training becomes necessary

---

## Target Audience: Future ML Systems Engineers

### Perfect For:

**Computer Science Students**
- Going beyond "use PyTorch" to "understand PyTorch"
- Building portfolio projects that demonstrate deep system knowledge
- Preparing for ML engineering roles (not just data science)

**Software Engineers → ML Engineers**
- Leveraging existing programming skills for ML systems
- Understanding performance, debugging, and optimization
- Learning production ML patterns and infrastructure

**ML Practitioners**
- Moving from model users to model builders
- Debugging training issues at the systems level  
- Optimizing models for production deployment

**Researchers & Advanced Users**
- Implementing custom operations and architectures
- Understanding framework limitations and workarounds
- Building specialized ML systems for unique domains

### Career Transformation:

**Before TinyTorch:** "I can train models with PyTorch"
**After TinyTorch:** "I can build and optimize ML systems"

You become the person your team asks:
- *"Why is our training bottlenecked?"* 
- *"Can we fit this model in memory?"*
- *"How do we implement this research paper?"*
- *"What's the best architecture for our constraints?"*

---

## Pedagogical Philosophy: Build → Use → Understand

### 1. Build First
Every component implemented from scratch:
- Tensors with broadcasting and memory management
- Automatic differentiation with computational graphs
- Optimizers with state management and memory profiling
- Complete training loops with checkpointing and monitoring

### 2. Use Immediately
No toy examples - recreate ML history with real results:
- **MLP Era**: Train MLPs to 52.7% CIFAR-10 accuracy (the baseline that motivated CNNs)
- **CNN Revolution**: Build LeNet-1 (39.4%) and LeNet-5 (47.5%) - witness the breakthrough
- **Modern CNNs**: Push beyond MLPs with optimized architectures (75%+ achievable)
- **Transformer Era**: Language models using 95% vision framework reuse

### 3. Understand Systems
Connect implementations to production reality:
- How your tensor maps to PyTorch's memory model
- Why your optimizer choices affect GPU utilization
- How your autograd compares to production frameworks
- When your implementations would need modification at scale

### 4. Reflect on Trade-offs
ML Systems Thinking sections in every module:
- Memory vs compute trade-offs in different architectures
- Accuracy vs efficiency considerations for deployment  
- Debugging strategies for common production issues
- Framework design principles and their implications

---

## Unique Value Proposition

### What Makes TinyTorch Different:

**Systems-First Approach**
- Not just "how does attention work" but "why does attention scale O(N²) and how do production systems handle this?"
- Not just "implement SGD" but "when do you choose SGD vs Adam in production?"

**Production Relevance**
- Memory profiling, performance optimization, deployment patterns
- Real datasets, realistic scale, professional development workflow
- Connection to industry practices and framework design decisions

**Framework Generalization**
- 20 modules that build ONE cohesive ML framework supporting vision AND language
- 95% component reuse from computer vision to language models
- Professional package structure with CLI tools and testing

**Proven Pedagogy**
- Build → Use → Understand cycle creates deep intuition
- Immediate testing and feedback for every component
- Progressive complexity with solid foundations
- NBGrader integration for classroom deployment

---

## Learning Outcomes: Becoming an ML Systems Engineer

### Technical Mastery
- **Implement any ML paper** from first principles
- **Debug training issues** at the systems level
- **Optimize models** for production deployment
- **Profile and improve** ML system performance
- **Design custom architectures** for specialized domains
- **Understand framework generalization** across vision and language

### Systems Understanding 
- **Memory management** in ML frameworks
- **Computational complexity** vs real-world performance
- **Hardware utilization** patterns and optimization
- **Distributed training** challenges and solutions
- **Production deployment** considerations and trade-offs

### Professional Skills
- **Test-driven development** for ML systems
- **Performance profiling** and optimization techniques
- **Code organization** and package development
- **Documentation** and API design
- **MLOps** and production monitoring

### Career Impact
- **Technical interviews**: Demonstrate deep ML systems knowledge
- **Job opportunities**: Qualify for ML engineer (not just data scientist) roles
- **Team leadership**: Become the go-to person for ML systems questions
- **Research ability**: Implement cutting-edge papers independently
- **Entrepreneurship**: Build ML products with full-stack understanding

---

## Success Stories: What Students Say

*"Finally understood what happens when I call `loss.backward()` - now I can debug gradient issues instead of just hoping they go away."*

*"Built my own attention mechanism from scratch, then extended my vision framework to language models with 95% component reuse. When GPT-4 came out, I actually understood both the technical details AND the framework unification."*

*"Got hired as an ML engineer specifically because I could explain how optimizers work at the memory level during the technical interview."*

*"Used TinyTorch concepts to optimize our production training pipeline for both vision and language models - saved 40% on cloud costs by understanding memory bottlenecks across modalities."*

*"Implemented a custom loss function for our research project in 30 minutes instead of spending days figuring out PyTorch internals."*

---

## Ready to Become an ML Systems Engineer?

**TinyTorch transforms ML users into ML builders.**

Stop wondering how frameworks work. Start building them.

**[Begin Your Journey →](chapters/00-introduction.md)**

---

*TinyTorch: Because understanding how to build ML systems makes you a more effective ML engineer.*