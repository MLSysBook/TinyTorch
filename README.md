# Tiny🔥Torch: Build ML Systems from Scratch

> A hands-on systems course where you implement every component of a modern ML system

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![nbdev](https://img.shields.io/badge/built%20with-nbdev-orange.svg)](https://nbdev.fast.ai/)

> **Disclaimer**: TinyTorch is an educational framework developed independently and is not affiliated with or endorsed by Meta or the PyTorch project.

**Tiny🔥Torch** is a hands-on companion to [*Machine Learning Systems*](https://mlsysbook.ai), providing practical coding exercises that complement the book's theoretical foundations. Rather than just learning *about* ML systems, you'll build one from scratch—implementing everything from tensors and autograd to hardware-aware optimization and deployment systems.

## 🎯 What You'll Build

By completing this course, you will have implemented a complete ML system:

**Core Framework** → **Training Pipeline** → **Production System**
- ✅ Tensors with automatic differentiation
- ✅ Neural network layers (MLP, CNN, Transformer)
- ✅ Training loops with optimizers (SGD, Adam)
- ✅ Data loading and preprocessing pipelines
- ✅ Model compression (pruning, quantization)
- ✅ Performance profiling and optimization
- ✅ Production deployment and monitoring

## 🚀 Quick Start

**Ready to build? Choose your path:**

### 🏃‍♂️ I want to start building now
→ **[QUICKSTART.md](QUICKSTART.md)** - Get coding in 10 minutes

### 📚 I want to understand the full course structure  
→ **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Complete learning roadmap

### 🔍 I want to see the course in action
→ **[modules/setup/](modules/setup/)** - Browse the first module

## 🎓 Learning Approach

**Module-First Development**: Each module is self-contained with its own notebook, tests, and learning objectives. You'll work in Jupyter notebooks using the [nbdev](https://nbdev.fast.ai/) workflow to build a real Python package.

**The Cycle**: `Write Code → Export → Test → Next Module`

```bash
# The rhythm you'll use for every module
jupyter lab tensor_dev.ipynb    # Write & test interactively  
python bin/tito.py sync         # Export to Python package
python bin/tito.py test         # Verify implementation
```

## 📚 Course Structure

| Phase | Modules | What You'll Build |
|-------|---------|-------------------|
| **Foundation** | Setup, Tensor, Autograd | Core mathematical engine |
| **Neural Networks** | MLP, CNN | Learning algorithms |
| **Training Systems** | Data, Training, Config | End-to-end pipelines |
| **Production** | Profiling, Compression, MLOps | Real-world deployment |

**Total Time**: 40-80 hours over several weeks • **Prerequisites**: Python basics

## 🛠️ Key Commands

```bash
python bin/tito.py info               # Check progress
python bin/tito.py sync               # Export notebooks  
python bin/tito.py test --module [name]  # Test implementation
```

## 🌟 Why Tiny🔥Torch?

**Systems Engineering Principles**: Learn to design ML systems from first principles
**Hardware-Software Co-design**: Understand how algorithms map to computational resources  
**Performance-Aware Development**: Build systems optimized for real-world constraints
**End-to-End Systems**: From mathematical foundations to production deployment

## 📖 Educational Approach

**Companion to [Machine Learning Systems](https://mlsysbook.ai)**: This course provides hands-on implementation exercises that bring the book's concepts to life through code.

**Learning by Building**: Following the educational philosophy of [Karpathy's micrograd](https://github.com/karpathy/micrograd), we learn complex systems by implementing them from scratch.

**Real-World Systems**: Drawing from production [PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io/) architectures to understand industry-proven design patterns.

## 🤔 Frequently Asked Questions

<details>
<summary><strong>Why should students build TinyTorch if AI agents can already generate similar code?</strong></summary>

Even though large language models can generate working ML code, building systems from scratch remains *pedagogically essential*:

- **Understanding vs. Using**: AI-generated code shows what works, but not *why* it works. TinyTorch teaches students to reason through tensor operations, memory flows, and training logic.
- **Systems Literacy**: Debugging and designing real ML pipelines requires understanding abstractions like autograd, data loaders, and parameter updates, not just calling APIs.
- **AI-Augmented Engineers**: The best AI engineers will *collaborate with* AI tools, not rely on them blindly. TinyTorch trains students to read, verify, and modify generated code responsibly.
- **Intentional Design**: Systems thinking can’t be outsourced. TinyTorch helps learners internalize how decisions about data layout, execution, and precision affect performance.

</details>

<details>
<summary><strong>Why not just study the PyTorch or TensorFlow source code instead?</strong></summary>

Industrial frameworks are optimized for scale, not clarity. They contain thousands of lines of code, hardware-specific kernels, and complex abstractions. 

TinyTorch, by contrast, is intentionally **minimal** and **educational** — like building a kernel in an operating systems course. It helps learners understand the essential components and build an end-to-end pipeline from first principles.

</details>

<details>
<summary><strong>Isn't it more efficient to just teach ML theory and use existing frameworks?</strong></summary>

Teaching only the math without implementation leaves students unable to debug or extend real-world systems. TinyTorch bridges that gap by making ML systems tangible:

- Students learn by doing, not just reading.
- Implementing backpropagation or a training loop exposes hidden assumptions and tradeoffs.
- Understanding how layers are built gives deeper insight into model behavior and performance.

</details>

<details>
<summary><strong>Why use TinyML in a Machine Learning Systems course?</strong></summary>

TinyML makes systems concepts concrete. By running ML models on constrained hardware, students encounter the real-world limits of memory, compute, latency, and energy — exactly the challenges modern ML engineers face at scale.

- ⚙️ **Hardware constraints** expose architectural tradeoffs that are hidden in cloud settings.
- 🧠 **Systems thinking** is deepened by understanding how models interact with sensors, microcontrollers, and execution runtimes.
- 🌍 **End-to-end ML** becomes tangible — from data ingestion to inference.

TinyML isn’t about toy problems — it’s about simplifying to the point of *clarity*, not abstraction. Students see the full system pipeline, not just the cloud endpoint.

</details>

<details>
<summary><strong>What do the hardware kits add to the learning experience?</strong></summary>

The hardware kits are where learning becomes **hands-on and embodied**. They bring several pedagogical advantages:

- 🔌 **Physicality**: Students see real data flowing through sensors and watch ML models respond — not just print outputs.
- 🧪 **Experimentation**: Kits enable tinkering with latency, power, and model size in ways that are otherwise abstract.
- 🚀 **Creativity**: Students can build real applications — from gesture detection to keyword spotting — using what they learned in TinyTorch.

The kits act as *debuggable, inspectable deployment targets*. They reveal what’s easy vs. hard in ML deployment — and why hardware-aware design matters.

</details>

---
## 🤝 Contributing

We welcome contributions! Whether you're a student who found a bug or an instructor wanting to add modules, see our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Ready to start building?** → [**QUICKSTART.md**](QUICKSTART.md) 🚀
