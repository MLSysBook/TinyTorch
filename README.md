# TinyTorch: Build a Machine Learning System from Scratch

TinyTorch is a pedagogical project designed to accompany the *Machine Learning Systems* textbook. Inspired by OS and compiler courses where students build entire systems from first principles, TinyTorch guides you through building a complete ML training and inference runtime — from autograd to data pipelines, optimizers to profilers — **entirely from scratch**.

This is not a PyTorch tutorial. In TinyTorch, you’ll **write the components that frameworks like PyTorch are built on.**

---

## 🧠 Project Goals

- **Systems Understanding**: Learn how modern ML systems are constructed, not just how to use them.
- **Full-Stack ML Infra**: Build core components such as tensor classes, autograd engine, data loaders, optimizers, profilers, and benchmarking tools.
- **Modularity**: Understand abstraction boundaries in ML systems — what belongs in a model vs. a trainer vs. a runtime.
- **Infrastructure Thinking**: Learn how design decisions impact performance, reproducibility, deployment, and maintainability.

---

## 📚 Curriculum Integration

TinyTorch aligns with **Chapters 1–13** of the *Machine Learning Systems* textbook. Each chapter is paired with a corresponding system component you’ll implement. This forms a progressively richer ML infrastructure, culminating in a working system that can train and evaluate models on real data.

| Chapter | Component |
|--------|-----------|
| 1–2 | CLI + system architecture scaffold |
| 3 | Forward/backward engine for MLP |
| 4 | CNN layers: Conv2D, MaxPool |
| 5 | Configs + artifact logging |
| 6 | DataLoader and Dataset |
| 7 | Autograd engine |
| 8 | Training loop and optimizers |
| 9 | Profiling tools |
| 10 | Pruning and quantization |
| 11 | Custom matmul kernels |
| 12 | Benchmarking harness |
| 13 | Checkpointing + experiment reproducibility |

---

## 📦 Repository Structure (early preview)

tinytorch/
├── core/
│ ├── tensor.py # Autograd and data storage
│ ├── modules.py # Layers and model composition
│ ├── dataloader.py # Data loading and batching
│ ├── optimizer.py # SGD, Adam, etc.
│ ├── trainer.py # Training loop
│ ├── profiler.py # Runtime measurement
│ └── benchmark.py # Latency, throughput, energy est.
├── configs/
│ └── default.yaml
├── logs/
│ └── run_YYYY_MM_DD/
├── main.py # CLI entrypoint
└── README.md

---

## 🚀 Getting Started

Each chapter will introduce a new concept and a programming assignment that builds toward the full TinyTorch runtime.

**To begin:**
1. Clone this repository.
2. Follow the instructions in `chapters/01-intro/README.md` to implement the CLI.
3. Proceed incrementally — each module builds on the last.

---

## 💡 Philosophy

> “You don’t really understand a system until you’ve built it.”

TinyTorch aims to give students that understanding — by reconstructing the guts of machine learning infrastructure in a clean, minimal, yet fully functional form.

By the end, you’ll not only have built a system capable of training neural networks — you’ll understand every line that makes it work.

---

## 🧰 Requirements

- Python 3.8+
- NumPy
- (Optional: Numba for kernel acceleration)

No PyTorch. No TensorFlow. No black boxes.

---

## 📬 License and Attribution

TinyTorch is part of the *Machine Learning Systems* course and textbook by Vijay Janapa Reddi et al. Inspired by systems-style projects like xv6, PintOS, and cs231n assignments.

License: MIT
