# TinyTorch December 2024 Release

## 🎉 Announcement: TinyTorch is Ready for Community Review

**TL;DR**: Complete ML systems course (20 modules: Tensors → Transformers → Optimization) now available for pedagogical review. Seeking feedback on implementation quality, learning progression, and systems thinking approach.

---

## What is TinyTorch?

A Harvard University course teaching ML systems engineering by building a complete framework from scratch—no PyTorch or TensorFlow dependencies. Students implement every component: tensors, autograd, optimizers, CNNs, transformers, and optimization systems.

**North Star Goal**: Train CNNs on CIFAR-10 to 75%+ accuracy using only your own code + NumPy.

---

## What's Released (December 2024)

### ✅ Complete Implementation (All 20 Modules)

**Part I: Neural Network Foundations (01-07)**
- Tensors, Activations, Layers, Losses, Autograd, Optimizers, Training
- Milestone: Train XOR solver and MNIST classifier

**Part II: Computer Vision (08-09)**
- DataLoader, Spatial Convolutions (Conv2d, MaxPool2d)
- Milestone: CIFAR-10 @ 75%+ accuracy

**Part III: Language Models (10-14)**
- Tokenization, Embeddings, Attention, Transformers, KV-Caching
- Milestone: TinyGPT text generation

**Part IV: System Optimization (15-20)**
- Profiling, Acceleration, Quantization, Compression, Benchmarking, Capstone
- Milestone: TinyMLPerf optimization competition

### 📚 Complete Documentation

- **Jupyter Book**: Full course website with learning guides
- **Inline Tests**: Immediate validation in every module
- **Historical Milestones**: 6 demos (1957 Perceptron → 2024 Systems)
- **CLI System**: `tito` command-line tool for student workflow

### 🔧 Infrastructure

- NBGrader integration for classroom deployment
- Comprehensive testing suite (200+ tests)
- Student version generation tooling (untested)
- GitHub Actions for book deployment

---

## What We're Seeking Feedback On

### 1. **Pedagogical Progression**
- Do modules build logically? (Tensor → Autograd → CNNs → Transformers)
- Are learning objectives clear?
- Does "Build → Use → Understand" framework work?

### 2. **Implementation Quality**
- Code clarity and readability
- Educational value of inline tests
- Balance of guidance vs. challenge

### 3. **Systems Thinking**
- Memory management lessons
- Performance analysis integration
- Real-world ML engineering patterns

### 4. **Documentation**
- Jupyter Book clarity
- Module README completeness
- Getting started experience

---

## How to Review

### Quick Look (15 minutes)
```bash
# Browse the Jupyter Book
open https://mlsysbook.github.io/TinyTorch/
```

### Deep Dive (2-4 hours)
```bash
# Clone and run implementations
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
./setup-environment.sh
source activate.sh

# Try building a module
cd modules/source/01_tensor
python tensor_dev.py

# Check a milestone
cd ../../../milestones/03_1986_mlp_revival
python mlp_mnist.py
```

### Provide Feedback
- **GitHub Issues**: Specific bugs or improvements
- **GitHub Discussions**: General feedback, pedagogical suggestions
- **Email**: vijay@seas.harvard.edu for detailed reviews

---

## What's NOT Ready (Yet)

### 🚧 Student Version Tooling
- Scripts exist to generate student versions (remove solutions)
- **Not yet validated with real students**
- Planned for testing: January-March 2025

### 🚧 Classroom Deployment
- NBGrader workflows need instructor testing
- Grading rubrics need validation
- First classroom use: Fall 2025 (tentative)

### 🚧 Known Issues
- Modules 15-20 (optimization tier) are functional but need polish
- Some inline tests could use better explanations
- Book could use more cross-referencing

**We're being honest**: This release prioritizes complete implementations for review over polished student experience.

---

## Why Solutions Are Public

**Philosophy**: Modern ML education values pedagogy over secrecy.

**For Reviewers**: You need to see complete implementations to evaluate educational quality.

**For Students**: TinyTorch's progressive complexity makes copying ineffective. Module 05 (Autograd) exposes shallow understanding from earlier modules. Learning comes from struggle, not copying.

**For Instructors**: See [STUDENT_VERSION_TOOLING.md](STUDENT_VERSION_TOOLING.md) for classroom strategies.

---

## Timeline

- **December 2024**: Community review of complete implementations
- **January-March 2025**: Incorporate feedback, test student version tooling
- **April-May 2025**: Finalize classroom workflows
- **Fall 2025**: Potential first classroom deployment

---

## Who Should Review This?

### ✅ Perfect For:
- **ML educators** considering systems-focused courses
- **ML engineers** evaluating educational materials
- **Students** interested in deep understanding (not just API usage)
- **Open-source contributors** wanting to improve ML education

### ⚠️ Not Yet For:
- Instructors needing classroom-ready materials immediately
- Students expecting polished MOOC experience
- Organizations requiring production-ready framework

---

## Acknowledgments

**Created by**: [Prof. Vijay Janapa Reddi](https://vijay.seas.harvard.edu), Harvard University

**Inspired by**: FastAI (pedagogy), MiniTorch (Cornell), micrograd (Karpathy), tinygrad (Hotz)

**Community**: Thanks to early testers and feedback providers

---

## Links

- **Jupyter Book**: https://mlsysbook.github.io/TinyTorch/
- **GitHub**: https://github.com/mlsysbook/TinyTorch
- **Issues**: https://github.com/mlsysbook/TinyTorch/issues
- **Discussions**: https://github.com/mlsysbook/TinyTorch/discussions

---

## Quick Facts

- **20 modules** (Tensor → Capstone)
- **6 historical milestones** (1957 Perceptron → 2024 Systems)
- **200+ tests** (integration + unit)
- **Zero external ML dependencies** (only NumPy)
- **MIT License** (open source)
- **Harvard course** (academic-quality materials)

---

## Call to Action

**We need your feedback to make TinyTorch exceptional.**

- 📖 Read the book: https://mlsysbook.github.io/TinyTorch/
- 💻 Try the code: `git clone https://github.com/mlsysbook/TinyTorch.git`
- 💬 Share feedback: GitHub Issues or Discussions
- 🌟 Star the repo: Help others discover it
- 📢 Spread the word: Share with ML educators and engineers

**Goal**: Build the best ML systems education materials through community collaboration.

---

**Thank you for helping us improve ML systems education!**

— Prof. Vijay Janapa Reddi, Harvard University


