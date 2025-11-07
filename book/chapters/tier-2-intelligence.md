# Intelligence Tier: Vision & Language

**Modules 08-13 | Estimated Time: 30-40 hours**

---

## Overview

The Intelligence Tier teaches you to build modern AI systems for computer vision and natural language processing. You'll implement CNNs for image understanding and Transformers for language modeling—the architectures powering today's AI applications.

By completing Intelligence, you'll understand how AI systems process real-world data and generate intelligent outputs.

---

## What You'll Build

### Modern AI Architectures

**Module 08: DataLoader**  
Production data pipelines for efficient batch loading and preprocessing

**Module 09: Spatial (CNNs)**  
Convolutional neural networks for computer vision tasks

**Module 10: Tokenization**  
Convert text to numerical representations for language models

**Module 11: Embeddings**  
Dense vector representations for words and tokens

**Module 12: Attention**  
Attention mechanisms that learn what information matters

**Module 13: Transformers**  
Complete transformer architecture (GPT-style) for language modeling

---

## Learning Approach

### Build → Use → Apply

In Intelligence Tier, you'll follow this pattern:

1. **Build** each architecture from mathematical foundations
2. **Use** it on real datasets (CIFAR-10 for vision, Shakespeare for language)
3. **Apply** to practical problems (image classification, text generation)

This tier emphasizes **application**—you'll see your implementations solve real problems.

---

## Why This Matters

### Industry Relevance

The architectures you'll build power production systems:

- **CNNs**: Tesla Autopilot, medical imaging, facial recognition
- **Transformers**: ChatGPT, GitHub Copilot, Google Translate
- **Attention**: All modern NLP systems use attention mechanisms
- **Embeddings**: Recommendation systems, semantic search

Understanding these architectures internally enables you to optimize, debug, and extend them.

### Real Data, Real Systems

Unlike Foundation Tier's mathematical focus, Intelligence Tier uses production datasets:

- **CIFAR-10**: 60,000 color images across 10 classes
- **Shakespeare**: Real text corpus for language modeling
- **Real preprocessing**: Normalization, augmentation, tokenization

This prepares you for real-world ML engineering.

---

## Module Roadmap

### Advanced (Modules 08-13)

**08. DataLoader** - 5-6 hours  
Build data pipelines with batching, shuffling, and preprocessing

**09. Spatial (CNNs)** - 6-8 hours  
Implement Conv2d, MaxPool, and complete CNN architectures

**10. Tokenization** - 3-4 hours  
Character-level and subword tokenization for NLP

**11. Embeddings** - 3-4 hours  
Token and positional embeddings for transformers

**12. Attention** - 6-8 hours  
Scaled dot-product attention and multi-head attention mechanisms

**13. Transformers** - 8-10 hours  
Complete GPT-style transformer for text generation

---

## Tier Milestones

**After completing specific modules**, unlock historical demonstrations:

**Module 09 (CNNs)**: **1998: LeNet-5**  
Use YOUR CNN to classify CIFAR-10 images

**Module 13 (Transformers)**: **2017: Vaswani's Transformer**  
Use YOUR transformer for text generation:

```bash
# ChatGPT-style demo
python milestones/05_2017_transformer/vaswani_chatgpt.py

# Code completion demo
python milestones/05_2017_transformer/vaswani_copilot.py
```

**Expected Results:**
- CNN achieves 65-75% accuracy on CIFAR-10
- Transformer generates coherent text after 10-15 minutes of training

These milestones prove your implementations work at production scale.

---

## Prerequisites

**Before starting Intelligence Tier:**

- Complete Foundation Tier (Modules 01-07)
- Understanding of:
  - Tensor operations and backpropagation
  - Training loops and optimization
  - Loss functions and gradient descent

**Verify Foundation is complete:**

```bash
tito test 01 02 03 04 05 06 07
```

All tests should pass before beginning Module 08.

---

## Getting Started

**Ready to build modern AI?**

Begin with Module 08: DataLoader - the data pipeline infrastructure that feeds neural networks.

[Start Module 08: DataLoader →](08-dataloader.html)

---

## Additional Resources

**Need Help?**
- [Ask in GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)
- [View Course Introduction](00-introduction.html)
- [Review Foundation Tier](tier-1-foundation.html)

**Related Content:**
- [Historical Milestones](milestones.html) - See transformer breakthroughs
- [Testing Framework](../testing-framework.html) - Validate CNN/Transformer implementations
- [Progress Tracking](../learning-progress.html) - Monitor your journey

