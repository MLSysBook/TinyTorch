---
title: "CNN"
description: "Convolutional Neural Network layers and operations"
difficulty: "Intermediate"
time_estimate: "2-4 hours"
prerequisites: []
next_steps: []
learning_objectives: ['Understand the convolution operation (sliding window, local connectivity, weight sharing)', 'Implement Conv2D with explicit for-loops (single channel, single filter, no stride/pad)', 'Visualize how convolution builds feature maps', 'Compose Conv2D with other layers to build a simple ConvNet', '(Stretch) Explore stride, padding, pooling, and multi-channel input']
---

# 🧠 Module X: CNN - Convolutional Neural Networks
---
**Course Navigation:** [Home](../intro.html) → [Module 6: 06 Cnn](#)

---


<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ ⭐⭐⭐⭐ | <strong>Time:</strong> 6-8 hours</p>
</div>



## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 6-8 hours
- **Prerequisites**: Tensor, Activations, Layers, Networks modules
- **Next Steps**: Training, Computer Vision modules

**Implement the core building block of modern computer vision: the convolutional layer.**

## 🎯 Learning Objectives
- Understand the convolution operation (sliding window, local connectivity, weight sharing)
- Implement Conv2D with explicit for-loops (single channel, single filter, no stride/pad)
- Visualize how convolution builds feature maps
- Compose Conv2D with other layers to build a simple ConvNet
- (Stretch) Explore stride, padding, pooling, and multi-channel input

## 🧠 Build → Use → Understand
1. **Build**: Implement Conv2D from scratch (for-loop)
2. **Use**: Compose Conv2D with ReLU, Flatten, Dense to build a ConvNet
3. **Understand**: Visualize and analyze how convolution works

## 📚 What You'll Build
- **Conv2D (for-loop):** The core operation, implemented by you
- **Conv2D Layer:** Wrap your function in a layer class
- **Simple ConvNet:** Compose Conv2D → ReLU → Flatten → Dense
- **Visualization:** See how the filter slides and builds the output

## 🛠️ Provided Functionality
- **Stride and Padding:** Provided as utilities or stretch goals
- **Multi-channel/Filter Support:** Provided or as stretch
- **Pooling (Max/Avg):** Optional, provided or as stretch
- **Flatten Layer:** Provided
- **Visualization:** Provided for learning
- **Tests:** Provided for feedback

## 🤔 Why Focus on the For-Loop?
Implementing the convolution for-loop is the best way to understand what makes CNNs powerful. You’ll see exactly how the filter slides, how local patterns are captured, and why this operation is so efficient for images. Other features (stride, padding, pooling) are important, but the core insight comes from building the basic operation yourself.

## 🚀 Getting Started
```bash
cd modules/cnn
jupyter notebook cnn_dev.ipynb  # or edit cnn_dev.py
```

## 📖 Module Structure
```
modules/cnn/
├── cnn_dev.py           # Main development file (work here!)
├── cnn_dev.ipynb        # Jupyter notebook version
├── tests/
│   └── test_cnn.py      # Tests for your implementation
├── README.md            # This file
```

## 🧪 Testing Your Implementation
```bash
# Run tests
python -m pytest tests/test_cnn.py -v
```

## 🌟 Stretch Goals
- Add stride and padding support
- Support multi-channel input/output
- Implement pooling layers
- Visualize learned filters and feature maps

## 💡 Key Insight
> **Convolution is a new, fundamental building block.**
> By implementing it yourself, you’ll understand the magic behind modern vision models! 

---

## 🚀 Interactive Learning

<div class="admonition tip">
<p class="admonition-title">💡 Try It Yourself</p>
<p>Ready to start building? Choose your preferred environment:</p>
</div>

### 🔧 **Builder Environment**
<div class="admonition note">
<p class="admonition-title">🏗️ Quick Start</p>
<p>Jump directly into the implementation with our guided builder:</p>
</div>

<a href="https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/06_cnn/cnn_dev.ipynb" target="_blank" class="btn btn-primary">
    🚀 Launch Builder
</a>

### 📓 **Jupyter Notebook**
<div class="admonition note">
<p class="admonition-title">📚 Full Development</p>
<p>Work with the complete development environment:</p>
</div>

<a href="https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/06_cnn/cnn_dev.ipynb" target="_blank" class="btn btn-success">
    📓 Open Jupyter
</a>

### 🎯 **Google Colab**
<div class="admonition note">
<p class="admonition-title">☁️ Cloud Environment</p>
<p>Use Google's cloud-based notebook environment:</p>
</div>

<a href="https://colab.research.google.com/github/MLSysBook/TinyTorch/blob/main/modules/source/06_cnn/cnn_dev.ipynb" target="_blank" class="btn btn-info">
    ☁️ Open in Colab
</a>

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/05_networks.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/07_dataloader.html" title="next page">Next Module →</a>
</div>
