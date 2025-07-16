# CNN - Convolutional Neural Networks

Welcome to the CNN module! Here you'll implement the core building block of modern computer vision: the convolutional layer.

```{admonition} 🎯 Learning Goals
:class: tip
- Understand the convolution operation and its importance in computer vision
- Implement Conv2D with explicit for-loops to understand the sliding window mechanism
- Build convolutional layers that can detect spatial patterns in images
- Compose Conv2D with other layers to build complete convolutional networks
- See how convolution enables parameter sharing and translation invariance
```


## Build → Use → Reflect
1. **Build**: Conv2D layer using sliding window convolution from scratch
2. **Use**: Transform images and see feature maps emerge
3. **Reflect**: How CNNs learn hierarchical spatial patterns

## What You'll Learn
By the end of this module, you'll understand:
- How convolution works as a sliding window operation
- Why convolution is perfect for spatial data like images
- How to build learnable convolutional layers
- The CNN pipeline: Conv2D → Activation → Flatten → Dense
- How parameter sharing makes CNNs efficient
## 🚀 Interactive Learning

Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/05_cnn/cnn_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/05_cnn/cnn_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/05_cnn/cnn_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

