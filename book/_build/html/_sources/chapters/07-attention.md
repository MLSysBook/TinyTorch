---
title: "Attention"
description: "Core attention mechanism and masking utilities"
difficulty: "⭐⭐⭐"
time_estimate: "4-5 hours"
prerequisites: []
next_steps: []
learning_objectives: []
---

# Module: Attention

```{div} badges
⭐⭐⭐ | ⏱️ 4-5 hours
```


## 📊 Module Info
- **Difficulty**: ⭐⭐⭐ Advanced
- **Time Estimate**: 4-5 hours
- **Prerequisites**: Tensor module
- **Next Steps**: Training, Transformers modules

Build the core attention mechanism that powers modern AI! This module implements the fundamental scaled dot-product attention that's used in ChatGPT, BERT, GPT-4, and virtually all state-of-the-art AI systems.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Master the attention formula**: Understand and implement `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Build self-attention**: Create the core component that enables global context understanding
- **Control information flow**: Implement masking for causal, padding, and bidirectional attention
- **Visualize attention patterns**: See what the model "pays attention to"
- **Understand modern AI**: Grasp the mechanism that revolutionized natural language processing

## 🧠 Build → Use → Understand

This module follows TinyTorch's **Build → Use → Understand** framework:

1. **Build**: Implement the core attention mechanism and masking utilities from mathematical foundations
2. **Use**: Apply attention to sequence tasks and visualize attention patterns
3. **Understand**: How attention enables dynamic, global context modeling that powers modern AI

## 📚 What You'll Build

### Scaled Dot-Product Attention
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    The fundamental attention operation:
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    This exact function powers ChatGPT, BERT, and all transformers.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = softmax(scores)
    return attention_weights @ V, attention_weights
```

### Self-Attention Wrapper
```python
class SelfAttention:
    """
    Convenient wrapper for self-attention where Q=K=V.
    The most common use case in transformer models.
    """
    def __init__(self, d_model):
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        # Self-attention: Q = K = V = x
        return scaled_dot_product_attention(x, x, x, mask)
```

### Attention Masking
```python
# Causal masking (GPT-style: can't see future tokens)
causal_mask = create_causal_mask(seq_len)

# Padding masking (ignore padding tokens)
padding_mask = create_padding_mask(lengths, max_length)

# Bidirectional masking (BERT-style: can see all tokens)
bidirectional_mask = create_bidirectional_mask(seq_len)
```

## 🔬 Key Concepts

### Why Attention Revolutionized AI
- **Global connectivity**: Unlike CNNs, attention connects any two positions directly
- **Dynamic weights**: Attention adapts to input content, not fixed like convolution kernels
- **Parallel processing**: Unlike RNNs, all positions computed simultaneously
- **Interpretability**: You can visualize what the model pays attention to

### The Attention Formula Explained
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What information is available?"  
- V (Value): "What is the actual content?"
- √d_k scaling: Prevents extreme softmax values
```

### Attention vs Convolution
| Aspect | Convolution | Attention |
|--------|-------------|-----------|
| **Receptive field** | Local, grows with depth | Global from layer 1 |
| **Computation** | O(n) with kernel size | O(n²) with sequence length |
| **Weights** | Fixed learned kernels | Dynamic input-dependent |
| **Best for** | Spatial data (images) | Sequential data (text) |

### Real-World Applications
- **Language Models**: GPT, BERT, ChatGPT use self-attention to understand context
- **Machine Translation**: Google Translate uses attention to align source and target words
- **Image Understanding**: Vision Transformers apply attention to image patches
- **Multimodal AI**: CLIP, DALL-E use attention to connect text and images

## 🚀 From Attention to Modern AI

This module teaches the **core building block** of modern AI:

**What you're building**: The fundamental attention mechanism  
**What it enables**: Multi-head attention, positional encoding, transformer blocks  
**What it powers**: ChatGPT, BERT, GPT-4, and contemporary AI systems

Understanding this module gives you the foundation to understand:
- How ChatGPT generates coherent text
- How BERT understands language bidirectionally
- How Vision Transformers work without convolution
- How modern AI achieves human-like language understanding

## 📈 Module Progression

```
Tensors → **ATTENTION** → Layers → Networks → CNNs → Training
  ↑              ↑
Foundation   Modern AI Core
```

After completing this module, you'll understand the mechanism that sparked the AI revolution, making you ready to work with state-of-the-art models and architectures.

## 🎯 Success Criteria

You'll know you've mastered this module when you can:
- [ ] Implement scaled dot-product attention from scratch
- [ ] Explain why the √d_k scaling prevents gradient problems
- [ ] Create different types of attention masks for various use cases
- [ ] Visualize and interpret attention weights
- [ ] Understand why attention enabled the transformer revolution
- [ ] Connect this foundation to modern AI systems like ChatGPT 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/07_attention/attention_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/07_attention/attention_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/07_attention/attention_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/06_dense.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/08_attention.html" title="next page">Next Module →</a>
</div>
