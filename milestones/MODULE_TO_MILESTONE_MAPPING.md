# Module to Milestone Mapping

## Clean Progression (After Removing Module 07: Training)

This document shows the correct module completion → milestone unlocking progression for TinyTorch.

---

## 🎯 Core Philosophy

**Students write MANUAL training loops** - no abstraction layer hiding the fundamentals.  
This matches how PyTorch actually works: you control the training loop yourself.

---

## 📚 Module Progression

### **Module 01: Tensor**
- Core data structure with automatic differentiation support
- Foundation for all computations

### **Module 02: Activations**
- ReLU, Sigmoid, Softmax
- Non-linear transformations

### **Module 03: Layers**
- Linear (fully connected) layers
- Parameter management

### **Module 04: Losses**
- MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
- Loss computation for training

### **Module 05: Autograd**
- Automatic differentiation
- Backward pass implementation
- Computational graph

### **Module 06: Optimizers**
- SGD, Adam
- Parameter update rules
- **Optional utilities**: Learning rate scheduling, gradient clipping

---

## 🏆 Milestone Unlocking

### **Milestone 01: Perceptron (1957)** 
📦 **Unlocks after Module 03** (Tensor, Activations, Layers)

**What it demonstrates:**
- First trainable neural network
- Linear classification
- Simple forward pass + weight update

**Training pattern:**
```python
# Simple gradient step
output = model(x)
loss = loss_fn(output, target)
loss.backward()
# Manual weight update
```

---

### **Milestone 02: XOR (1969)**
📦 **Unlocks after Module 06** (+ Losses, Autograd, Optimizers)

**What it demonstrates:**
- Non-linear problem solving
- Hidden layers
- Complete manual training loop

**Training pattern:**
```python
for epoch in range(epochs):
    predictions = model(X)
    loss = loss_fn(predictions, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

### **Milestone 03: MNIST MLP (1986)**
📦 **Unlocks after Module 06** (Same as XOR)

**What it demonstrates:**
- Multi-class classification
- Real vision dataset
- Mini-batch training

**Training pattern:**
```python
for epoch in range(epochs):
    for batch in batches:
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

### **Module 07: DataLoader** (was Module 08)
- Batching and shuffling
- Dataset abstraction
- Iteration protocol

---

### **Milestone 04: CNN (1998)**
📦 **Unlocks after Module 08** (+ DataLoader, Spatial)

### **Module 08: Spatial** (was Module 09)
- Conv2d, MaxPool2D
- Spatial operations for vision

**What it demonstrates:**
- Convolutional networks
- Spatial feature extraction
- YOUR DataLoader for efficient batching

**Training pattern:**
```python
for epoch in range(epochs):
    for batch_images, batch_labels in dataloader:  # YOUR DataLoader!
        outputs = model(batch_images)
        loss = criterion.forward(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

### **Modules 09-12: Language Modeling Stack**
- Module 09: Tokenization
- Module 10: Embeddings (token + positional)
- Module 11: Attention (multi-head self-attention)
- Module 12: Transformers (LayerNorm, TransformerBlock)

---

### **Milestone 05: Transformer (2017)**
📦 **Unlocks after Module 12** (+ Tokenization, Embeddings, Attention, Transformers)

**What it demonstrates:**
- Sequence modeling
- Attention mechanisms
- Autoregressive generation
- Foundation for ChatGPT/GPT-4

**Training pattern:**
```python
for epoch in range(epochs):
    for batch_idx, (batch_input, batch_target) in enumerate(train_loader):
        logits = model.forward(batch_input)
        loss = compute_loss(logits, batch_target)  # Cross-entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## ✅ Key Design Decisions

### **Why Remove Module 07 (Training)?**

1. **PyTorch doesn't have it**: PyTorch has no built-in `Trainer` class. Users write manual loops.

2. **Pedagogically stronger**: Manual loops force understanding of the core cycle:
   - Forward pass
   - Loss computation
   - Backward pass
   - Parameter update

3. **Matches production code**: Research and production code use manual loops for maximum control.

4. **All milestones already use it**: Every milestone already demonstrates manual training loops.

### **What happened to training utilities?**

Useful utilities from Module 07 (LR scheduling, gradient clipping) are available as **optional helpers** in Module 06 (Optimizers):

```python
from tinytorch.optim import CosineScheduler, clip_grad_norm

# Use when needed
scheduler = CosineScheduler(max_lr=0.1, min_lr=0.01, total_epochs=100)
lr = scheduler.get_lr(epoch)

# Gradient clipping for stability
clip_grad_norm(model.parameters(), max_norm=1.0)
```

---

## 🎓 Pedagogical Flow

| Stage | Modules | Milestone | Learning Goal |
|-------|---------|-----------|---------------|
| **Foundation** | 01-03 | Perceptron (1957) | Basic neural network |
| **Training Basics** | 04-06 | XOR (1969) | Complete training loop |
| **Real Vision** | 01-06 | MNIST MLP (1986) | Multi-class classification |
| **Efficient Loading** | 07 | - | Batching & iteration |
| **Spatial Understanding** | 08 | CNN (1998) | Convolutional networks |
| **Language Modeling** | 09-12 | Transformer (2017) | Attention & sequences |

---

## 🚀 Student Journey

```
Complete Module 01-03 → Run Perceptron milestone
    ↓
Complete Module 04-06 → Run XOR + MNIST milestones
    ↓
Complete Module 07-08 → Run CNN milestone
    ↓
Complete Module 09-12 → Run Transformer milestone
    ↓
Advanced modules (KV-caching, profiling, etc.)
```

Each milestone proves mastery by combining multiple modules into working systems.

---

## 💡 Why This Works

1. **Progressive complexity**: Each milestone builds on previous capabilities
2. **Manual control**: Students see exactly what's happening in training
3. **Production relevant**: Same patterns used in research and production
4. **Clear dependencies**: Each milestone requires specific completed modules
5. **No magic**: No abstraction layers hiding fundamentals
6. **PyTorch-aligned**: Matches how PyTorch actually works

---

**Remember**: The goal is to understand ML systems deeply, not just use high-level abstractions. Manual training loops are the path to mastery.

