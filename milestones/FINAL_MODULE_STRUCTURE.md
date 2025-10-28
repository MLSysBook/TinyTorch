# TinyTorch Final Module Structure

## 📚 Core Learning Path (For All Milestones)

### **Phase 1: Building Blocks (Modules 01-04)**
```
01. Tensor           → Data structure with autograd support
02. Activations      → ReLU, Sigmoid, Softmax
03. Layers           → Linear layers, parameter management
04. Losses           → MSE, CrossEntropy, BinaryCrossEntropy
```

**Unlocks:** 
- 🏆 Milestone 01: Perceptron (1957) - after Module 03

---

### **Phase 2: Learning Systems (Modules 05-06)**
```
05. Autograd         → Backward passes, computational graph
06. Optimizers       → SGD, Adam parameter updates
```

**Unlocks:**
- 🏆 Milestone 02: XOR (1969) - after Module 06
- 🏆 Milestone 03: MNIST MLP (1986) - after Module 06

**Training Pattern:** Manual loops
```python
for epoch in range(epochs):
    for batch in batches:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

### **Phase 3: Vision & Data (Modules 08-09)**
```
08. DataLoader       → Batching, shuffling, Dataset abstraction
09. Spatial          → Conv2d, MaxPool2D for vision
```

**Note:** Module 07 is intentionally skipped in core progression (see Advanced Topics)

**Unlocks:**
- 🏆 Milestone 04: CNN (1998) - after Module 09

---

### **Phase 4: Language & Attention (Modules 10-13)**
```
10. Tokenization     → Character & BPE tokenizers
11. Embeddings       → Token + positional embeddings
12. Attention        → Multi-head self-attention
13. Transformers     → LayerNorm, TransformerBlock
```

**Unlocks:**
- 🏆 Milestone 05: Transformer (2017) - after Module 13

---

## 🎓 Advanced Topics (Optional - After Completing All Milestones)

### **Module 07: Training Abstractions ⭐**
```
⚠️ OPTIONAL - Skip in your first pass!

This module teaches abstraction patterns like:
- PyTorch Lightning-style Trainer classes
- Learning rate scheduling
- Gradient clipping
- Checkpointing systems

Come back to this AFTER you've written manual training 
loops many times and understand the fundamentals.
```

**Why it's optional:**
- Manual training loops teach fundamentals better
- All milestones use manual loops (matching PyTorch patterns)
- Abstractions make sense AFTER you understand what they're abstracting
- PyTorch itself doesn't have a Trainer class - you write manual loops

---

### **Advanced Systems (Modules 14-20)**
```
14. KV-Caching       → Efficient transformer inference
15. Profiling        → Performance measurement
16. Acceleration     → Speed optimizations
17. Quantization     → INT8 quantization
18. Compression      → Pruning techniques
19. Benchmarking     → Performance comparison
20. Capstone         → Complete TinyGPT system
```

---

## 🎯 Module → Milestone Mapping

| Milestone | Name | Required Modules | Training Style |
|-----------|------|------------------|----------------|
| 01 | Perceptron (1957) | 01-03 | Simple gradient step |
| 02 | XOR (1969) | 01-06 | Manual loop |
| 03 | MNIST MLP (1986) | 01-06 | Manual loop + batching |
| 04 | CNN (1998) | 01-06, 08-09 | Manual loop + DataLoader |
| 05 | Transformer (2017) | 01-06, 08-13 | Manual loop + Attention |

---

## 📖 Table of Contents Structure

The course book now follows this progression:

```
🧱 Building Blocks
  ├─ 01. Tensor
  ├─ 02. Activations
  ├─ 03. Layers
  └─ 04. Losses

🧠 Learning Systems
  ├─ 05. Autograd
  └─ 06. Optimizers
      └─ 🏆 Milestones 01-03 unlock here

🖼️ Vision & Data
  ├─ 08. DataLoader       [Comes right after 06!]
  └─ 09. Spatial
      └─ 🏆 Milestone 04 unlocks here

🗣️ Language & Attention
  ├─ 10. Tokenization
  ├─ 11. Embeddings
  ├─ 12. Attention
  └─ 13. Transformers
      └─ 🏆 Milestone 05 unlocks here

🎓 Advanced Topics
  ├─ 07. Training Abstractions ⭐ Optional
  ├─ 14. KV Caching
  ├─ 15. Profiling
  ├─ 16. Acceleration
  ├─ 17. Quantization
  ├─ 18. Compression
  └─ 19. Benchmarking

🏅 Competition
  └─ 20. AI Olympics (Capstone)
```

---

## 💡 Design Rationale

### Why Skip Module 07 in Core Path?

1. **PyTorch doesn't have it**: PyTorch has no built-in `Trainer` class. Research code uses manual loops.

2. **Pedagogical superiority**: Writing manual loops 5 times teaches you more than using an abstraction once.

3. **Real-world relevance**: Production code needs custom training loops for:
   - Mixed precision training
   - Gradient accumulation
   - Custom loss functions
   - Complex data pipelines
   - Debugging and monitoring

4. **Learn fundamentals first**: Understand what you're abstracting before learning abstractions.

### Why Keep the Number 07?

1. **Zero breaking changes**: Students mid-course aren't affected
2. **Git history stays clean**: No massive refactoring needed
3. **Semantic meaning**: The gap documents an intentional design decision
4. **Version precedent**: Software versions often have gaps (Python 2.7 → 3.0)
5. **Future value**: Advanced students can still learn abstraction patterns

---

## 🚀 Student Learning Journey

```
Week 1-2: Modules 01-03
   └─ Run Milestone 01 (Perceptron)

Week 3-4: Modules 04-06
   └─ Run Milestones 02-03 (XOR, MNIST)
   └─ Master manual training loops!

Week 5-6: Modules 08-09
   └─ Run Milestone 04 (CNN)
   └─ Learn DataLoader + spatial ops

Week 7-9: Modules 10-13
   └─ Run Milestone 05 (Transformer)
   └─ Understand attention mechanisms

Week 10+: Advanced Topics
   └─ Optional: Module 07 (Training Abstractions)
   └─ Modules 14-20 (Systems optimization)
   └─ Capstone project
```

---

## ✅ What Changed?

### From Previous Structure:
- ❌ Module 07 was in core path but unused by any milestone
- ❌ Students confused about whether to use Trainer or manual loops
- ❌ TOC showed modules 07, 08, 09 in sequence

### To Current Structure:
- ✅ Module 07 moved to Advanced Topics (still available)
- ✅ Core path 01-06 → 08-13 (clear progression)
- ✅ All milestones use manual loops (consistent pedagogy)
- ✅ TOC clearly shows the learning path
- ✅ Zero breaking changes (numbers preserved)

---

## 🎯 Key Takeaway

**Manual training loops are not a limitation - they're a feature.**

Every ML engineer should be comfortable writing:
```python
for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

This is how PyTorch works. This is how research works. This is what you'll do in production.

Learn this pattern deeply before learning when to abstract it.

---

**Module 07 is there when you're ready for it - but fundamentals come first. 🎓**

