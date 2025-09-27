# 🎯 TinyTorch Master Plan V2: Minimal Viable Learning
*Build ML Systems Through Implementation, Not Over-Engineering*

## Core Philosophy
**Build JUST ENOUGH to understand WHY PyTorch works the way it does.**

Students implement minimal but complete systems that demonstrate core algorithmic and engineering concepts underlying modern AI frameworks.

---

## 📚 **15-Module Curriculum: From Tensors to Transformers**

### **PHASE 1: MINIMAL WORKING NETWORK** (Modules 1-4)
*Milestone: XOR network inference in 4 modules*

| Module | Name | What Students Build (Just Enough) | Engineering Concepts to Emphasize |
|--------|------|-----------------------------------|-----------------------------------|
| **1** | **Setup** | • Virtual environment setup<br>• Basic memory profiler (tracemalloc)<br>• Simple test runner | • Development environment = foundation<br>• Measure before optimizing<br>• Reproducible environments |
| **2** | **Tensor** | • Basic Tensor class with .data<br>• Shape, dtype properties<br>• Essential ops: +, -, *, /<br>• Basic indexing [i, j] | • Memory layout (row vs column major)<br>• Views vs copies demonstration<br>• NumPy vectorization = 10-100x speedup<br>• O(N) memory scaling |
| **3** | **Activations** | • ReLU, Sigmoid (forward only)<br>• Broadcasting for element-wise ops<br>• XOR impossibility proof | • Nonlinearity = intelligence<br>• Broadcasting memory implications<br>• Numerical stability (sigmoid overflow)<br>• Why linear networks can't learn XOR |
| **4** | **Layers** | • Parameter class (tensor + grad flag)<br>• Linear layer (W·x + b)<br>• Sequential container<br>• Forward pass only | • Matrix multiplication O(N³)<br>• Parameter memory quadratic scaling<br>• Composition enables depth<br>• Memory per layer analysis |

**🎯 Phase 1 Milestone**: Run XOR network inference
```python
# Students can execute:
net = Sequential([Linear(2,4), ReLU(), Linear(4,1)])
output = net(xor_input)  # Works without training!
```

---

### **PHASE 2: INTELLIGENT LEARNING** (Modules 5-8)
*Milestone: Self-training XOR network with 100% accuracy*

| Module | Name | What Students Build (Just Enough) | Engineering Concepts to Emphasize |
|--------|------|-----------------------------------|-----------------------------------|
| **5** | **Autograd** | • Computational graph nodes<br>• Chain rule implementation<br>• Backward for +, *, Linear<br>• Gradient accumulation | • Memory explosion during backprop<br>• Reverse-mode AD efficiency<br>• Graph retention = memory cost<br>• O(N) memory for gradients |
| **6** | **Losses** | • MSE Loss (for XOR)<br>• CrossEntropy (preview)<br>• loss.backward() integration | • Scalar loss enables backprop<br>• Loss choice affects convergence<br>• Gradient magnitude analysis |
| **7** | **Optimizers** | • SGD only (w = w - lr*grad)<br>• Parameter update loop<br>• Gradient zeroing | • Learning rate = critical hyperparameter<br>• Why zero gradients (accumulation bug)<br>• O(parameters) update cost |
| **8** | **Training** | • Basic train() function<br>• Forward→loss→backward→step<br>• Simple validation loop | • Training memory = activations + gradients<br>• Train vs eval modes<br>• Gradient accumulation for memory |

**🎯 Phase 2 Milestone**: Train XOR to convergence
```python
# Students watch learning happen:
for epoch in range(100):
    pred = net(X)
    loss = mse_loss(pred, y)
    loss.backward()  # Autograd magic!
    optimizer.step()  # Parameters update!
    print(f"Epoch {epoch}: Loss = {loss.data}")
# Loss: 1.0 → 0.01 (network learned!)
```

---

### **PHASE 3: REAL DATA MASTERY** (Modules 9-12)  
*Milestone: MNIST CNN with >95% accuracy*

| Module | Name | What Students Build (Just Enough) | Engineering Concepts to Emphasize |
|--------|------|-----------------------------------|-----------------------------------|
| **9** | **Spatial** | • Conv2d (simple, unoptimized)<br>• MaxPool2d<br>• Flatten layer<br>• Basic CNN architecture | • Conv memory O(batch×C×H×W×K²)<br>• Pooling reduces params exponentially<br>• Receptive field growth<br>• Why CNNs for images |
| **10** | **DataLoader** | • Dataset class for MNIST<br>• Basic batch iteration<br>• Simple preprocessing | • I/O bottlenecks from disk<br>• Batch size vs memory tradeoff<br>• Why preprocessing matters<br>• Data pipeline optimization |
| **11** | **Advanced Opt** | • Adam optimizer<br>• CrossEntropy loss<br>• Image training loop<br>• Validation metrics | • Adam = 3× parameter memory<br>• Adaptive learning rates<br>• Momentum accumulation cost<br>• Validation prevents overfitting |
| **12** | **Production** | • Model checkpointing<br>• Early stopping<br>• Learning rate decay<br>• Accuracy tracking | • Checkpoint size = model params<br>• Early stopping as regularization<br>• LR scheduling for convergence<br>• Metric computation cost |

**🎯 Phase 3 Milestone**: MNIST digit recognition
```python
# Real computer vision:
cnn = Sequential([
    Conv2d(1, 16, 3), ReLU(), MaxPool2d(2),
    Conv2d(16, 32, 3), ReLU(), MaxPool2d(2),
    Flatten(), Linear(32*5*5, 10)
])
trainer.fit(mnist_train, epochs=5)
accuracy = evaluate(mnist_test)  # >95%!
```

---

### **PHASE 4: MODERN AI** (Modules 13-15)
*Milestone: TinyGPT text generation*

| Module | Name | What Students Build (Just Enough) | Engineering Concepts to Emphasize |
|--------|------|-----------------------------------|-----------------------------------|
| **13** | **Attention** | • Scaled dot-product attention<br>• Single-head Q,K,V<br>• Causal masking<br>• Position encoding | • O(N²) memory scaling<br>• Sequence length bottlenecks<br>• Causal masks prevent leakage<br>• Why attention > recurrence |
| **14** | **Transformers** | • Multi-head attention<br>• LayerNorm<br>• Transformer block<br>• GPT architecture | • Multi-head = parallel attention<br>• LayerNorm vs BatchNorm<br>• Residuals prevent vanishing<br>• Layer memory accumulation |
| **15** | **Generation** | • Character tokenization<br>• Embedding layers<br>• Autoregressive generation<br>• Temperature sampling | • Sequential inference cost<br>• Embedding lookup efficiency<br>• Generation memory patterns<br>• Temperature controls diversity |

**🎯 Phase 4 Milestone**: Generate text with TinyGPT
```python
# Modern AI from scratch:
model = TinyGPT(vocab_size=1000, layers=6, heads=8)
train_on_shakespeare(model)
generated = model.generate("To be or not to be")
print(generated)  # Coherent continuation!
```

---

## 🎯 **What Students DON'T Build (But Understand)**

### **Deferred Complexity**
- **GPU/CUDA**: Understand device abstraction, implement CPU-only
- **Optimized kernels**: Use NumPy, understand why optimization matters  
- **Dynamic graphs**: Simple static graphs, understand flexibility tradeoff
- **Production features**: Focus on algorithms, not deployment

### **Integrated Simplifications**
- **Memory profiling**: Built into every module with tracemalloc
- **Performance timing**: Simple time.time(), not complex profiling
- **Batch normalization**: Mentioned but not implemented (complexity)
- **Dropout**: Brief mention in CNNs, not full implementation

---

## 📊 **Learning Validation Metrics**

### **Concrete Success Criteria**
| Phase | Module | Success Metric | Systems Understanding |
|-------|--------|---------------|----------------------|
| 1 | 4 | XOR inference runs | Memory layout, matrix ops |
| 2 | 8 | XOR trains to <0.01 loss | Gradient flow, optimization |
| 3 | 12 | MNIST >95% accuracy | CNN efficiency, data pipelines |
| 4 | 15 | Coherent text generation | Attention scaling, generation |

### **Time Investment**
- **Per module**: 3-4 hours (read, implement, test)
- **Per phase**: 12-16 hours  
- **Total**: 48-64 hours (realistic semester)
- **Complexity curve**: ▁▂▃▄ ▅▅▆▆ ▇▇██ ███ (gradual increase)

---

## 🔬 **Systems Engineering Thread**

### **Every Module Teaches**
1. **Memory patterns**: Where does memory go? When are copies made?
2. **Computational complexity**: O(N), O(N²), O(N³) analysis
3. **Performance bottlenecks**: What breaks first at scale?
4. **PyTorch comparison**: How does real PyTorch handle this?

### **Key Systems Insights Students Gain**
- Why matrix multiplication dominates neural network compute
- Why autograd requires retaining intermediate activations  
- Why convolution is memory-bandwidth limited
- Why attention creates quadratic scaling challenges
- Why batch size affects GPU utilization
- Why data loading becomes the bottleneck at scale

---

## 🚀 **Why This Structure Works**

### **Pedagogical Advantages**
- **Immediate validation**: Every phase produces working code
- **Progressive complexity**: Each phase builds on the last
- **Industry relevance**: Uses standard benchmarks (XOR, MNIST)
- **Modern relevance**: Ends with transformer architecture

### **Engineering Focus**
- **Just enough implementation**: Learn concepts without over-engineering
- **Memory-first thinking**: Understand resource constraints
- **Production awareness**: Know how real systems differ
- **Debugging skills**: Build systems that can be understood

### **Student Outcomes**
After completing TinyTorch, students can:
- Read and understand PyTorch source code
- Debug training failures in production ML systems
- Make informed architecture decisions based on resource constraints
- Understand the engineering tradeoffs in modern AI systems

---

## 📝 **Implementation Notes**

### **Module Structure**
Each module follows consistent pattern:
1. **Minimal implementation** of core concepts
2. **Unit tests** validating functionality  
3. **Memory/performance analysis** section
4. **PyTorch comparison** showing production version
5. **Systems thinking questions** for reflection

### **Code Philosophy**
- **Readable > Optimized**: Clear code that teaches
- **Explicit > Magic**: Show how things work
- **Working > Complete**: Just enough to achieve milestone
- **Tested > Assumed**: Validate everything works

---

## ✅ **Success Metrics**

**Students successfully complete TinyTorch when they can:**
1. Explain why neural networks need nonlinear activations (Phase 1)
2. Debug gradient flow problems in training (Phase 2)  
3. Choose appropriate architectures for data types (Phase 3)
4. Understand transformer memory scaling (Phase 4)
5. Read PyTorch source with comprehension (Overall)

**The Ultimate Test**: Can students build and train a working model from scratch that achieves meaningful results on a real dataset?

---

*This plan eliminates over-engineering while maintaining the core insight: students learn ML systems by building minimal but complete implementations that demonstrate the key algorithmic and systems concepts underlying modern AI frameworks.*