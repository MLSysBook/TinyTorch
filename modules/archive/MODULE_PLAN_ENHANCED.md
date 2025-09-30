# TinyTorch Module Development Plan - Enhanced for Implementation

## 🎯 Overview
19 modules building to 5 milestones, each with concrete deliverables and systems analysis.

---

## 📦 Module Specifications

### Module 01: Tensor
**Learning Objective:** Can I create and manipulate the building blocks of ML?

**Implementation Requirements:**
```python
class Tensor:
    def __init__(self, data, requires_grad=False)
    def __add__(self, other)
    def __mul__(self, other)
    def matmul(self, other)
    def reshape(self, *shape)
    def transpose(self, dim0, dim1)
    # Broadcasting support
```

**Dependencies:** None (foundation module)

**Systems Analysis Required:**
- Memory layout (row-major vs column-major)
- Broadcasting memory overhead
- Matmul complexity: O(n³) naive vs optimized BLAS

**Tests Required:**
- Shape manipulation
- Broadcasting rules
- Numerical accuracy

**NBGrader Points:** 20 points
- Implementation: 15 points
- Systems analysis: 5 points

---

### Module 02: Activations
**Learning Objective:** Can I add nonlinearity - the key to neural network intelligence?

**Implementation Requirements:**
```python
class Sigmoid:
    def forward(self, x: Tensor) -> Tensor
    def backward(self, grad: Tensor) -> Tensor

class ReLU:
    def forward(self, x: Tensor) -> Tensor
    def backward(self, grad: Tensor) -> Tensor

class GELU:  # For GPT
    def forward(self, x: Tensor) -> Tensor
    def backward(self, grad: Tensor) -> Tensor
```

**Dependencies:** Module 01 (Tensor)

**Systems Analysis Required:**
- Numerical stability (sigmoid overflow/underflow)
- ReLU sparsity benefits for memory/compute
- GELU approximations (tanh vs erf)

**Tests Required:**
- Gradient correctness
- Numerical stability tests
- Performance comparison

---

### Module 03: Layers
**Learning Objective:** Can I build the fundamental building blocks of neural networks?

**Implementation Requirements:**
```python
class Linear:
    def __init__(self, in_features, out_features, bias=True)
    def forward(self, x: Tensor) -> Tensor
    def parameters(self) -> List[Tensor]

class Sequential:
    def __init__(self, *layers)
    def forward(self, x: Tensor) -> Tensor

class Dropout:
    def __init__(self, p=0.5)
    def forward(self, x: Tensor, training=True) -> Tensor
```

**Dependencies:** Modules 01-02

**Systems Analysis Required:**
- Weight initialization impact (Xavier, He)
- Memory: weights + activations + gradients
- Dropout as regularization vs ensemble

---

### Module 04: Losses
**Learning Objective:** Can I measure how wrong my model is?

**Implementation Requirements:**
```python
class CrossEntropyLoss:
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor
    def backward(self) -> Tensor

def log_softmax(x: Tensor, dim=-1) -> Tensor  # Numerical stability
```

**Dependencies:** Modules 01-02

**Systems Analysis Required:**
- Log-sum-exp trick for numerical stability
- Memory efficient loss computation
- Relationship to KL divergence and entropy

---

## 🪜 **Milestone 1: Perceptron (After Module 04)**
**Deliverable Requirements:**
- Train Linear + Activation on 2D toy dataset
- Visualize decision boundary
- Compare sigmoid vs ReLU convergence
- Memory profile the training loop
- **Success Criteria:** 95% accuracy on linearly separable data

---

### Module 05: Autograd
**Learning Objective:** Can I automatically compute gradients for learning?

**Implementation Requirements:**
```python
# Modify Tensor class to support:
class Tensor:
    def __init__(self, data, requires_grad=False)
    def backward(self, grad=None)
    @property
    def grad(self)

# Computational graph tracking
class Function:
    def forward(self, *inputs)
    def backward(self, grad_output)
```

**Dependencies:** Modules 01-04 (retrofits Tensor)

**Systems Analysis Required:**
- Graph memory growth with depth
- Gradient checkpointing trade-offs
- Compare to PyTorch's autograd

---

### Module 06: Optimizers
**Learning Objective:** Can I optimize neural networks with sophisticated algorithms?

**Implementation Requirements:**
```python
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9)
    def step(self)
    def zero_grad(self)

class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    def step(self)
```

**Dependencies:** Modules 01-05

**Systems Analysis Required:**
- Adam memory: 3× parameter memory (params + m + v)
- Momentum vs adaptive learning rates
- Weight decay vs L2 regularization

---

### Module 07: Training
**Learning Objective:** Can I build complete training loops for end-to-end learning?

**Implementation Requirements:**
```python
class Trainer:
    def __init__(self, model, optimizer, loss_fn)
    def train_epoch(self, dataloader)
    def evaluate(self, dataloader)

# Learning rate schedules
class CosineSchedule:
    def get_lr(self, epoch)

# Gradient clipping
def clip_grad_norm(parameters, max_norm)
```

**Dependencies:** Modules 01-06

**Systems Analysis Required:**
- Batch size vs memory vs convergence
- Gradient accumulation for large models
- Learning rate warmup importance

---

## 🪜 **Milestone 2: MLP (After Module 07)**
**Deliverable Requirements:**
- 2-layer MLP on MNIST (flattened)
- Compare to perceptron baseline
- Profile memory per batch size
- Implement early stopping
- **Success Criteria:** >95% accuracy on MNIST

---

### Module 08: DataLoader
**Learning Objective:** Can I efficiently load and batch data for training?

**Implementation Requirements:**
```python
class Dataset:
    def __len__(self)
    def __getitem__(self, idx)

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False)
    def __iter__(self)

# Specific datasets
class MNIST(Dataset)
class CIFAR10(Dataset)
```

**Dependencies:** Modules 01

**Systems Analysis Required:**
- Memory mapping vs loading into RAM
- Prefetching and parallelism
- Data augmentation compute trade-offs

---

### Module 09: Spatial
**Learning Objective:** Can I process spatial data like images with convolutions?

**Implementation Requirements:**
```python
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0)
    def forward(self, x: Tensor) -> Tensor

class MaxPool2d:
    def __init__(self, kernel_size, stride=None)
    def forward(self, x: Tensor) -> Tensor
```

**Dependencies:** Modules 01-07

**Systems Analysis Required:**
- im2col memory explosion
- Winograd convolution trade-offs
- Depthwise separable efficiency

---

## 🪜 **Milestone 3: CNN (After Module 09)**
**Deliverable Requirements:**
- 3-layer CNN on CIFAR-10
- Visualize learned filters
- Compare parameter efficiency to MLP
- Profile convolution vs FC layers
- **Success Criteria:** >75% accuracy on CIFAR-10

---

### Module 10: Tokenization
**Learning Objective:** Can I convert text into numerical representations?

**Implementation Requirements:**
```python
class CharTokenizer:
    def encode(self, text: str) -> List[int]
    def decode(self, tokens: List[int]) -> str

class BPETokenizer:  # Optional/stub
    def train(self, corpus)
    def encode(self, text)
```

**Dependencies:** None (standalone)

**Systems Analysis Required:**
- Vocabulary size vs sequence length trade-off
- Unicode handling complexity
- Subword vs character vs word trade-offs

---

### Module 11: Embeddings
**Learning Objective:** Can I create learnable representations of discrete tokens?

**Implementation Requirements:**
```python
class Embedding:
    def __init__(self, vocab_size, embed_dim)
    def forward(self, indices: Tensor) -> Tensor

class PositionalEncoding:
    def __init__(self, max_seq_len, embed_dim)
    def forward(self, x: Tensor) -> Tensor
```

**Dependencies:** Modules 01-03

**Systems Analysis Required:**
- Embedding table memory: vocab_size × embed_dim
- Learned vs sinusoidal position encodings
- Embedding dimension scaling laws

---

### Module 12: Attention
**Learning Objective:** Can I build attention mechanisms for sequence understanding?

**Implementation Requirements:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Complete implementation

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads)
    def forward(self, x: Tensor, mask=None) -> Tensor
```

**Dependencies:** Modules 01-03

**Systems Analysis Required:**
- O(n²) memory complexity with sequence length
- FlashAttention optimizations
- Attention pattern sparsity

---

### Module 13: Transformers
**Learning Objective:** Can I build complete transformer architectures?

**Implementation Requirements:**
```python
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, mlp_ratio=4)
    # Attention + MLP + Residual + LayerNorm

class GPT:
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads)
    def forward(self, indices: Tensor) -> Tensor
    def generate(self, prompt, max_length)
```

**Dependencies:** Modules 01-12

**Systems Analysis Required:**
- Parameter count scaling
- Activation memory with depth
- Gradient accumulation strategies

---

### Module 14: KV Caching
**Learning Objective:** Can I optimize autoregressive generation?

**Implementation Requirements:**
```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_layers, embed_dim)
    def update(self, layer_idx, key, value)
    def get(self, layer_idx)

# Modified attention with cache
def attention_with_cache(Q, K, V, cache, layer_idx)
```

**Dependencies:** Modules 12-13

**Systems Analysis Required:**
- Cache memory: batch × layers × seq_len × embed_dim
- Cache reuse vs recomputation trade-off
- Multi-query attention benefits

---

## 🪜 **Milestone 4: TinyGPT (After Module 14)**
**Deliverable Requirements:**
- Character-level GPT on Shakespeare
- Generate coherent text samples
- Compare with/without KV caching speed
- Perplexity < 2.0 on validation
- **Success Criteria:** Coherent 100-token generations

---

### Module 15: Profiling
**Learning Objective:** Can I measure what matters in ML systems?

**Implementation Requirements:**
```python
class Profiler:
    def count_parameters(model)
    def count_flops(model, input_shape)
    def measure_memory(model, input_shape)
    def measure_latency(model, input)
```

**Dependencies:** All previous

**Systems Analysis Required:**
- FLOPs vs MACs vs actual runtime
- Memory bandwidth bottlenecks
- Roofline model analysis

---

### Module 16: Acceleration
**Learning Objective:** Can I make models run faster?

**Implementation Requirements:**
```python
# Vectorization examples
def vectorized_matmul(a, b)

# Mixed precision
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, loss_scale=1024)
```

**Dependencies:** Modules 01-07

**Systems Analysis Required:**
- Compute intensity and bandwidth limits
- Mixed precision numerical stability
- Batch size scaling efficiency

---

### Module 17: Quantization
**Learning Objective:** Can I reduce model precision without breaking it?

**Implementation Requirements:**
```python
def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, float]:
    # Scale and zero point

class QuantizedLinear:
    def forward(self, x: Tensor) -> Tensor
```

**Dependencies:** Modules 01-03

**Systems Analysis Required:**
- Quantization error accumulation
- Activations vs weights sensitivity
- INT8 vs FP16 trade-offs

---

### Module 18: Compression
**Learning Objective:** Can I make models smaller?

**Implementation Requirements:**
```python
def magnitude_prune(model, sparsity=0.9):
    # Remove small weights

def measure_sparsity(model):
    # Count zeros
```

**Dependencies:** All previous

**Systems Analysis Required:**
- Structured vs unstructured sparsity
- Lottery ticket hypothesis
- Fine-tuning after pruning

---

### Module 19: Benchmarking
**Learning Objective:** Can I fairly compare different approaches?

**Implementation Requirements:**
```python
class Benchmark:
    def compare_models(models, metrics=['accuracy', 'latency', 'memory'])
    def plot_results()
    def generate_report()
```

**Dependencies:** All previous

**Systems Analysis Required:**
- Latency vs throughput
- Energy efficiency metrics
- Pareto frontiers

---

## 🪜 **Milestone 5: Systems Capstone (After Module 19)**
**Deliverable Requirements:**
- Profile CNN vs TinyGPT
- Apply quantization to both
- Apply pruning to both
- Generate comparison report:
  - Accuracy vs model size
  - Latency vs accuracy
  - Memory vs throughput
- **Success Criteria:** 2× speedup with <5% accuracy loss

---

## 📋 Module Development Checklist

For EACH module, the developer must:

### Implementation
- [ ] Follow exact API signatures specified
- [ ] Use only prior module dependencies
- [ ] Add proper export directives (#| default_exp)
- [ ] Include NBGrader metadata

### Systems Analysis (MANDATORY)
- [ ] Memory profiling section with code
- [ ] Computational complexity analysis
- [ ] Scaling behavior experiments
- [ ] Production context (PyTorch/TensorFlow comparison)

### Testing
- [ ] Unit tests after each implementation
- [ ] Performance benchmarks
- [ ] Integration test with prior modules
- [ ] Edge cases and error handling

### Documentation
- [ ] Mathematical background
- [ ] Clear code comments
- [ ] ML Systems Thinking questions
- [ ] Module summary

### Validation
- [ ] Run through QA Agent
- [ ] Export with `tito module complete`
- [ ] Verify checkpoint passes
- [ ] Check no forward dependencies

---

## 🚀 Implementation Order

**Phase 1: Foundation (Modules 01-04)**
→ Milestone 1: Perceptron

**Phase 2: Learning (Modules 05-07)**
→ Milestone 2: MLP

**Phase 3: Vision (Modules 08-09)**
→ Milestone 3: CNN

**Phase 4: Language (Modules 10-14)**
→ Milestone 4: TinyGPT

**Phase 5: Systems (Modules 15-19)**
→ Milestone 5: Systems Capstone

---

## 🎯 Success Criteria

Each module is complete when:
1. All tests pass
2. Systems analysis included
3. QA Agent approves
4. Checkpoint validates
5. Integration tests pass
6. Documentation complete