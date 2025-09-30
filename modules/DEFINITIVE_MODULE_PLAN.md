# TinyTorch Definitive Module Plan

## 🎯 Overview
19 modules building to 5 milestones, teaching ML systems through implementation.

## 📚 Module Specifications

### Module 01: Tensor
**Learning Objective:** Can I create and manipulate the building blocks of ML?

**Implementation Requirements:**
```python
class Tensor:
    """Educational tensor that grows with student knowledge."""

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.shape = self.data.shape

        # Gradient features (dormant until Module 05)
        self.requires_grad = requires_grad
        self.grad = None

    def __add__(self, other): return Tensor(self.data + other.data)
    def __mul__(self, other): return Tensor(self.data * other.data)
    def matmul(self, other): return Tensor(np.dot(self.data, other.data))
    def reshape(self, *shape): return Tensor(self.data.reshape(shape))
    def transpose(self, dim0, dim1): # Implement transpose
    def sum(self, axis=None): return Tensor(self.data.sum(axis=axis))

    def backward(self):
        """Compute gradients (implemented in Module 05)."""
        pass  # Students: ignore until Module 05
```

**Student Introduction:**
```
We're building a Tensor class that will grow throughout the course.
For now, focus on: data, shape, and operations.
Ignore for now: requires_grad, grad, backward() (we'll use them in Module 05)
```

**Dependencies:** None
**Export:** `#| default_exp core.tensor`
**Tests:** Shape manipulation, broadcasting, matmul correctness
**Systems Focus:** Memory layout, broadcasting overhead, matmul complexity O(n³)

---

### Module 02: Activations
**Learning Objective:** Can I add nonlinearity - the key to neural network intelligence?

**Implementation Requirements:**
```python
class Sigmoid:
    def forward(self, x: Tensor) -> Tensor
    def backward(self, grad: Tensor) -> Tensor  # Stub until Module 05

class ReLU:
    def forward(self, x: Tensor) -> Tensor
    def backward(self, grad: Tensor) -> Tensor  # Stub until Module 05

class GELU:  # For GPT later
    def forward(self, x: Tensor) -> Tensor
    def backward(self, grad: Tensor) -> Tensor  # Stub until Module 05
```

**Dependencies:** Module 01 (Tensor)
**Export:** `#| default_exp core.activations`
**Tests:** Output ranges, gradient shapes (once implemented)
**Systems Focus:** ReLU sparsity benefits, sigmoid saturation, GELU approximations

---

### Module 03: Layers
**Learning Objective:** Can I build the fundamental building blocks of neural networks?

**Implementation Requirements:**
```python
class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(randn(in_features, out_features))
        self.bias = Tensor(zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor
    def parameters(self) -> List[Tensor]

class Sequential:
    def __init__(self, *layers)
    def forward(self, x: Tensor) -> Tensor
    def parameters(self) -> List[Tensor]

class Dropout:
    def __init__(self, p=0.5)
    def forward(self, x: Tensor, training=True) -> Tensor
```

**Dependencies:** Modules 01-02
**Export:** `#| default_exp core.layers`
**Tests:** Shape preservation, parameter counting
**Systems Focus:** Weight initialization (Xavier/He), memory per layer

---

### Module 04: Losses
**Learning Objective:** Can I measure how wrong my model is?

**Implementation Requirements:**
```python
class CrossEntropyLoss:
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor
    def backward(self) -> Tensor  # Stub until Module 05

class MSELoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor
    def backward(self) -> Tensor  # Stub until Module 05

def log_softmax(x: Tensor, dim=-1) -> Tensor  # Numerical stability
```

**Dependencies:** Modules 01-03
**Export:** `#| default_exp core.losses`
**Tests:** Numerical stability, correct loss values
**Systems Focus:** Log-sum-exp trick, memory efficient computation

---

## 🪜 **Milestone 1: Perceptron (After Module 04)**
**Location:** `milestones/01_perceptron/`
**Deliverable:** Train Linear + Sigmoid on 2D dataset, visualize decision boundary
**Success Criteria:** 95% accuracy on linearly separable data
**Unlock:** Complete modules 01-04 + integration test

---

### Module 05: Autograd
**Learning Objective:** Can I automatically compute gradients for learning?

**Implementation Requirements:**
```python
# Activate the dormant gradient features in Tensor
# No new Tensor class - enhance existing one!

def implement_backward_for_tensor():
    """Fill in the Tensor.backward() method"""
    # Track computation graph
    # Compute gradients via chain rule
    # Update tensor.grad attributes

class Function:
    """Base class for differentiable operations"""
    def forward(self, *inputs)
    def backward(self, grad_output)

# Wrap existing operations to track gradients
class AddBackward(Function): ...
class MulBackward(Function): ...
class MatmulBackward(Function): ...
```

**Dependencies:** Modules 01-04 (enhances Tensor from Module 01)
**Export:** `#| default_exp core.autograd`
**Tests:** Gradient correctness, chain rule, graph building
**Systems Focus:** Graph memory growth, gradient checkpointing

---

### Module 06: Optimizers
**Learning Objective:** Can I optimize neural networks with sophisticated algorithms?

**Implementation Requirements:**
```python
class Optimizer:
    def __init__(self, params)
    def zero_grad(self)
    def step(self)

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9)

class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
```

**Dependencies:** Modules 01-05 (uses gradients from Module 05)
**Export:** `#| default_exp core.optimizers`
**Tests:** Parameter updates, momentum accumulation
**Systems Focus:** Adam's 3× memory usage, momentum vs adaptive

---

### Module 07: Training
**Learning Objective:** Can I build complete training loops for end-to-end learning?

**Implementation Requirements:**
```python
class Trainer:
    def __init__(self, model, optimizer, loss_fn)
    def train_epoch(self, dataloader)
    def evaluate(self, dataloader)
    def save_checkpoint(self, path)
    def load_checkpoint(self, path)

class CosineSchedule:
    def get_lr(self, epoch)

def clip_grad_norm(parameters, max_norm)
```

**Dependencies:** Modules 01-06
**Export:** `#| default_exp core.training`
**Tests:** Training loop, checkpointing, scheduling
**Systems Focus:** Batch size vs memory, gradient accumulation

---

## 🪜 **Milestone 2: MLP (After Module 07)**
**Location:** `milestones/02_mlp/`
**Deliverable:** 2-layer MLP on MNIST, compare to perceptron
**Success Criteria:** >95% accuracy on MNIST
**Unlock:** Complete modules 05-07 + integration test

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
    def __len__(self)

class TensorDataset(Dataset):
    def __init__(self, *tensors)

def download_mnist() -> Tuple[Dataset, Dataset]
def download_cifar10() -> Tuple[Dataset, Dataset]
```

**Dependencies:** Modules 01-07
**Export:** `#| default_exp data.loader`
**Tests:** Batching, shuffling, iteration
**Systems Focus:** Memory mapping, prefetching, data pipeline

---

### Module 09: Spatial
**Learning Objective:** Can I process spatial data like images with convolutions?

**Implementation Requirements:**
```python
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0)
    def forward(self, x: Tensor) -> Tensor
    def parameters(self) -> List[Tensor]

class MaxPool2d:
    def __init__(self, kernel_size, stride=None)
    def forward(self, x: Tensor) -> Tensor

class BatchNorm2d:
    def __init__(self, num_features)
    def forward(self, x: Tensor, training=True) -> Tensor
```

**Dependencies:** Modules 01-08
**Export:** `#| default_exp core.spatial`
**Tests:** Output shapes, receptive fields
**Systems Focus:** Convolution complexity O(N²M²K²), im2col memory trade-off, depthwise separable

---

## 🪜 **Milestone 3: CNN (After Module 09)**
**Location:** `milestones/03_cnn/`
**Deliverable:** 3-layer CNN on CIFAR-10, visualize filters
**Success Criteria:** >75% accuracy on CIFAR-10
**Unlock:** Complete modules 08-09 + integration test

---

### Module 10: Tokenization
**Learning Objective:** Can I convert text into numerical representations?

**Implementation Requirements:**
```python
class Tokenizer:
    def encode(self, text: str) -> List[int]
    def decode(self, tokens: List[int]) -> str

class CharTokenizer(Tokenizer):
    def __init__(self, vocab: List[str])
    def build_vocab(self, corpus: List[str])

class BPETokenizer(Tokenizer):  # Optional/advanced
    def train(self, corpus: List[str], vocab_size: int)
```

**Dependencies:** Module 01
**Export:** `#| default_exp text.tokenization`
**Tests:** Encode/decode round-trip, vocabulary building
**Systems Focus:** Vocab size vs sequence length trade-off

---

### Module 11: Embeddings
**Learning Objective:** Can I create learnable representations of discrete tokens?

**Implementation Requirements:**
```python
class Embedding:
    def __init__(self, vocab_size, embed_dim)
    def forward(self, indices: Tensor) -> Tensor
    def parameters(self) -> List[Tensor]

class PositionalEncoding:
    def __init__(self, max_seq_len, embed_dim)
    def forward(self, x: Tensor) -> Tensor

def create_sinusoidal_embeddings(max_seq_len, embed_dim) -> Tensor
```

**Dependencies:** Modules 01-10
**Export:** `#| default_exp text.embeddings`
**Tests:** Embedding lookup, position encoding
**Systems Focus:** Embedding table memory, learned vs fixed

---

### Module 12: Attention
**Learning Objective:** Can I build attention mechanisms for sequence understanding?

**Implementation Requirements:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None) -> Tensor

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads)
    def forward(self, x: Tensor, mask=None) -> Tensor
    def parameters(self) -> List[Tensor]
```

**Dependencies:** Modules 01-11
**Export:** `#| default_exp core.attention`
**Tests:** Attention weights sum to 1, masking
**Systems Focus:** O(n²) memory complexity with sequence length, FlashAttention concepts

---

### Module 13: Transformers
**Learning Objective:** Can I build complete transformer architectures?

**Implementation Requirements:**
```python
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, embed_dim * mlp_ratio)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor

class GPT:
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads)
    def forward(self, indices: Tensor) -> Tensor
    def generate(self, prompt: Tensor, max_length: int) -> Tensor
```

**Dependencies:** Modules 01-12
**Export:** `#| default_exp models.transformer`
**Tests:** Shape preservation, generation
**Systems Focus:** Parameter scaling, activation memory

---

### Module 14: KV Caching
**Learning Objective:** Can I optimize autoregressive generation?

**Implementation Requirements:**
```python
class KVCache:
    def __init__(self, batch_size, max_seq_len, num_layers, num_heads, head_dim)
    def update(self, layer_idx, key, value, seq_pos)
    def get(self, layer_idx) -> Tuple[Tensor, Tensor]

# Modified attention to use cache
def attention_with_cache(Q, K, V, cache, layer_idx, seq_pos) -> Tensor
```

**Dependencies:** Modules 01-13
**Export:** `#| default_exp generation.kv_cache`
**Tests:** Cache correctness, memory usage
**Systems Focus:** Cache memory vs recomputation trade-off

---

## 🪜 **Milestone 4: TinyGPT (After Module 14)**
**Location:** `milestones/04_tinygpt/`
**Deliverable:** Character-level GPT on Shakespeare, generate text
**Success Criteria:** Perplexity < 2.0, coherent generation
**Unlock:** Complete modules 10-14 + integration test

---

### Module 15: Profiling
**Learning Objective:** Can I measure what matters in ML systems?

**Implementation Requirements:**
```python
class Profiler:
    def count_parameters(self, model) -> int
    def count_flops(self, model, input_shape) -> int
    def measure_memory(self, model, input_shape) -> Dict[str, float]
    def measure_latency(self, model, input, warmup=10, iterations=100) -> float

def profile_forward_pass(model, input) -> Dict[str, Any]
def profile_backward_pass(model, input, loss_fn) -> Dict[str, Any]
```

**Dependencies:** All previous
**Export:** `#| default_exp profiling.profiler`
**Tests:** Accurate counting, timing consistency
**Systems Focus:** FLOPs vs runtime, roofline model

---

### Module 16: Acceleration
**Learning Objective:** Can I make models run faster?

**Implementation Requirements:**
```python
# Vectorization examples
def vectorized_matmul(a: Tensor, b: Tensor) -> Tensor
def fused_gelu(x: Tensor) -> Tensor  # Fuse operations

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, loss_scale=1024)
    def train_step(self, batch)
    def scale_loss(self, loss)
```

**Dependencies:** All previous
**Export:** `#| default_exp optimization.acceleration`
**Tests:** Speedup measurement, numerical stability
**Systems Focus:** Compute intensity, bandwidth limits

---

### Module 17: Quantization
**Learning Objective:** Can I reduce model precision without breaking it?

**Implementation Requirements:**
```python
def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """Return quantized tensor, scale, zero_point"""

class QuantizedLinear:
    def __init__(self, linear_layer: Linear)
    def forward(self, x: Tensor) -> Tensor

def quantize_model(model) -> None:
    """In-place quantization of all Linear layers"""
```

**Dependencies:** All previous
**Export:** `#| default_exp optimization.quantization`
**Tests:** Accuracy preservation, actual memory reduction
**Systems Focus:** Quantization error, INT8 vs FP16

---

### Module 18: Compression
**Learning Objective:** Can I make models smaller?

**Implementation Requirements:**
```python
def magnitude_prune(model, sparsity=0.9):
    """Remove weights below threshold"""

def structured_prune(model, prune_ratio=0.5):
    """Remove entire channels/neurons"""

def measure_sparsity(model) -> float:
    """Calculate percentage of zero weights"""
```

**Dependencies:** All previous
**Export:** `#| default_exp optimization.compression`
**Tests:** Sparsity achieved, model still works
**Systems Focus:** Structured vs unstructured, lottery ticket

---

### Module 19: Benchmarking
**Learning Objective:** Can I fairly compare different approaches?

**Implementation Requirements:**
```python
class Benchmark:
    def __init__(self, models: List, datasets: List, metrics: List[str])
    def run(self) -> pd.DataFrame
    def plot_results(self)
    def generate_report(self) -> str

def compare_models(model1, model2, test_data) -> Dict[str, float]
def plot_pareto_frontier(results: pd.DataFrame)
```

**Dependencies:** All previous
**Export:** `#| default_exp benchmarking.benchmark`
**Tests:** Metric calculation, report generation
**Systems Focus:** Latency vs throughput, energy efficiency

---

## 🪜 **Milestone 5: Systems Capstone (After Module 19)**
**Location:** `milestones/05_systems_capstone/`
**Deliverable:** Profile and optimize CNN vs TinyGPT
- Apply quantization and pruning
- Generate comparison report
- Show accuracy vs speed trade-offs
**Success Criteria:** 2× speedup with <5% accuracy loss
**Unlock:** Complete modules 15-19 + integration test

---

## 📋 Implementation Checklist for Module Developer

### For EACH Module:

**Setup:**
- [ ] Create `modules/XX_name/name_dev.py`
- [ ] Add jupytext headers
- [ ] Add export directive (#| default_exp)

**Implementation:**
- [ ] Follow API specs exactly
- [ ] Use ONLY prior modules
- [ ] Include dormant features in Module 01
- [ ] NO monkey-patching ever

**Testing:**
- [ ] Unit tests after each function
- [ ] Integration test at module end
- [ ] Test in isolation (only prior deps)

**Systems Analysis:**
- [ ] Memory profiling (if appropriate)
- [ ] Complexity analysis
- [ ] Production comparison

**Documentation:**
- [ ] Clear student introduction
- [ ] Explain dormant features properly
- [ ] NBGrader metadata

**Validation:**
- [ ] Run `test_module()`
- [ ] Export with `tito module complete XX`
- [ ] Verify checkpoint passes

---

## 🚀 Implementation Order

1. **Phase 1:** Modules 01-04 → Milestone 1 (Perceptron)
2. **Phase 2:** Modules 05-07 → Milestone 2 (MLP)
3. **Phase 3:** Modules 08-09 → Milestone 3 (CNN)
4. **Phase 4:** Modules 10-14 → Milestone 4 (TinyGPT)
5. **Phase 5:** Modules 15-19 → Milestone 5 (Systems)

---

## 🎯 Critical Design Decisions

### 1. **Single Tensor Class**
- Module 01 creates Tensor with dormant gradient features
- Module 05 activates these features (no new class!)
- No Variable class, no monkey-patching

### 2. **Progressive Dependencies**
- Each module uses ONLY previous modules
- No forward references allowed
- Tests work at each stage

### 3. **Milestone Structure**
- Separate `milestones/` directory
- Unlocked after module groups complete
- Colab-compatible notebooks

### 4. **Systems Focus**
- Every module includes performance analysis
- Memory profiling where appropriate
- Production context comparisons

This is the complete, definitive plan for TinyTorch development.