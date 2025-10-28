# Transformer Configuration Analysis

## Current Configuration

```python
# Default hyperparameters in vaswani_shakespeare.py
embed_dim = 128
num_layers = 4
num_heads = 4
seq_length = 64
batch_size = 32
learning_rate = 0.001
epochs = 5
ffn_hidden_dim = embed_dim * 4 = 512
```

**Total parameters: ~500K (as stated in docstring)**

---

## Comparison with Known Architectures

### 1. GPT-1 (2018) - Character-level equivalent
- **embed_dim**: 768 (6x larger)
- **num_layers**: 12 (3x deeper)
- **num_heads**: 12 (3x more)
- **context**: 512 tokens
- **Total params**: 117M

### 2. GPT-2 Small (2019)
- **embed_dim**: 768
- **num_layers**: 12
- **num_heads**: 12
- **context**: 1024
- **Total params**: 124M

### 3. Karpathy's nanoGPT (Educational, Character-level Shakespeare)
```python
# Recommended for Shakespeare character-level
embed_dim = 384          # 3x our current
num_layers = 6           # 1.5x our current
num_heads = 6            # 1.5x our current
seq_length = 256         # 4x our current
batch_size = 64          # 2x our current
learning_rate = 3e-4     # 0.3x our current (lower!)
dropout = 0.2
```
**Total params: ~10.65M** (20x our current)

### 4. minGPT (Karpathy) - Minimal viable
```python
embed_dim = 192          # 1.5x our current
num_layers = 6           # 1.5x our current
num_heads = 6            # 1.5x our current
seq_length = 128         # 2x our current
batch_size = 32          # Same as ours
learning_rate = 6e-4
```
**Total params: ~2.7M** (5.4x our current)

---

## Analysis of Current Config

### ✅ **What's Good:**

1. **Head dimension is correct**: 
   - `head_dim = embed_dim / num_heads = 128 / 4 = 32`
   - Standard practice: 32-64 per head ✅
   - Powers-of-2 for memory alignment ✅

2. **FFN ratio is correct**:
   - `ffn_hidden = embed_dim * 4 = 512`
   - This is the standard 4x expansion ✅

3. **Heads divide evenly**:
   - `128 % 4 = 0` ✅

4. **Batch size is reasonable**:
   - 32 is standard for educational implementations ✅

### ⚠️ **Potential Issues:**

1. **Model is VERY small** (500K params):
   - May struggle to capture complex patterns
   - Might underfit on Shakespeare corpus (~5MB)
   - Recommendation: **10-20x larger** for good Shakespeare generation

2. **Context length is SHORT** (64 chars):
   - Average English sentence: 15-20 words = 75-100 chars
   - Shakespeare lines often longer
   - **Recommendation: 128-256** for better context

3. **Learning rate might be HIGH** (0.001):
   - Standard for transformers: 3e-4 to 6e-4
   - **Current: 1e-3** (3x higher than typical)
   - Risk: Unstable training, missing optimal solutions
   - **Recommendation: 3e-4** (0.0003)

4. **Too few epochs** (5):
   - Character-level models typically need 20-50 epochs
   - With limited batches (100/epoch), gets very little data
   - **Recommendation: 20-50 epochs**

5. **No dropout**:
   - Standard practice: 0.1-0.2 dropout in attention and FFN
   - Risk: Overfitting (though model is small)
   - **Recommendation: Add 0.1 dropout** if model is larger

6. **No learning rate schedule**:
   - Standard: Warmup + cosine decay or linear decay
   - Helps convergence
   - **Optional but recommended** for longer training

---

## Recommended Configurations

### Option A: **Minimal (Current, for fast testing)**
```python
embed_dim = 128       # Keep for speed
num_layers = 4        # Keep
num_heads = 4         # Keep
seq_length = 64       # Keep
batch_size = 32       # Keep
learning_rate = 3e-4  # REDUCE from 1e-3
epochs = 20           # INCREASE from 5
max_batches = 500     # INCREASE from 100
```
**Expected**: Basic Shakespeare patterns, some coherence
**Training time**: ~10 minutes
**Use case**: Testing gradient flow and basic learning

### Option B: **Balanced (Good Shakespeare generation)**
```python
embed_dim = 256       # 2x increase
num_layers = 6        # 1.5x increase
num_heads = 8         # 2x increase
seq_length = 128      # 2x increase
batch_size = 32       # Keep (memory constraint)
learning_rate = 3e-4  # Standard
epochs = 30           # More training
max_batches = None    # Full dataset
dropout = 0.1         # Add regularization
```
**Expected**: Good Shakespeare-like text, proper word patterns
**Training time**: ~45 minutes (NumPy, CPU)
**Params**: ~2.5M (5x current)

### Option C: **Strong (Publication-quality)**
```python
embed_dim = 384       # 3x increase (like nanoGPT)
num_layers = 6        # 1.5x increase
num_heads = 6         # 1.5x increase
seq_length = 256      # 4x increase
batch_size = 64       # 2x increase (if memory allows)
learning_rate = 3e-4  # Standard
epochs = 50           # Full training
max_batches = None    # Full dataset
dropout = 0.2         # More regularization
```
**Expected**: Excellent Shakespeare, multi-sentence coherence
**Training time**: ~2-3 hours (NumPy, CPU)
**Params**: ~10M (20x current)

---

## Critical Issues to Fix IMMEDIATELY

### 1. ❗ Learning Rate is TOO HIGH
```python
# Current (in training function):
learning_rate = 0.001  # 1e-3

# Should be:
learning_rate = 0.0003  # 3e-4 (standard for transformers)
```

**Why this matters:**
- Adam with 1e-3 can overshoot optimal solutions
- Transformers are sensitive to LR (attention mechanism)
- Industry standard: 3e-4 to 6e-4

### 2. ❗ Too Few Training Steps
```python
# Current:
epochs = 5
max_batches_per_epoch = 100

# Total updates: 5 * 100 = 500 steps
```

**For 500K param model on 5MB text, recommend:**
- **Minimum: 2000-5000 steps** (4-10x current)
- **Better: 10000+ steps** (20x current)

```python
# Quick fix:
epochs = 20           # 4x more
# Remove max_batches limit OR increase to 500
```

### 3. ⚠️ Context Length Limiting Model
```python
# Current:
seq_length = 64  # ~10 words of context

# Better:
seq_length = 128  # ~20 words - more reasonable
```

---

## Implementation Priority

### Phase 1: IMMEDIATE (Fix Critical Issues)
1. ✅ **Lower learning rate** to 3e-4
2. ✅ **Increase epochs** to 20
3. ✅ **Remove or raise** max_batches limit

### Phase 2: IMPORTANT (Better Performance)
4. ⚠️ **Increase seq_length** to 128
5. ⚠️ **Increase embed_dim** to 256
6. ⚠️ **Add gradient clipping** (max_norm=1.0)

### Phase 3: OPTIONAL (Publication Quality)
7. 💡 Add learning rate warmup (100-500 steps)
8. 💡 Add dropout (0.1-0.2)
9. 💡 Increase to 6 layers
10. 💡 Add validation set monitoring

---

## Expected Learning Behavior

### Current Config (500K params):
- **Steps 0-100**: Loss drops quickly (4.5 → 2.0)
- **Steps 100-500**: Slow improvement (2.0 → 1.5)
- **Steps 500+**: Should plateau or improve slowly
- **Final Loss**: ~1.2-1.5 (character-level)
- **Generated Text**: Basic patterns, some real words, not coherent

### Better Config (2.5M params, Config B):
- **Steps 0-500**: Fast drop (4.5 → 1.8)
- **Steps 500-2000**: Steady improvement (1.8 → 1.2)
- **Steps 2000-5000**: Fine-tuning (1.2 → 0.9)
- **Final Loss**: ~0.8-1.0
- **Generated Text**: Coherent sentences, Shakespeare-like style

### Strong Config (10M params, Config C):
- **Final Loss**: ~0.6-0.8
- **Generated Text**: Multi-sentence coherence, proper grammar, stylistic

---

## References

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original Transformer: 512 embed_dim, 8 heads, 6 layers (encoder/decoder each)

2. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019 - GPT-2)
   - Smallest model: 124M params, 768 embed_dim

3. **Karpathy's nanoGPT** (2022)
   - https://github.com/karpathy/nanoGPT
   - Battle-tested configs for character-level Shakespeare

4. **Karpathy's minGPT** (2020)
   - https://github.com/karpathy/minGPT
   - Educational minimal implementation

5. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018 - GPT-1)
   - 117M params, learning_rate=2.5e-4

---

## TL;DR - What to Change Right Now

```python
# In vaswani_shakespeare.py, change defaults:

# Line 452: epochs
parser.add_argument('--epochs', type=int, default=20,  # was: 5
                   help='Training epochs')

# Line 458: embed-dim  
parser.add_argument('--embed-dim', type=int, default=256,  # was: 128
                   help='Embedding dimension')

# Line 456: seq-length
parser.add_argument('--seq-length', type=int, default=128,  # was: 64
                   help='Sequence length')

# Line 298: learning_rate
def train_shakespeare_gpt(model, train_loader, dataset, epochs=5, learning_rate=0.0003):  # was: 0.001

# Line 318: max_batches (in training loop)
if batch_idx >= 500:  # was: 100, OR remove entirely
    break
```

**These 4 changes will 10x your results!**

