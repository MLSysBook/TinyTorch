---
title: "Attention - The Mechanism That Powers Modern AI"
description: "Build scaled dot-product and multi-head attention from scratch"
difficulty: 3
time_estimate: "5-6 hours"
prerequisites: ["Tensor", "Layers", "Embeddings"]
next_steps: ["Transformers"]
learning_objectives:
  - "Implement scaled dot-product attention with query, key, and value matrices"
  - "Design multi-head attention for parallel attention subspaces"
  - "Understand masking strategies for causal, padding, and bidirectional attention"
  - "Build self-attention mechanisms for sequence-to-sequence modeling"
  - "Apply attention patterns that power GPT, BERT, and modern transformers"
---

# 12. Attention

**🏛️ ARCHITECTURE TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 5-6 hours

## Overview

Implement the attention mechanism that revolutionized AI. This module builds scaled dot-product attention and multi-head attention—the core components of GPT, BERT, and all modern transformer models.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement scaled dot-product attention** with query, key, and value matrices following the Transformer paper formula
2. **Design multi-head attention** for parallel attention in multiple representation subspaces
3. **Understand masking strategies** for causal (GPT-style), padding, and bidirectional (BERT-style) attention
4. **Build self-attention mechanisms** for sequence-to-sequence modeling with global context
5. **Apply attention patterns** that power all modern transformers from GPT-4 to Claude to Gemini

## Why This Matters

### Production Context

Attention is the core of modern AI:

- **GPT-4** uses 96 attention layers with 128 heads each; attention is 70% of compute
- **BERT** pioneered bidirectional attention; powers Google Search ranking
- **AlphaFold2** uses attention over protein sequences; solved 50-year protein folding problem
- **Vision Transformers** replaced CNNs in production at Google, Meta, OpenAI

### Historical Context

Attention revolutionized machine learning:

- **RNN Era (pre-2017)**: Sequential processing; no parallelism; gradient vanishing in long sequences
- **Attention is All You Need (2017)**: Pure attention architecture; parallelizable; global context
- **BERT/GPT (2018)**: Transformers dominate NLP; attention beats all previous approaches
- **Beyond NLP (2020+)**: Attention powers vision (ViT), biology (AlphaFold), multimodal (CLIP)

The attention mechanism you're implementing sparked the current AI revolution.

## Pedagogical Pattern: Build → Use → Analyze

### 1. Build

Implement from first principles:
- Scaled dot-product attention: `softmax(QK^T/√d_k)V`
- Multi-head attention with parallel heads
- Masking for causal and padding patterns
- Self-attention wrapper (Q=K=V)
- Attention visualization and interpretation

### 2. Use

Apply to real problems:
- Build language model with causal attention
- Implement BERT-style bidirectional attention
- Visualize attention patterns on real text
- Compare single-head vs multi-head performance
- Measure O(n²) computational scaling

### 3. Analyze

Deep-dive into design choices:
- Why does attention scale quadratically with sequence length?
- How do multiple heads capture different linguistic patterns?
- Why is the 1/√d_k scaling factor critical for training?
- When would you use causal vs bidirectional attention?
- What are the memory vs computation trade-offs?

## Implementation Guide

### Core Components

**Scaled Dot-Product Attention - The Heart of Transformers**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """The fundamental attention operation from 'Attention is All You Need'.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    This exact formula powers GPT, BERT, and all transformers.
    
    Args:
        Q: Query matrix (batch, heads, seq_len_q, d_k)
        K: Key matrix (batch, heads, seq_len_k, d_k)
        V: Value matrix (batch, heads, seq_len_v, d_v)
        mask: Optional mask (batch, 1, seq_len_q, seq_len_k)
    
    Returns:
        output: Attended values (batch, heads, seq_len_q, d_v)
        attention_weights: Attention probabilities (batch, heads, seq_len_q, seq_len_k)
    
    Intuition:
        Q = "What am I looking for?"
        K = "What information is available?"
        V = "What is the actual content?"
        
        Attention computes: for each query, how much should I focus on each key?
        Then uses those weights to mix the values.
    """
    # d_k = dimension of keys (and queries)
    d_k = Q.shape[-1]
    
    # Compute attention scores: QK^T
    # Shape: (batch, heads, seq_len_q, seq_len_k)
    scores = Q @ K.transpose(-2, -1)
    
    # Scale by sqrt(d_k) to prevent extreme softmax saturation
    scores = scores / math.sqrt(d_k)
    
    # Apply mask if provided (for causal or padding masking)
    if mask is not None:
        # Set masked positions to large negative value
        # After softmax, these become ~0
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention probabilities
    # Each row sums to 1: how much attention to pay to each position
    attention_weights = softmax(scores, dim=-1)
    
    # Weighted sum of values based on attention
    output = attention_weights @ V
    
    return output, attention_weights
```

**Multi-Head Attention - Parallel Attention Subspaces**
```python
class MultiHeadAttention:
    """Multi-head attention from 'Attention is All You Need'.
    
    Allows model to jointly attend to information from different
    representation subspaces at different positions.
    
    Architecture:
        Input (batch, seq_len, d_model)
          → Project to Q, K, V (each batch, seq_len, d_model)
          → Split into H heads (batch, H, seq_len, d_model/H)
          → Attention for each head in parallel
          → Concatenate heads
          → Final linear projection
        Output (batch, seq_len, d_model)
    
    Example:
        d_model = 512, num_heads = 8
        Each head processes 512/8 = 64 dimensions
        8 heads learn different attention patterns in parallel
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        
        # Output projection
        self.W_o = Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        """Multi-head attention forward pass.
        
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: Optional mask
        
        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.shape[0]
        
        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq_len_q, d_model)
        K = self.W_k(key)    # (batch, seq_len_k, d_model)
        V = self.W_v(value)  # (batch, seq_len_v, d_model)
        
        # 2. Split into multiple heads
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        # Transpose: → (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention for each head in parallel
        attended, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attended: (batch, num_heads, seq_len_q, d_k)
        
        # 4. Concatenate heads
        # Transpose: (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        # Reshape: → (batch, seq_len, d_model)
        attended = attended.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attended)
        
        return output, attention_weights
```

**Masking Utilities**
```python
def create_causal_mask(seq_len):
    """Create causal mask for autoregressive (GPT-style) attention.
    
    Prevents positions from attending to future positions.
    Position i can only attend to positions <= i.
    
    Returns:
        mask: (seq_len, seq_len) lower triangular matrix
        
    Example (seq_len=4):
        [[1, 0, 0, 0],     # Position 0 sees only position 0
         [1, 1, 0, 0],     # Position 1 sees 0,1
         [1, 1, 1, 0],     # Position 2 sees 0,1,2
         [1, 1, 1, 1]]     # Position 3 sees all
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return Tensor(mask)

def create_padding_mask(lengths, max_length):
    """Create padding mask to ignore padding tokens.
    
    Args:
        lengths: (batch_size,) actual sequence lengths
        max_length: maximum sequence length in batch
    
    Returns:
        mask: (batch_size, 1, 1, max_length) where 1=real, 0=padding
    """
    batch_size = lengths.shape[0]
    mask = np.zeros((batch_size, max_length))
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return Tensor(mask).reshape(batch_size, 1, 1, max_length)
```

### Step-by-Step Implementation

1. **Implement Scaled Dot-Product Attention**
   - Compute QK^T matmul
   - Apply 1/√d_k scaling
   - Add masking support
   - Apply softmax and value weighting
   - Verify attention weights sum to 1

2. **Build Multi-Head Attention**
   - Create Q, K, V projection layers
   - Split embeddings into multiple heads
   - Apply attention to each head in parallel
   - Concatenate head outputs
   - Add final projection layer

3. **Add Masking Utilities**
   - Implement causal mask for GPT-style models
   - Create padding mask for variable-length sequences
   - Test mask shapes and broadcasting
   - Verify masking prevents information leak

4. **Create Self-Attention Wrapper**
   - Build convenience class where Q=K=V
   - Add optional masking parameter
   - Test with real embedded sequences
   - Profile computational cost

5. **Visualize Attention Patterns**
   - Extract attention weights from forward pass
   - Plot heatmaps for different heads
   - Analyze what patterns each head learns
   - Interpret attention on real text examples

## Testing

### Inline Tests (During Development)

Run inline tests while building:
```bash
cd modules/source/12_attention
python attention_dev.py
```

Expected output:
```
Unit Test: Scaled dot-product attention...
✅ Attention scores computed correctly
✅ Softmax normalization verified (sums to 1)
✅ Output shape matches expected dimensions
Progress: Attention Mechanism ✓

Unit Test: Multi-head attention...
✅ 8 heads process 512 dims in parallel
✅ Head splitting and concatenation correct
✅ Output projection applied properly
Progress: Multi-Head Attention ✓

Unit Test: Causal masking...
✅ Future positions blocked correctly
✅ Past positions accessible
✅ Autoregressive property verified
Progress: Masking ✓
```

### Export and Validate

After completing the module:
```bash
# Export to tinytorch package
tito export 12_attention

# Run integration tests
tito test 12_attention
```

## Where This Code Lives

```
tinytorch/
├── nn/
│   └── attention.py            # Your implementation goes here
└── __init__.py                 # Exposes MultiHeadAttention, etc.

Usage in other modules:
>>> from tinytorch.nn import MultiHeadAttention
>>> attn = MultiHeadAttention(d_model=512, num_heads=8)
>>> output, weights = attn(query, key, value, mask=causal_mask)
```

## Systems Thinking Questions

1. **Quadratic Complexity**: Attention is O(n²) in sequence length. For n=1024, we compute ~1M attention scores. For n=4096 (GPT-3 context), how many? Why is this a problem for long documents?

2. **Multi-Head Benefits**: Why 8 heads of 64 dims each instead of 1 head of 512 dims? What different patterns might different heads learn (syntax vs semantics vs coreference)?

3. **Scaling Factor Impact**: Without 1/√d_k scaling, softmax gets extreme values (nearly one-hot). Why? How does this hurt gradient flow? (Hint: softmax derivative)

4. **Memory vs Compute**: Attention weights matrix is (batch × heads × seq × seq). For batch=32, heads=8, seq=1024, this is 256M values. At FP32, how much memory? Why is this a bottleneck?

5. **Causal vs Bidirectional**: GPT uses causal masking (can't see future). BERT uses bidirectional (can see all positions). Why does this architectural choice define fundamentally different models?

## Real-World Connections

### Industry Applications

**Large Language Models (OpenAI, Anthropic, Google)**
- GPT-4: 96 layers × 128 heads = 12,288 attention computations
- Attention optimizations (FlashAttention) critical for training at scale
- Multi-query attention reduces inference cost in production
- Attention is the primary computational bottleneck

**Machine Translation (Google Translate, DeepL)**
- Cross-attention aligns source and target languages
- Attention weights show word alignment (interpretability)
- Multi-head attention captures different translation patterns
- Real-time translation requires optimized attention kernels

**Vision Models (Google ViT, Meta DINOv2)**
- Self-attention over image patches replaces convolution
- Global receptive field from layer 1 (vs deep CNN stacks)
- Attention scales better to high-resolution images
- Now dominant architecture for vision tasks

### Research Impact

This module implements patterns from:
- Attention is All You Need (Vaswani et al., 2017): The transformer paper
- BERT (Devlin et al., 2018): Bidirectional attention for NLP
- GPT-2/3 (Radford et al., 2019): Causal attention for generation
- ViT (Dosovitskiy et al., 2020): Attention for computer vision

## What's Next?

In **Module 13: Transformers**, you'll compose attention into complete transformer blocks:

- Stack multi-head attention with feedforward networks
- Add layer normalization and residual connections
- Build encoder (BERT-style) and decoder (GPT-style) architectures
- Train full transformer on text generation tasks

The attention mechanism you built is the core component of every transformer!

---

**Ready to build the AI revolution from scratch?** Open `modules/source/12_attention/attention_dev.py` and start implementing.
