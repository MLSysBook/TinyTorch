---
title: "Embeddings - Token to Vector Representations"
description: "Build embedding layers that convert discrete tokens to dense vectors"
difficulty: 2
time_estimate: "4-5 hours"
prerequisites: ["Tensor", "Tokenization"]
next_steps: ["Attention"]
learning_objectives:
  - "Implement embedding layers with efficient lookup table operations"
  - "Design positional encodings to capture sequence order information"
  - "Understand memory scaling with vocabulary size and embedding dimensions"
  - "Optimize embedding lookups for cache efficiency and bandwidth"
  - "Apply dimensionality principles to semantic vector representations"
---

# 11. Embeddings

**🏛️ ARCHITECTURE TIER** | Difficulty: ⭐⭐ (2/4) | Time: 4-5 hours

## Overview

Build embedding systems that transform discrete token IDs into dense vector representations. This module implements lookup tables, positional encodings, and optimization techniques that power all modern language models.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement embedding layers** with efficient lookup table operations for token-to-vector conversion
2. **Design positional encodings** (learned and sinusoidal) to capture sequence order information
3. **Understand memory scaling** with vocabulary size and embedding dimensions in production models
4. **Optimize embedding lookups** for cache efficiency and memory bandwidth utilization
5. **Apply dimensionality principles** to balance expressiveness and computational efficiency

## Why This Matters

### Production Context

Embeddings are the foundation of all modern NLP:

- **GPT-3's embedding table**: 50K vocab × 12K dims = 600M parameters (20% of total model)
- **BERT's embeddings**: Token + position + segment embeddings enable bidirectional understanding
- **Word2Vec/GloVe**: Pioneered semantic embeddings; "king - man + woman ≈ queen"
- **Recommendation systems**: Embedding tables for billions of items (YouTube, Netflix, Spotify)

### Historical Context

Embeddings evolved from sparse to dense representations:

- **One-Hot Encoding (pre-2013)**: Vocabulary-sized vectors; no semantic similarity
- **Word2Vec (2013)**: Dense embeddings capture semantic relationships; revolutionized NLP
- **GloVe (2014)**: Global co-occurrence statistics improve quality
- **Contextual Embeddings (2018)**: BERT/GPT embeddings depend on context; same word, different vectors
- **Modern Scale (2020+)**: 100K+ vocabulary embeddings in production language models

The embeddings you're building are the input layer of transformers and all modern NLP.

## Pedagogical Pattern: Build → Use → Analyze

### 1. Build

Implement from first principles:
- Embedding layer with learnable lookup table
- Sinusoidal positional encoding (Transformer-style)
- Learned positional embeddings (GPT-style)
- Combined token + position embeddings
- Gradient flow through embedding lookups

### 2. Use

Apply to real problems:
- Convert token sequences to dense vectors
- Add positional information for sequence order
- Visualize embedding spaces with t-SNE
- Measure semantic similarity with cosine distance
- Integrate with attention mechanisms (Module 12)

### 3. Analyze

Deep-dive into design trade-offs:
- How does embedding dimension affect model capacity?
- Why do transformers need positional encodings?
- What's the memory cost of large vocabularies?
- How do embeddings capture semantic relationships?
- Why sinusoidal vs learned position encodings?

## Implementation Guide

### Core Components

**Embedding Layer - Token Lookup Table**
```python
class Embedding:
    """Learnable embedding layer for token-to-vector conversion.
    
    Implements efficient lookup table that maps token IDs to dense vectors.
    The core component of all language models.
    
    Args:
        vocab_size: Size of vocabulary (e.g., 50,000 for GPT-2)
        embedding_dim: Dimension of dense vectors (e.g., 768 for BERT-base)
    
    Memory: vocab_size × embedding_dim parameters
    Example: 50K vocab × 768 dim = 38M parameters
    """
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embedding table randomly
        # Shape: (vocab_size, embedding_dim)
        self.weight = Tensor.randn(vocab_size, embedding_dim) * 0.02
    
    def forward(self, token_ids):
        """Look up embeddings for token IDs.
        
        Args:
            token_ids: (batch_size, seq_len) tensor of token IDs
        
        Returns:
            embeddings: (batch_size, seq_len, embedding_dim) dense vectors
        """
        batch_size, seq_len = token_ids.shape
        
        # Lookup operation: index into embedding table
        embeddings = self.weight[token_ids]  # Advanced indexing
        
        return embeddings
    
    def backward(self, grad_output):
        """Gradients accumulate in embedding table.
        
        Only embeddings that were looked up receive gradients.
        This is sparse gradient update - critical for efficiency.
        """
        batch_size, seq_len, embed_dim = grad_output.shape
        
        # Accumulate gradients for each unique token ID
        grad_weight = Tensor.zeros_like(self.weight)
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = token_ids[b, s]
                grad_weight[token_id] += grad_output[b, s]
        
        return grad_weight
```

**Positional Encoding - Sinusoidal (Transformer-Style)**
```python
class SinusoidalPositionalEncoding:
    """Fixed sinusoidal positional encoding.
    
    Used in original Transformer (Vaswani et al., 2017).
    Encodes absolute position using sine/cosine functions of different frequencies.
    
    Advantages:
    - No learned parameters
    - Can generalize to longer sequences than training length
    - Mathematically elegant relative position representation
    """
    def __init__(self, max_seq_len, embedding_dim):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Pre-compute positional encodings
        self.encodings = self._compute_encodings()
    
    def _compute_encodings(self):
        """Compute sinusoidal position encodings.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                         -(np.log(10000.0) / self.embedding_dim))
        
        encodings = np.zeros((self.max_seq_len, self.embedding_dim))
        encodings[:, 0::2] = np.sin(position * div_term)  # Even indices
        encodings[:, 1::2] = np.cos(position * div_term)  # Odd indices
        
        return Tensor(encodings)
    
    def forward(self, seq_len):
        """Return positional encodings for sequence length.
        
        Args:
            seq_len: Length of input sequence
        
        Returns:
            pos_encodings: (seq_len, embedding_dim) positional vectors
        """
        return self.encodings[:seq_len]
```

**Learned Positional Embeddings (GPT-Style)**
```python
class LearnedPositionalEmbedding:
    """Learned positional embeddings.
    
    Used in GPT models. Learns absolute position representations during training.
    
    Advantages:
    - Can learn task-specific position patterns
    - Often performs slightly better than sinusoidal
    
    Disadvantages:
    - Cannot generalize beyond max trained sequence length
    - Requires additional parameters
    """
    def __init__(self, max_seq_len, embedding_dim):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Learnable position embedding table
        self.weight = Tensor.randn(max_seq_len, embedding_dim) * 0.02
    
    def forward(self, seq_len):
        """Look up learned position embeddings.
        
        Args:
            seq_len: Length of input sequence
        
        Returns:
            pos_embeddings: (seq_len, embedding_dim) learned vectors
        """
        return self.weight[:seq_len]
```

**Combined Token + Position Embeddings**
```python
def get_combined_embeddings(token_ids, token_embeddings, pos_embeddings):
    """Combine token and position embeddings.
    
    Used as input to transformer models.
    
    Args:
        token_ids: (batch_size, seq_len) token indices
        token_embeddings: Embedding layer for tokens
        pos_embeddings: Positional encoding layer
    
    Returns:
        combined: (batch_size, seq_len, embedding_dim) token + position
    """
    batch_size, seq_len = token_ids.shape
    
    # Get token embeddings
    token_vecs = token_embeddings(token_ids)  # (B, L, D)
    
    # Get position embeddings
    pos_vecs = pos_embeddings(seq_len)        # (L, D)
    
    # Add them together (broadcasting handles batch dimension)
    combined = token_vecs + pos_vecs          # (B, L, D)
    
    return combined
```

### Step-by-Step Implementation

1. **Create Embedding Layer**
   - Initialize weight matrix (vocab_size × embedding_dim)
   - Implement forward pass with indexing
   - Add backward pass with sparse gradient accumulation
   - Test with small vocabulary

2. **Implement Sinusoidal Positions**
   - Compute sine/cosine encodings
   - Handle even/odd indices correctly
   - Verify periodicity properties
   - Test generalization to longer sequences

3. **Add Learned Positions**
   - Create learnable position table
   - Initialize with small random values
   - Implement forward and backward passes
   - Compare with sinusoidal encodings

4. **Combine Token + Position**
   - Add token and position embeddings
   - Handle batch broadcasting correctly
   - Verify gradient flow through both
   - Test with real tokenized sequences

5. **Analyze Embedding Spaces**
   - Visualize embeddings with t-SNE or PCA
   - Measure cosine similarity between tokens
   - Verify semantic relationships emerge
   - Profile memory and lookup efficiency

## Testing

### Inline Tests (During Development)

Run inline tests while building:
```bash
cd modules/11_embeddings
python embeddings_dev.py
```

Expected output:
```
Unit Test: Embedding layer...
✅ Lookup table created: 10K vocab × 256 dims = 2.5M parameters
✅ Forward pass shape correct: (32, 20, 256)
✅ Backward pass accumulates gradients correctly
Progress: Embedding Layer ✓

Unit Test: Sinusoidal positional encoding...
✅ Encodings computed for 512 positions
✅ Sine/cosine patterns verified
✅ Generalization to longer sequences works
Progress: Sinusoidal Positions ✓

Unit Test: Combined embeddings...
✅ Token + position addition works
✅ Gradient flows through both components
✅ Batch broadcasting handled correctly
Progress: Combined Embeddings ✓
```

### Export and Validate

After completing the module:
```bash
# Export to tinytorch package
tito export 11_embeddings

# Run integration tests
tito test 11_embeddings
```

## Where This Code Lives

```
tinytorch/
├── nn/
│   └── embeddings.py           # Your implementation goes here
└── __init__.py                 # Exposes Embedding, PositionalEncoding, etc.

Usage in other modules:
>>> from tinytorch.nn import Embedding, SinusoidalPositionalEncoding
>>> token_emb = Embedding(vocab_size=50000, embedding_dim=768)
>>> pos_emb = SinusoidalPositionalEncoding(max_len=512, dim=768)
```

## Systems Thinking Questions

1. **Memory Scaling**: GPT-3 has 50K vocab × 12K dims = 600M embedding parameters. At FP32 (4 bytes), how much memory? At FP16? Why does this matter for training vs inference?

2. **Sparse Gradients**: During training, only ~1% of vocabulary appears in each batch. How does sparse gradient accumulation save computation compared to dense updates?

3. **Embedding Dimension Choice**: BERT-base uses 768 dims, BERT-large uses 1024. How does dimension affect: (a) model capacity, (b) computation, (c) memory bandwidth?

4. **Position Encoding Trade-offs**: Sinusoidal allows generalization to any length. Learned positions are limited to max training length. When would you choose each?

5. **Semantic Geometry**: Why do word embeddings exhibit linear relationships like "king - man + woman ≈ queen"? What property of the training objective causes this?

## Real-World Connections

### Industry Applications

**Large Language Models (OpenAI, Anthropic, Google)**
- GPT-4: 100K+ vocabulary embeddings
- Embedding tables often 20-40% of total model parameters
- Optimized embedding access critical for inference latency
- Mixed-precision (FP16) embeddings save memory

**Recommendation Systems (YouTube, Netflix, Spotify)**
- Billion-scale item embeddings for personalization
- Embedding retrieval systems for fast nearest-neighbor search
- Continuous embedding updates with online learning
- Embedding quantization for serving efficiency

**Multilingual Models (Google Translate, Facebook M2M)**
- Shared embedding spaces across 100+ languages
- Cross-lingual embeddings enable zero-shot transfer
- Vocabulary size optimization for multilingual coverage
- Embedding alignment techniques for language pairs

### Research Impact

This module implements patterns from:
- Word2Vec (2013): Pioneered dense semantic embeddings
- GloVe (2014): Global co-occurrence matrix factorization
- Transformer (2017): Sinusoidal positional encodings
- BERT (2018): Contextual embeddings revolutionized NLP
- GPT (2018): Learned positional embeddings for autoregressive models

## What's Next?

In **Module 12: Attention**, you'll use these embeddings as input to attention mechanisms:

- Query, Key, Value projections from embeddings
- Scaled dot-product attention over embedded sequences
- Multi-head attention for different representation subspaces
- Self-attention that relates all positions in a sequence

The embeddings you built are the foundation input to every transformer!

---

**Ready to build embedding systems from scratch?** Open `modules/11_embeddings/embeddings_dev.py` and start implementing.
