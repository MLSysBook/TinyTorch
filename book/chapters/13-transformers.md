---
title: "Transformers - Complete Encoder-Decoder Architecture"
description: "Build full transformer models with encoder and decoder stacks"
difficulty: 4
time_estimate: "6-8 hours"
prerequisites: ["Embeddings", "Attention"]
next_steps: ["KV Caching (Performance Tier)"]
learning_objectives:
  - "Implement complete transformer blocks with attention and feedforward layers"
  - "Design encoder stacks for bidirectional understanding (BERT-style)"
  - "Build decoder stacks for autoregressive generation (GPT-style)"
  - "Understand layer normalization and residual connections for deep networks"
  - "Apply transformer architectures to language modeling and generation tasks"
---

# 13. Transformers

**🏛️ ARCHITECTURE TIER** | Difficulty: ⭐⭐⭐⭐ (4/4) | Time: 6-8 hours

## Overview

Build complete transformer models by composing attention, feedforward, and normalization layers. This module implements encoder stacks (BERT-style) and decoder stacks (GPT-style) that power all modern language models.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement complete transformer blocks** with multi-head attention, feedforward networks, and normalization
2. **Design encoder stacks** for bidirectional understanding using masked self-attention (BERT-style)
3. **Build decoder stacks** for autoregressive text generation with causal masking (GPT-style)
4. **Understand layer normalization and residual connections** critical for training deep transformer networks
5. **Apply transformer architectures** to language modeling, text generation, and sequence-to-sequence tasks

## Why This Matters

### Production Context

Transformers are the architecture of modern AI:

- **GPT-4**: 96-layer decoder-only transformer; powers ChatGPT and GitHub Copilot
- **BERT**: 12-layer encoder-only transformer; ranks billions of web pages for Google Search
- **T5**: Encoder-decoder transformer; Google's universal text-to-text model
- **Claude, Gemini, Llama**: All transformer-based; billions of users daily

### Historical Context

Transformers unified and dominated AI:

- **Pre-Transformer (pre-2017)**: RNNs/LSTMs for sequences; CNNs for vision; separate architectures
- **Attention is All You Need (2017)**: Pure transformer beats RNNs; parallelizable; scales efficiently
- **BERT/GPT (2018)**: Transformers dominate NLP; pre-training + fine-tuning paradigm
- **Transformers Everywhere (2020+)**: Vision (ViT), speech (Whisper), protein folding (AlphaFold), multimodal (GPT-4)

The architecture you're implementing powers virtually all modern AI systems.

## Pedagogical Pattern: Build → Use → Analyze

### 1. Build

Implement from first principles:
- Feedforward network with two linear layers and activation
- Layer normalization for training stability
- Transformer block: attention → residual → norm → FFN → residual → norm
- Encoder stack (bidirectional, BERT-style)
- Decoder stack (autoregressive, GPT-style)

### 2. Use

Apply to real problems:
- Train GPT-style decoder on Shakespeare text generation
- Build BERT-style encoder for sequence classification
- Implement encoder-decoder for sequence-to-sequence tasks
- Generate text autoregressively with sampling
- Compare encoder-only vs decoder-only architectures

### 3. Analyze

Deep-dive into architectural choices:
- Why are residual connections critical for deep transformers?
- How does layer normalization differ from batch normalization?
- When would you use encoder-only vs decoder-only vs encoder-decoder?
- Why pre-norm vs post-norm transformer blocks?
- What's the compute/memory trade-off in stacking many layers?

## Implementation Guide

### Core Components

**Feedforward Network - Position-Wise FFN**
```python
class FeedForward:
    """Position-wise feedforward network in transformer.
    
    Two linear transformations with ReLU activation:
        FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
    
    Applied identically to each position independently.
    Typically d_ff = 4 × d_model (expansion factor).
    
    Args:
        d_model: Input/output dimension (e.g., 512)
        d_ff: Hidden dimension (e.g., 2048 = 4 × 512)
        dropout: Dropout probability for regularization
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)      # (batch, seq_len, d_ff)
        x = self.relu(x)          # Nonlinearity
        x = self.dropout(x)       # Regularization
        x = self.linear2(x)       # (batch, seq_len, d_model)
        return x
```

**Layer Normalization - Training Stability**
```python
class LayerNorm:
    """Layer normalization for transformer training stability.
    
    Normalizes across feature dimension for each sample independently.
    Unlike BatchNorm, works with any batch size including batch=1.
    
    Formula: y = γ(x - μ)/√(σ² + ε) + β
    where μ, σ² computed per sample across features
    
    Why not BatchNorm?
    - Transformers process variable-length sequences
    - LayerNorm independent of batch size (better for inference)
    - Empirically works better for NLP tasks
    """
    def __init__(self, d_model, eps=1e-6):
        self.gamma = Parameter(Tensor.ones(d_model))   # Learned scale
        self.beta = Parameter(Tensor.zeros(d_model))   # Learned shift
        self.eps = eps
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta
```

**Transformer Block - Complete Layer**
```python
class TransformerBlock:
    """Single transformer layer with attention and feedforward.
    
    Architecture (Pre-Norm variant):
        x → LayerNorm → MultiHeadAttention → Residual
          → LayerNorm → FeedForward → Residual
    
    Pre-Norm (shown above) vs Post-Norm:
        - Pre-Norm: Normalize before sub-layers; better gradient flow
        - Post-Norm: Normalize after sub-layers; original Transformer paper
        - Pre-Norm generally preferred for deep models (>12 layers)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # Attention sub-layer
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        
        # Feedforward sub-layer
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
    
    def forward(self, x, mask=None):
        """Forward pass with residual connections.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Attention sub-layer with residual
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, mask)
        x = x + self.dropout1(attended)  # Residual connection
        
        # Feedforward sub-layer with residual
        normed = self.norm2(x)
        fed_forward = self.feedforward(normed)
        x = x + self.dropout2(fed_forward)  # Residual connection
        
        return x
```

**GPT-Style Decoder - Autoregressive Generation**
```python
class GPTDecoder:
    """GPT-style decoder for autoregressive language modeling.
    
    Architecture:
        Input tokens → Embed + PositionalEncoding
        → TransformerBlocks (with causal masking)
        → Linear projection to vocabulary
    
    Features:
        - Causal masking: position i can only attend to positions ≤ i
        - Autoregressive: generates one token at a time
        - Pre-training objective: predict next token
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        # Embedding layers
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = LearnedPositionalEmbedding(max_len, d_model)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) 
                       for _ in range(num_layers)]
        
        # Output projection
        self.norm = LayerNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size)
    
    def forward(self, token_ids):
        """Forward pass through decoder.
        
        Args:
            token_ids: (batch, seq_len) token indices
        
        Returns:
            logits: (batch, seq_len, vocab_size) unnormalized predictions
        """
        batch_size, seq_len = token_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(token_ids)
        pos_embeds = self.position_embedding(seq_len)
        x = token_embeds + pos_embeds  # (batch, seq_len, d_model)
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        # Output projection
        x = self.norm(x)
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """Autoregressive text generation.
        
        Args:
            start_tokens: (batch, start_len) initial sequence
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            generated: (batch, start_len + max_new_tokens) full sequence
        """
        generated = start_tokens
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(generated)  # (batch, seq_len, vocab_size)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = softmax(next_token_logits, dim=-1)
            next_token = sample(probs)  # (batch, 1)
            
            # Append to sequence
            generated = concat([generated, next_token], dim=1)
        
        return generated
```

**BERT-Style Encoder - Bidirectional Understanding**
```python
class BERTEncoder:
    """BERT-style encoder for bidirectional sequence understanding.
    
    Architecture:
        Input tokens → Embed + PositionalEncoding
        → TransformerBlocks (no causal masking)
        → Task-specific head (classification, QA, etc.)
    
    Features:
        - Bidirectional: each position attends to all positions
        - Pre-training: masked language modeling (MLM)
        - Fine-tuning: task-specific heads added
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = LearnedPositionalEmbedding(max_len, d_model)
        
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) 
                       for _ in range(num_layers)]
        
        self.norm = LayerNorm(d_model)
    
    def forward(self, token_ids, attention_mask=None):
        """Forward pass through encoder.
        
        Args:
            token_ids: (batch, seq_len)
            attention_mask: Optional mask for padding tokens
        
        Returns:
            embeddings: (batch, seq_len, d_model) contextualized representations
        """
        # Embeddings
        token_embeds = self.token_embedding(token_ids)
        pos_embeds = self.position_embedding(token_ids.shape[1])
        x = token_embeds + pos_embeds
        
        # Transformer blocks (bidirectional - no causal mask)
        for block in self.blocks:
            x = block(x, mask=attention_mask)
        
        x = self.norm(x)
        return x
```

### Step-by-Step Implementation

1. **Build Feedforward Network**
   - Two linear layers with expansion factor (4×)
   - Add ReLU activation between layers
   - Include dropout for regularization
   - Test with different d_ff values

2. **Implement Layer Normalization**
   - Compute mean and std across feature dimension
   - Add learnable scale (gamma) and shift (beta)
   - Handle numerical stability with epsilon
   - Compare with batch normalization

3. **Create Transformer Block**
   - Add multi-head attention sub-layer
   - Implement residual connections
   - Add layer normalization (pre-norm placement)
   - Include feedforward sub-layer
   - Test forward and backward passes

4. **Build GPT Decoder**
   - Stack transformer blocks
   - Add token and position embeddings
   - Implement causal masking
   - Add output projection to vocabulary
   - Implement autoregressive generation

5. **Build BERT Encoder**
   - Stack transformer blocks without causal mask
   - Add bidirectional attention
   - Implement padding mask handling
   - Test on classification tasks
   - Compare with decoder architecture

## Testing

### Inline Tests (During Development)

Run inline tests while building:
```bash
cd modules/source/13_transformers
python transformers_dev.py
```

Expected output:
```
Unit Test: Transformer block...
✅ Attention + FFN sub-layers work correctly
✅ Residual connections preserve gradient flow
✅ Layer normalization stabilizes training
Progress: Transformer Block ✓

Unit Test: GPT decoder...
✅ 12-layer decoder initialized successfully
✅ Causal masking prevents future information leak
✅ Text generation produces coherent sequences
Progress: GPT Decoder ✓

Unit Test: BERT encoder...
✅ Bidirectional attention accesses all positions
✅ Padding mask ignores padding tokens correctly
✅ Encoder outputs contextualized representations
Progress: BERT Encoder ✓
```

### Export and Validate

After completing the module:
```bash
# Export to tinytorch package
tito export 13_transformers

# Run integration tests
tito test 13_transformers
```

## Where This Code Lives

```
tinytorch/
├── models/
│   ├── transformer.py          # Transformer blocks
│   ├── gpt.py                  # GPT decoder
│   └── bert.py                 # BERT encoder
└── __init__.py                 # Exposes transformer models

Usage in other modules:
>>> from tinytorch.models import GPTDecoder, BERTEncoder
>>> gpt = GPTDecoder(vocab_size=50000, d_model=768, num_layers=12, num_heads=12, d_ff=3072, max_len=1024)
>>> generated_text = gpt.generate(start_tokens, max_new_tokens=100)
```

## Systems Thinking Questions

1. **Layer Depth Trade-offs**: GPT-3 has 96 layers. What are the benefits? What are the challenges (training stability, memory, gradients)? Why can't we just use 1000 layers?

2. **Residual Connections Necessity**: Remove residual connections from a 12-layer transformer. What happens during training? Why do gradients vanish? How do residuals solve this?

3. **Pre-Norm vs Post-Norm**: Original Transformer used post-norm (norm after sub-layer). Modern transformers use pre-norm (norm before). Why? What's the gradient flow difference?

4. **Encoder vs Decoder Choice**: When would you use encoder-only (BERT), decoder-only (GPT), or encoder-decoder (T5)? What tasks suit each architecture?

5. **Memory Scaling**: A 12-layer transformer with d_model=768 has how many parameters? How does this scale with layers, dimensions, and vocabulary size? What's the memory footprint?

## Real-World Connections

### Industry Applications

**Large Language Models (OpenAI, Anthropic, Google)**
- GPT-4: 96-layer decoder stack, trained on trillions of tokens
- Claude: Decoder-only architecture with constitutional AI training
- PaLM 2: Decoder with 340B parameters across 64 layers
- Gemini: Multimodal transformer processing text, images, audio

**Search and Understanding (Google, Microsoft)**
- BERT powers Google Search ranking for billions of queries daily
- Bing uses transformer encoder for semantic search
- Question-answering systems built on BERT fine-tuning
- Document understanding and summarization

**Code Generation (GitHub, Google, Meta)**
- Copilot: GPT-based decoder trained on GitHub code
- AlphaCode: Transformer decoder for competitive programming
- CodeLlama: Specialized decoder for code completion
- All use decoder-only transformer architecture

### Research Impact

This module implements patterns from:
- Transformer (Vaswani et al., 2017): The foundational architecture
- BERT (Devlin et al., 2018): Bidirectional encoder pre-training
- GPT-2/3 (Radford et al., 2019): Decoder-only scaling
- T5 (Raffel et al., 2020): Unified encoder-decoder framework

## What's Next?

In **Module 14: KV Caching** (Performance Tier), you'll optimize transformers for production:

- Cache key and value matrices to avoid recomputation
- Reduce inference latency by 10-100× for long sequences
- Understand memory vs compute trade-offs in production serving
- Implement the optimization used by ChatGPT and all production LLMs

The transformers you built are complete—now it's time to make them fast!

---

**Ready to build GPT and BERT from scratch?** Open `modules/source/13_transformers/transformers_dev.py` and start implementing.
