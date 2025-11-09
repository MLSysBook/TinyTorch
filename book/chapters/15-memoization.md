---
title: "Memoization - Computational Reuse for Inference"
description: "Apply memoization pattern to transformers through KV caching for 10-15x faster generation"
difficulty: 2
time_estimate: "4-5 hours"
prerequisites: ["Profiling", "Transformers"]
next_steps: ["Quantization"]
learning_objectives:
  - "Understand memoization as a fundamental optimization pattern"
  - "Apply memoization to transformers through KV caching"
  - "Implement cache management for efficient inference"
  - "Measure O(n²) to O(n) performance improvement"
  - "Recognize when computational reuse applies to other problems"
---

# 15. Memoization

**⚡ OPTIMIZATION TIER** | Difficulty: ⭐⭐ (2/4) | Time: 4-5 hours

## Overview

Learn memoization - a fundamental optimization pattern that caches computational results to avoid redundant work. You'll apply this pattern to transformers through KV (Key-Value) caching, achieving 10-15× speedup for autoregressive generation by storing and reusing attention keys and values.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement KV caching** to eliminate redundant attention key/value computations during generation
2. **Design cache management systems** for efficient multi-turn conversation handling
3. **Understand memory-speed trade-offs** between caching everything vs recomputing on-the-fly
4. **Optimize transformer latency** from O(n²) to O(n) per generated token
5. **Apply caching patterns** used in ChatGPT, Claude, and all production language models

## Why This Matters

### Production Context

KV caching is mandatory for production LLM serving:

- **ChatGPT** uses KV caching for all multi-turn conversations; without it, latency would be unusable
- **Claude** caches up to 100K tokens of context; enables long document processing
- **GitHub Copilot** caches code context; provides real-time completions
- **Google Gemini** uses multi-level caching; serves billions of requests daily

### Historical Context

Caching evolved with transformer deployment:

- **Early Transformers (2017-2019)**: No caching; research focused on training, not inference
- **GPT-2 Deployment (2019)**: KV caching implemented; enabled practical text generation
- **Production Scale (2020+)**: Multi-level caching (KV + intermediate layers); critical for economics
- **Modern Systems (2023+)**: Distributed caching across GPUs; 100K+ token contexts

Without KV caching, ChatGPT would be 50-100× slower and economically infeasible.

## Pedagogical Pattern: Build → Use → Optimize

### 1. Build

Implement from first principles:
- KV cache data structure for attention
- Cache management (append, reuse, clear)
- Cached attention forward pass
- Multi-turn conversation caching
- Memory-efficient cache storage

### 2. Use

Apply to real problems:
- Optimize GPT decoder for text generation
- Cache conversation history for multi-turn chat
- Measure latency improvement (10-100× speedup)
- Profile memory usage vs cache size
- Compare cached vs non-cached inference

### 3. Optimize

Production-ready enhancements:
- Implement cache eviction policies (LRU, FIFO)
- Add distributed caching across GPUs
- Optimize memory layout for cache hits
- Compress cached values (quantization)
- Build cache warmup strategies

## Implementation Guide

### Core Components

**Understanding the Problem - Why Caching Helps**
```python
# WITHOUT KV caching (naive autoregressive generation):
# Generate token 1: compute attention for [t0]
# Generate token 2: compute attention for [t0, t1]  ← recomputes t0
# Generate token 3: compute attention for [t0, t1, t2]  ← recomputes t0, t1
# Generate token n: compute attention for [t0, ..., tn]  ← recomputes everything
# 
# Complexity: O(n²) - quadratic in sequence length
# For 100 tokens: ~5000 attention operations

# WITH KV caching:
# Generate token 1: compute K,V for [t0], cache them
# Generate token 2: reuse cached K,V for t0, compute only for t1
# Generate token 3: reuse cached K,V for t0,t1, compute only for t2
# Generate token n: reuse all cached, compute only for tn
#
# Complexity: O(n) - linear in sequence length
# For 100 tokens: ~100 attention operations (50× speedup!)
```

**KV Cache Data Structure**
```python
class KVCache:
    """Cache for attention keys and values.
    
    Stores computed K,V matrices to avoid recomputation during
    autoregressive generation.
    
    Memory layout:
        keys: (num_layers, batch, num_heads, seq_len, d_k)
        values: (num_layers, batch, num_heads, seq_len, d_v)
    
    For GPT-2:
        12 layers × 12 heads × 1024 seq × 64 dims = ~9M values
        At FP16 (2 bytes): 18MB per batch item
    """
    def __init__(self, num_layers, batch_size, num_heads, d_k, d_v, max_seq_len):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Pre-allocate cache tensors
        self.keys = {}   # {layer_idx: (batch, heads, seq_len, d_k)}
        self.values = {} # {layer_idx: (batch, heads, seq_len, d_v)}
        
        # Track current sequence length
        self.seq_len = 0
    
    def append(self, layer_idx, new_keys, new_values):
        """Append new keys/values to cache for a layer.
        
        Args:
            layer_idx: Which transformer layer
            new_keys: (batch, heads, 1, d_k) - single new position
            new_values: (batch, heads, 1, d_v) - single new position
        """
        if layer_idx not in self.keys:
            # Initialize cache for this layer
            self.keys[layer_idx] = new_keys
            self.values[layer_idx] = new_values
        else:
            # Concatenate with existing cache
            self.keys[layer_idx] = concat([self.keys[layer_idx], new_keys], dim=2)
            self.values[layer_idx] = concat([self.values[layer_idx], new_values], dim=2)
        
        # Update sequence length (same across all layers)
        self.seq_len = self.keys[layer_idx].shape[2]
    
    def get(self, layer_idx):
        """Retrieve cached keys/values for a layer.
        
        Returns:
            keys: (batch, heads, seq_len, d_k)
            values: (batch, heads, seq_len, d_v)
        """
        return self.keys.get(layer_idx), self.values.get(layer_idx)
    
    def clear(self):
        """Clear all cached data."""
        self.keys.clear()
        self.values.clear()
        self.seq_len = 0
    
    def memory_usage(self):
        """Calculate cache memory usage in bytes."""
        total_elements = 0
        for k, v in zip(self.keys.values(), self.values.values()):
            total_elements += k.numel() + v.numel()
        # Assume FP16 (2 bytes per element)
        return total_elements * 2
```

**Cached Attention Layer**
```python
class CachedMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with KV caching support.
    
    Extends MultiHeadAttention to cache K,V matrices during generation.
    """
    def forward(self, query, key=None, value=None, kv_cache=None, layer_idx=None):
        """Forward pass with optional KV caching.
        
        Args:
            query: (batch, 1, d_model) - single new position
            key: (batch, seq_len, d_model) - optional, for initial pass
            value: (batch, seq_len, d_model) - optional, for initial pass
            kv_cache: KVCache object
            layer_idx: Which layer (for cache indexing)
        
        Returns:
            output: (batch, 1, d_model) - attended output
            attention_weights: (batch, heads, 1, seq_len) - for analysis
        """
        batch_size = query.shape[0]
        
        # Project query for new position
        Q = self.W_q(query)  # (batch, 1, d_model)
        Q = Q.reshape(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch, heads, 1, d_k)
        
        if kv_cache is not None and layer_idx is not None:
            # Check if cache exists for this layer
            cached_K, cached_V = kv_cache.get(layer_idx)
            
            if cached_K is None:
                # First token: compute and cache K,V
                K = self.W_k(key)
                V = self.W_v(value)
                K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                
                # Cache for future tokens
                kv_cache.append(layer_idx, K, V)
            else:
                # Subsequent tokens: compute only new K,V, concat with cache
                new_K = self.W_k(key)  # key is just new position
                new_V = self.W_v(value)
                new_K = new_K.reshape(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
                new_V = new_V.reshape(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
                
                # Append to cache
                kv_cache.append(layer_idx, new_K, new_V)
                
                # Use full cached K,V
                K, V = kv_cache.get(layer_idx)
        else:
            # No caching: regular attention
            K = self.W_k(key)
            V = self.W_v(value)
            K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention with cached K,V
        attended, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        # Reshape output
        attended = attended.transpose(1, 2).reshape(batch_size, 1, self.d_model)
        output = self.W_o(attended)
        
        return output, attention_weights
```

**Cached Generation - The Full Pipeline**
```python
def generate_with_cache(model, start_tokens, max_new_tokens, temperature=1.0):
    """Autoregressive generation with KV caching.
    
    Achieves 10-100× speedup over non-cached generation.
    
    Args:
        model: Transformer with KV cache support
        start_tokens: (batch, start_len) initial sequence
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
    
    Returns:
        generated: (batch, start_len + max_new_tokens) full sequence
    """
    batch_size = start_tokens.shape[0]
    generated = start_tokens
    
    # Initialize KV cache
    kv_cache = KVCache(
        num_layers=model.num_layers,
        batch_size=batch_size,
        num_heads=model.num_heads,
        d_k=model.d_k,
        d_v=model.d_k,
        max_seq_len=start_tokens.shape[1] + max_new_tokens
    )
    
    # Process initial sequence (fills cache)
    _ = model.forward(start_tokens, kv_cache=kv_cache)
    
    # Generate tokens one at a time (uses cache)
    for _ in range(max_new_tokens):
        # Forward pass on ONLY the last token
        # Cache provides context from all previous tokens
        last_token = generated[:, -1:]  # (batch, 1)
        logits = model.forward(last_token, kv_cache=kv_cache)  # (batch, 1, vocab_size)
        
        # Sample next token
        next_token_logits = logits[:, -1, :] / temperature
        probs = softmax(next_token_logits, dim=-1)
        next_token = sample(probs)
        
        # Append to sequence
        generated = concat([generated, next_token], dim=1)
    
    return generated
```

### Step-by-Step Implementation

1. **Design KV Cache Structure**
   - Create storage for keys and values per layer
   - Support appending new keys/values efficiently
   - Add retrieval and clearing methods
   - Calculate memory usage

2. **Modify Attention for Caching**
   - Add KV cache parameter to forward pass
   - Check if cache exists for current layer
   - Compute only new K,V when cache present
   - Concat new K,V with cached values

3. **Implement Cached Generation**
   - Initialize cache before generation loop
   - Process initial tokens (fill cache)
   - Generate new tokens using cached context
   - Measure speedup vs non-cached

4. **Add Cache Management**
   - Implement cache clearing between conversations
   - Add cache size limits and eviction
   - Support batch processing with caching
   - Handle variable sequence lengths

5. **Optimize Memory Layout**
   - Use contiguous tensors for cache hits
   - Implement FP16 caching for memory savings
   - Add cache compression (quantization)
   - Profile memory bandwidth bottlenecks

## Testing

### Inline Tests (During Development)

Run inline tests while building:
```bash
cd modules/source/14_kvcaching
python kvcaching_dev.py
```

Expected output:
```
Unit Test: KV cache data structure...
✅ Cache initialization successful
✅ Append and retrieval work correctly
✅ Memory usage calculated: 18MB per batch
Progress: KV Cache ✓

Unit Test: Cached attention...
✅ First token: K,V computed and cached
✅ Subsequent tokens: reuse cached K,V
✅ Attention output matches non-cached version
Progress: Cached Attention ✓

Unit Test: Generation with caching...
✅ Generated 100 tokens with caching
✅ Speedup: 47× faster than without cache
✅ Output quality: identical to non-cached
Progress: Cached Generation ✓
```

### Export and Validate

After completing the module:
```bash
# Export to tinytorch package
tito export 14_kvcaching

# Run integration tests
tito test 14_kvcaching
```

## Where This Code Lives

```
tinytorch/
├── nn/
│   └── kvcache.py              # Your implementation goes here
└── __init__.py                 # Exposes KVCache, CachedMultiHeadAttention

Usage in other modules:
>>> from tinytorch.nn import KVCache, CachedMultiHeadAttention
>>> cache = KVCache(num_layers=12, batch_size=1, num_heads=12, d_k=64, d_v=64, max_seq_len=1024)
>>> generated = generate_with_cache(model, start_tokens, max_new_tokens=100)
```

## Systems Thinking Questions

1. **Memory-Speed Trade-off**: KV cache uses 18MB per batch for GPT-2. For batch=32, that's 576MB. What if you have 8GB GPU? How many concurrent users can you serve? What's the trade-off?

2. **Cache Invalidation**: In multi-turn chat, when should you clear the cache? What if context exceeds max_seq_len? How do production systems handle this?

3. **Distributed Caching**: For models too large for one GPU, you need tensor parallelism. How do you partition the KV cache across GPUs? What's the communication overhead?

4. **Quantized Caching**: Storing cache in INT8 instead of FP16 saves 50% memory. What's the accuracy impact? When is this worth it?

5. **Speculation and Prefetching**: What if you predict the next query and pre-compute KV cache? How would you implement speculative caching?

## Real-World Connections

### Industry Applications

**Conversational AI (OpenAI ChatGPT, Anthropic Claude)**
- KV caching for all multi-turn conversations
- Cache eviction policies for context window limits
- Memory-speed trade-offs define pricing ($/1M tokens)
- Without caching, latency would be 50-100× worse

**Code Completion (GitHub Copilot, Cursor)**
- Real-time caching of code context
- Incremental updates as user types
- Low-latency requirements (< 100ms) mandate caching
- Cache hit rates directly impact user experience

**Search and Retrieval (Perplexity, Bing AI)**
- Cache document embeddings and attention
- Multi-stage caching (retrieval + generation)
- Distributed caching across data centers
- Cache warmup for popular queries

### Research Impact

This module implements patterns from:
- GPT-2 (2019): First large-scale use of KV caching
- Megatron-LM (2020): Distributed KV caching across GPUs
- FlashAttention (2022): Memory-efficient attention without full caching
- PagedAttention (2023): Virtual memory for KV cache management

## What's Next?

In **Module 15: Profiling**, you'll measure where time goes in your transformer:

- Profile attention, feedforward, and embedding operations
- Identify computational bottlenecks beyond caching
- Measure FLOPs, memory bandwidth, and latency
- Understand performance characteristics across architectures

The caching you implemented solves the biggest inference bottleneck—now let's find what else to optimize!

---

**Ready to implement production-critical caching?** Open `modules/source/14_kvcaching/kvcaching_dev.py` and start implementing.
