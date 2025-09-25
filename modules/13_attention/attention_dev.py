# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Attention - The Mechanism That Revolutionized Language Understanding

Welcome to the Attention module! You'll implement the scaled dot-product attention and multi-head attention mechanisms that power modern transformer architectures and enable language models to understand complex relationships in sequences.

## Learning Goals
- Systems understanding: How attention's O(N²) complexity affects memory usage and computational scaling
- Core implementation skill: Build attention mechanisms with efficient memory management
- Pattern recognition: Understand how attention enables sequence modeling and long-range dependencies
- Framework connection: See how your implementations match PyTorch's attention systems
- Performance insight: Learn how attention patterns affect training efficiency and model capabilities

## Build → Use → Reflect
1. **Build**: Scaled dot-product attention and multi-head attention with masking and KV-cache
2. **Use**: Process sequences to capture dependencies between distant tokens
3. **Reflect**: How does attention's quadratic scaling determine practical limits of sequence length?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how attention enables transformers to model sequence relationships
- Practical capability to implement attention with memory-efficient patterns and causal masking
- Systems insight into how attention's O(N²) scaling affects model architecture and deployment
- Performance consideration of how attention optimization determines transformer feasibility
- Connection to production systems like GPT's attention layers and their optimization techniques

## Systems Reality Check
💡 **Production Context**: Attention is the memory bottleneck in transformers - GPT-3 uses 96 attention heads across 96 layers
⚡ **Performance Note**: O(N²) memory scaling means 2x sequence length = 4x attention memory - this fundamentally limits transformer sequence length
"""

# %% nbgrader={"grade": false, "grade_id": "attention-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.attention

#| export
import math
import numpy as np
import os
import sys
from typing import Union, List, Optional, Tuple, Dict

# Import our Tensor class - try from package first, then from local module
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local tensor module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    from tensor_dev import Tensor

# Try to import embedding classes
try:
    from tinytorch.core.embeddings import Embedding, PositionalEncoding
except ImportError:
    # For development, import from local module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '12_embeddings'))
    try:
        from embeddings_dev import Embedding, PositionalEncoding
    except ImportError:
        # Create minimal mock classes if not available
        class Embedding:
            def __init__(self, vocab_size, embedding_dim):
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
        class PositionalEncoding:
            def __init__(self, embedding_dim, max_seq_length=5000):
                self.embedding_dim = embedding_dim

# %% nbgrader={"grade": false, "grade_id": "attention-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🎯 TinyTorch Attention Module")
print(f"NumPy version: {np.__version__}")
print("Ready to build attention mechanisms!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/13_attention/attention_dev.py`  
**Building Side:** Code exports to `tinytorch.core.attention`

```python
# Final package structure:
from tinytorch.core.attention import ScaledDotProductAttention, MultiHeadAttention
from tinytorch.core.embeddings import Embedding, PositionalEncoding  # Previous module
from tinytorch.core.transformers import TransformerBlock  # Next module
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.MultiheadAttention`
- **Consistency:** All attention mechanisms live together in `core.attention`
- **Integration:** Works seamlessly with embeddings and transformer architectures
"""

# %% [markdown]
"""
## What is Attention?

### The Problem: Sequence Dependencies
Traditional RNNs process sequences step-by-step, making it hard to capture long-range dependencies:
```
"The cat, which was sitting on the mat, was hungry"
    ^                                      ^
    Subject must agree with verb - but they're far apart!
```

### Attention Solution
Attention allows every position to directly attend to every other position:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What can I attend to?"  
- **V (Value)**: "What information do I get?"

### Why Attention Works
- **Parallelization**: All positions computed simultaneously
- **Long-range**: Direct connections between distant tokens
- **Flexible**: Attention weights learned during training
- **Interpretable**: Attention patterns show what the model focuses on

### Systems Trade-offs
- **Memory**: O(N²) scaling with sequence length
- **Computation**: Matrix multiplications scale with sequence length²
- **Parallelization**: Highly parallelizable on GPUs
- **Sequence limits**: Quadratic scaling limits practical sequence length
"""

# %% [markdown]
"""
## Scaled Dot-Product Attention Implementation

Let's start with the core attention mechanism - scaled dot-product attention that forms the foundation of transformers.
"""

# %% nbgrader={"grade": false, "grade_id": "scaled-attention", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention mechanism.
    
    The fundamental attention computation used in transformers:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    This allows each position to attend to all positions in the sequence.
    """
    
    def __init__(self, dropout: float = 0.0, temperature: float = 1.0):
        """
        Initialize scaled dot-product attention.
        
        Args:
            dropout: Dropout rate for attention weights (not implemented in basic version)
            temperature: Temperature scaling for attention distribution
        """
        self.dropout = dropout
        self.temperature = temperature
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                mask: Optional[Tensor] = None, 
                return_attention_weights: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute scaled dot-product attention.
        
        TODO: Implement scaled dot-product attention.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Compute attention scores: query @ key.transpose()
        2. Scale by sqrt(key_dim) for numerical stability
        3. Apply mask if provided (set masked positions to large negative values)
        4. Apply softmax to get attention weights
        5. Apply attention weights to values: attention_weights @ value
        6. Return attended values (and optionally attention weights)
        
        MATHEMATICAL FOUNDATION:
        scores = QK^T / sqrt(d_k)
        attention_weights = softmax(scores)
        output = attention_weights @ V
        
        MASKING:
        - Set masked positions to -1e9 before softmax
        - This makes them effectively zero after softmax
        - Used for causal (autoregressive) attention
        
        Args:
            query: Query tensor with shape (batch_size, seq_len_q, d_k)
            key: Key tensor with shape (batch_size, seq_len_k, d_k)
            value: Value tensor with shape (batch_size, seq_len_v, d_v)
            mask: Optional mask tensor with shape (seq_len_q, seq_len_k) or broadcastable
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Attended values with shape (batch_size, seq_len_q, d_v)
            Optionally also attention weights with shape (batch_size, seq_len_q, seq_len_k)
        """
        ### BEGIN SOLUTION
        # Get dimensions
        batch_size, seq_len_q, d_k = query.shape
        _, seq_len_k, _ = key.shape
        _, seq_len_v, d_v = value.shape
        
        assert seq_len_k == seq_len_v, "Key and Value must have same sequence length"
        
        # Step 1: Compute attention scores QK^T
        # query: (batch, seq_q, d_k), key: (batch, seq_k, d_k)
        # We need key^T, so we transpose the last two dimensions
        key_transposed = np.transpose(key.data, (0, 2, 1))  # (batch, d_k, seq_k)
        
        # Batch matrix multiplication: (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
        scores = np.matmul(query.data, key_transposed)
        
        # Step 2: Scale by sqrt(d_k) for numerical stability
        scores = scores / math.sqrt(d_k) / self.temperature
        
        # Step 3: Apply mask if provided
        if mask is not None:
            mask_value = -1e9  # Large negative value that becomes ~0 after softmax
            
            # Handle different mask shapes
            if isinstance(mask, Tensor):
                mask_array = mask.data
            else:
                mask_array = mask
                
            # Apply mask: set masked positions to large negative values
            # mask should be 1 for positions to keep, 0 for positions to mask
            masked_scores = np.where(mask_array == 0, mask_value, scores)
            scores = masked_scores
        
        # Step 4: Apply softmax to get attention weights
        # Numerical stable softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Step 5: Apply attention weights to values
        # attention_weights: (batch, seq_q, seq_k), value: (batch, seq_k, d_v)
        # Result: (batch, seq_q, d_v)
        attended_values = np.matmul(attention_weights, value.data)
        
        output = Tensor(attended_values)
        
        if return_attention_weights:
            return output, Tensor(attention_weights)
        else:
            return output
        ### END SOLUTION
    
    def __call__(self, query: Tensor, key: Tensor, value: Tensor, 
                 mask: Optional[Tensor] = None, 
                 return_attention_weights: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Make the class callable."""
        return self.forward(query, key, value, mask, return_attention_weights)

# %% [markdown]
"""
### 🧪 Test Your Scaled Dot-Product Attention Implementation

Once you implement the ScaledDotProductAttention forward method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-scaled-attention-immediate", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_scaled_attention():
    """Unit test for scaled dot-product attention."""
    print("🔬 Unit Test: Scaled Dot-Product Attention...")
    
    # Create attention layer
    attention = ScaledDotProductAttention()
    
    # Test basic attention computation
    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 6
    
    # Create test inputs
    query = Tensor(np.random.randn(batch_size, seq_len, d_k))
    key = Tensor(np.random.randn(batch_size, seq_len, d_k))
    value = Tensor(np.random.randn(batch_size, seq_len, d_v))
    
    # Test forward pass
    output = attention.forward(query, key, value)
    expected_shape = (batch_size, seq_len, d_v)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test with different sequence lengths
    seq_len_k = 6
    key_diff = Tensor(np.random.randn(batch_size, seq_len_k, d_k))
    value_diff = Tensor(np.random.randn(batch_size, seq_len_k, d_v))
    
    output_diff = attention.forward(query, key_diff, value_diff)
    expected_shape_diff = (batch_size, seq_len, d_v)
    assert output_diff.shape == expected_shape_diff, f"Expected shape {expected_shape_diff}, got {output_diff.shape}"
    
    # Test with attention weights return
    output, attn_weights = attention.forward(query, key, value, return_attention_weights=True)
    expected_attn_shape = (batch_size, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Expected attention shape {expected_attn_shape}, got {attn_weights.shape}"
    
    # Verify attention weights sum to 1 (softmax property)
    attn_sums = np.sum(attn_weights.data, axis=-1)  # Sum over keys for each query
    assert np.allclose(attn_sums, 1.0), "Attention weights should sum to 1"
    
    # Test with causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # Upper triangular mask
    causal_mask = 1 - causal_mask  # Flip: 1 for allowed, 0 for masked
    
    output_masked, attn_masked = attention.forward(query, key, value, 
                                                  mask=Tensor(causal_mask),
                                                  return_attention_weights=True)
    
    # Verify causal mask works - future positions should have ~0 attention
    # Upper triangular part (excluding diagonal) should be close to 0
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            assert np.all(attn_masked.data[:, i, j] < 1e-6), f"Future position ({i},{j}) should have near-zero attention"
    
    # Test callable interface
    output_callable = attention(query, key, value)
    assert np.allclose(output_callable.data, output.data), "Callable interface should work"
    
    # Test numerical stability with extreme values
    extreme_query = Tensor(np.ones((1, 2, 4)) * 100)  # Large values
    extreme_key = Tensor(np.ones((1, 2, 4)) * 100)
    extreme_value = Tensor(np.random.randn(1, 2, 4))
    
    extreme_output = attention.forward(extreme_query, extreme_key, extreme_value)
    assert not np.any(np.isnan(extreme_output.data)), "Should handle extreme values without NaN"
    assert not np.any(np.isinf(extreme_output.data)), "Should handle extreme values without inf"
    
    print("✅ Scaled dot-product attention tests passed!")
    print(f"✅ Handles various input shapes and sequence lengths")
    print(f"✅ Attention weights sum to 1 (softmax property)")
    print(f"✅ Causal masking works correctly")
    print(f"✅ Numerical stability with extreme values")

# Test function defined (called in main block)

# %% [markdown]
"""
## Multi-Head Attention Implementation

Now let's implement multi-head attention, which runs multiple attention heads in parallel and concatenates their outputs. This allows the model to attend to different types of information simultaneously.
"""

# %% nbgrader={"grade": false, "grade_id": "multi-head-attention", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    Runs multiple attention heads in parallel and combines their outputs.
    This allows the model to attend to different representation subspaces
    simultaneously, capturing diverse types of relationships.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        TODO: Implement multi-head attention initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store configuration parameters
        2. Calculate head dimension (embed_dim must be divisible by num_heads)
        3. Initialize linear projection layers for Q, K, V, and output
        4. Create scaled dot-product attention layer
        
        DESIGN DECISIONS:
        - Each head gets embed_dim // num_heads dimensions
        - Separate linear layers for Q, K, V projections
        - Output projection to combine all heads
        
        Args:
            embed_dim: Embedding dimension (total across all heads)
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        ### BEGIN SOLUTION
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Check that embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        
        # Initialize projection layers (these would be proper Linear layers in full implementation)
        # For now, we'll use simple weight matrices
        self.w_q = Tensor(np.random.randn(embed_dim, embed_dim) / math.sqrt(embed_dim))
        self.w_k = Tensor(np.random.randn(embed_dim, embed_dim) / math.sqrt(embed_dim))
        self.w_v = Tensor(np.random.randn(embed_dim, embed_dim) / math.sqrt(embed_dim))
        self.w_o = Tensor(np.random.randn(embed_dim, embed_dim) / math.sqrt(embed_dim))
        
        # Store parameters for optimization
        self.parameters = [self.w_q, self.w_k, self.w_v, self.w_o]
        
        # Create scaled dot-product attention
        self.scaled_attention = ScaledDotProductAttention(dropout=dropout)
        ### END SOLUTION
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[Tensor] = None,
                return_attention_weights: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute multi-head attention.
        
        TODO: Implement multi-head attention forward pass.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Linear projections: compute Q, K, V from inputs
        2. Reshape for multiple heads: (batch, seq, embed) -> (batch, heads, seq, head_dim)
        3. Apply scaled dot-product attention for all heads simultaneously
        4. Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed)
        5. Apply output projection
        
        RESHAPING DETAILS:
        - Input: (batch_size, seq_len, embed_dim)
        - After projection: (batch_size, seq_len, embed_dim)
        - Reshaped for heads: (batch_size, seq_len, num_heads, head_dim)
        - Transposed for attention: (batch_size, num_heads, seq_len, head_dim)
        
        Args:
            query: Query tensor with shape (batch_size, seq_len, embed_dim)
            key: Key tensor with shape (batch_size, seq_len, embed_dim)
            value: Value tensor with shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Multi-head attention output with shape (batch_size, seq_len, embed_dim)
            Optionally also attention weights from all heads
        """
        ### BEGIN SOLUTION
        batch_size, seq_len, embed_dim = query.shape
        
        # Step 1: Linear projections
        # query @ w_q: (batch, seq, embed) @ (embed, embed) -> (batch, seq, embed)
        Q = Tensor(np.matmul(query.data, self.w_q.data))
        K = Tensor(np.matmul(key.data, self.w_k.data))
        V = Tensor(np.matmul(value.data, self.w_v.data))
        
        # Step 2: Reshape for multiple heads
        # Get actual sequence lengths (may differ for cross-attention)
        query_seq_len = Q.shape[1]
        key_seq_len = K.shape[1] 
        value_seq_len = V.shape[1]
        
        # (batch, seq, embed) -> (batch, seq, num_heads, head_dim)
        Q_reshaped = Q.data.reshape(batch_size, query_seq_len, self.num_heads, self.head_dim)
        K_reshaped = K.data.reshape(batch_size, key_seq_len, self.num_heads, self.head_dim)
        V_reshaped = V.data.reshape(batch_size, value_seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq, head_dim) for easier processing
        Q_heads = np.transpose(Q_reshaped, (0, 2, 1, 3))
        K_heads = np.transpose(K_reshaped, (0, 2, 1, 3))
        V_heads = np.transpose(V_reshaped, (0, 2, 1, 3))
        
        # Step 3: Apply attention to all heads simultaneously
        # We need to reshape to (batch*num_heads, seq, head_dim) for the attention function
        batch_heads = batch_size * self.num_heads
        Q_flat = Q_heads.reshape(batch_heads, query_seq_len, self.head_dim)
        K_flat = K_heads.reshape(batch_heads, key_seq_len, self.head_dim)
        V_flat = V_heads.reshape(batch_heads, value_seq_len, self.head_dim)
        
        # Apply attention
        if return_attention_weights:
            attn_output_flat, attn_weights_flat = self.scaled_attention.forward(
                Tensor(Q_flat), Tensor(K_flat), Tensor(V_flat), 
                mask=mask, return_attention_weights=True
            )
        else:
            attn_output_flat = self.scaled_attention.forward(
                Tensor(Q_flat), Tensor(K_flat), Tensor(V_flat), mask=mask
            )
        
        # Step 4: Reshape back to separate heads
        # (batch*num_heads, seq, head_dim) -> (batch, num_heads, seq, head_dim)
        attn_output_heads = attn_output_flat.data.reshape(batch_size, self.num_heads, query_seq_len, self.head_dim)
        
        # Transpose back to (batch, seq, num_heads, head_dim)
        attn_output_reshaped = np.transpose(attn_output_heads, (0, 2, 1, 3))
        
        # Concatenate heads: (batch, seq, num_heads, head_dim) -> (batch, seq, embed_dim)
        attn_output_concat = attn_output_reshaped.reshape(batch_size, query_seq_len, embed_dim)
        
        # Step 5: Apply output projection
        output = np.matmul(attn_output_concat, self.w_o.data)
        
        if return_attention_weights:
            # Reshape attention weights back to per-head format
            # Attention weights shape: (query_seq_len, key_seq_len)
            attn_weights_heads = attn_weights_flat.data.reshape(batch_size, self.num_heads, query_seq_len, key_seq_len)
            return Tensor(output), Tensor(attn_weights_heads)
        else:
            return Tensor(output)
        ### END SOLUTION
    
    def __call__(self, query: Tensor, key: Tensor, value: Tensor,
                 mask: Optional[Tensor] = None,
                 return_attention_weights: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Make the class callable."""
        return self.forward(query, key, value, mask, return_attention_weights)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of multi-head attention parameters.
        
        This function is PROVIDED to show memory analysis.
        """
        # Parameter memory
        param_memory_mb = sum(param.data.nbytes for param in self.parameters) / (1024 * 1024)
        
        # Memory per head
        memory_per_head_mb = param_memory_mb / self.num_heads
        
        return {
            'total_parameter_memory_mb': param_memory_mb,
            'memory_per_head_mb': memory_per_head_mb,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'total_parameters': sum(param.data.size for param in self.parameters)
        }

# %% [markdown]
"""
### 🧪 Test Your Multi-Head Attention Implementation

Once you implement the MultiHeadAttention methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-multi-head-attention-immediate", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_multi_head_attention():
    """Unit test for multi-head attention."""
    print("🔬 Unit Test: Multi-Head Attention...")
    
    # Test basic configuration
    embed_dim = 64
    num_heads = 8
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Verify initialization
    assert mha.embed_dim == embed_dim, "Should store embedding dimension"
    assert mha.num_heads == num_heads, "Should store number of heads"
    assert mha.head_dim == embed_dim // num_heads, "Should calculate head dimension correctly"
    
    # Verify parameter tracking
    assert len(mha.parameters) == 4, "Should have 4 parameter matrices (Q, K, V, O)"
    for param in mha.parameters:
        assert param.shape == (embed_dim, embed_dim), "All parameters should be square matrices"
    
    # Test forward pass
    batch_size = 2
    seq_len = 6
    
    query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    
    output = mha.forward(query, key, value)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test with attention weights return
    output, attn_weights = mha.forward(query, key, value, return_attention_weights=True)
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Expected attention shape {expected_attn_shape}, got {attn_weights.shape}"
    
    # Test different head configurations
    for test_heads in [1, 2, 4]:
        if embed_dim % test_heads == 0:
            test_mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=test_heads)
            test_output = test_mha.forward(query, key, value)
            assert test_output.shape == expected_shape, f"Should work with {test_heads} heads"
    
    # Test invalid head configuration
    try:
        invalid_mha = MultiHeadAttention(embed_dim=65, num_heads=8)  # 65 not divisible by 8
        assert False, "Should raise error for invalid head configuration"
    except ValueError:
        pass  # Expected behavior
    
    # Test with causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    causal_mask = 1 - causal_mask  # Flip: 1 for allowed, 0 for masked
    
    output_masked, attn_masked = mha.forward(query, key, value,
                                           mask=Tensor(causal_mask),
                                           return_attention_weights=True)
    
    # Verify masking works across all heads
    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                assert np.all(attn_masked.data[:, head, i, j] < 1e-5), \
                    f"Head {head}: Future position ({i},{j}) should have near-zero attention"
    
    # Test callable interface
    output_callable = mha(query, key, value)
    assert output_callable.shape == expected_shape, "Callable interface should work"
    
    # Test memory usage calculation
    memory_stats = mha.get_memory_usage()
    assert 'total_parameter_memory_mb' in memory_stats, "Should provide memory statistics"
    assert memory_stats['num_heads'] == num_heads, "Should report correct number of heads"
    assert memory_stats['head_dim'] == embed_dim // num_heads, "Should report correct head dimension"
    
    # Test self-attention (Q=K=V)
    self_attn_output = mha.forward(query, query, query)
    assert self_attn_output.shape == expected_shape, "Self-attention should work"
    
    print("✅ Multi-head attention tests passed!")
    print(f"✅ Handles {num_heads} heads with {mha.head_dim} dimensions each")
    print(f"✅ Parameter memory: {memory_stats['total_parameter_memory_mb']:.2f}MB")
    print(f"✅ Causal masking works across all heads")
    print(f"✅ Self-attention capability verified")

# Test function defined (called in main block)

# %% [markdown]
"""
## KV-Cache for Efficient Inference

For autoregressive generation (like GPT), we can cache key and value computations to avoid recomputing them for each new token. Let's implement a simple KV-cache system:
"""

# %% nbgrader={"grade": false, "grade_id": "kv-cache", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    
    During text generation, we generate one token at a time. Instead of
    recomputing K and V for all previous tokens, we can cache them and
    only compute K and V for the new token.
    """
    
    def __init__(self, max_batch_size: int, max_seq_length: int, 
                 num_heads: int, head_dim: int):
        """
        Initialize KV cache with pre-allocated memory.
        
        TODO: Implement KV cache initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store cache configuration parameters
        2. Pre-allocate memory for cached keys and values
        3. Initialize cache position tracking
        4. Set up cache state management
        
        PRE-ALLOCATION BENEFITS:
        - Avoids memory allocation during generation
        - Enables efficient memory reuse
        - Predictable memory usage
        
        Args:
            max_batch_size: Maximum batch size for generation
            max_seq_length: Maximum sequence length to cache
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
        """
        ### BEGIN SOLUTION
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate cache memory
        # Shape: (max_batch_size, num_heads, max_seq_length, head_dim)
        cache_shape = (max_batch_size, num_heads, max_seq_length, head_dim)
        self.cached_keys = np.zeros(cache_shape, dtype=np.float32)
        self.cached_values = np.zeros(cache_shape, dtype=np.float32)
        
        # Track current cache length for each sequence in batch
        self.cache_lengths = np.zeros(max_batch_size, dtype=int)
        
        # Track whether cache is active
        self.is_active = False
        ### END SOLUTION
    
    def update(self, batch_idx: int, new_keys: Tensor, new_values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Update cache with new keys and values, return full cached K,V.
        
        TODO: Implement cache update.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Get current cache position for this batch
        2. Add new keys and values to cache at current position
        3. Update cache length
        4. Return full cached keys and values up to current length
        
        GENERATION PATTERN:
        - First call: cache is empty, add initial K,V
        - Subsequent calls: add one new token's K,V
        - Always return all cached K,V for attention computation
        
        Args:
            batch_idx: Index of sequence in batch
            new_keys: New keys to add with shape (num_heads, new_seq_len, head_dim)
            new_values: New values to add with shape (num_heads, new_seq_len, head_dim)
            
        Returns:
            Full cached keys and values with shape (num_heads, total_cached_len, head_dim)
        """
        ### BEGIN SOLUTION
        # Get current cache position
        current_pos = self.cache_lengths[batch_idx]
        new_seq_len = new_keys.shape[1]  # Assuming shape (num_heads, seq_len, head_dim)
        
        # Check bounds
        if current_pos + new_seq_len > self.max_seq_length:
            raise ValueError(f"Cache overflow: {current_pos + new_seq_len} > {self.max_seq_length}")
        
        # Update cache with new keys and values
        end_pos = current_pos + new_seq_len
        self.cached_keys[batch_idx, :, current_pos:end_pos, :] = new_keys.data
        self.cached_values[batch_idx, :, current_pos:end_pos, :] = new_values.data
        
        # Update cache length
        self.cache_lengths[batch_idx] = end_pos
        self.is_active = True
        
        # Return full cached keys and values
        full_keys = self.cached_keys[batch_idx, :, :end_pos, :]
        full_values = self.cached_values[batch_idx, :, :end_pos, :]
        
        return Tensor(full_keys), Tensor(full_values)
        ### END SOLUTION
    
    def reset(self, batch_idx: Optional[int] = None):
        """
        Reset cache for specific batch index or entire cache.
        
        This function is PROVIDED for cache management.
        """
        if batch_idx is not None:
            # Reset specific sequence
            self.cache_lengths[batch_idx] = 0
            self.cached_keys[batch_idx] = 0
            self.cached_values[batch_idx] = 0
        else:
            # Reset entire cache
            self.cache_lengths.fill(0)
            self.cached_keys.fill(0)
            self.cached_values.fill(0)
            self.is_active = False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of KV cache.
        
        This function is PROVIDED to show memory analysis.
        """
        # Cache memory in bytes
        cache_memory_bytes = self.cached_keys.nbytes + self.cached_values.nbytes
        cache_memory_mb = cache_memory_bytes / (1024 * 1024)
        
        # Memory per sequence
        memory_per_sequence_mb = cache_memory_mb / self.max_batch_size
        
        return {
            'total_cache_memory_mb': cache_memory_mb,
            'memory_per_sequence_mb': memory_per_sequence_mb,
            'max_batch_size': self.max_batch_size,
            'max_seq_length': self.max_seq_length,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'cache_utilization': np.mean(self.cache_lengths / self.max_seq_length) if self.is_active else 0.0
        }

# %% [markdown]
"""
### 🧪 Test Your KV-Cache Implementation

Once you implement the KVCache methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-kv-cache-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_kv_cache():
    """Unit test for KV cache."""
    print("🔬 Unit Test: KV-Cache...")
    
    # Create KV cache
    max_batch_size = 4
    max_seq_length = 16
    num_heads = 8
    head_dim = 64
    
    kv_cache = KVCache(max_batch_size=max_batch_size, max_seq_length=max_seq_length,
                       num_heads=num_heads, head_dim=head_dim)
    
    # Test initialization
    assert kv_cache.max_batch_size == max_batch_size, "Should store max batch size"
    assert kv_cache.max_seq_length == max_seq_length, "Should store max sequence length"
    assert kv_cache.cached_keys.shape == (max_batch_size, num_heads, max_seq_length, head_dim), "Should pre-allocate key cache"
    assert kv_cache.cached_values.shape == (max_batch_size, num_heads, max_seq_length, head_dim), "Should pre-allocate value cache"
    assert not kv_cache.is_active, "Should start inactive"
    
    # Test first update (initial sequence)
    batch_idx = 0
    initial_seq_len = 5
    initial_keys = Tensor(np.random.randn(num_heads, initial_seq_len, head_dim))
    initial_values = Tensor(np.random.randn(num_heads, initial_seq_len, head_dim))
    
    cached_keys, cached_values = kv_cache.update(batch_idx, initial_keys, initial_values)
    
    # Verify cache update
    assert cached_keys.shape == (num_heads, initial_seq_len, head_dim), f"Expected cached keys shape (num_heads, {initial_seq_len}, head_dim)"
    assert cached_values.shape == (num_heads, initial_seq_len, head_dim), f"Expected cached values shape (num_heads, {initial_seq_len}, head_dim)"
    assert kv_cache.cache_lengths[batch_idx] == initial_seq_len, f"Should update cache length to {initial_seq_len}"
    assert kv_cache.is_active, "Should be active after first update"
    
    # Verify cached data matches input
    assert np.allclose(cached_keys.data, initial_keys.data), "Cached keys should match input"
    assert np.allclose(cached_values.data, initial_values.data), "Cached values should match input"
    
    # Test incremental update (add one token)
    new_token_keys = Tensor(np.random.randn(num_heads, 1, head_dim))
    new_token_values = Tensor(np.random.randn(num_heads, 1, head_dim))
    
    cached_keys_updated, cached_values_updated = kv_cache.update(batch_idx, new_token_keys, new_token_values)
    
    # Verify incremental update
    expected_new_length = initial_seq_len + 1
    assert cached_keys_updated.shape == (num_heads, expected_new_length, head_dim), "Should include new token in cached keys"
    assert cached_values_updated.shape == (num_heads, expected_new_length, head_dim), "Should include new token in cached values"
    assert kv_cache.cache_lengths[batch_idx] == expected_new_length, f"Should update cache length to {expected_new_length}"
    
    # Verify old data is preserved and new data is appended
    assert np.allclose(cached_keys_updated.data[:, :initial_seq_len, :], initial_keys.data), "Should preserve old cached keys"
    assert np.allclose(cached_keys_updated.data[:, initial_seq_len:, :], new_token_keys.data), "Should append new keys"
    
    # Test multiple sequences in batch
    batch_idx_2 = 1
    seq2_keys = Tensor(np.random.randn(num_heads, 3, head_dim))
    seq2_values = Tensor(np.random.randn(num_heads, 3, head_dim))
    
    cached_keys_seq2, cached_values_seq2 = kv_cache.update(batch_idx_2, seq2_keys, seq2_values)
    
    # Verify independent cache management
    assert cached_keys_seq2.shape == (num_heads, 3, head_dim), "Second sequence should have correct shape"
    assert kv_cache.cache_lengths[batch_idx_2] == 3, "Second sequence should have correct length"
    assert kv_cache.cache_lengths[batch_idx] == expected_new_length, "First sequence length should be unchanged"
    
    # Test cache overflow protection
    try:
        # Try to add more tokens than max_seq_length allows
        overflow_keys = Tensor(np.random.randn(num_heads, max_seq_length, head_dim))
        overflow_values = Tensor(np.random.randn(num_heads, max_seq_length, head_dim))
        kv_cache.update(batch_idx, overflow_keys, overflow_values)
        assert False, "Should raise error for cache overflow"
    except ValueError:
        pass  # Expected behavior
    
    # Test cache reset
    kv_cache.reset(batch_idx)
    assert kv_cache.cache_lengths[batch_idx] == 0, "Should reset cache length to 0"
    assert kv_cache.cache_lengths[batch_idx_2] == 3, "Should not affect other sequences"
    
    # Test full cache reset
    kv_cache.reset()
    assert np.all(kv_cache.cache_lengths == 0), "Should reset all cache lengths"
    assert not kv_cache.is_active, "Should be inactive after full reset"
    
    # Test memory usage calculation
    memory_stats = kv_cache.get_memory_usage()
    assert 'total_cache_memory_mb' in memory_stats, "Should provide memory statistics"
    assert memory_stats['max_batch_size'] == max_batch_size, "Should report correct batch size"
    assert memory_stats['max_seq_length'] == max_seq_length, "Should report correct sequence length"
    
    print("✅ KV-Cache tests passed!")
    print(f"✅ Handles {max_batch_size} sequences of up to {max_seq_length} tokens")
    print(f"✅ Memory usage: {memory_stats['total_cache_memory_mb']:.2f}MB total")
    print(f"✅ Cache overflow protection works")
    print(f"✅ Independent batch sequence management")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🎯 ML Systems: Performance Analysis & Attention Scaling

Now let's develop systems engineering skills by analyzing attention performance and understanding how attention's quadratic scaling affects practical transformer deployment.

### **Learning Outcome**: *"I understand how attention's O(N²) complexity determines the practical limits of transformer sequence length and deployment strategies"*
"""

# %% nbgrader={"grade": false, "grade_id": "attention-profiler", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import time

class AttentionProfiler:
    """
    Performance profiling toolkit for attention mechanisms.
    
    Helps ML engineers understand computational costs, memory scaling,
    and bottlenecks in attention-based architectures.
    """
    
    def __init__(self):
        self.results = {}
    
    def measure_attention_scaling(self, attention_layer, seq_lengths: List[int], 
                                 embed_dim: int = 256, batch_size: int = 1) -> Dict:
        """
        Measure how attention performance scales with sequence length.
        
        TODO: Implement attention scaling measurement.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create test inputs for each sequence length
        2. Measure computation time for attention forward pass
        3. Calculate memory usage for attention matrices
        4. Analyze scaling patterns (should be O(N²))
        5. Return comprehensive scaling analysis
        
        METRICS TO CALCULATE:
        - Computation time vs sequence length
        - Memory usage vs sequence length  
        - Attention matrix size scaling
        - Throughput degradation patterns
        
        Args:
            attention_layer: Attention layer to test (ScaledDotProductAttention or MultiHeadAttention)
            seq_lengths: List of sequence lengths to test
            embed_dim: Embedding dimension for test inputs
            batch_size: Batch size for testing
            
        Returns:
            Dictionary with scaling analysis results
        """
        ### BEGIN SOLUTION
        scaling_results = {}
        
        for seq_len in seq_lengths:
            # Create test inputs
            query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
            key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
            value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
            
            # Measure computation time
            start_time = time.time()
            if hasattr(attention_layer, 'forward'):
                output = attention_layer.forward(query, key, value)
            else:
                output = attention_layer(query, key, value)
            end_time = time.time()
            
            computation_time_ms = (end_time - start_time) * 1000
            
            # Calculate memory usage
            input_memory_mb = (query.data.nbytes + key.data.nbytes + value.data.nbytes) / (1024 * 1024)
            output_memory_mb = output.data.nbytes / (1024 * 1024)
            
            # Attention matrix memory (batch_size * seq_len * seq_len)
            attention_matrix_memory_mb = (batch_size * seq_len * seq_len * 4) / (1024 * 1024)  # 4 bytes per float32
            
            # Calculate throughput
            total_operations = batch_size * seq_len * seq_len * embed_dim  # Rough estimate
            operations_per_second = total_operations / (end_time - start_time) if end_time > start_time else 0
            
            scaling_results[seq_len] = {
                'seq_length': seq_len,
                'computation_time_ms': computation_time_ms,
                'input_memory_mb': input_memory_mb,
                'output_memory_mb': output_memory_mb,
                'attention_matrix_memory_mb': attention_matrix_memory_mb,
                'total_memory_mb': input_memory_mb + output_memory_mb + attention_matrix_memory_mb,
                'operations_per_second': operations_per_second,
                'time_per_token_us': computation_time_ms * 1000 / (batch_size * seq_len) if seq_len > 0 else 0
            }
        
        return scaling_results
        ### END SOLUTION
    
    def analyze_quadratic_scaling(self, scaling_results: Dict) -> Dict:
        """
        Analyze quadratic scaling patterns in attention results.
        
        This function is PROVIDED to show scaling pattern analysis.
        """
        print("📈 ATTENTION QUADRATIC SCALING ANALYSIS")
        print("=" * 60)
        
        seq_lengths = sorted(scaling_results.keys())
        
        if len(seq_lengths) < 2:
            print("Need at least 2 sequence lengths for scaling analysis")
            return {}
        
        print(f"{'Seq Length':<10} {'Time (ms)':<12} {'Memory (MB)':<12} {'Attn Matrix':<12} {'Time/Token':<12}")
        print("-" * 70)
        
        for seq_len in seq_lengths:
            result = scaling_results[seq_len]
            print(f"{seq_len:<10} {result['computation_time_ms']:<12.2f} "
                  f"{result['total_memory_mb']:<12.2f} {result['attention_matrix_memory_mb']:<12.2f} "
                  f"{result['time_per_token_us']:<12.2f}")
        
        # Analyze scaling ratios
        base_seq = seq_lengths[0]
        base_result = scaling_results[base_seq]
        
        scaling_analysis = {'base_sequence_length': base_seq}
        
        print(f"\n📊 SCALING ANALYSIS (relative to {base_seq} tokens):")
        print(f"{'Length Ratio':<12} {'Time Ratio':<12} {'Memory Ratio':<12} {'Theory (N²)':<12}")
        print("-" * 50)
        
        for seq_len in seq_lengths[1:]:
            result = scaling_results[seq_len]
            
            length_ratio = seq_len / base_seq
            time_ratio = result['computation_time_ms'] / base_result['computation_time_ms']
            memory_ratio = result['attention_matrix_memory_mb'] / base_result['attention_matrix_memory_mb']
            theoretical_ratio = length_ratio ** 2
            
            scaling_analysis[seq_len] = {
                'length_ratio': length_ratio,
                'time_ratio': time_ratio,
                'memory_ratio': memory_ratio,
                'theoretical_ratio': theoretical_ratio,
                'time_efficiency': theoretical_ratio / time_ratio if time_ratio > 0 else 0
            }
            
            print(f"{length_ratio:<12.1f} {time_ratio:<12.1f} {memory_ratio:<12.1f} {theoretical_ratio:<12.1f}")
        
        # Analysis insights
        print(f"\n💡 SCALING INSIGHTS:")
        avg_memory_efficiency = np.mean([scaling_analysis[seq]['memory_ratio'] / scaling_analysis[seq]['theoretical_ratio'] 
                                       for seq in seq_lengths[1:] if seq in scaling_analysis])
        
        print(f"   - Memory scaling: ~{avg_memory_efficiency:.1f}x theoretical O(N²)")
        print(f"   - Attention matrix dominates memory usage")
        print(f"   - Time scaling may deviate from O(N²) due to hardware effects")
        print(f"   - Practical sequence limit determined by available GPU memory")
        
        return scaling_analysis
    
    def compare_attention_types(self, seq_length: int = 128, embed_dim: int = 256) -> Dict:
        """
        Compare performance of different attention implementations.
        
        This function is PROVIDED to show attention type comparison.
        """
        print(f"\n🔍 ATTENTION TYPE COMPARISON")
        print("=" * 50)
        
        batch_size = 8
        
        # Create test inputs
        query = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
        
        results = {}
        
        # Test scaled dot-product attention
        scaled_attention = ScaledDotProductAttention()
        start_time = time.time()
        scaled_output = scaled_attention.forward(query, key, value)
        scaled_time = (time.time() - start_time) * 1000
        
        results['scaled_dot_product'] = {
            'computation_time_ms': scaled_time,
            'parameters': 0,  # No learnable parameters
            'memory_mb': scaled_output.data.nbytes / (1024 * 1024),
            'description': 'Basic attention mechanism'
        }
        
        # Test multi-head attention
        num_heads = 8
        mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        start_time = time.time()
        mha_output = mha.forward(query, key, value)
        mha_time = (time.time() - start_time) * 1000
        
        mha_memory = mha.get_memory_usage()
        
        results['multi_head'] = {
            'computation_time_ms': mha_time,
            'parameters': mha_memory['total_parameters'],
            'memory_mb': mha_output.data.nbytes / (1024 * 1024) + mha_memory['total_parameter_memory_mb'],
            'description': f'{num_heads}-head attention with projections'
        }
        
        # Display comparison
        print(f"Test configuration: {batch_size} batch × {seq_length} seq × {embed_dim} dim")
        print(f"{'Type':<15} {'Time (ms)':<10} {'Parameters':<12} {'Memory (MB)':<12} {'Description'}")
        print("-" * 70)
        
        for name, stats in results.items():
            print(f"{name:<15} {stats['computation_time_ms']:<10.2f} "
                  f"{stats['parameters']:<12,} {stats['memory_mb']:<12.2f} {stats['description']}")
        
        # Analysis
        time_overhead = results['multi_head']['computation_time_ms'] / results['scaled_dot_product']['computation_time_ms']
        memory_overhead = results['multi_head']['memory_mb'] / results['scaled_dot_product']['memory_mb']
        
        print(f"\n📊 OVERHEAD ANALYSIS:")
        print(f"   Multi-head vs Scaled: {time_overhead:.1f}x time, {memory_overhead:.1f}x memory")
        print(f"   Trade-off: Multi-head provides richer representations at cost of computation")
        print(f"   Parameters: Multi-head adds {results['multi_head']['parameters']:,} learnable parameters")
        
        return results
    
    def simulate_kv_cache_benefits(self, seq_lengths: List[int], embed_dim: int = 256, 
                                  num_heads: int = 8) -> Dict:
        """
        Simulate memory and computation benefits of KV-cache during generation.
        
        This function is PROVIDED to show KV-cache analysis.
        """
        print(f"\n💾 KV-CACHE BENEFITS ANALYSIS")
        print("=" * 50)
        
        head_dim = embed_dim // num_heads
        batch_size = 1  # Typical generation batch size
        
        results = {}
        
        print(f"{'Seq Length':<10} {'No Cache (MB)':<14} {'With Cache (MB)':<16} {'Savings':<10} {'Speedup'}")
        print("-" * 65)
        
        for seq_len in seq_lengths:
            # Without cache: recompute K,V for all tokens every generation step
            # Memory: attention matrices for all positions
            no_cache_attention_memory = batch_size * seq_len * seq_len * 4 / (1024 * 1024)  # bytes -> MB
            no_cache_kv_memory = batch_size * seq_len * embed_dim * 2 * 4 / (1024 * 1024)  # K + V
            no_cache_total = no_cache_attention_memory + no_cache_kv_memory
            
            # With cache: store K,V, only compute attention for new token
            cache_storage = batch_size * seq_len * embed_dim * 2 * 4 / (1024 * 1024)  # K + V storage
            cache_attention_memory = batch_size * 1 * seq_len * 4 / (1024 * 1024)  # Only new token attention
            cache_total = cache_storage + cache_attention_memory
            
            # Compute benefits
            memory_savings = (no_cache_total - cache_total) / no_cache_total * 100
            speedup_estimate = seq_len  # Rough estimate: avoid recomputing seq_len tokens
            
            results[seq_len] = {
                'no_cache_memory_mb': no_cache_total,
                'cache_memory_mb': cache_total,
                'memory_savings_percent': memory_savings,
                'estimated_speedup': speedup_estimate
            }
            
            print(f"{seq_len:<10} {no_cache_total:<14.2f} {cache_total:<16.2f} "
                  f"{memory_savings:<10.1f}% {speedup_estimate:<10.1f}x")
        
        print(f"\n💡 KV-CACHE INSIGHTS:")
        print(f"   - Memory: Significant savings for long sequences")
        print(f"   - Speed: Avoid recomputing K,V for all previous tokens")
        print(f"   - Trade-off: Cache storage vs recomputation")
        print(f"   - Essential for: Real-time text generation and interactive systems")
        
        return results

def analyze_attention_system_design():
    """
    Comprehensive analysis of attention system design choices and scaling implications.
    
    This function is PROVIDED to show systems-level design thinking.
    """
    print("🏗️ ATTENTION SYSTEM DESIGN ANALYSIS")
    print("=" * 60)
    
    # Model configurations with different attention strategies
    model_configs = [
        {
            'name': 'Small GPT',
            'seq_length': 512,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6
        },
        {
            'name': 'Medium GPT', 
            'seq_length': 1024,
            'embed_dim': 512,
            'num_heads': 16,
            'num_layers': 12
        },
        {
            'name': 'Large GPT',
            'seq_length': 2048,
            'embed_dim': 1024, 
            'num_heads': 32,
            'num_layers': 24
        }
    ]
    
    print(f"📋 ATTENTION MEMORY SCALING ANALYSIS:")
    print(f"{'Model':<12} {'Seq Len':<8} {'Heads':<6} {'Layers':<7} {'Attn Memory':<12} {'Total Attn':<12}")
    print("-" * 75)
    
    for config in model_configs:
        # Calculate attention memory per layer
        batch_size = 1
        seq_len = config['seq_length']
        attention_matrix_memory_mb = (batch_size * seq_len * seq_len * 4) / (1024 * 1024)
        
        # Total attention memory across all layers
        total_attention_memory_mb = attention_matrix_memory_mb * config['num_layers']
        
        print(f"{config['name']:<12} {seq_len:<8} {config['num_heads']:<6} "
              f"{config['num_layers']:<7} {attention_matrix_memory_mb:<12.1f} {total_attention_memory_mb:<12.1f}")
    
    print(f"\n🎯 KEY DESIGN IMPLICATIONS:")
    print(f"   1. Sequence Length Scaling:")
    print(f"      - Memory scales O(N²) with sequence length")
    print(f"      - 2x sequence length = 4x attention memory")
    print(f"      - Practical limit: GPU memory capacity")
    
    print(f"   2. Multi-Head Benefits:")
    print(f"      - Multiple attention patterns in parallel")
    print(f"      - Linear scaling with number of heads")
    print(f"      - Trade-off: representation richness vs computation")
    
    print(f"   3. Layer Depth Impact:")
    print(f"      - Attention memory scales linearly with layers")
    print(f"      - Deep models need efficient attention implementations")
    print(f"      - Memory checkpointing may be necessary")
    
    print(f"   4. Production Constraints:")
    print(f"      - GPU memory limits maximum sequence length")
    print(f"      - Attention is the memory bottleneck in transformers")
    print(f"      - KV-cache essential for generation workloads")
    
    print(f"\n🏭 OPTIMIZATION STRATEGIES:")
    print(f"   - Flash Attention: Memory-efficient attention computation")
    print(f"   - Sparse Attention: Reduce O(N²) to O(N√N) or O(N log N)")
    print(f"   - Linear Attention: Approximate attention with linear complexity")
    print(f"   - Sliding Window: Local attention with fixed window size")
    print(f"   - KV-Cache: Essential for autoregressive generation")

# %% [markdown]
"""
### 🧪 Test: Attention Performance Analysis

Let's test our attention profiler with realistic performance scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "test-attention-profiler", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_attention_profiler():
    """Test attention profiler with various scenarios."""
    print("🔬 Unit Test: Attention Performance Profiler...")
    
    profiler = AttentionProfiler()
    
    # Test scaling measurement with scaled attention
    scaled_attention = ScaledDotProductAttention()
    seq_lengths = [32, 64, 128]
    embed_dim = 128
    
    scaling_results = profiler.measure_attention_scaling(scaled_attention, seq_lengths, embed_dim)
    
    # Verify results structure
    assert len(scaling_results) == len(seq_lengths), f"Should test {len(seq_lengths)} sequence lengths"
    
    for seq_len in seq_lengths:
        assert seq_len in scaling_results, f"Should include results for sequence length {seq_len}"
        result = scaling_results[seq_len]
        
        # Verify required metrics
        required_keys = ['seq_length', 'computation_time_ms', 'input_memory_mb', 
                        'output_memory_mb', 'attention_matrix_memory_mb', 'total_memory_mb']
        for key in required_keys:
            assert key in result, f"Missing metric: {key} for seq_len {seq_len}"
            assert isinstance(result[key], (int, float)), f"Invalid type for {key}"
        
        # Verify reasonable values
        assert result['seq_length'] == seq_len, "Should store correct sequence length"
        assert result['computation_time_ms'] >= 0, "Time should be non-negative"
        assert result['total_memory_mb'] > 0, "Memory usage should be positive"
    
    print("✅ Scaling measurement test passed")
    
    # Test quadratic scaling analysis
    scaling_analysis = profiler.analyze_quadratic_scaling(scaling_results)
    
    # Verify scaling analysis
    assert 'base_sequence_length' in scaling_analysis, "Should include base sequence length"
    
    # Check that longer sequences show increased ratios
    for seq_len in seq_lengths[1:]:
        if seq_len in scaling_analysis:
            analysis = scaling_analysis[seq_len]
            assert analysis['length_ratio'] > 1, f"Length ratio should be > 1 for {seq_len}"
            assert analysis['theoretical_ratio'] > 1, f"Theoretical ratio should be > 1 for {seq_len}"
    
    print("✅ Quadratic scaling analysis test passed")
    
    # Test attention type comparison
    comparison_results = profiler.compare_attention_types(seq_length=64, embed_dim=128)
    
    # Verify comparison results
    assert 'scaled_dot_product' in comparison_results, "Should test scaled dot-product attention"
    assert 'multi_head' in comparison_results, "Should test multi-head attention"
    
    for attn_type, metrics in comparison_results.items():
        assert 'computation_time_ms' in metrics, "Should measure computation time"
        assert 'parameters' in metrics, "Should count parameters"
        assert 'memory_mb' in metrics, "Should measure memory usage"
        assert metrics['computation_time_ms'] > 0, "Should have positive computation time"
    
    print("✅ Attention type comparison test passed")
    
    # Test KV-cache benefits simulation
    cache_results = profiler.simulate_kv_cache_benefits([64, 128], embed_dim=128)
    
    # Verify cache simulation results
    for seq_len, result in cache_results.items():
        assert 'no_cache_memory_mb' in result, "Should calculate no-cache memory"
        assert 'cache_memory_mb' in result, "Should calculate cache memory"
        assert 'memory_savings_percent' in result, "Should calculate savings"
        assert result['memory_savings_percent'] > 0, "Should show memory savings"
    
    print("✅ KV-cache benefits simulation test passed")
    print("🎯 Attention Profiler: All tests passed!")

# Test function defined (called in main block)

# %% [markdown]
"""
## Integration Testing: Complete Attention Pipeline

Let's test how all our attention components work together in a realistic transformer-like pipeline:
"""

# %% nbgrader={"grade": false, "grade_id": "test-attention-integration", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_attention_integration():
    """Test complete attention pipeline with embeddings integration."""
    print("🧪 Integration Test: Complete Attention Pipeline...")
    
    # Configuration
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    seq_length = 32
    batch_size = 4
    
    # Create embedding components (mock minimal versions if not available)
    try:
        from embeddings_dev import Embedding, PositionalEncoding
        embedding = Embedding(vocab_size=vocab_size, embedding_dim=embed_dim)
        pos_encoding = PositionalEncoding(embedding_dim=embed_dim, max_seq_length=seq_length*2)
        embeddings_available = True
    except:
        # Create mock embeddings for testing
        embedding = None
        pos_encoding = None
        embeddings_available = False
        print("  Using mock embeddings for testing...")
    
    # Create attention components
    scaled_attention = ScaledDotProductAttention()
    multi_head_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create test data
    if embeddings_available:
        # Use real embedding pipeline
        token_ids = np.random.randint(0, vocab_size, (batch_size, seq_length))
        embeddings = embedding.forward(token_ids)
        pos_embeddings = pos_encoding.forward(embeddings)
        input_representations = pos_embeddings
        print(f"  Using real embeddings: {input_representations.shape}")
    else:
        # Use mock input data
        input_representations = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
        print(f"  Using mock input: {input_representations.shape}")
    
    # Test 1: Self-attention with scaled dot-product
    print("  Testing scaled dot-product self-attention...")
    self_attn_output = scaled_attention.forward(
        input_representations, input_representations, input_representations
    )
    
    expected_shape = (batch_size, seq_length, embed_dim)
    assert self_attn_output.shape == expected_shape, f"Expected {expected_shape}, got {self_attn_output.shape}"
    print(f"    Self-attention output: {self_attn_output.shape}")
    
    # Test 2: Multi-head self-attention
    print("  Testing multi-head self-attention...")
    mha_output, mha_weights = multi_head_attention.forward(
        input_representations, input_representations, input_representations,
        return_attention_weights=True
    )
    
    assert mha_output.shape == expected_shape, f"Expected {expected_shape}, got {mha_output.shape}"
    expected_attn_shape = (batch_size, num_heads, seq_length, seq_length)
    assert mha_weights.shape == expected_attn_shape, f"Expected attention {expected_attn_shape}, got {mha_weights.shape}"
    print(f"    Multi-head output: {mha_output.shape}")
    print(f"    Attention weights: {mha_weights.shape}")
    
    # Test 3: Causal (autoregressive) attention
    print("  Testing causal attention masking...")
    causal_mask = np.triu(np.ones((seq_length, seq_length)), k=1)
    causal_mask = 1 - causal_mask  # Convert to attention mask
    
    causal_output, causal_weights = multi_head_attention.forward(
        input_representations, input_representations, input_representations,
        mask=Tensor(causal_mask), return_attention_weights=True
    )
    
    # Verify causal masking works
    for head in range(num_heads):
        for i in range(seq_length):
            for j in range(i+1, seq_length):
                assert np.all(causal_weights.data[:, head, i, j] < 1e-5), \
                    f"Position ({i},{j}) should be masked in head {head}"
    
    print(f"    Causal attention works correctly across {num_heads} heads")
    
    # Test 4: Cross-attention (encoder-decoder style)
    print("  Testing cross-attention...")
    # Create different key/value inputs (simulating encoder-decoder)
    encoder_seq_length = seq_length + 8  # Different length
    encoder_representations = Tensor(np.random.randn(batch_size, encoder_seq_length, embed_dim))
    
    cross_attn_output = multi_head_attention.forward(
        input_representations,  # Query from decoder
        encoder_representations,  # Key from encoder
        encoder_representations   # Value from encoder
    )
    
    # Output should have decoder sequence length, encoder information
    expected_cross_shape = (batch_size, seq_length, embed_dim)
    assert cross_attn_output.shape == expected_cross_shape, \
        f"Expected {expected_cross_shape}, got {cross_attn_output.shape}"
    print(f"    Cross-attention output: {cross_attn_output.shape}")
    
    # Test 5: KV-Cache integration
    print("  Testing KV-cache integration...")
    head_dim = embed_dim // num_heads
    kv_cache = KVCache(max_batch_size=batch_size, max_seq_length=seq_length*2,
                       num_heads=num_heads, head_dim=head_dim)
    
    # Simulate autoregressive generation
    for step in range(3):  # Generate 3 tokens
        if step == 0:
            # First step: process initial sequence
            step_input = input_representations
        else:
            # Subsequent steps: process one new token
            new_token_repr = Tensor(np.random.randn(batch_size, 1, embed_dim))
            step_input = new_token_repr
        
        # In real implementation, we'd integrate KV-cache with attention
        # For now, just test that cache operations work
        batch_idx = 0
        step_keys = Tensor(np.random.randn(num_heads, step_input.shape[1], head_dim))
        step_values = Tensor(np.random.randn(num_heads, step_input.shape[1], head_dim))
        
        cached_keys, cached_values = kv_cache.update(batch_idx, step_keys, step_values)
        
        expected_cache_length = sum(input_representations.shape[1] if i == 0 else 1 for i in range(step + 1))
        assert cached_keys.shape[1] == expected_cache_length, \
            f"Cache should have {expected_cache_length} tokens at step {step}"
    
    print(f"    KV-cache successfully caches keys/values across generation steps")
    
    # Test 6: Memory usage analysis
    print("  Analyzing memory usage...")
    mha_memory = multi_head_attention.get_memory_usage()
    cache_memory = kv_cache.get_memory_usage()
    
    total_memory_mb = mha_memory['total_parameter_memory_mb'] + cache_memory['total_cache_memory_mb']
    
    print(f"    Multi-head attention parameters: {mha_memory['total_parameter_memory_mb']:.2f}MB")
    print(f"    KV-cache storage: {cache_memory['total_cache_memory_mb']:.2f}MB")
    print(f"    Total attention system memory: {total_memory_mb:.2f}MB")
    
    # Test 7: Performance characteristics
    print("  Testing performance characteristics...")
    start_time = time.time()
    
    # Process multiple steps to measure throughput
    for _ in range(10):
        output = multi_head_attention.forward(
            input_representations, input_representations, input_representations
        )
    
    total_time = time.time() - start_time
    throughput = (batch_size * seq_length * 10) / total_time  # tokens per second
    
    print(f"    Attention throughput: {throughput:.0f} tokens/second")
    
    print("✅ Complete attention pipeline integration test passed!")
    print(f"✅ Self-attention, cross-attention, and causal masking work correctly")
    print(f"✅ KV-cache integration ready for autoregressive generation")
    print(f"✅ Memory usage and performance characteristics measured")

# Test function defined (called in main block)

# %% [markdown]
"""
## Main Execution Block

All attention tests and demonstrations are run from here when the module is executed directly:
"""

# %% nbgrader={"grade": false, "grade_id": "attention-main", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    # Run all unit tests
    test_unit_scaled_attention()
    test_unit_multi_head_attention()
    test_unit_kv_cache()
    test_attention_profiler()
    test_attention_integration()
    
    print("\n" + "="*60)
    print("🔍 ATTENTION SYSTEMS ANALYSIS")
    print("="*60)
    
    # Performance analysis
    profiler = AttentionProfiler()
    
    # Test attention scaling with different sequence lengths
    print("📈 ATTENTION SCALING ANALYSIS:")
    scaled_attention = ScaledDotProductAttention()
    seq_lengths = [64, 128, 256, 512]
    embed_dim = 256
    
    scaling_results = profiler.measure_attention_scaling(scaled_attention, seq_lengths, embed_dim)
    quadratic_analysis = profiler.analyze_quadratic_scaling(scaling_results)
    
    # Compare attention types
    print("\n" + "="*60)
    attention_comparison = profiler.compare_attention_types(seq_length=128, embed_dim=256)
    
    # KV-cache benefits analysis
    print("\n" + "="*60)
    kv_cache_analysis = profiler.simulate_kv_cache_benefits([128, 256, 512], embed_dim=256)
    
    # Systems design analysis
    print("\n" + "="*60)
    analyze_attention_system_design()
    
    # Demonstrate realistic transformer attention setup
    print("\n" + "="*60)
    print("🏗️ REALISTIC TRANSFORMER ATTENTION SETUP")
    print("="*60)
    
    # Create realistic transformer configuration
    embed_dim = 512
    num_heads = 8
    seq_length = 256
    batch_size = 16
    
    print(f"Transformer configuration:")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Head dimension: {embed_dim // num_heads}")
    
    # Create attention components
    multi_head_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    kv_cache = KVCache(max_batch_size=batch_size, max_seq_length=seq_length*2,
                       num_heads=num_heads, head_dim=embed_dim//num_heads)
    
    # Memory analysis
    mha_memory = multi_head_attention.get_memory_usage()
    cache_memory = kv_cache.get_memory_usage()
    
    print(f"\nMemory analysis:")
    print(f"  Multi-head attention parameters: {mha_memory['total_parameters']:,}")
    print(f"  Parameter memory: {mha_memory['total_parameter_memory_mb']:.1f}MB")
    print(f"  KV-cache memory: {cache_memory['total_cache_memory_mb']:.1f}MB")
    
    # Performance simulation
    input_representations = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
    
    start_time = time.time()
    output, attention_weights = multi_head_attention.forward(
        input_representations, input_representations, input_representations,
        return_attention_weights=True
    )
    processing_time = time.time() - start_time
    
    # Calculate attention matrix memory
    attention_memory_mb = (batch_size * num_heads * seq_length * seq_length * 4) / (1024 * 1024)
    output_memory_mb = output.data.nbytes / (1024 * 1024)
    
    print(f"\nPerformance analysis:")
    print(f"  Processing time: {processing_time*1000:.2f}ms")
    print(f"  Throughput: {(batch_size * seq_length) / processing_time:.0f} tokens/second")
    print(f"  Attention matrix memory: {attention_memory_mb:.1f}MB")
    print(f"  Output memory: {output_memory_mb:.1f}MB")
    
    # Scaling limits analysis
    print(f"\nScaling limits:")
    max_gpu_memory_gb = 24  # Typical high-end GPU
    max_attention_memory_gb = max_gpu_memory_gb * 0.5  # Assume 50% for attention
    max_seq_len_theoretical = int(math.sqrt(max_attention_memory_gb * 1024 * 1024 * 1024 / (batch_size * num_heads * 4)))
    
    print(f"  Theoretical max sequence (24GB GPU): ~{max_seq_len_theoretical} tokens")
    print(f"  Current sequence uses: {attention_memory_mb:.1f}MB")
    print(f"  Memory efficiency critical for longer sequences")
    
    print("\n" + "="*60)
    print("🎯 ATTENTION MODULE COMPLETE!")
    print("="*60)
    print("All attention tests passed!")
    print("Ready for transformer architecture integration!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built the attention mechanisms that revolutionized language understanding, let's connect this work to broader ML systems challenges. These questions help you think critically about how attention's quadratic scaling affects production transformer deployment.

Take time to reflect thoughtfully on each question - your insights will help you understand how attention connects to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Attention Memory Scaling and Sequence Length Optimization

**Context**: Your attention implementations demonstrate the fundamental O(N²) memory scaling that limits transformer sequence length. Production language models must balance sequence length capabilities with memory constraints, leading to complex architectural decisions about attention patterns, memory optimization, and deployment strategies.

**Reflection Question**: Design an attention system for a production language model that needs to efficiently process documents up to 32k tokens while operating within 80GB GPU memory constraints. How would you implement attention optimization techniques like Flash Attention or sparse attention patterns, design memory-efficient attention computation that minimizes intermediate storage, and handle variable sequence lengths in production batches? Consider the challenges of maintaining attention quality while reducing memory footprint and optimizing for both training and inference workloads.

Think about: attention optimization techniques, memory-efficient computation patterns, sparse attention strategies, and variable-length batch processing.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-attention-memory", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON ATTENTION MEMORY SCALING AND OPTIMIZATION:

TODO: Replace this text with your thoughtful response about attention memory optimization system design.

Consider addressing:
- How would you implement attention optimization for 32k tokens within 80GB GPU memory?
- What techniques would you use to reduce attention's O(N²) memory scaling?
- How would you design memory-efficient attention computation with minimal intermediate storage?
- What approaches would you use for handling variable sequence lengths in production batches?
- How would you maintain attention quality while optimizing for memory constraints?

Write a technical analysis connecting your attention implementations to real memory optimization challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of attention memory scaling and optimization techniques (3 points)
- Designs practical approaches to memory-efficient attention computation (3 points)
- Addresses variable-length processing and production deployment constraints (2 points)
- Shows systems thinking about attention optimization trade-offs (2 points)
- Clear technical reasoning with memory optimization insights (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring technical analysis of attention memory optimization
# Students should demonstrate understanding of attention scaling challenges and optimization techniques
### END SOLUTION

# %% [markdown]
"""
### Question 2: Multi-Head Attention Parallelization and Hardware Optimization

**Context**: Your multi-head attention implementation shows how attention heads can process different representation subspaces in parallel. Production transformer systems must optimize multi-head attention for diverse hardware platforms (CPUs, GPUs, TPUs) while maximizing throughput and minimizing latency for both training and inference workloads.

**Reflection Question**: Architect a multi-head attention system optimized for distributed training across 64 GPUs and efficient inference on various hardware platforms. How would you implement attention head parallelization that maximizes GPU utilization, design efficient attention kernel fusion to minimize memory bandwidth bottlenecks, and optimize for different inference scenarios (batch processing vs single-token generation)? Consider the challenges of maintaining numerical consistency across hardware platforms while achieving optimal performance for both training throughput and inference latency.

Think about: multi-GPU attention parallelization, kernel fusion optimization, hardware-specific tuning, and inference optimization strategies.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-attention-parallelization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON MULTI-HEAD ATTENTION PARALLELIZATION:

TODO: Replace this text with your thoughtful response about multi-head attention hardware optimization.

Consider addressing:
- How would you implement attention head parallelization across 64 GPUs for training?
- What kernel fusion techniques would you use to minimize memory bandwidth bottlenecks?
- How would you optimize attention for different hardware platforms (CPU, GPU, TPU)?
- What strategies would you use to optimize for batch processing vs single-token generation?
- How would you maintain numerical consistency across diverse hardware configurations?

Write an architectural analysis connecting your attention implementations to hardware optimization challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of multi-head attention parallelization and hardware optimization (3 points)
- Designs practical approaches to distributed training and kernel fusion (3 points)
- Addresses platform-specific optimization and inference scenarios (2 points)
- Demonstrates systems thinking about hardware-software co-optimization (2 points)
- Clear architectural reasoning with parallelization insights (bonus points for comprehensive system design)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of attention parallelization and hardware optimization
# Students should demonstrate knowledge of distributed training and platform-specific optimization
### END SOLUTION

# %% [markdown]
"""
### Question 3: KV-Cache Optimization and Generation Efficiency

**Context**: Your KV-cache implementation demonstrates how caching key-value computations can significantly improve autoregressive generation efficiency. Production language models must optimize KV-cache strategies for diverse generation workloads while managing memory usage, cache consistency, and throughput across different deployment scenarios.

**Reflection Question**: Design a KV-cache optimization system for a production language model serving that handles diverse generation workloads: real-time chat (low latency), batch document processing (high throughput), and interactive code generation (variable length patterns). How would you implement adaptive cache management that optimizes memory usage based on generation patterns, design efficient cache sharing across multiple requests, and handle cache eviction strategies for long-running services? Consider the challenges of balancing cache hit rates with memory efficiency while maintaining consistent generation quality across different workload types.

Think about: adaptive cache management, multi-request cache sharing, eviction strategies, and workload-specific optimization.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-kv-cache-optimization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON KV-CACHE OPTIMIZATION AND GENERATION EFFICIENCY:

TODO: Replace this text with your thoughtful response about KV-cache optimization for diverse generation workloads.

Consider addressing:
- How would you design adaptive cache management for real-time chat, batch processing, and code generation?
- What strategies would you use for efficient cache sharing across multiple requests?
- How would you implement cache eviction strategies for long-running production services?
- What approaches would you use to optimize memory usage based on generation patterns?
- How would you balance cache hit rates with memory efficiency across different workloads?

Write a design analysis connecting your KV-cache implementation to production generation system optimization.

GRADING RUBRIC (Instructor Use):
- Understands KV-cache optimization challenges and adaptive management strategies (3 points)
- Designs practical approaches to multi-request cache sharing and eviction (3 points)
- Addresses workload-specific optimization and memory efficiency considerations (2 points)
- Shows systems thinking about production generation service optimization (2 points)
- Clear design reasoning with cache optimization insights (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of KV-cache optimization for production systems
# Students should demonstrate knowledge of cache management and generation efficiency optimization
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Attention

Congratulations! You have successfully implemented the attention mechanisms that revolutionized language understanding:

### ✅ What You Have Built
- **Scaled Dot-Product Attention**: The fundamental attention mechanism with proper masking support
- **Multi-Head Attention**: Parallel attention heads for richer representation learning
- **KV-Cache System**: Efficient caching for autoregressive generation workloads
- **Causal Masking**: Support for autoregressive language modeling
- **Performance Analysis**: Comprehensive scaling and optimization analysis tools
- **🆕 Memory Optimization**: Understanding and measuring attention's O(N²) scaling characteristics
- **🆕 Systems Integration**: Complete attention pipeline with embeddings and generation support

### ✅ Key Learning Outcomes
- **Understanding**: How attention enables transformers to model sequence relationships
- **Implementation**: Built attention mechanisms with memory-efficient patterns and causal masking
- **Systems Insight**: How attention's quadratic scaling affects model architecture and deployment
- **Performance Engineering**: Measured and analyzed attention bottlenecks and optimization techniques
- **Production Context**: Understanding real-world attention challenges and optimization strategies

### ✅ Technical Mastery
- **Attention Mathematics**: Attention(Q,K,V) = softmax(QK^T/√d_k)V with proper scaling
- **Multi-Head Architecture**: Parallel attention computation with head dimension management
- **Causal Masking**: Autoregressive attention patterns for language generation
- **Memory Scaling**: Understanding O(N²) complexity and its implications for sequence length
- **🆕 KV-Cache Efficiency**: Optimizing attention computation for generation workloads

### ✅ Professional Skills Developed
- **Systems Architecture**: Designing attention systems for production scale and efficiency
- **Memory Engineering**: Understanding and optimizing attention's memory bottlenecks
- **Performance Analysis**: Measuring and improving attention computation throughput
- **Integration Design**: Building attention systems that work with embeddings and transformers

### ✅ Ready for Next Steps
Your attention systems are now ready to power:
- **Transformer Blocks**: Complete transformer architectures with attention and feedforward layers
- **Language Generation**: Autoregressive text generation with efficient attention patterns
- **Sequence Modeling**: Advanced sequence processing for various NLP tasks
- **🧠 Modern AI Systems**: Foundation for GPT, BERT, and other transformer-based models

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch Attention**: `torch.nn.MultiheadAttention` and `torch.nn.functional.scaled_dot_product_attention`
- **Flash Attention**: Memory-efficient attention computation used in production systems
- **KV-Cache Optimization**: Essential for efficient language model serving and generation
- **Industry Applications**: Every modern language model relies on optimized attention mechanisms

### 🎯 The Revolution of Attention
You have built the mechanism that transformed AI:
- **Before**: RNNs struggled with long-range dependencies and sequential computation
- **After**: Attention enables parallel processing and direct long-range connections

**Next Module**: Transformers - Combining your embeddings and attention into complete transformer architectures!

Your attention mechanisms are the computational core that enables transformers to understand and generate language. Now let's build the complete transformer blocks that use them!
"""