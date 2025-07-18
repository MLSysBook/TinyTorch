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
# Attention - The Foundation of Modern AI

Welcome to the Attention module! This is where you'll implement the revolutionary mechanism that powers ChatGPT, BERT, GPT-4, and virtually all state-of-the-art AI systems.

## Learning Goals
- Understand attention as dynamic pattern matching with Query, Key, Value projections
- Implement scaled dot-product attention from mathematical foundations
- Build multi-head attention to capture diverse relationship patterns
- Create positional encoding to give transformers sequence awareness
- Compose transformer blocks that combine attention with feed-forward networks
- Compare attention's global connectivity with CNN's local receptive fields

## Build → Use → Reflect
1. **Build**: Implement attention mechanisms from scratch using mathematical principles
2. **Use**: Apply attention to sequence tasks and visualize attention patterns
3. **Reflect**: Understand how attention revolutionized AI by enabling global context modeling

## What You'll Learn
By the end of this module, you'll understand:
- How attention enables dynamic focus on relevant input parts
- Why multi-head attention captures diverse relationship types
- How positional encoding gives transformers sequence understanding
- The transformer architecture that powers modern AI systems
- Computational trade-offs between attention and convolution
"""

# %% nbgrader={"grade": false, "grade_id": "attention-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.attention

#| export
import numpy as np
import math
import sys
import os
from typing import List, Union, Optional, Tuple
import matplotlib.pyplot as plt

# Import all the building blocks we need - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import ReLU, Softmax
    from tinytorch.core.networks import Sequential
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_activations'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '04_layers'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_networks'))
    from tensor_dev import Tensor
    from activations_dev import ReLU, Softmax
    from layers_dev import Dense
    from networks_dev import Sequential

# %% nbgrader={"grade": false, "grade_id": "attention-setup", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| hide
#| export
def _should_show_plots():
    """Check if we should show plots (disable during testing)"""
    # Check multiple conditions that indicate we're in test mode
    is_pytest = (
        'pytest' in sys.modules or
        'test' in sys.argv or
        os.environ.get('PYTEST_CURRENT_TEST') is not None or
        any('test' in arg for arg in sys.argv) or
        any('pytest' in arg for arg in sys.argv)
    )
    
    # Show plots in development mode (when not in test mode)
    return not is_pytest

# %% nbgrader={"grade": false, "grade_id": "attention-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔥 TinyTorch Attention Module")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Ready to build attention mechanisms that power modern AI!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/06_attention/attention_dev.py`  
**Building Side:** Code exports to `tinytorch.core.attention`

```python
# Final package structure:
from tinytorch.core.attention import (
    scaled_dot_product_attention,  # Core attention function
    MultiHeadAttention,           # Multi-head attention layer
    PositionalEncoding,           # Position information
    TransformerBlock,             # Complete transformer layer
    SelfAttention                 # Self-attention wrapper
)
from tinytorch.core.layers import Dense  # Building blocks
from tinytorch.core.tensor import Tensor  # Foundation
```

**Why this matters:**
- **Learning:** Focused module for deep understanding of attention
- **Production:** Proper organization like PyTorch's `torch.nn.MultiheadAttention`
- **Consistency:** All attention mechanisms live together in `core.attention`
- **Integration:** Works seamlessly with tensors, layers, and networks
"""

# %% [markdown]
"""
## Step 1: Understanding Attention - The Revolutionary Mechanism

### What is Attention?
**Attention** is a mechanism that allows models to dynamically focus on relevant parts of the input. It's like having a spotlight that can shine on different parts of a sequence based on what's most important for the current task.

### The Fundamental Insight: Query, Key, Value
Attention works through three projections:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What information is available?"
- **Value (V)**: "What is the actual content?"

### Real-World Analogy: Library Search
Imagine searching in a library:
```
Query: "machine learning books"     ← What you're looking for
Keys: ["AI", "ML", "physics", ...] ← Book category labels  
Values: [book1, book2, book3, ...]  ← Actual book contents

Attention: Look at all keys, find matches with query, 
          return weighted combination of corresponding values
```

### The Attention Formula
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Step by step:**
1. **Compute scores**: `QK^T` measures similarity between queries and keys
2. **Scale**: Divide by `√d_k` to prevent extremely large values
3. **Normalize**: `softmax` converts scores to probabilities
4. **Combine**: Weight the values by attention probabilities

### Why This Is Revolutionary
- **Dynamic weights**: Unlike fixed convolution kernels, attention adapts to input
- **Global connectivity**: Any position can attend to any other position directly
- **Interpretability**: Attention weights show what the model focuses on
- **Scalability**: Works for sequences of varying lengths

Let's implement this step by step!
"""

# %% [markdown]
"""
## Step 2: Implementing Scaled Dot-Product Attention

### The Core Attention Operation
This is the mathematical heart of all modern AI systems. Every transformer model (GPT, BERT, etc.) uses this exact operation.

### Mathematical Foundation
```
scores = QK^T / √d_k
attention_weights = softmax(scores)
output = attention_weights @ V
```

### Why Scale by √d_k?
- **Prevents saturation**: Large dot products → extreme softmax values → vanishing gradients
- **Stable training**: Keeps attention weights in a reasonable range
- **Mathematical insight**: Compensates for variance growth with dimension

Let's build the fundamental attention function!
"""

# %% nbgrader={"grade": false, "grade_id": "scaled-dot-product-attention", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention - The foundation of all transformer models.
    
    This is the exact mechanism used in GPT, BERT, and all modern language models.
    
    Args:
        Q: Query matrix of shape (..., seq_len_q, d_k)
        K: Key matrix of shape (..., seq_len_k, d_k)  
        V: Value matrix of shape (..., seq_len_v, d_v)
        mask: Optional mask of shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output (..., seq_len_q, d_v)
        attention_weights: Attention probabilities (..., seq_len_q, seq_len_k)
    
    Mathematical operation:
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    # Get the dimension for scaling
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores (QK^T)
    # This measures similarity between each query and each key
    scores = np.matmul(Q, np.swapaxes(K, -2, -1))  # (..., seq_len_q, seq_len_k)
    
    # Step 2: Scale by √d_k to prevent exploding gradients
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask if provided (for padding or causality)
    if mask is not None:
        # Replace masked positions with large negative values
        # This makes softmax output ~0 for these positions
        scores = np.where(mask == 0, -1e9, scores)
    
    # Step 4: Apply softmax to get attention probabilities
    # Each row sums to 1, representing where to focus attention
    # Using numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    
    # Step 5: Apply attention weights to values
    # This gives us the weighted combination of values
    output = np.matmul(attention_weights, V)  # (..., seq_len_q, d_v)
    
    return output, attention_weights

# %% [markdown]
"""
### 🧪 Unit Test: Scaled Dot-Product Attention

**This is a unit test** - it tests the core attention mechanism in isolation.

Let's verify our attention implementation works correctly with a simple example.
"""

# %% nbgrader={"grade": false, "grade_id": "test-attention", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Scaled Dot-Product Attention...")

# Create simple test data
seq_len, d_model = 4, 6
np.random.seed(42)

# Create Q, K, V matrices
Q = np.random.randn(seq_len, d_model) * 0.1
K = np.random.randn(seq_len, d_model) * 0.1  
V = np.random.randn(seq_len, d_model) * 0.1

print(f"📊 Input shapes: Q{Q.shape}, K{K.shape}, V{V.shape}")

# Test attention
output, weights = scaled_dot_product_attention(Q, K, V)

print(f"📊 Output shapes: output{output.shape}, weights{weights.shape}")

# Verify properties
weights_sum = np.sum(weights, axis=-1)
print(f"✅ Attention weights sum to 1: {np.allclose(weights_sum, 1.0)}")
print(f"✅ Output has correct shape: {output.shape == (seq_len, d_model)}")
print(f"✅ All weights are non-negative: {np.all(weights >= 0)}")

# Test with mask
mask = np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0], 
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])
output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)

# Check that masked positions have near-zero attention
masked_positions = (mask == 0)
masked_weights = weights_masked[masked_positions]
print(f"✅ Masked positions have near-zero weights: {np.all(masked_weights < 1e-6)}")

print("📈 Progress: Scaled Dot-Product Attention ✓")

# %% [markdown]
"""
## Step 3: Multi-Head Attention - Capturing Diverse Relationships

### Why Multiple Heads?
A single attention head captures one type of relationship. Multiple heads allow the model to attend to different types of patterns simultaneously:

- **Head 1**: Syntactic relationships (subject-verb)
- **Head 2**: Semantic relationships (word meanings)
- **Head 3**: Long-range dependencies
- **Head 4**: Local context patterns

### The Multi-Head Architecture
```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ)W^O

where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```

### Implementation Strategy
1. **Project**: Apply learned projections to create Q, K, V for each head
2. **Split**: Divide into multiple heads with smaller dimensions
3. **Attend**: Apply attention for each head independently
4. **Combine**: Concatenate heads and apply output projection

Let's build this powerful attention mechanism!
"""

# %% nbgrader={"grade": false, "grade_id": "multi-head-attention", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class MultiHeadAttention:
    """
    Multi-Head Attention - Enables models to attend to different representation 
    subspaces simultaneously. This is the core component of transformer models.
    
    In transformers, each head learns to focus on different types of relationships:
    - Syntactic patterns
    - Semantic relationships  
    - Long-range dependencies
    - Local context
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # For testing purposes, use simple weight matrices instead of Dense layers
        # In production, these would be Dense layers
        np.random.seed(42)  # For reproducible testing
        self.W_q = np.random.randn(d_model, d_model) * 0.1  # Query projection
        self.W_k = np.random.randn(d_model, d_model) * 0.1  # Key projection  
        self.W_v = np.random.randn(d_model, d_model) * 0.1  # Value projection
        self.W_o = np.random.randn(d_model, d_model) * 0.1  # Output projection
        
        print(f"🔧 MultiHeadAttention: {num_heads} heads, {self.d_k} dims per head")
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (..., seq_len_q, d_model)
            key: Key tensor (..., seq_len_k, d_model)
            value: Value tensor (..., seq_len_v, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attention output (..., seq_len_q, d_model)
            attention_weights: Average attention weights across heads
        """
        batch_dims = query.shape[:-2]
        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]
        
        # Step 1: Apply linear projections to get Q, K, V for all heads
        Q = np.matmul(query, self.W_q)  # (..., seq_len_q, d_model)
        K = np.matmul(key, self.W_k)    # (..., seq_len_k, d_model)  
        V = np.matmul(value, self.W_v)  # (..., seq_len_v, d_model)
        
        # Step 2: Reshape and transpose for multi-head processing
        # Split d_model into num_heads * d_k
        Q = self._reshape_for_heads(Q)  # (..., num_heads, seq_len_q, d_k)
        K = self._reshape_for_heads(K)  # (..., num_heads, seq_len_k, d_k)
        V = self._reshape_for_heads(V)  # (..., num_heads, seq_len_v, d_k)
        
        # Step 3: Apply attention for each head
        attention_output, attention_weights = self._apply_attention_heads(Q, K, V, mask)
        
        # Step 4: Concatenate heads and apply output projection
        # Reshape back to (..., seq_len_q, d_model)
        attention_output = self._concatenate_heads(attention_output)
        
        # Final linear projection
        output = np.matmul(attention_output, self.W_o)
        
        return output, attention_weights
    
    def _reshape_for_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape tensor for multi-head processing."""
        # Input: (..., seq_len, d_model)
        # Output: (..., num_heads, seq_len, d_k)
        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # Reshape to (..., seq_len, num_heads, d_k)
        x = x.reshape(*batch_dims, seq_len, self.num_heads, self.d_k)
        
        # Transpose to (..., num_heads, seq_len, d_k)
        x = x.transpose(*range(len(batch_dims)), -2, -3, -1)
        
        return x
    
    def _apply_attention_heads(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                              mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply scaled dot-product attention for each head."""
        # Input shapes: (..., num_heads, seq_len, d_k)
        
        # Apply attention to each head
        attention_outputs = []
        attention_weights_list = []
        
        for head in range(self.num_heads):
            # Extract tensors for this head
            Q_head = Q[..., head, :, :]  # (..., seq_len_q, d_k)
            K_head = K[..., head, :, :]  # (..., seq_len_k, d_k)
            V_head = V[..., head, :, :]  # (..., seq_len_v, d_k)
            
            # Apply attention for this head
            head_output, head_weights = scaled_dot_product_attention(Q_head, K_head, V_head, mask)
            
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)
        
        # Stack outputs: (..., num_heads, seq_len_q, d_k)
        attention_output = np.stack(attention_outputs, axis=-3)
        
        # Average attention weights across heads for visualization
        attention_weights = np.stack(attention_weights_list, axis=-3)
        attention_weights = np.mean(attention_weights, axis=-3)
        
        return attention_output, attention_weights
    
    def _concatenate_heads(self, x: np.ndarray) -> np.ndarray:
        """Concatenate attention heads back to original dimension."""
        # Input: (..., num_heads, seq_len, d_k)
        # Output: (..., seq_len, d_model)
        batch_dims = x.shape[:-3]
        seq_len = x.shape[-2]
        
        # Transpose to (..., seq_len, num_heads, d_k)
        x = x.transpose(*range(len(batch_dims)), -2, -3, -1)
        
        # Reshape to (..., seq_len, d_model)
        x = x.reshape(*batch_dims, seq_len, self.d_model)
        
        return x
    
    def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                 mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make the class callable."""
        return self.forward(query, key, value, mask)

# %% [markdown]
"""
### 🧪 Unit Test: Multi-Head Attention

**This is a unit test** - it tests multi-head attention composition in isolation.

Let's verify that our multi-head attention correctly splits, processes, and combines attention heads.
"""

# %% nbgrader={"grade": false, "grade_id": "test-multi-head", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Multi-Head Attention...")

# Test parameters
d_model = 64
num_heads = 8
seq_len = 10
np.random.seed(42)

# Create test data
query = np.random.randn(seq_len, d_model) * 0.1
key = np.random.randn(seq_len, d_model) * 0.1
value = np.random.randn(seq_len, d_model) * 0.1

print(f"📊 Test setup: d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}")

# Create multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Test forward pass
output, weights = mha(query, key, value)

print(f"📊 Output shapes: output{output.shape}, weights{weights.shape}")

# Verify properties
print(f"✅ Output shape correct: {output.shape == (seq_len, d_model)}")
print(f"✅ Attention weights shape correct: {weights.shape == (seq_len, seq_len)}")
print(f"✅ Attention weights sum to 1: {np.allclose(np.sum(weights, axis=-1), 1.0)}")
print(f"✅ d_k per head correct: {mha.d_k == d_model // num_heads}")

# Test self-attention (Q = K = V)
self_output, self_weights = mha(query, query, query)
print(f"✅ Self-attention works: {self_output.shape == (seq_len, d_model)}")

print("📈 Progress: Multi-Head Attention ✓")

# %% [markdown]
"""
## Step 4: Positional Encoding - Teaching Transformers About Order

### The Position Problem
Unlike RNNs or CNNs, attention is **position-agnostic**. The operation is symmetric - swapping two input positions gives the same result. This is both a strength (parallelizable) and weakness (no understanding of order).

### Why Position Matters
For sequences, order is crucial:
- "The cat sat on the mat" ≠ "The mat sat on the cat"
- "I didn't say she stole my money" has different meanings based on emphasis
- Code execution order matters: `x = 1; y = x + 1` ≠ `y = x + 1; x = 1`

### Sinusoidal Positional Encoding
The Transformer paper uses sinusoidal functions to encode position:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Why Sinusoidal?
- **Deterministic**: Same position always gets same encoding
- **Extrapolation**: Can handle sequences longer than training
- **Smooth**: Similar positions get similar encodings
- **Learnable patterns**: Model can learn to use positional relationships

Let's implement this crucial component!
"""

# %% nbgrader={"grade": false, "grade_id": "positional-encoding", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class PositionalEncoding:
    """
    Positional Encoding using sinusoidal functions.
    
    Adds position information to transformer inputs so the model
    can understand sequence order. Uses the same approach as the
    original Transformer paper.
    """
    
    def __init__(self, d_model: int, max_length: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length to precompute
        """
        self.d_model = d_model
        self.max_length = max_length
        
        # Precompute positional encodings
        self.pe = self._create_positional_encoding()
        print(f"🔧 PositionalEncoding: d_model={d_model}, max_length={max_length}")
    
    def _create_positional_encoding(self) -> np.ndarray:
        """
        Create sinusoidal positional encoding matrix.
        
        Returns:
            pe: Positional encoding matrix (max_length, d_model)
        """
        pe = np.zeros((self.max_length, self.d_model))
        
        # Create position indices
        position = np.arange(self.max_length).reshape(-1, 1)  # (max_length, 1)
        
        # Create dimension indices for the sinusoidal pattern
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(math.log(10000.0) / self.d_model))  # (d_model//2,)
        
        # Apply sinusoidal functions
        pe[:, 0::2] = np.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = np.cos(position * div_term)  # Odd indices: cos
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor (..., seq_len, d_model)
            
        Returns:
            output: Input + positional encoding (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_length}")
        
        # Get positional encoding for this sequence length
        pos_encoding = self.pe[:seq_len, :]  # (seq_len, d_model)
        
        # Add to input (broadcasting handles batch dimensions)
        output = x + pos_encoding
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the class callable."""
        return self.forward(x)

# %% [markdown]
"""
### 🧪 Unit Test: Positional Encoding

**This is a unit test** - it tests positional encoding addition in isolation.

Let's verify that positional encoding adds meaningful position information.
"""

# %% nbgrader={"grade": false, "grade_id": "test-positional-encoding", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Positional Encoding...")

# Test parameters
d_model = 32
max_length = 100
seq_len = 8
np.random.seed(42)

# Create test data (like word embeddings)
embeddings = np.random.randn(seq_len, d_model) * 0.1

print(f"📊 Test setup: d_model={d_model}, seq_len={seq_len}")

# Create positional encoding
pos_enc = PositionalEncoding(d_model, max_length)

# Test forward pass
output = pos_enc(embeddings)

print(f"📊 Shapes: input{embeddings.shape}, output{output.shape}")

# Verify properties
print(f"✅ Output shape preserved: {output.shape == embeddings.shape}")
print(f"✅ Positional encoding has correct shape: {pos_enc.pe.shape == (max_length, d_model)}")

# Test that different positions get different encodings
pos_0 = pos_enc.pe[0, :]
pos_1 = pos_enc.pe[1, :] 
pos_10 = pos_enc.pe[10, :]

print(f"✅ Different positions have different encodings: {not np.allclose(pos_0, pos_1)}")
print(f"✅ Position encoding bounded: {np.all(np.abs(pos_enc.pe) <= 1.1)}")

print("📈 Progress: Positional Encoding ✓")

# %% [markdown]
"""
## Step 5: Layer Normalization - Stabilizing Training

### Why Normalization Matters
Deep networks suffer from **internal covariate shift** - as parameters change during training, the distribution of layer inputs changes, making training unstable.

### Layer Normalization vs Batch Normalization
- **Batch Norm**: Normalizes across the batch dimension
- **Layer Norm**: Normalizes across the feature dimension
- **Why Layer Norm for Transformers**: Works better with variable sequence lengths and smaller batches

### The Layer Norm Operation
```
LayerNorm(x) = γ * (x - μ) / σ + β

where:
  μ = mean(x, axis=-1)    # Mean across features
  σ = std(x, axis=-1)     # Standard deviation across features  
  γ, β = learnable parameters
```

Let's implement this essential component!
"""

# %% nbgrader={"grade": false, "grade_id": "layer-normalization", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class LayerNorm:
    """
    Layer Normalization - Normalizes inputs across the feature dimension.
    
    Essential for stable transformer training. Unlike batch normalization,
    layer norm works consistently across different batch sizes and
    sequence lengths.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize Layer Normalization.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(d_model)   # Scale parameter
        self.beta = np.zeros(d_model)   # Shift parameter
        
        print(f"🔧 LayerNorm: d_model={d_model}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor (..., d_model)
            
        Returns:
            output: Normalized tensor (..., d_model)
        """
        # Compute mean and variance across the last dimension (features)
        mean = np.mean(x, axis=-1, keepdims=True)      # (..., 1)
        variance = np.var(x, axis=-1, keepdims=True)   # (..., 1)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        # Apply learnable transformation
        output = self.gamma * x_normalized + self.beta
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the class callable."""
        return self.forward(x) 
# %% [markdown]
"""
## Step 6: Complete Transformer Block

### The Transformer Architecture
A transformer block combines all the components we've built:

1. **Multi-Head Self-Attention** - Global context modeling
2. **Residual Connection** - Gradient flow and training stability  
3. **Layer Normalization** - Input distribution stabilization
4. **Feed-Forward Network** - Non-linear transformation
5. **Another Residual + LayerNorm** - More stability

### Pre-Norm vs Post-Norm
We'll use **Pre-Norm** (LayerNorm before attention/FFN) as it's more stable:
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Let's build the complete transformer block!
"""

# %% nbgrader={"grade": false, "grade_id": "transformer-block", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class TransformerBlock:
    """
    Complete Transformer Block - The fundamental building block of transformer models.
    
    Combines multi-head attention, feed-forward networks, residual connections,
    and layer normalization. This is the exact architecture used in GPT, BERT,
    and other transformer models.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize Transformer Block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension (usually 4 * d_model)
            dropout: Dropout rate (not implemented in this version)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network (simplified for testing)
        np.random.seed(42)
        self.ff_w1 = np.random.randn(d_model, d_ff) * 0.1
        self.ff_b1 = np.zeros(d_ff)
        self.ff_w2 = np.random.randn(d_ff, d_model) * 0.1
        self.ff_b2 = np.zeros(d_model)
        
        # Layer normalization layers
        self.ln1 = LayerNorm(d_model)  # Before attention
        self.ln2 = LayerNorm(d_model)  # Before feed-forward
        
        print(f"🔧 TransformerBlock: d_model={d_model}, heads={num_heads}, d_ff={d_ff}")
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor (..., seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Transformed tensor (..., seq_len, d_model)
            attention_weights: Attention weights from self-attention
        """
        # Self-attention with residual connection and layer norm (Pre-Norm)
        ln1_output = self.ln1(x)
        attn_output, attention_weights = self.self_attention(ln1_output, ln1_output, ln1_output, mask)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with residual connection and layer norm (Pre-Norm)
        ln2_output = self.ln2(x)
        # Simple feed-forward: Linear -> ReLU -> Linear
        ff_hidden = np.matmul(ln2_output, self.ff_w1) + self.ff_b1
        ff_hidden = np.maximum(0, ff_hidden)  # ReLU activation
        ff_output = np.matmul(ff_hidden, self.ff_w2) + self.ff_b2
        x = x + ff_output  # Residual connection
        
        return x, attention_weights
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make the class callable."""
        return self.forward(x, mask)

# %% [markdown]
"""
### 🧪 Unit Test: Complete Transformer Block

**This is a unit test** - it tests the complete transformer block integration.

Let's verify that our transformer block properly combines all components.
"""

# %% nbgrader={"grade": false, "grade_id": "test-transformer-block", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Complete Transformer Block...")

# Test parameters
d_model = 64
num_heads = 8
d_ff = 256
seq_len = 12
np.random.seed(42)

# Create test data (like embeddings + positional encoding)
x = np.random.randn(seq_len, d_model) * 0.1

print(f"📊 Test setup: d_model={d_model}, heads={num_heads}, d_ff={d_ff}, seq_len={seq_len}")

# Create transformer block
transformer = TransformerBlock(d_model, num_heads, d_ff)

# Test forward pass
output, attention_weights = transformer(x)

print(f"📊 Output shapes: output{output.shape}, attention{attention_weights.shape}")

# Verify properties
print(f"✅ Output shape preserved: {output.shape == x.shape}")
print(f"✅ Attention weights correct shape: {attention_weights.shape == (seq_len, seq_len)}")
print(f"✅ Attention weights sum to 1: {np.allclose(np.sum(attention_weights, axis=-1), 1.0)}")

# Test with causal mask (for autoregressive models like GPT)
causal_mask = np.tril(np.ones((seq_len, seq_len)))  # Lower triangular mask
output_masked, attention_masked = transformer(x, causal_mask)

print(f"✅ Masked transformer works: {output_masked.shape == x.shape}")

# Verify causal masking worked
upper_triangle = np.triu(attention_masked, k=1)  # Upper triangle should be ~0
print(f"✅ Causal masking applied: {np.all(upper_triangle < 1e-6)}")

print("📈 Progress: Complete Transformer Block ✓")

print("\n" + "="*50)
print("🔥 ATTENTION MODULE COMPLETE!")
print("="*50)
print("✅ Scaled dot-product attention")
print("✅ Multi-head attention") 
print("✅ Positional encoding")
print("✅ Layer normalization")
print("✅ Complete transformer block")
print("✅ Self-attention wrapper")
print("✅ Masking utilities")
print("✅ Integration tests")
print("\nYou now understand the core mechanism powering modern AI! 🚀")
print("Next: Apply these attention mechanisms to real datasets and tasks.")
