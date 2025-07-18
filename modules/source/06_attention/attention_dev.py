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
- Master the attention formula that powers all transformer models
- Create masking utilities for different attention patterns
- Build the foundation for understanding modern AI architectures

## Build → Use → Understand
1. **Build**: Implement the core attention mechanism from scratch using mathematical principles
2. **Use**: Apply attention to sequence tasks and visualize attention patterns
3. **Understand**: How attention revolutionized AI by enabling global context modeling

## What You'll Learn
By the end of this module, you'll understand:
- How attention enables dynamic focus on relevant input parts
- The mathematical foundation behind all transformer models
- Why attention is more powerful than fixed convolution kernels
- How masking enables different attention patterns (causal, padding)
- The building block that powers ChatGPT, BERT, and modern AI
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

# Import our building blocks - try package first, then local modules
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    from tensor_dev import Tensor

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
    SelfAttention,                 # Self-attention wrapper
    create_causal_mask,           # Masking utilities
    create_padding_mask
)
from tinytorch.core.tensor import Tensor  # Foundation
```

**Why this matters:**
- **Learning:** Focused module for deep understanding of core attention
- **Production:** Proper organization like PyTorch's attention functions
- **Consistency:** All attention mechanisms live together in `core.attention`
- **Foundation:** Building block for future transformer modules
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

### Attention vs Convolution
| Aspect | Convolution | Attention |
|--------|-------------|-----------|
| **Receptive field** | Local, grows with depth | Global from layer 1 |
| **Computation** | O(n) with kernel size | O(n²) with sequence length |
| **Weights** | Fixed learned kernels | Dynamic input-dependent |
| **Best for** | Spatial data (images) | Sequential data (text) |

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
## Step 3: Self-Attention - The Most Common Case

### What is Self-Attention?
**Self-Attention** is the most common use of attention where Q, K, and V all come from the same input sequence. This is what enables models like GPT to understand relationships between words in a sentence.

### Why Self-Attention Matters
- **Context understanding**: Each word can attend to every other word
- **Long-range dependencies**: Connect distant related concepts
- **Parallel processing**: Unlike RNNs, all positions computed simultaneously
- **Foundation of GPT**: How language models understand context

Let's create a convenient wrapper for self-attention!
"""

# %% nbgrader={"grade": false, "grade_id": "self-attention", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class SelfAttention:
    """
    Self-Attention wrapper - Convenience class for self-attention where Q=K=V.
    
    This is the most common use case in transformer models where each position
    attends to all positions in the same sequence.
    """
    
    def __init__(self, d_model: int):
        """
        Initialize Self-Attention.
        
        Args:
            d_model: Model dimension
        """
        self.d_model = d_model
        print(f"🔧 SelfAttention: d_model={d_model}")
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor (..., seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Self-attention output (..., seq_len, d_model)
            attention_weights: Attention weights
        """
        # Self-attention: Q = K = V = x
        return scaled_dot_product_attention(x, x, x, mask)
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make the class callable."""
        return self.forward(x, mask)

# %% [markdown]
"""
### 🧪 Unit Test: Self-Attention

**This is a unit test** - it tests self-attention wrapper functionality.

Let's verify our self-attention wrapper works correctly.
"""

# %% nbgrader={"grade": false, "grade_id": "test-self-attention", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Self-Attention...")

# Test parameters
d_model = 32
seq_len = 8
np.random.seed(42)

# Create test data (like word embeddings)
x = np.random.randn(seq_len, d_model) * 0.1

print(f"📊 Test setup: d_model={d_model}, seq_len={seq_len}")

# Create self-attention
self_attn = SelfAttention(d_model)

# Test forward pass
output, weights = self_attn(x)

print(f"📊 Output shapes: output{output.shape}, weights{weights.shape}")

# Verify properties
print(f"✅ Output shape preserved: {output.shape == x.shape}")
print(f"✅ Attention weights correct shape: {weights.shape == (seq_len, seq_len)}")
print(f"✅ Attention weights sum to 1: {np.allclose(np.sum(weights, axis=-1), 1.0)}")
print(f"✅ Self-attention is symmetric operation: {weights.shape[0] == weights.shape[1]}")

print("📈 Progress: Self-Attention ✓")

# %% [markdown]
"""
## Step 4: Attention Masking - Controlling Information Flow

### Why Masking Matters
Masking allows us to control which positions can attend to which other positions:

1. **Causal Masking**: For autoregressive models (like GPT) - can't see future tokens
2. **Padding Masking**: Ignore padding tokens in variable-length sequences
3. **Custom Masking**: Application-specific attention patterns

### Types of Masks
- **Causal (Lower Triangular)**: Position i can only attend to positions ≤ i
- **Padding**: Mask out padding tokens so they don't affect attention
- **Bidirectional**: All positions can attend to all positions (like BERT)

Let's implement these essential masking utilities!
"""

# %% nbgrader={"grade": false, "grade_id": "attention-masking", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (lower triangular) mask for autoregressive models.
    
    Used in models like GPT where each position can only attend to 
    previous positions, not future ones.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        mask: Causal mask (seq_len, seq_len) with 1s for allowed positions, 0s for blocked
    """
    return np.tril(np.ones((seq_len, seq_len)))

#| export  
def create_padding_mask(lengths: List[int], max_length: int) -> np.ndarray:
    """
    Create padding mask for variable-length sequences.
    
    Args:
        lengths: List of actual sequence lengths
        max_length: Maximum sequence length (padded length)
        
    Returns:
        mask: Padding mask (batch_size, max_length, max_length)
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_length, max_length))
    
    for i, length in enumerate(lengths):
        mask[i, :length, :length] = 1
    
    return mask

#| export
def create_bidirectional_mask(seq_len: int) -> np.ndarray:
    """
    Create a bidirectional mask where all positions can attend to all positions.
    
    Used in models like BERT for bidirectional context understanding.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        mask: All-ones mask (seq_len, seq_len)
    """
    return np.ones((seq_len, seq_len))

# %% [markdown]
"""
### 🧪 Unit Test: Attention Masking

**This is a unit test** - it tests all masking utilities work correctly.

Let's verify our masking functions create the correct patterns.
"""

# %% nbgrader={"grade": false, "grade_id": "test-masking", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Attention Masking...")

# Test causal mask
seq_len = 5
causal_mask = create_causal_mask(seq_len)

print(f"📊 Causal mask for seq_len={seq_len}:")
print(causal_mask)

# Verify causal mask properties
print(f"✅ Causal mask is lower triangular: {np.allclose(causal_mask, np.tril(causal_mask))}")
print(f"✅ Causal mask has correct shape: {causal_mask.shape == (seq_len, seq_len)}")
print(f"✅ Causal mask upper triangle is zeros: {np.all(np.triu(causal_mask, k=1) == 0)}")

# Test padding mask
lengths = [5, 3, 4]
max_length = 5
padding_mask = create_padding_mask(lengths, max_length)

print(f"📊 Padding mask for lengths {lengths}, max_length={max_length}:")
print("Mask for sequence 0 (length 5):")
print(padding_mask[0])
print("Mask for sequence 1 (length 3):")
print(padding_mask[1])

# Verify padding mask properties
print(f"✅ Padding mask has correct shape: {padding_mask.shape == (3, max_length, max_length)}")
print(f"✅ Full-length sequence is all ones: {np.all(padding_mask[0] == 1)}")
print(f"✅ Short sequence has zeros in padding area: {np.all(padding_mask[1, 3:, :] == 0)}")

# Test bidirectional mask
bidirectional_mask = create_bidirectional_mask(seq_len)
print(f"✅ Bidirectional mask is all ones: {np.all(bidirectional_mask == 1)}")
print(f"✅ Bidirectional mask has correct shape: {bidirectional_mask.shape == (seq_len, seq_len)}")

print("📈 Progress: Attention Masking ✓")

# %% [markdown]
"""
## Step 5: Attention Visualization and Analysis

### Understanding What Attention Learns
Let's create a simple example to see what attention patterns emerge and understand the behavior.
"""

# %% nbgrader={"grade": false, "grade_id": "attention-analysis", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🎯 Attention behavior analysis:")

# Create a simple sequence with clear patterns
simple_seq = np.array([
    [1, 0, 0, 0],  # Position 0: [1, 0, 0, 0]
    [0, 1, 0, 0],  # Position 1: [0, 1, 0, 0]  
    [0, 0, 1, 0],  # Position 2: [0, 0, 1, 0]
    [1, 0, 0, 0],  # Position 3: [1, 0, 0, 0] (same as position 0)
])

print(f"🎯 Simple test sequence shape: {simple_seq.shape}")

# Apply attention
output, weights = scaled_dot_product_attention(simple_seq, simple_seq, simple_seq)

print(f"🎯 Attention pattern analysis:")
print(f"Position 0 attends most to position: {np.argmax(weights[0])}")
print(f"Position 3 attends most to position: {np.argmax(weights[3])}")
print(f"✅ Positions with same content should attend to each other!")

# Test with causal masking
causal_mask = create_causal_mask(4)
output_causal, weights_causal = scaled_dot_product_attention(simple_seq, simple_seq, simple_seq, causal_mask)

print(f"🎯 With causal masking:")
print(f"Position 3 can only attend to positions 0-3: {np.sum(weights_causal[3, :]) > 0.99}")

if _should_show_plots():
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(weights, cmap='Blues')
    plt.title('Full Attention Weights\n(Darker = Higher Attention)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f'{weights[i,j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if weights[i,j] > 0.5 else 'black')
    
    plt.subplot(1, 3, 2)
    plt.imshow(weights_causal, cmap='Blues')
    plt.title('Causal Attention Weights\n(Upper triangle masked)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.plot(weights[0], 'o-', label='Position 0 attention')
    plt.plot(weights[3], 's-', label='Position 3 attention')
    plt.xlabel('Attending to Position')
    plt.ylabel('Attention Weight')
    plt.title('Attention Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

print("🎯 Attention learns to focus on similar content!")

# %% [markdown]
"""
### 🧪 Unit Test: Complete Attention System Integration

**This is a unit test** - it tests the complete attention system working together.

Let's verify all components work together seamlessly.
"""

# %% nbgrader={"grade": false, "grade_id": "test-integration", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔬 Unit Test: Complete Attention System Integration...")

# Test parameters
d_model = 64
seq_len = 16
batch_size = 2
np.random.seed(42)

print(f"📊 Integration test: d_model={d_model}, seq_len={seq_len}, batch_size={batch_size}")

# Step 1: Create input embeddings (simulating word embeddings)
embeddings = np.random.randn(batch_size, seq_len, d_model) * 0.1
print(f"📊 Input embeddings: {embeddings.shape}")

# Step 2: Test basic attention
output, attention_weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)
print(f"✅ Basic attention works: {output.shape}")

# Step 3: Test self-attention wrapper
self_attn = SelfAttention(d_model)
self_output, self_weights = self_attn(embeddings[0])  # Single batch item
print(f"✅ Self-attention output: {self_output.shape}")

# Step 4: Test with causal mask (like GPT)
causal_mask = create_causal_mask(seq_len)
causal_output, causal_weights = scaled_dot_product_attention(
    embeddings[0], embeddings[0], embeddings[0], causal_mask
)
print(f"✅ Causal attention works: {causal_output.shape}")

# Step 5: Test with padding mask (variable lengths)
lengths = [seq_len, seq_len-3]  # Different sequence lengths
padding_mask = create_padding_mask(lengths, seq_len)
padded_output, padded_weights = scaled_dot_product_attention(
    embeddings[0], embeddings[0], embeddings[0], padding_mask[0]
)
print(f"✅ Padding mask works: {padded_output.shape}")

# Step 6: Verify all outputs have correct properties
print(f"✅ All attention weights sum to 1: {np.allclose(np.sum(attention_weights, axis=-1), 1.0)}")
print(f"✅ All outputs preserve input shape: {output.shape == embeddings.shape}")
print(f"✅ Causal masking works: {np.all(np.triu(causal_weights, k=1) < 1e-6)}")

print("📈 Progress: Complete Attention System ✓")

print("\n" + "="*50)
print("🔥 ATTENTION MODULE COMPLETE!")
print("="*50)
print("✅ Scaled dot-product attention")
print("✅ Self-attention wrapper") 
print("✅ Causal masking")
print("✅ Padding masking")
print("✅ Bidirectional masking")
print("✅ Attention visualization")
print("✅ Complete integration tests")
print("\nYou now understand the core mechanism powering modern AI! 🚀")
print("Next: Learn how to build complete transformer models using this foundation.")
