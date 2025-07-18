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
    
    TODO: Implement the core attention mechanism.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Get d_k (dimension of keys) from Q.shape[-1]
    2. Compute attention scores: Q @ K^T (matrix multiplication)
    3. Scale by √d_k: scores / sqrt(d_k)
    4. Apply mask if provided: set masked positions to -1e9
    5. Apply softmax to get attention weights (probabilities)
    6. Apply attention weights to values: weights @ V
    7. Return (output, attention_weights)
    
    MATHEMATICAL OPERATION:
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    IMPLEMENTATION HINTS:
    - Use np.matmul() for matrix multiplication
    - Use np.swapaxes(K, -2, -1) to transpose last two dimensions
    - Use math.sqrt() for square root
    - Use np.where() for masking: np.where(mask == 0, -1e9, scores)
    - Implement softmax manually: exp(x) / sum(exp(x))
    - Use keepdims=True for broadcasting
    
    LEARNING CONNECTIONS:
    - This exact function powers ChatGPT, BERT, GPT-4
    - The scaling prevents gradient vanishing in deep networks
    - Masking enables causal (GPT) and bidirectional (BERT) models
    - Attention weights are interpretable - you can visualize them!
    
    Args:
        Q: Query matrix of shape (..., seq_len_q, d_k)
        K: Key matrix of shape (..., seq_len_k, d_k)  
        V: Value matrix of shape (..., seq_len_v, d_v)
        mask: Optional mask of shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output (..., seq_len_q, d_v)
        attention_weights: Attention probabilities (..., seq_len_q, seq_len_k)
    """
    ### BEGIN SOLUTION
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
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Attention Implementation

Once you implement the `scaled_dot_product_attention` function above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-attention-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_scaled_dot_product_attention():
    """Test scaled dot-product attention implementation"""
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
    assert np.allclose(weights_sum, 1.0), f"Attention weights should sum to 1, got {weights_sum}"
    assert output.shape == (seq_len, d_model), f"Output shape should be {(seq_len, d_model)}, got {output.shape}"
    assert np.all(weights >= 0), "All attention weights should be non-negative"

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
    assert np.all(masked_weights < 1e-6), "Masked positions should have near-zero weights"

    print("✅ Attention weights sum to 1: True")
    print("✅ Output has correct shape: True")
    print("✅ All weights are non-negative: True")
    print("✅ Masked positions have near-zero weights: True")
    print("📈 Progress: Scaled Dot-Product Attention ✓")

# Run the test
test_scaled_dot_product_attention()

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
        
        TODO: Store the model dimension for this self-attention layer.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store d_model as an instance variable (self.d_model)
        2. Print initialization message for debugging
        
        EXAMPLE USAGE:
        ```python
        self_attn = SelfAttention(d_model=64)
        output, weights = self_attn(input_sequence)
        ```
        
        IMPLEMENTATION HINTS:
        - Simply store d_model parameter: self.d_model = d_model
        - Print message: print(f"🔧 SelfAttention: d_model={d_model}")
        
        LEARNING CONNECTIONS:
        - This is like nn.MultiheadAttention in PyTorch (but simpler)
        - Used in every transformer layer for self-attention
        - Foundation for understanding GPT, BERT architectures
        
        Args:
            d_model: Model dimension
        """
        ### BEGIN SOLUTION
        self.d_model = d_model
        print(f"🔧 SelfAttention: d_model={d_model}")
        ### END SOLUTION
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of self-attention.
        
        TODO: Apply self-attention where Q=K=V=x.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Call scaled_dot_product_attention with Q=K=V=x
        2. Pass the mask parameter through
        3. Return the output and attention weights
        
        EXAMPLE USAGE:
        ```python
        x = np.random.randn(seq_len, d_model)  # Input sequence
        output, weights = self_attn.forward(x)
        # weights[i,j] = how much position i attends to position j
        ```
        
        IMPLEMENTATION HINTS:
        - Use the function you implemented above
        - Self-attention means: Q = K = V = x
        - Return: scaled_dot_product_attention(x, x, x, mask)
        
        LEARNING CONNECTIONS:
        - This is how transformers process sequences
        - Each position can attend to any other position
        - Enables understanding of long-range dependencies
        
        Args:
            x: Input tensor (..., seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Self-attention output (..., seq_len, d_model)
            attention_weights: Attention weights
        """
        ### BEGIN SOLUTION
        # Self-attention: Q = K = V = x
        return scaled_dot_product_attention(x, x, x, mask)
        ### END SOLUTION
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make the class callable."""
        return self.forward(x, mask)

# %% [markdown]
"""
### 🧪 Test Your Self-Attention Implementation

Once you implement the SelfAttention class above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-self-attention-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_self_attention():
    """Test self-attention wrapper"""
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
    assert output.shape == x.shape, f"Output shape should match input shape {x.shape}, got {output.shape}"
    assert weights.shape == (seq_len, seq_len), f"Attention weights shape should be {(seq_len, seq_len)}, got {weights.shape}"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Attention weights should sum to 1"
    assert weights.shape[0] == weights.shape[1], "Self-attention weights should be square matrix"

    print("✅ Output shape preserved: True")
    print("✅ Attention weights correct shape: True")
    print("✅ Attention weights sum to 1: True")
    print("✅ Self-attention is symmetric operation: True")
    print("📈 Progress: Self-Attention ✓")

# Run the test
test_self_attention()

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
    
    TODO: Create a lower triangular matrix of ones.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Use np.tril() to create lower triangular matrix
    2. Create matrix of ones with shape (seq_len, seq_len)
    3. Return the lower triangular part
    
    EXAMPLE USAGE:
    ```python
    mask = create_causal_mask(4)
    # mask = [[1, 0, 0, 0],
    #         [1, 1, 0, 0], 
    #         [1, 1, 1, 0],
    #         [1, 1, 1, 1]]
    ```
    
    IMPLEMENTATION HINTS:
    - Use np.ones((seq_len, seq_len)) to create matrix of ones
    - Use np.tril() to get lower triangular part
    - Or combine: np.tril(np.ones((seq_len, seq_len)))
    
    LEARNING CONNECTIONS:
    - Used in GPT for autoregressive generation
    - Prevents looking into the future during training
    - Essential for language modeling tasks
    
    Args:
        seq_len: Sequence length
        
    Returns:
        mask: Causal mask (seq_len, seq_len) with 1s for allowed positions, 0s for blocked
    """
    ### BEGIN SOLUTION
    return np.tril(np.ones((seq_len, seq_len)))
    ### END SOLUTION

#| export  
def create_padding_mask(lengths: List[int], max_length: int) -> np.ndarray:
    """
    Create padding mask for variable-length sequences.
    
    TODO: Create mask that ignores padding tokens.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Initialize zero array with shape (batch_size, max_length, max_length)
    2. For each sequence in the batch, set valid positions to 1
    3. Valid positions are [:length, :length] for each sequence
    4. Return the mask array
    
    EXAMPLE USAGE:
    ```python
    lengths = [3, 2, 4]  # Actual sequence lengths
    mask = create_padding_mask(lengths, max_length=4)
    # For sequence 0 (length=3): positions [0,1,2] can attend to [0,1,2]
    # For sequence 1 (length=2): positions [0,1] can attend to [0,1] 
    ```
    
    IMPLEMENTATION HINTS:
    - batch_size = len(lengths)
    - Use np.zeros((batch_size, max_length, max_length))
    - Loop through lengths: for i, length in enumerate(lengths)
    - Set valid region: mask[i, :length, :length] = 1
    
    LEARNING CONNECTIONS:
    - Used when sequences have different lengths
    - Prevents attention to padding tokens
    - Essential for efficient batch processing
    
    Args:
        lengths: List of actual sequence lengths
        max_length: Maximum sequence length (padded length)
        
    Returns:
        mask: Padding mask (batch_size, max_length, max_length)
    """
    ### BEGIN SOLUTION
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_length, max_length))
    
    for i, length in enumerate(lengths):
        mask[i, :length, :length] = 1
    
    return mask
    ### END SOLUTION

#| export
def create_bidirectional_mask(seq_len: int) -> np.ndarray:
    """
    Create a bidirectional mask where all positions can attend to all positions.
    
    Used in models like BERT for bidirectional context understanding.
    
    TODO: Create a matrix of all ones.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Use np.ones() to create matrix of all ones
    2. Shape should be (seq_len, seq_len)
    3. Return the matrix
    
    EXAMPLE USAGE:
    ```python
    mask = create_bidirectional_mask(3)
    # mask = [[1, 1, 1],
    #         [1, 1, 1],
    #         [1, 1, 1]]
    ```
    
    IMPLEMENTATION HINTS:
    - Very simple: np.ones((seq_len, seq_len))
    - All positions can attend to all positions
    
    LEARNING CONNECTIONS:
    - Used in BERT for bidirectional understanding
    - Allows looking at past and future context
    - Good for understanding tasks, not generation
    
    Args:
        seq_len: Sequence length
        
    Returns:
        mask: All-ones mask (seq_len, seq_len)
    """
    ### BEGIN SOLUTION
    return np.ones((seq_len, seq_len))
    ### END SOLUTION

# %% [markdown]
"""
### 🧪 Test Your Masking Functions

Once you implement the masking functions above, run this cell to test them:
"""

# %% nbgrader={"grade": true, "grade_id": "test-masking-immediate", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
def test_attention_masking():
    """Test attention masking utilities"""
    print("🔬 Unit Test: Attention Masking...")

    # Test causal mask
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)

    print(f"📊 Causal mask for seq_len={seq_len}:")
    print(causal_mask)

    # Verify causal mask properties
    assert np.allclose(causal_mask, np.tril(causal_mask)), "Causal mask should be lower triangular"
    assert causal_mask.shape == (seq_len, seq_len), f"Causal mask should have shape {(seq_len, seq_len)}"
    assert np.all(np.triu(causal_mask, k=1) == 0), "Causal mask upper triangle should be zeros"

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
    assert padding_mask.shape == (3, max_length, max_length), f"Padding mask should have shape {(3, max_length, max_length)}"
    assert np.all(padding_mask[0] == 1), "Full-length sequence should be all ones"
    assert np.all(padding_mask[1, 3:, :] == 0), "Short sequence should have zeros in padding area"

    # Test bidirectional mask
    bidirectional_mask = create_bidirectional_mask(seq_len)
    assert np.all(bidirectional_mask == 1), "Bidirectional mask should be all ones"
    assert bidirectional_mask.shape == (seq_len, seq_len), f"Bidirectional mask should have shape {(seq_len, seq_len)}"

    print("✅ Causal mask is lower triangular: True")
    print("✅ Causal mask has correct shape: True")
    print("✅ Causal mask upper triangle is zeros: True")
    print("✅ Padding mask has correct shape: True")
    print("✅ Full-length sequence is all ones: True")
    print("✅ Short sequence has zeros in padding area: True")
    print("✅ Bidirectional mask is all ones: True")
    print("✅ Bidirectional mask has correct shape: True")
    print("📈 Progress: Attention Masking ✓")

# Run the test
test_attention_masking()

# %% [markdown]
"""
## Step 5: Complete System Integration Test

### Bringing It All Together
Let's test all components working together in a realistic scenario similar to how they would be used in actual transformer models.
"""

# %% nbgrader={"grade": true, "grade_id": "test-integration-final", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_complete_attention_system():
    """Test the complete attention system working together"""
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
    assert output.shape == embeddings.shape, "Basic attention should preserve shape"
    print(f"✅ Basic attention works: {output.shape}")

    # Step 3: Test self-attention wrapper
    self_attn = SelfAttention(d_model)
    self_output, self_weights = self_attn(embeddings[0])  # Single batch item
    assert self_output.shape == (seq_len, d_model), "Self-attention should preserve shape"
    print(f"✅ Self-attention output: {self_output.shape}")

    # Step 4: Test with causal mask (like GPT)
    causal_mask = create_causal_mask(seq_len)
    causal_output, causal_weights = scaled_dot_product_attention(
        embeddings[0], embeddings[0], embeddings[0], causal_mask
    )
    assert causal_output.shape == (seq_len, d_model), "Causal attention should preserve shape"
    print(f"✅ Causal attention works: {causal_output.shape}")

    # Step 5: Test with padding mask (variable lengths)
    lengths = [seq_len, seq_len-3]  # Different sequence lengths
    padding_mask = create_padding_mask(lengths, seq_len)
    padded_output, padded_weights = scaled_dot_product_attention(
        embeddings[0], embeddings[0], embeddings[0], padding_mask[0]
    )
    assert padded_output.shape == (seq_len, d_model), "Padding attention should preserve shape"
    print(f"✅ Padding mask works: {padded_output.shape}")

    # Step 6: Verify all outputs have correct properties
    assert np.allclose(np.sum(attention_weights, axis=-1), 1.0), "All attention weights should sum to 1"
    assert output.shape == embeddings.shape, "All outputs should preserve input shape"
    assert np.all(np.triu(causal_weights, k=1) < 1e-6), "Causal masking should work"

    print("✅ All attention weights sum to 1: True")
    print("✅ All outputs preserve input shape: True")
    print("✅ Causal masking works: True")
    print("📈 Progress: Complete Attention System ✓")

# Run the test
test_complete_attention_system()

# %% [markdown]
"""
## 🎯 Attention Behavior Analysis

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

def test_attention_mechanism():
    """Test attention mechanism implementation."""
    print("🔬 Unit Test: Attention Mechanism...")
    
    # Test basic attention
    Q = np.random.randn(4, 6) * 0.1
    K = np.random.randn(4, 6) * 0.1  
    V = np.random.randn(4, 6) * 0.1
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (4, 6), "Attention should produce correct output shape"
    assert weights.shape == (4, 4), "Attention weights should be square matrix"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Attention weights should sum to 1"
    
    print("✅ Attention mechanism works correctly")

def test_self_attention_wrapper():
    """Test self-attention wrapper implementation."""
    print("🔬 Unit Test: Self-Attention Wrapper...")
    
    # Test self-attention
    self_attn = SelfAttention(d_model=32)
    x = np.random.randn(8, 32) * 0.1
    output, weights = self_attn(x)
    
    assert output.shape == x.shape, "Self-attention should preserve input shape"
    assert weights.shape == (8, 8), "Self-attention weights should be square"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Weights should sum to 1"
    
    print("✅ Self-attention wrapper works correctly")

def test_masking_utilities():
    """Test attention masking utilities."""
    print("🔬 Unit Test: Masking Utilities...")
    
    # Test causal mask
    causal_mask = create_causal_mask(4)
    assert np.allclose(causal_mask, np.tril(causal_mask)), "Causal mask should be lower triangular"
    
    # Test padding mask  
    padding_mask = create_padding_mask([3, 2], 4)
    assert padding_mask.shape == (2, 4, 4), "Padding mask should have correct shape"
    
    # Test bidirectional mask
    bidirectional_mask = create_bidirectional_mask(3)
    assert np.all(bidirectional_mask == 1), "Bidirectional mask should be all ones"
    
    print("✅ Masking utilities work correctly")

# %% [markdown]
"""
## 🧪 Module Testing

Time to test your implementation! This section uses TinyTorch's standardized testing framework to ensure your implementation works correctly.

**This testing section is locked** - it provides consistent feedback across all modules and cannot be modified.
"""

# %% nbgrader={"grade": false, "grade_id": "standardized-testing", "locked": true, "schema_version": 3, "solution": false, "task": false}
# =============================================================================
# STANDARDIZED MODULE TESTING - DO NOT MODIFY
# This cell is locked to ensure consistent testing across all TinyTorch modules
# =============================================================================

if __name__ == "__main__":
    from tito.tools.testing import run_module_tests_auto
    
    # Automatically discover and run all tests in this module
    success = run_module_tests_auto("Attention")

# %% [markdown]
"""
## 🎯 Module Summary

Congratulations! You've successfully implemented the revolutionary attention mechanism that powers all modern AI systems:

### What You've Accomplished
✅ **Scaled Dot-Product Attention**: Implemented the mathematical core of all transformer models  
✅ **Self-Attention Wrapper**: Built the mechanism that enables sequence understanding  
✅ **Attention Masking**: Created causal, padding, and bidirectional attention patterns  
✅ **Complete Integration**: Tested all components working together seamlessly  
✅ **Real Applications**: Applied attention to sequence processing and pattern matching

### Key Concepts You've Learned
- **Attention as dynamic pattern matching**: Query-Key-Value projections enable adaptive focus
- **Mathematical foundation**: Attention(Q,K,V) = softmax(QK^T/√d_k)V powers all modern AI
- **Global connectivity**: Unlike convolution, attention connects all positions directly
- **Interpretability**: Attention weights reveal what the model focuses on
- **Masking mechanisms**: Control information flow for different model architectures

### Mathematical Foundations
- **Attention formula**: The exact operation used in ChatGPT, BERT, GPT-4
- **Scaling factor**: √d_k prevents gradient vanishing in deep networks
- **Softmax normalization**: Converts similarity scores to probability distributions
- **Matrix operations**: Efficient parallel computation of all attention heads

### Real-World Applications
- **Language models**: ChatGPT, GPT-4, BERT use this exact mechanism
- **Machine translation**: Google Translate's transformer architecture
- **Computer vision**: Vision Transformers (ViTs) for image classification
- **Multimodal AI**: DALL-E, CLIP combining text and image understanding

### Attention vs. Convolution Insights
- **Receptive field**: Attention is global from layer 1, convolution is local
- **Computation**: Attention is O(n²), convolution is O(n) with kernel size
- **Weights**: Attention weights are dynamic and input-dependent
- **Best applications**: Attention excels at sequential/relational data

### Architecture Design Patterns
- **Self-attention**: Most common pattern where Q=K=V=input
- **Causal masking**: Enables autoregressive generation (GPT-style models)
- **Bidirectional**: Allows full context access (BERT-style models)
- **Padding masks**: Handle variable-length sequences efficiently

### Performance Characteristics
- **Quadratic scaling**: Memory and computation grow with sequence length squared
- **Parallelization**: All positions computed simultaneously (unlike RNNs)
- **Memory efficiency**: Attention weights require careful management
- **Gradient flow**: Direct connections enable training very deep networks

### Transformer Building Blocks
Your attention implementation is the foundation for:
- **Multi-head attention**: Multiple attention heads in parallel
- **Transformer blocks**: Attention + feedforward + residual connections
- **Positional encoding**: Adding sequence position information
- **Complete transformers**: Full encoder-decoder architectures

### Next Steps
1. **Export your code**: Use NBDev to export to the `tinytorch` package
2. **Test your implementation**: Run the complete test suite
3. **Build transformer architectures**: 
   ```python
   from tinytorch.core.attention import scaled_dot_product_attention, SelfAttention
   from tinytorch.core.attention import create_causal_mask, create_padding_mask
   
   # Create self-attention
   self_attn = SelfAttention(d_model=512)
   
   # Process sequence with causal masking (GPT-style)
   mask = create_causal_mask(seq_len)
   output, weights = self_attn(embeddings, mask)
   
   # Visualize attention patterns
   plt.imshow(weights, cmap='Blues')
   plt.title('Attention Patterns')
   ```
4. **Explore advanced transformers**: Multi-head attention, positional encoding, full transformer blocks!

### The Revolutionary Impact
You've implemented the mechanism that:
- **Revolutionized NLP**: Enabled ChatGPT, GPT-4, BERT breakthrough performance
- **Transformed computer vision**: Vision Transformers (ViTs) now compete with CNNs
- **Powers modern AI**: Almost every state-of-the-art model uses attention
- **Enables interpretability**: Attention weights show what AI models focus on

**Ready for the next challenge?** Let's build complete transformer architectures using your attention foundation!
"""
