# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#| default_exp core.attention
#| export

# %% [markdown]
"""
# Module 12: Attention - Learning to Focus

Welcome to Module 12! You're about to build the attention mechanism that revolutionized deep learning and powers GPT, BERT, and modern transformers.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor, activations, layers, losses, autograd, optimizers, training, dataloaders, spatial layers, tokenization, and embeddings
**You'll Build**: Scaled dot-product attention and multi-head attention mechanisms
**You'll Enable**: Transformer architectures, GPT-style language models, and sequence-to-sequence processing

**Connection Map**:
```
Embeddings → Attention → Transformers → Language Models
(representations) (focus mechanism) (complete architecture) (text generation)
```

## Learning Objectives
By the end of this module, you will:
1. Implement scaled dot-product attention with explicit O(n²) complexity
2. Build multi-head attention for parallel processing streams
3. Understand attention weight computation and interpretation
4. Experience attention's quadratic memory scaling firsthand
5. Test attention mechanisms with masking and sequence processing

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/12_attention/attention_dev.py
**Building Side:** Code exports to tinytorch.core.attention

```python
# Final package structure:
from tinytorch.core.attention import scaled_dot_product_attention, MultiHeadAttention  # This module
from tinytorch.core.tensor import Tensor  # Module 01 - foundation
from tinytorch.core.layers import Linear  # Module 03 - transformations
from tinytorch.text.embeddings import Embedding, PositionalEncoding  # Module 11 - representations
```

**Why this matters:**
- **Learning:** Complete attention system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn.functional and torch.nn with attention operations
- **Consistency:** All attention computations and multi-head mechanics in core.attention
- **Integration:** Works seamlessly with embeddings for complete sequence processing pipelines
"""

# %% nbgrader={"grade": false, "grade_id": "imports", "locked": false, "solution": true}
import numpy as np
import math
import time
import sys
import os
from typing import Optional, Tuple, List

# Import dependencies from other modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_layers'))
from layers_dev import Linear

# Note: Keeping simplified implementations for reference during development
class _SimplifiedTensor:
        """Simplified tensor for attention operations development."""

        def __init__(self, data, requires_grad=False):
            self.data = np.array(data, dtype=np.float32)
            self.shape = self.data.shape
            self.requires_grad = requires_grad
            self.grad = None

        def __repr__(self):
            return f"Tensor(shape={self.shape}, data=\n{self.data})"

        def __add__(self, other):
            if isinstance(other, Tensor):
                return Tensor(self.data + other.data)
            return Tensor(self.data + other)

        def __mul__(self, other):
            if isinstance(other, Tensor):
                return Tensor(self.data * other.data)
            return Tensor(self.data * other)

        def sum(self, axis=None):
            return Tensor(np.sum(self.data, axis=axis))

        def mean(self, axis=None):
            return Tensor(np.mean(self.data, axis=axis))

        def matmul(self, other):
            return Tensor(np.matmul(self.data, other.data))

        def softmax(self, axis=-1):
            """Apply softmax along specified axis."""
            # Subtract max for numerical stability
            shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
            exp_values = np.exp(shifted)
            return Tensor(exp_values / np.sum(exp_values, axis=axis, keepdims=True))

    # Simplified Linear layer for development
    class Linear:
        """Simplified linear layer for attention projections."""

        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
            # Initialize weights and bias (simplified Xavier initialization)
            self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
            self.bias = Tensor(np.zeros(out_features))

        def forward(self, x):
            """Forward pass: y = xW + b"""
            output = x.matmul(self.weight)
            # Add bias (broadcast across batch and sequence dimensions)
            return Tensor(output.data + self.bias.data)

        def parameters(self):
            """Return list of parameters for this layer."""
            return [self.weight, self.bias]

# %% [markdown]
"""
## Part 1: Introduction - What is Attention?

Attention is the mechanism that allows models to focus on relevant parts of the input when processing sequences. Think of it as a search engine inside your neural network - given a query, attention finds the most relevant keys and retrieves their associated values.

### The Attention Intuition

When you read "The cat sat on the ___", your brain automatically focuses on "cat" and "sat" to predict "mat". This selective focus is exactly what attention mechanisms provide to neural networks.

Imagine attention as a library research system:
- **Query (Q)**: "I need information about machine learning"
- **Keys (K)**: Index cards describing each book's content
- **Values (V)**: The actual books on the shelves
- **Attention Process**: Find books whose descriptions match your query, then retrieve those books

### Why Attention Changed Everything

Before attention, RNNs processed sequences step-by-step, creating an information bottleneck:

```
RNN Processing (Sequential):
Token 1 → Hidden → Token 2 → Hidden → ... → Final Hidden
         ↓              ↓                      ↓
    Limited Info   Compressed State    All Information Lost
```

Attention allows direct connections between any two positions:

```
Attention Processing (Parallel):
Token 1 ←─────────→ Token 2 ←─────────→ Token 3 ←─────────→ Token 4
   ↑                   ↑                   ↑                   ↑
   └─────────────── Direct Connections ──────────────────────┘
```

This enables:
- **Long-range dependencies**: Connecting words far apart
- **Parallel computation**: No sequential dependencies
- **Interpretable focus patterns**: We can see what the model attends to

### The Mathematical Foundation

Attention computes a weighted sum of values, where weights are determined by the similarity between queries and keys:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

This simple formula powers GPT, BERT, and virtually every modern language model.
"""

# %% [markdown]
"""
## Part 2: Foundations - Attention Mathematics

### The Three Components Visualized

Think of attention like a sophisticated address book lookup:

```
Query: "What information do I need?"
┌─────────────────────────────────────┐
│ Q: [0.1, 0.8, 0.3, 0.2]            │ ← Query vector (what we're looking for)
└─────────────────────────────────────┘

Keys: "What information is available at each position?"
┌─────────────────────────────────────┐
│ K₁: [0.2, 0.7, 0.1, 0.4]           │ ← Key 1 (description of position 1)
│ K₂: [0.1, 0.9, 0.2, 0.1]           │ ← Key 2 (description of position 2)
│ K₃: [0.3, 0.1, 0.8, 0.3]           │ ← Key 3 (description of position 3)
│ K₄: [0.4, 0.2, 0.1, 0.9]           │ ← Key 4 (description of position 4)
└─────────────────────────────────────┘

Values: "What actual content can I retrieve?"
┌─────────────────────────────────────┐
│ V₁: [content from position 1]       │ ← Value 1 (actual information)
│ V₂: [content from position 2]       │ ← Value 2 (actual information)
│ V₃: [content from position 3]       │ ← Value 3 (actual information)
│ V₄: [content from position 4]       │ ← Value 4 (actual information)
└─────────────────────────────────────┘
```

### The Attention Process Step by Step

```
Step 1: Compute Similarity Scores
Q · K₁ = 0.64    Q · K₂ = 0.81    Q · K₃ = 0.35    Q · K₄ = 0.42
  ↓               ↓               ↓               ↓
Raw similarity scores (higher = more relevant)

Step 2: Scale and Normalize
Scores / √d_k = [0.32, 0.41, 0.18, 0.21]  ← Scale for stability
     ↓
Softmax = [0.20, 0.45, 0.15, 0.20]        ← Convert to probabilities

Step 3: Weighted Combination
Output = 0.20×V₁ + 0.45×V₂ + 0.15×V₃ + 0.20×V₄
```

### Dimensions and Shapes

```
Input Shapes:
Q: (batch_size, seq_len, d_model)  ← Each position has a query
K: (batch_size, seq_len, d_model)  ← Each position has a key
V: (batch_size, seq_len, d_model)  ← Each position has a value

Intermediate Shapes:
QK^T: (batch_size, seq_len, seq_len)  ← Attention matrix (the O(n²) part!)
Weights: (batch_size, seq_len, seq_len)  ← After softmax
Output: (batch_size, seq_len, d_model)  ← Weighted combination of values
```

### Why O(n²) Complexity?

For sequence length n, we compute:
1. **QK^T**: n queries × n keys = n² similarity scores
2. **Softmax**: n² weights to normalize
3. **Weights×V**: n² weights × n values = n² operations for aggregation

This quadratic scaling is attention's blessing (global connectivity) and curse (memory/compute limits).

### The Attention Matrix Visualization

For a 4-token sequence "The cat sat down":

```
Attention Matrix (after softmax):
        The   cat   sat  down
The   [0.30  0.20  0.15  0.35]  ← "The" attends mostly to "down"
cat   [0.10  0.60  0.25  0.05]  ← "cat" focuses on itself and "sat"
sat   [0.05  0.40  0.50  0.05]  ← "sat" attends to "cat" and itself
down  [0.25  0.15  0.10  0.50]  ← "down" focuses on itself and "The"

Each row sums to 1.0 (probability distribution)
```
"""

# %% [markdown]
"""
## Part 3: Implementation - Building Scaled Dot-Product Attention

Now let's implement the core attention mechanism that powers all transformer models. We'll use explicit loops first to make the O(n²) complexity visible and educational.

### Understanding the Algorithm Visually

```
Step-by-Step Attention Computation:

1. Score Computation (Q @ K^T):
   For each query position i and key position j:
   score[i,j] = Σ(Q[i,d] × K[j,d]) for d in embedding_dims

   Query i    Key j      Dot Product
   [0.1,0.8] · [0.2,0.7] = 0.1×0.2 + 0.8×0.7 = 0.58

2. Scaling (÷ √d_k):
   scaled_scores = scores / √embedding_dim
   (Prevents softmax saturation for large dimensions)

3. Masking (optional):
   For causal attention: scores[i,j] = -∞ if j > i

   Causal Mask (lower triangular):
   [  OK  -∞  -∞  -∞ ]
   [  OK   OK  -∞  -∞ ]
   [  OK   OK   OK  -∞ ]
   [  OK   OK   OK   OK ]

4. Softmax (normalize each row):
   weights[i,j] = exp(scores[i,j]) / Σ(exp(scores[i,k])) for all k

5. Apply to Values:
   output[i] = Σ(weights[i,j] × V[j]) for all j
```
"""

# %% nbgrader={"grade": false, "grade_id": "attention-function", "locked": false, "solution": true}
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    This is the fundamental attention operation that powers all transformer models.
    We'll implement it with explicit loops first to show the O(n²) complexity.

    TODO: Implement scaled dot-product attention step by step

    APPROACH:
    1. Extract dimensions and validate inputs
    2. Compute attention scores with explicit nested loops (show O(n²) complexity)
    3. Scale by 1/√d_k for numerical stability
    4. Apply causal mask if provided (set masked positions to -inf)
    5. Apply softmax to get attention weights
    6. Apply values with attention weights (another O(n²) operation)
    7. Return output and attention weights

    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_model)
        K: Key tensor of shape (batch_size, seq_len, d_model)
        V: Value tensor of shape (batch_size, seq_len, d_model)
        mask: Optional causal mask, True=allow, False=mask (batch_size, seq_len, seq_len)

    Returns:
        output: Attended values (batch_size, seq_len, d_model)
        attention_weights: Attention matrix (batch_size, seq_len, seq_len)

    EXAMPLE:
    >>> Q = Tensor(np.random.randn(2, 4, 64))  # batch=2, seq=4, dim=64
    >>> K = Tensor(np.random.randn(2, 4, 64))
    >>> V = Tensor(np.random.randn(2, 4, 64))
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>> print(output.shape)  # (2, 4, 64)
    >>> print(weights.shape)  # (2, 4, 4)
    >>> print(weights.data[0].sum(axis=1))  # Each row sums to ~1.0

    HINTS:
    - Use explicit nested loops to compute Q[i] @ K[j] for educational purposes
    - Scale factor is 1/√d_k where d_k is the last dimension of Q
    - Masked positions should be set to -1e9 before softmax
    - Remember that softmax normalizes along the last dimension
    """
    ### BEGIN SOLUTION
    # Step 1: Extract dimensions and validate
    batch_size, seq_len, d_model = Q.shape
    assert K.shape == (batch_size, seq_len, d_model), f"K shape {K.shape} doesn't match Q shape {Q.shape}"
    assert V.shape == (batch_size, seq_len, d_model), f"V shape {V.shape} doesn't match Q shape {Q.shape}"

    # Step 2: Compute attention scores with explicit loops (educational O(n²) demonstration)
    scores = np.zeros((batch_size, seq_len, seq_len))

    # Show the quadratic complexity explicitly
    for b in range(batch_size):           # For each batch
        for i in range(seq_len):          # For each query position
            for j in range(seq_len):      # Attend to each key position
                # Compute dot product between query i and key j
                score = 0.0
                for d in range(d_model):  # Dot product across embedding dimension
                    score += Q.data[b, i, d] * K.data[b, j, d]
                scores[b, i, j] = score

    # Step 3: Scale by 1/√d_k for numerical stability
    scale_factor = 1.0 / math.sqrt(d_model)
    scores = scores * scale_factor

    # Step 4: Apply causal mask if provided
    if mask is not None:
        # mask[i,j] = False means position j should not attend to position i
        mask_value = -1e9  # Large negative value becomes 0 after softmax
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if not mask.data[b, i, j]:  # If mask is False, block attention
                        scores[b, i, j] = mask_value

    # Step 5: Apply softmax to get attention weights (probability distribution)
    attention_weights = np.zeros_like(scores)
    for b in range(batch_size):
        for i in range(seq_len):
            # Softmax over the j dimension (what this query attends to)
            row = scores[b, i, :]
            max_val = np.max(row)  # Numerical stability
            exp_row = np.exp(row - max_val)
            sum_exp = np.sum(exp_row)
            attention_weights[b, i, :] = exp_row / sum_exp

    # Step 6: Apply attention weights to values (another O(n²) operation)
    output = np.zeros((batch_size, seq_len, d_model))

    # Again, show the quadratic complexity
    for b in range(batch_size):           # For each batch
        for i in range(seq_len):          # For each output position
            for j in range(seq_len):      # Weighted sum over all value positions
                weight = attention_weights[b, i, j]
                for d in range(d_model):  # Accumulate across embedding dimension
                    output[b, i, d] += weight * V.data[b, j, d]

    return Tensor(output), Tensor(attention_weights)
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-attention-basic", "locked": true, "points": 10}
def test_unit_scaled_dot_product_attention():
    """🔬 Unit Test: Scaled Dot-Product Attention"""
    print("🔬 Unit Test: Scaled Dot-Product Attention...")

    # Test basic functionality
    batch_size, seq_len, d_model = 2, 4, 8
    Q = Tensor(np.random.randn(batch_size, seq_len, d_model))
    K = Tensor(np.random.randn(batch_size, seq_len, d_model))
    V = Tensor(np.random.randn(batch_size, seq_len, d_model))

    output, weights = scaled_dot_product_attention(Q, K, V)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape {output.shape} incorrect"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Weights shape {weights.shape} incorrect"

    # Check attention weights sum to 1 (probability distribution)
    weights_sum = weights.data.sum(axis=2)  # Sum over last dimension
    expected_sum = np.ones((batch_size, seq_len))
    assert np.allclose(weights_sum, expected_sum, atol=1e-6), "Attention weights don't sum to 1"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len)), k=0))  # Lower triangular
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)

    # Check that future positions have zero attention
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):  # Future positions
                assert abs(weights_masked.data[b, i, j]) < 1e-6, f"Future attention not masked at ({i},{j})"

    print("✅ scaled_dot_product_attention works correctly!")

test_unit_scaled_dot_product_attention()

# %% [markdown]
"""
### 🧪 Unit Test: Scaled Dot-Product Attention

This test validates our core attention mechanism:
- **Output shapes**: Ensures attention preserves sequence dimensions
- **Probability constraint**: Attention weights must sum to 1 per query
- **Causal masking**: Future positions should have zero attention weight

**Why attention weights sum to 1**: Each query position creates a probability distribution over all key positions. This ensures the output is a proper weighted average of values.

**Why causal masking matters**: In language modeling, positions shouldn't attend to future tokens (information they wouldn't have during generation).

**The O(n²) complexity you just witnessed**: Our explicit loops show exactly why attention scales quadratically - every query position must compare with every key position.
"""

# %% [markdown]
"""
## Part 4: Implementation - Multi-Head Attention

Multi-head attention runs multiple attention "heads" in parallel, each learning to focus on different types of relationships. Think of it as having multiple specialists: one for syntax, one for semantics, one for long-range dependencies, etc.

### Understanding Multi-Head Architecture

```
Single-Head vs Multi-Head Attention:

SINGLE HEAD (Limited):
Input → [Linear] → Q,K,V → [Attention] → Output
         512×512         512×512         512

MULTI-HEAD (Rich):
Input → [Linear] → Q₁,K₁,V₁ → [Attention₁] → Head₁ (64 dims)
     → [Linear] → Q₂,K₂,V₂ → [Attention₂] → Head₂ (64 dims)
     → [Linear] → Q₃,K₃,V₃ → [Attention₃] → Head₃ (64 dims)
     ...
     → [Linear] → Q₈,K₈,V₈ → [Attention₈] → Head₈ (64 dims)
                                              ↓
                                        [Concatenate]
                                              ↓
                                        [Linear Mix] → Output (512)
```

### The Multi-Head Process Detailed

```
Step 1: Project to Q, K, V
Input (512 dims) → Linear → Q, K, V (512 dims each)

Step 2: Split into Heads
Q (512) → Reshape → 8 heads × 64 dims per head
K (512) → Reshape → 8 heads × 64 dims per head
V (512) → Reshape → 8 heads × 64 dims per head

Step 3: Parallel Attention (for each of 8 heads)
Head 1: Q₁(64) attends to K₁(64) → weights₁ → output₁(64)
Head 2: Q₂(64) attends to K₂(64) → weights₂ → output₂(64)
...
Head 8: Q₈(64) attends to K₈(64) → weights₈ → output₈(64)

Step 4: Concatenate and Mix
[output₁ ∥ output₂ ∥ ... ∥ output₈] (512) → Linear → Final(512)
```

### Why Multiple Heads Are Powerful

Each head can specialize in different patterns:
- **Head 1**: Short-range syntax ("the cat" → subject-article relationship)
- **Head 2**: Long-range coreference ("John...he" → pronoun resolution)
- **Head 3**: Semantic similarity ("dog" ↔ "pet" connections)
- **Head 4**: Positional patterns (attending to specific distances)

This parallelization allows the model to attend to different representation subspaces simultaneously.
"""

# %% nbgrader={"grade": false, "grade_id": "multihead-attention", "locked": false, "solution": true}
class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel, each learning different relationships.
    This is the core component of transformer architectures.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize multi-head attention.

        TODO: Set up linear projections and validate configuration

        APPROACH:
        1. Validate that embed_dim is divisible by num_heads
        2. Calculate head_dim (embed_dim // num_heads)
        3. Create linear layers for Q, K, V projections
        4. Create output projection layer
        5. Store configuration parameters

        Args:
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of parallel attention heads

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> mha.head_dim  # 64 (512 / 8)
        >>> len(mha.parameters())  # 4 linear layers * 2 params each = 8 tensors

        HINTS:
        - head_dim = embed_dim // num_heads must be integer
        - Need 4 Linear layers: q_proj, k_proj, v_proj, out_proj
        - Each projection maps embed_dim → embed_dim
        """
        ### BEGIN SOLUTION
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        # Output projection to mix information across heads
        self.out_proj = Linear(embed_dim, embed_dim)
        ### END SOLUTION

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through multi-head attention.

        TODO: Implement the complete multi-head attention forward pass

        APPROACH:
        1. Extract input dimensions (batch_size, seq_len, embed_dim)
        2. Project input to Q, K, V using linear layers
        3. Reshape projections to separate heads: (batch, seq, heads, head_dim)
        4. Transpose to (batch, heads, seq, head_dim) for parallel processing
        5. Apply scaled dot-product attention to each head
        6. Transpose back and reshape to merge heads
        7. Apply output projection

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (batch_size, seq_len, seq_len)

        Returns:
            output: Attended representation (batch_size, seq_len, embed_dim)

        EXAMPLE:
        >>> mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 64))  # batch=2, seq=10, dim=64
        >>> output = mha.forward(x)
        >>> print(output.shape)  # (2, 10, 64) - same as input

        HINTS:
        - Reshape: (batch, seq, embed_dim) → (batch, seq, heads, head_dim)
        - Transpose: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
        - After attention: reverse the process to merge heads
        - Use scaled_dot_product_attention for each head
        """
        ### BEGIN SOLUTION
        # Step 1: Extract dimensions
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Input dim {embed_dim} doesn't match expected {self.embed_dim}"

        # Step 2: Project to Q, K, V
        Q = self.q_proj.forward(x)  # (batch, seq, embed_dim)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # Step 3: Reshape to separate heads
        # From (batch, seq, embed_dim) to (batch, seq, num_heads, head_dim)
        Q_heads = Q.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K_heads = K.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V_heads = V.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step 4: Transpose to (batch, num_heads, seq, head_dim) for parallel processing
        Q_heads = np.transpose(Q_heads, (0, 2, 1, 3))
        K_heads = np.transpose(K_heads, (0, 2, 1, 3))
        V_heads = np.transpose(V_heads, (0, 2, 1, 3))

        # Step 5: Apply attention to each head
        head_outputs = []
        for h in range(self.num_heads):
            # Extract this head's Q, K, V
            Q_h = Tensor(Q_heads[:, h, :, :])  # (batch, seq, head_dim)
            K_h = Tensor(K_heads[:, h, :, :])
            V_h = Tensor(V_heads[:, h, :, :])

            # Apply attention for this head
            head_out, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
            head_outputs.append(head_out.data)

        # Step 6: Concatenate heads back together
        # Stack: list of (batch, seq, head_dim) → (batch, num_heads, seq, head_dim)
        concat_heads = np.stack(head_outputs, axis=1)

        # Transpose back: (batch, num_heads, seq, head_dim) → (batch, seq, num_heads, head_dim)
        concat_heads = np.transpose(concat_heads, (0, 2, 1, 3))

        # Reshape: (batch, seq, num_heads, head_dim) → (batch, seq, embed_dim)
        concat_output = concat_heads.reshape(batch_size, seq_len, self.embed_dim)

        # Step 7: Apply output projection
        output = self.out_proj.forward(Tensor(concat_output))

        return output
        ### END SOLUTION

    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters.

        TODO: Collect parameters from all linear layers

        APPROACH:
        1. Get parameters from q_proj, k_proj, v_proj, out_proj
        2. Combine into single list

        Returns:
            List of all parameter tensors
        """
        ### BEGIN SOLUTION
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-multihead", "locked": true, "points": 15}
def test_unit_multihead_attention():
    """🔬 Unit Test: Multi-Head Attention"""
    print("🔬 Unit Test: Multi-Head Attention...")

    # Test initialization
    embed_dim, num_heads = 64, 8
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Check configuration
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads
    assert mha.head_dim == embed_dim // num_heads

    # Test parameter counting (4 linear layers, each has weight + bias)
    params = mha.parameters()
    assert len(params) == 8, f"Expected 8 parameters (4 layers × 2), got {len(params)}"

    # Test forward pass
    batch_size, seq_len = 2, 6
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

    output = mha.forward(x)

    # Check output shape preservation
    assert output.shape == (batch_size, seq_len, embed_dim), f"Output shape {output.shape} incorrect"

    # Test with causal mask
    mask = Tensor(np.tril(np.ones((batch_size, seq_len, seq_len))))
    output_masked = mha.forward(x, mask)
    assert output_masked.shape == (batch_size, seq_len, embed_dim)

    # Test different head configurations
    mha_small = MultiHeadAttention(embed_dim=32, num_heads=4)
    x_small = Tensor(np.random.randn(1, 5, 32))
    output_small = mha_small.forward(x_small)
    assert output_small.shape == (1, 5, 32)

    print("✅ MultiHeadAttention works correctly!")

test_unit_multihead_attention()

# %% [markdown]
"""
### 🧪 Unit Test: Multi-Head Attention

This test validates our multi-head attention implementation:
- **Configuration**: Correct head dimension calculation and parameter setup
- **Parameter counting**: 4 linear layers × 2 parameters each = 8 total
- **Shape preservation**: Output maintains input dimensions
- **Masking support**: Causal masks work correctly with multiple heads

**Why multi-head attention works**: Different heads can specialize in different types of relationships (syntactic, semantic, positional), providing richer representations than single-head attention.

**Architecture insight**: The split → attend → concat pattern allows parallel processing of different representation subspaces, dramatically increasing the model's capacity to understand complex relationships.
"""

# %% [markdown]
"""
## Part 5: Systems Analysis - Attention's Computational Reality

Now let's analyze the computational and memory characteristics that make attention both powerful and challenging at scale.

### Memory Complexity Visualization

```
Attention Memory Scaling (per layer):

Sequence Length = 128:
┌────────────────────────────────┐
│ Attention Matrix: 128×128      │ = 16K values
│ Memory: 64 KB (float32)        │
└────────────────────────────────┘

Sequence Length = 512:
┌────────────────────────────────┐
│ Attention Matrix: 512×512      │ = 262K values
│ Memory: 1 MB (float32)         │ ← 16× larger!
└────────────────────────────────┘

Sequence Length = 2048 (GPT-3):
┌────────────────────────────────┐
│ Attention Matrix: 2048×2048    │ = 4.2M values
│ Memory: 16 MB (float32)        │ ← 256× larger than 128!
└────────────────────────────────┘

For a 96-layer model (GPT-3):
Total Attention Memory = 96 layers × 16 MB = 1.5 GB
Just for attention matrices!
```
"""

# %% nbgrader={"grade": false, "grade_id": "attention-complexity", "locked": false, "solution": true}
def analyze_attention_complexity():
    """📊 Analyze attention computational complexity and memory scaling."""
    print("📊 Analyzing Attention Complexity...")

    # Test different sequence lengths to show O(n²) scaling
    embed_dim = 64
    sequence_lengths = [16, 32, 64, 128, 256]

    print("\nSequence Length vs Attention Matrix Size:")
    print("Seq Len | Attention Matrix | Memory (KB) | Complexity")
    print("-" * 55)

    for seq_len in sequence_lengths:
        # Calculate attention matrix size
        attention_matrix_size = seq_len * seq_len

        # Memory for attention weights (float32 = 4 bytes)
        attention_memory_kb = (attention_matrix_size * 4) / 1024

        # Total complexity (Q@K + softmax + weights@V)
        complexity = 2 * seq_len * seq_len * embed_dim + seq_len * seq_len

        print(f"{seq_len:7d} | {attention_matrix_size:14d} | {attention_memory_kb:10.2f} | {complexity:10.0f}")

    print(f"\n💡 Attention memory scales as O(n²) with sequence length")
    print(f"🚀 For seq_len=1024, attention matrix alone needs {(1024*1024*4)/1024/1024:.1f} MB")

# %% nbgrader={"grade": false, "grade_id": "attention-timing", "locked": false, "solution": true}
def analyze_attention_timing():
    """📊 Measure attention computation time vs sequence length."""
    print("\n📊 Analyzing Attention Timing...")

    embed_dim, num_heads = 64, 8
    sequence_lengths = [32, 64, 128, 256]

    print("\nSequence Length vs Computation Time:")
    print("Seq Len | Time (ms) | Ops/sec | Scaling")
    print("-" * 40)

    prev_time = None
    for seq_len in sequence_lengths:
        # Create test input
        x = Tensor(np.random.randn(1, seq_len, embed_dim))
        mha = MultiHeadAttention(embed_dim, num_heads)

        # Time multiple runs for stability
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = mha.forward(x)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        ops_per_sec = 1000 / avg_time if avg_time > 0 else 0

        # Calculate scaling factor vs previous
        scaling = avg_time / prev_time if prev_time else 1.0

        print(f"{seq_len:7d} | {avg_time:8.2f} | {ops_per_sec:7.0f} | {scaling:6.2f}x")
        prev_time = avg_time

    print(f"\n💡 Attention time scales roughly as O(n²) with sequence length")
    print(f"🚀 This is why efficient attention (FlashAttention) is crucial for long sequences")

# Call the analysis functions
analyze_attention_complexity()
analyze_attention_timing()

# %% [markdown]
"""
### 📊 Systems Analysis: The O(n²) Reality

Our analysis reveals the fundamental challenge that drives modern attention research:

**Memory Scaling Crisis:**
- Attention matrix grows as n² with sequence length
- For GPT-3 context (2048 tokens): 16MB just for attention weights per layer
- With 96 layers: 1.5GB just for attention matrices!
- This excludes activations, gradients, and other tensors

**Time Complexity Validation:**
- Each sequence length doubling roughly quadruples computation time
- This matches the theoretical O(n²) complexity we implemented with explicit loops
- Real bottleneck shifts from computation to memory at scale

**The Production Reality:**
```
Model Scale Impact:

Small Model (6 layers, 512 context):
Attention Memory = 6 × 1MB = 6MB ✅ Manageable

GPT-3 Scale (96 layers, 2048 context):
Attention Memory = 96 × 16MB = 1.5GB ⚠️ Significant

GPT-4 Scale (hypothetical: 120 layers, 32K context):
Attention Memory = 120 × 4GB = 480GB ❌ Impossible on single GPU!
```

**Why This Matters:**
- **FlashAttention**: Reformulates computation to reduce memory without changing results
- **Sparse Attention**: Only compute attention for specific patterns (local, strided)
- **Linear Attention**: Approximate attention with linear complexity
- **State Space Models**: Alternative architectures that avoid attention entirely

The quadratic wall is why long-context AI is an active research frontier, not a solved problem.
"""

# %% [markdown]
"""
## Part 6: Integration - Attention Patterns in Action

Let's test our complete attention system with realistic scenarios and visualize actual attention patterns.

### Understanding Attention Patterns

Real transformer models learn interpretable attention patterns:

```
Example Attention Patterns in Language:

1. Local Syntax Attention:
   "The quick brown fox"
   The → quick (determiner-adjective)
   quick → brown (adjective-adjective)
   brown → fox (adjective-noun)

2. Long-Range Coreference:
   "John went to the store. He bought milk."
   He → John (pronoun resolution across sentence boundary)

3. Compositional Structure:
   "The cat in the hat sat"
   sat → cat (verb attending to subject, skipping prepositional phrase)

4. Causal Dependencies:
   "I think therefore I"
   I → think (causal reasoning patterns)
   I → I (self-reference at end)
```

Let's see these patterns emerge in our implementation.
"""

# %% nbgrader={"grade": false, "grade_id": "attention-scenarios", "locked": false, "solution": true}
def test_attention_scenarios():
    """Test attention mechanisms in realistic scenarios."""
    print("🔬 Testing Attention Scenarios...")

    # Scenario 1: Small transformer block setup
    print("\n1. Small Transformer Setup:")
    embed_dim, num_heads, seq_len = 128, 8, 32

    # Create embeddings (simulating token embeddings + positional)
    embeddings = Tensor(np.random.randn(2, seq_len, embed_dim))

    # Multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)
    attended = mha.forward(embeddings)

    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {attended.shape}")
    print(f"   Parameters: {len(mha.parameters())} tensors")

    # Scenario 2: Causal language modeling
    print("\n2. Causal Language Modeling:")

    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    mask = Tensor(np.broadcast_to(causal_mask, (2, seq_len, seq_len)))

    # Apply causal attention
    causal_output = mha.forward(embeddings, mask)

    print(f"   Masked output shape: {causal_output.shape}")
    print(f"   Causal mask applied: {mask.shape}")

    # Scenario 3: Compare attention patterns
    print("\n3. Attention Pattern Analysis:")

    # Create simple test sequence
    simple_embed = Tensor(np.random.randn(1, 4, 16))
    simple_mha = MultiHeadAttention(16, 4)

    # Get attention weights by calling the base function
    Q = simple_mha.q_proj.forward(simple_embed)
    K = simple_mha.k_proj.forward(simple_embed)
    V = simple_mha.v_proj.forward(simple_embed)

    # Reshape for single head analysis
    Q_head = Tensor(Q.data[:, :, :4])  # First head only
    K_head = Tensor(K.data[:, :, :4])
    V_head = Tensor(V.data[:, :, :4])

    _, weights = scaled_dot_product_attention(Q_head, K_head, V_head)

    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Attention weights (first batch, 4x4 matrix):")
    weight_matrix = weights.data[0, :, :].round(3)

    # Format the attention matrix nicely
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{weight_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print(f"   Row sums: {weights.data[0].sum(axis=1).round(3)} (should be ~1.0)")

    # Scenario 4: Attention with masking visualization
    print("\n4. Causal Masking Effect:")

    # Apply causal mask to the simple example
    simple_mask = Tensor(np.tril(np.ones((1, 4, 4))))
    _, masked_weights = scaled_dot_product_attention(Q_head, K_head, V_head, simple_mask)

    print("   Causal attention matrix (lower triangular):")
    masked_matrix = masked_weights.data[0, :, :].round(3)
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{masked_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print("   Notice: Upper triangle is zero (can't attend to future)")

    print("\n✅ All attention scenarios work correctly!")

test_attention_scenarios()

# %% [markdown]
"""
### 🧪 Integration Test: Attention Scenarios

This comprehensive test validates attention in realistic use cases:

**Transformer Setup**: Standard configuration matching real architectures
- 128-dimensional embeddings with 8 attention heads
- 16 dimensions per head (128 ÷ 8 = 16)
- Proper parameter counting and shape preservation

**Causal Language Modeling**: Essential for GPT-style models
- Lower triangular mask ensures autoregressive property
- Position i cannot attend to positions j > i (future tokens)
- Critical for language generation and training stability

**Attention Pattern Visualization**: Understanding what the model "sees"
- Each row sums to 1.0 (valid probability distribution)
- Patterns reveal which positions the model finds relevant
- Causal masking creates structured sparsity in attention

**Real-World Implications**:
- These patterns are interpretable in trained models
- Attention heads often specialize (syntax, semantics, position)
- Visualization tools like BertViz use these matrices for model interpretation

The attention matrices you see here are the foundation of model interpretability in transformers.
"""

# %% [markdown]
"""
## 🧪 Module Integration Test

Final validation that everything works together correctly.
"""

# %% nbgrader={"grade": true, "grade_id": "module-test", "locked": true, "points": 20}
def test_module():
    """
    Comprehensive test of entire attention module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_scaled_dot_product_attention()
    test_unit_multihead_attention()

    print("\nRunning integration scenarios...")
    test_attention_scenarios()

    print("\nRunning performance analysis...")
    analyze_attention_complexity()

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 12")

# Call before module summary
test_module()

# %%
if __name__ == "__main__":
    print("🚀 Running Attention module...")
    test_module()
    print("✅ Module validation complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Attention Mechanics

### Question 1: Memory Scaling Impact
You implemented scaled dot-product attention with explicit O(n²) loops.
If you have a sequence of length 1024 with 8-byte float64 attention weights:
- How many MB does the attention matrix use? _____ MB
- For a 12-layer transformer, what's the total attention memory? _____ MB

### Question 2: Multi-Head Efficiency
Your MultiHeadAttention splits embed_dim=512 into num_heads=8.
- How many parameters does each head's Q/K/V projection have? _____ parameters
- What's the head_dim for each attention head? _____ dimensions
- Why is this more efficient than 8 separate attention mechanisms?

### Question 3: Computational Bottlenecks
From your timing analysis, attention time roughly quadruples when sequence length doubles.
- For seq_len=128, if attention takes 10ms, estimate time for seq_len=512: _____ ms
- Which operation dominates: QK^T computation or attention×V? _____
- Why does this scaling limit make long-context models challenging?

### Question 4: Causal Masking Design
Your causal mask prevents future positions from attending to past positions.
- In a 4-token sequence, how many attention connections are blocked? _____ connections
- Why is this essential for language modeling but not for BERT-style encoding?
- How would you modify the mask for local attention (only nearby positions)?

### Question 5: Attention Pattern Interpretation
Your attention visualization shows weight matrices where each row sums to 1.0.
- If position 2 has weights [0.1, 0.2, 0.5, 0.2], which position gets the most attention? _____
- What would uniform attention [0.25, 0.25, 0.25, 0.25] suggest about the model's focus?
- Why might some heads learn sparse attention patterns while others are more diffuse?
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Attention

Congratulations! You've built the attention mechanism that revolutionized deep learning!

### Key Accomplishments
- Built scaled dot-product attention with explicit O(n²) complexity demonstration
- Implemented multi-head attention for parallel relationship learning
- Experienced attention's quadratic memory scaling firsthand through analysis
- Tested causal masking for language modeling applications
- Visualized actual attention patterns and weight distributions
- All tests pass ✅ (validated by `test_module()`)

### Systems Insights Gained
- **Computational Complexity**: Witnessed O(n²) scaling in both memory and time through explicit loops
- **Memory Bottlenecks**: Attention matrices dominate memory usage in transformers (1.5GB+ for GPT-3 scale)
- **Parallel Processing**: Multi-head attention enables diverse relationship learning across representation subspaces
- **Production Challenges**: Understanding why FlashAttention and efficient attention research are crucial
- **Interpretability Foundation**: Attention matrices provide direct insight into model focus patterns

### Ready for Next Steps
Your attention implementation is the core mechanism that enables modern language models!
Export with: `tito module complete 12`

**Next**: Module 13 will combine attention with feed-forward layers to build complete transformer blocks, leading to GPT-style language models!

### What You Just Built Powers
- **GPT models**: Your attention mechanism is the exact pattern used in ChatGPT and GPT-4
- **BERT and variants**: Bidirectional attention for understanding tasks
- **Vision Transformers**: The same attention applied to image patches
- **Modern AI systems**: Nearly every state-of-the-art language and multimodal model

The mechanism you just implemented with explicit loops is mathematically identical to the attention in production language models - you've built the foundation of modern AI!
"""