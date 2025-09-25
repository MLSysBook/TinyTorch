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
# Transformers - Complete Transformer Architecture Implementation

Welcome to the Transformers module! You'll implement complete transformer blocks with LayerNorm, residual connections, and feed-forward networks, building the architecture that powers modern language models like GPT and BERT.

## Learning Goals
- Systems understanding: How transformer blocks scale memory and computation with model depth
- Core implementation skill: Build complete transformer architectures with proper normalization
- Pattern recognition: Understand how residual connections enable training of deep transformer models
- Framework connection: See how your implementations match production transformer systems
- Performance insight: Learn how transformer layer memory accumulation affects model deployment

## Build → Use → Reflect
1. **Build**: LayerNorm, transformer blocks, and complete transformer models
2. **Use**: Process sequences through multi-layer transformer architectures
3. **Reflect**: How do transformer design choices affect scalability and training dynamics?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how transformer blocks enable powerful sequence modeling
- Practical capability to implement complete transformer architectures with proper layer organization
- Systems insight into how transformer depth affects memory usage and training efficiency
- Performance consideration of how layer normalization and residual connections affect convergence
- Connection to production systems like GPT's transformer blocks and their optimization techniques

## Systems Reality Check
💡 **Production Context**: GPT-3 has 96 transformer layers, each with 12k-dimensional representations and complex memory management
⚡ **Performance Note**: Transformer layer memory accumulates linearly with depth - deep models require careful activation checkpointing
"""

# %% nbgrader={"grade": false, "grade_id": "transformers-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.transformers

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

# Try to import attention classes
try:
    from tinytorch.core.attention import ScaledDotProductAttention, MultiHeadAttention, KVCache
except ImportError:
    # For development, import from local module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '13_attention'))
    try:
        from attention_dev import ScaledDotProductAttention, MultiHeadAttention, KVCache
    except ImportError:
        # Create minimal mock classes if not available
        class MultiHeadAttention:
            def __init__(self, embed_dim, num_heads):
                self.embed_dim = embed_dim
                self.num_heads = num_heads
            def forward(self, q, k, v, mask=None, past_key_value=None, return_attention_weights=False):
                # Mock implementation - supports KV caching interface but doesn't use it
                if return_attention_weights:
                    fake_weights = q  # Mock attention weights
                    if past_key_value is not None:
                        return q, fake_weights, (k, v)  # Mock new key-value
                    else:
                        return q, fake_weights
                else:
                    if past_key_value is not None:
                        return q, (k, v)  # Mock new key-value
                    else:
                        return q
        class ScaledDotProductAttention:
            def __init__(self):
                pass
        class KVCache:
            def __init__(self, *args, **kwargs):
                pass

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

# %% nbgrader={"grade": false, "grade_id": "transformers-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🏗️ TinyTorch Transformers Module")
print(f"NumPy version: {np.__version__}")
print("Ready to build complete transformer architectures!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/14_transformers/transformers_dev.py`  
**Building Side:** Code exports to `tinytorch.core.transformers`

```python
# Final package structure:
from tinytorch.core.transformers import LayerNorm, TransformerBlock, Transformer
from tinytorch.core.attention import MultiHeadAttention  # Previous module
from tinytorch.core.embeddings import Embedding, PositionalEncoding  # Foundation
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's transformer implementations
- **Consistency:** All transformer components live together in `core.transformers`
- **Integration:** Works seamlessly with attention, embeddings, and tokenization systems
"""

# %% [markdown]
"""
## What are Transformers?

### The Architecture Revolution
Transformers revolutionized AI by replacing recurrent connections with attention mechanisms:

**Traditional RNN/LSTM:**
```
h₁ → h₂ → h₃ → h₄  (Sequential processing)
```

**Transformer:**
```
All positions attend to all positions simultaneously (Parallel processing)
```

### Transformer Block Components
Each transformer block contains:

1. **Multi-Head Self-Attention**: Captures sequence relationships
2. **Layer Normalization**: Stabilizes training of deep networks
3. **Residual Connections**: Enables gradient flow through many layers
4. **Position-wise Feed-Forward**: Applies non-linear transformations

### The Complete Architecture
```
Input Embeddings + Positional Encoding
    ↓
[Transformer Block] × N layers
    ↓
Output Layer (Language Modeling Head)
```

### Systems Trade-offs
- **Layer depth**: More layers = more capacity, more memory
- **Attention heads**: More heads = richer representations, more computation
- **Feed-forward size**: Larger FFN = more parameters, better performance
- **Layer normalization**: Pre-norm vs post-norm affects training dynamics
"""

# %% [markdown]
"""
## Layer Normalization Implementation

Layer normalization is crucial for training stable transformers. Unlike batch normalization, it normalizes across the feature dimension for each sample independently.
"""

# %% nbgrader={"grade": false, "grade_id": "layer-norm", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class LayerNorm:
    """
    Layer Normalization for transformers.
    
    Normalizes across the feature dimension (last axis) for each sample,
    making training more stable and enabling deeper networks.
    """
    
    def __init__(self, normalized_shape: Union[int, Tuple[int]], eps: float = 1e-5):
        """
        Initialize layer normalization with learnable parameters.
        
        TODO: Implement layer normalization initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store normalization configuration
        2. Initialize learnable scale (gamma) and shift (beta) parameters
        3. Set epsilon for numerical stability
        4. Set up parameter tracking for optimization
        
        MATHEMATICAL FOUNDATION:
        LayerNorm(x) = γ * (x - μ) / σ + β
        
        Where:
        - μ = mean across feature dimensions
        - σ = std across feature dimensions  
        - γ = learnable scale parameter
        - β = learnable shift parameter
        
        Args:
            normalized_shape: Shape of features to normalize (e.g., embedding_dim)
            eps: Small value for numerical stability
        """
        ### BEGIN SOLUTION
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape
        
        self.eps = eps
        
        # Initialize learnable parameters
        # Gamma (scale): initialized to ones
        # Beta (bias): initialized to zeros
        self.gamma = Tensor(np.ones(self.normalized_shape))
        self.beta = Tensor(np.zeros(self.normalized_shape))
        
        # Track parameters for optimization
        self.parameters = [self.gamma, self.beta]
        ### END SOLUTION
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to input tensor.
        
        TODO: Implement layer normalization forward pass.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Calculate mean across feature dimensions
        2. Calculate standard deviation across feature dimensions
        3. Normalize: (x - mean) / (std + eps)
        4. Apply learnable scale and shift: gamma * normalized + beta
        
        NUMERICAL STABILITY:
        - Add eps to variance before taking sqrt
        - Use unbiased variance calculation
        
        EXAMPLE:
        layer_norm = LayerNorm(256)
        x = Tensor(np.random.randn(32, 128, 256))  # (batch, seq, features)
        normalized = layer_norm.forward(x)  # Same shape as input
        
        Args:
            x: Input tensor with shape (..., *normalized_shape)
            
        Returns:
            Normalized tensor with same shape as input
        """
        ### BEGIN SOLUTION
        # Calculate mean and variance across the feature dimensions (last axes)
        # For shape (..., *normalized_shape), we want to normalize over the last len(normalized_shape) axes
        
        # Determine axes to normalize over
        axes_to_normalize = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        
        # Calculate mean
        mean = np.mean(x.data, axis=axes_to_normalize, keepdims=True)
        
        # Calculate variance
        variance = np.var(x.data, axis=axes_to_normalize, keepdims=True)
        
        # Normalize
        normalized = (x.data - mean) / np.sqrt(variance + self.eps)
        
        # Apply learnable scale and shift
        # Reshape gamma and beta to be broadcastable
        gamma_broadcasted = self.gamma.data.reshape([1] * (len(x.shape) - len(self.normalized_shape)) + list(self.normalized_shape))
        beta_broadcasted = self.beta.data.reshape([1] * (len(x.shape) - len(self.normalized_shape)) + list(self.normalized_shape))
        
        output = gamma_broadcasted * normalized + beta_broadcasted
        
        return Tensor(output)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable."""
        return self.forward(x)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of layer normalization parameters.
        
        This function is PROVIDED to show memory analysis.
        """
        # Parameter memory
        param_memory_mb = sum(param.data.nbytes for param in self.parameters) / (1024 * 1024)
        
        return {
            'parameter_memory_mb': param_memory_mb,
            'total_parameters': sum(param.data.size for param in self.parameters),
            'normalized_shape': self.normalized_shape
        }

# %% [markdown]
"""
### 🧪 Test Your Layer Normalization Implementation

Once you implement the LayerNorm methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-layer-norm-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_layer_norm():
    """Unit test for layer normalization."""
    print("🔬 Unit Test: Layer Normalization...")
    
    # Test 1: Basic functionality
    embed_dim = 256
    layer_norm = LayerNorm(embed_dim)
    
    # Verify initialization
    assert layer_norm.normalized_shape == (embed_dim,), "Should store normalized shape"
    assert len(layer_norm.parameters) == 2, "Should have gamma and beta parameters"
    assert layer_norm.gamma.shape == (embed_dim,), "Gamma should match normalized shape"
    assert layer_norm.beta.shape == (embed_dim,), "Beta should match normalized shape"
    
    # Verify parameter initialization
    assert np.allclose(layer_norm.gamma.data, 1.0), "Gamma should be initialized to ones"
    assert np.allclose(layer_norm.beta.data, 0.0), "Beta should be initialized to zeros"
    
    # Test 2: Forward pass with 2D input
    batch_size = 16
    x_2d = Tensor(np.random.randn(batch_size, embed_dim))
    output_2d = layer_norm.forward(x_2d)
    
    assert output_2d.shape == x_2d.shape, "Output shape should match input shape"
    
    # Test 3: Forward pass with 3D input (typical transformer use)
    seq_length = 32
    x_3d = Tensor(np.random.randn(batch_size, seq_length, embed_dim))
    output_3d = layer_norm.forward(x_3d)
    
    assert output_3d.shape == x_3d.shape, "3D output shape should match input shape"
    
    # Test 4: Normalization properties
    # For each sample, the normalized features should have ~zero mean and ~unit variance
    for i in range(batch_size):
        for j in range(seq_length):
            sample_output = output_3d.data[i, j, :]
            sample_mean = np.mean(sample_output)
            sample_var = np.var(sample_output)
            
            assert abs(sample_mean) < 1e-6, f"Normalized mean should be ~0, got {sample_mean}"
            assert abs(sample_var - 1.0) < 1e-6, f"Normalized variance should be ~1, got {sample_var}"
    
    # Test 5: Different normalized shapes
    multi_dim_shape = (64, 4)  # Multi-dimensional normalization
    layer_norm_multi = LayerNorm(multi_dim_shape)
    
    x_multi = Tensor(np.random.randn(8, 32, 64, 4))
    output_multi = layer_norm_multi.forward(x_multi)
    
    assert output_multi.shape == x_multi.shape, "Multi-dim normalization should preserve shape"
    
    # Test 6: Callable interface
    output_callable = layer_norm(x_3d)
    assert np.allclose(output_callable.data, output_3d.data), "Callable interface should work"
    
    # Test 7: Numerical stability with extreme values
    extreme_x = Tensor(np.ones((4, embed_dim)) * 1e6)  # Very large values
    extreme_output = layer_norm.forward(extreme_x)
    
    assert not np.any(np.isnan(extreme_output.data)), "Should handle extreme values without NaN"
    assert not np.any(np.isinf(extreme_output.data)), "Should handle extreme values without inf"
    
    # Test 8: Memory usage calculation
    memory_stats = layer_norm.get_memory_usage()
    assert 'parameter_memory_mb' in memory_stats, "Should provide memory statistics"
    assert memory_stats['total_parameters'] == 2 * embed_dim, "Should count gamma and beta parameters"
    
    print("✅ Layer normalization tests passed!")
    print(f"✅ Properly normalizes across feature dimensions")
    print(f"✅ Handles 2D and 3D inputs correctly")
    print(f"✅ Maintains ~0 mean and ~1 variance after normalization")
    print(f"✅ Parameter memory: {memory_stats['parameter_memory_mb']:.4f}MB")

# Test function defined (called in main block)

# %% [markdown]
"""
## Position-wise Feed-Forward Network

Each transformer block contains a position-wise feed-forward network that applies the same transformation to each position independently.
"""

# %% nbgrader={"grade": false, "grade_id": "feed-forward", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class PositionwiseFeedForward:
    """
    Position-wise feed-forward network used in transformer blocks.
    
    Applies the same feed-forward network to each position in the sequence:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        """
        Initialize position-wise feed-forward network.
        
        TODO: Implement feed-forward network initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store network configuration
        2. Initialize weight matrices and bias vectors for two linear layers
        3. Set up parameter tracking for optimization
        4. Store dropout rate for training
        
        ARCHITECTURE:
        - Input: (batch, seq_len, embed_dim)
        - Linear 1: embed_dim → hidden_dim
        - ReLU activation
        - Linear 2: hidden_dim → embed_dim
        - Output: (batch, seq_len, embed_dim)
        
        PARAMETER INITIALIZATION:
        Use Xavier/Glorot initialization for stable training
        
        Args:
            embed_dim: Embedding dimension (input and output size)
            hidden_dim: Hidden layer dimension (typically 4 * embed_dim)
            dropout: Dropout rate for regularization
        """
        ### BEGIN SOLUTION
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Initialize weights using Xavier initialization
        # W1: embed_dim → hidden_dim
        xavier_bound_1 = math.sqrt(6.0 / (embed_dim + hidden_dim))
        self.w1 = Tensor(np.random.uniform(-xavier_bound_1, xavier_bound_1, (embed_dim, hidden_dim)))
        self.b1 = Tensor(np.zeros(hidden_dim))
        
        # W2: hidden_dim → embed_dim
        xavier_bound_2 = math.sqrt(6.0 / (hidden_dim + embed_dim))
        self.w2 = Tensor(np.random.uniform(-xavier_bound_2, xavier_bound_2, (hidden_dim, embed_dim)))
        self.b2 = Tensor(np.zeros(embed_dim))
        
        # Track parameters for optimization
        self.parameters = [self.w1, self.b1, self.w2, self.b2]
        ### END SOLUTION
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply position-wise feed-forward transformation.
        
        TODO: Implement feed-forward forward pass.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Apply first linear transformation: x @ W1 + b1
        2. Apply ReLU activation: max(0, linear1)
        3. Apply second linear transformation: relu @ W2 + b2
        4. Return result with same shape as input
        
        MATHEMATICAL FORMULATION:
        hidden = ReLU(x @ W1 + b1)
        output = hidden @ W2 + b2
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor with shape (batch_size, seq_len, embed_dim)
        """
        ### BEGIN SOLUTION
        # Reshape input for matrix multiplication if needed
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_len, embed_dim = x.shape
            # Reshape to (batch_size * seq_len, embed_dim) for efficient computation
            x_reshaped = x.data.reshape(-1, embed_dim)
        else:
            x_reshaped = x.data
        
        # First linear transformation: x @ W1 + b1
        hidden = np.matmul(x_reshaped, self.w1.data) + self.b1.data
        
        # ReLU activation
        hidden_relu = np.maximum(0, hidden)
        
        # Second linear transformation: hidden @ W2 + b2
        output = np.matmul(hidden_relu, self.w2.data) + self.b2.data
        
        # Reshape back to original shape
        if len(original_shape) == 3:
            output = output.reshape(original_shape)
        
        return Tensor(output)
        ### END SOLUTION
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the class callable."""
        return self.forward(x)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of feed-forward parameters.
        
        This function is PROVIDED to show memory analysis.
        """
        # Parameter memory
        param_memory_mb = sum(param.data.nbytes for param in self.parameters) / (1024 * 1024)
        
        # Calculate parameter counts
        w1_params = self.embed_dim * self.hidden_dim
        w2_params = self.hidden_dim * self.embed_dim
        bias_params = self.hidden_dim + self.embed_dim
        total_params = w1_params + w2_params + bias_params
        
        return {
            'parameter_memory_mb': param_memory_mb,
            'total_parameters': total_params,
            'w1_parameters': w1_params,
            'w2_parameters': w2_params,
            'bias_parameters': bias_params,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim
        }

# %% [markdown]
"""
### 🧪 Test Your Feed-Forward Network Implementation

Once you implement the PositionwiseFeedForward methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-feed-forward-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_feed_forward():
    """Unit test for position-wise feed-forward network."""
    print("🔬 Unit Test: Position-wise Feed-Forward Network...")
    
    # Test configuration
    embed_dim = 256
    hidden_dim = 1024  # Typical 4x expansion
    ffn = PositionwiseFeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim)
    
    # Verify initialization
    assert ffn.embed_dim == embed_dim, "Should store embedding dimension"
    assert ffn.hidden_dim == hidden_dim, "Should store hidden dimension"
    assert len(ffn.parameters) == 4, "Should have W1, b1, W2, b2 parameters"
    
    # Verify parameter shapes
    assert ffn.w1.shape == (embed_dim, hidden_dim), f"W1 should be ({embed_dim}, {hidden_dim})"
    assert ffn.b1.shape == (hidden_dim,), f"b1 should be ({hidden_dim},)"
    assert ffn.w2.shape == (hidden_dim, embed_dim), f"W2 should be ({hidden_dim}, {embed_dim})"
    assert ffn.b2.shape == (embed_dim,), f"b2 should be ({embed_dim},)"
    
    # Test forward pass with 3D input (typical transformer use)
    batch_size = 8
    seq_len = 32
    x_3d = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    output_3d = ffn.forward(x_3d)
    
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output_3d.shape == expected_shape, f"Expected shape {expected_shape}, got {output_3d.shape}"
    
    # Test forward pass with 2D input
    x_2d = Tensor(np.random.randn(batch_size, embed_dim))
    output_2d = ffn.forward(x_2d)
    
    expected_2d_shape = (batch_size, embed_dim)
    assert output_2d.shape == expected_2d_shape, f"Expected 2D shape {expected_2d_shape}, got {output_2d.shape}"
    
    # Test that FFN is applied position-wise (same transformation at each position)
    # Extract two positions from the sequence
    pos_1_input = Tensor(x_3d.data[:, 0, :])  # First position
    pos_2_input = Tensor(x_3d.data[:, 1, :])  # Second position
    
    pos_1_output = ffn.forward(pos_1_input)
    pos_2_output = ffn.forward(pos_2_input)
    
    # Compare with full sequence output
    assert np.allclose(pos_1_output.data, output_3d.data[:, 0, :]), "Position 0 should match individual processing"
    assert np.allclose(pos_2_output.data, output_3d.data[:, 1, :]), "Position 1 should match individual processing"
    
    # Test ReLU activation (some outputs should be zero for negative intermediate values)
    # Create input that will definitely produce some negative values after first linear layer
    negative_input = Tensor(-np.ones((4, embed_dim)) * 10)  # Very negative input
    negative_output = ffn.forward(negative_input)
    
    # Not all outputs should be negative (ReLU should clip some values)
    assert not np.all(negative_output.data < 0), "ReLU should prevent all outputs from being negative"
    
    # Test callable interface
    output_callable = ffn(x_3d)
    assert np.allclose(output_callable.data, output_3d.data), "Callable interface should work"
    
    # Test different hidden dimensions
    for test_hidden_dim in [512, 2048]:
        test_ffn = PositionwiseFeedForward(embed_dim=embed_dim, hidden_dim=test_hidden_dim)
        test_output = test_ffn.forward(x_3d)
        assert test_output.shape == expected_shape, f"Should work with hidden_dim={test_hidden_dim}"
    
    # Test memory usage calculation
    memory_stats = ffn.get_memory_usage()
    assert 'parameter_memory_mb' in memory_stats, "Should provide memory statistics"
    
    # Verify parameter counts
    expected_w1_params = embed_dim * hidden_dim
    expected_w2_params = hidden_dim * embed_dim
    expected_total = expected_w1_params + expected_w2_params + hidden_dim + embed_dim
    
    assert memory_stats['w1_parameters'] == expected_w1_params, "Should count W1 parameters correctly"
    assert memory_stats['w2_parameters'] == expected_w2_params, "Should count W2 parameters correctly"
    assert memory_stats['total_parameters'] == expected_total, "Should count total parameters correctly"
    
    print("✅ Position-wise feed-forward tests passed!")
    print(f"✅ Handles 2D and 3D inputs correctly")
    print(f"✅ Position-wise processing verified")
    print(f"✅ ReLU activation working properly")
    print(f"✅ Total parameters: {memory_stats['total_parameters']:,}")
    print(f"✅ Parameter memory: {memory_stats['parameter_memory_mb']:.2f}MB")

# Test function defined (called in main block)

# %% [markdown]
"""
## Transformer Block Implementation

Now let's build the complete transformer block that combines multi-head attention, layer normalization, and position-wise feed-forward networks with residual connections.
"""

# %% nbgrader={"grade": false, "grade_id": "transformer-block", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class TransformerBlock:
    """
    Complete transformer block with self-attention and feed-forward layers.
    
    Combines multi-head self-attention, layer normalization, residual connections,
    and position-wise feed-forward networks into the standard transformer architecture.
    
    SUPPORTS KV CACHING (Module 19 integration):
    - Forward method accepts optional past_key_value parameter for caching
    - Returns new key-value pairs when caching is enabled
    - Backward compatible: works with or without caching
    """
    
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, 
                 dropout: float = 0.0, pre_norm: bool = True):
        """
        Initialize transformer block with all components.
        
        TODO: Implement transformer block initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store block configuration
        2. Create multi-head attention layer
        3. Create two layer normalization layers (for attention and FFN)
        4. Create position-wise feed-forward network
        5. Set up parameter tracking from all sub-components
        
        ARCHITECTURE CHOICE: Pre-norm vs Post-norm
        - Pre-norm: LayerNorm → Attention → Residual (more stable)
        - Post-norm: Attention → LayerNorm → Residual (original paper)
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            hidden_dim: Feed-forward hidden dimension (typically 4 * embed_dim)
            dropout: Dropout rate for regularization
            pre_norm: Whether to use pre-normalization (recommended)
        """
        ### BEGIN SOLUTION
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pre_norm = pre_norm
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        # Layer normalization layers
        self.norm1 = LayerNorm(embed_dim)  # For attention
        self.norm2 = LayerNorm(embed_dim)  # For feed-forward
        
        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        
        # Collect all parameters from sub-components
        self.parameters = []
        if hasattr(self.attention, 'parameters'):
            self.parameters.extend(self.attention.parameters)
        self.parameters.extend(self.norm1.parameters)
        self.parameters.extend(self.norm2.parameters)
        self.parameters.extend(self.ffn.parameters)
        ### END SOLUTION
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                return_attention_weights: bool = False, past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]]:
        """
        Process input through complete transformer block.
        
        TODO: Implement transformer block forward pass.
        
        STEP-BY-STEP IMPLEMENTATION (Pre-norm):
        1. Self-attention with residual: x + attention(norm1(x))
        2. Feed-forward with residual: attn_out + ffn(norm2(attn_out))
        3. Return final output (and optionally attention weights)
        
        RESIDUAL CONNECTIONS:
        Essential for training deep networks - allow gradients to flow directly
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            past_key_value: Optional cached key-value pair from previous forward pass
            
        Returns:
            Transformer block output with same shape as input
            Optionally also attention weights
            Optionally also new key-value pair for caching (if past_key_value provided)
        """
        ### BEGIN SOLUTION
        if self.pre_norm:
            # Pre-normalization: LayerNorm before attention/FFN
            
            # Self-attention with residual connection
            norm1_x = self.norm1(x)
            
            # Handle KV caching - try to pass past_key_value to attention if supported
            if past_key_value is not None:
                # Try to use KV caching - gracefully fall back if not supported
                try:
                    if return_attention_weights:
                        attn_result = self.attention.forward(
                            norm1_x, norm1_x, norm1_x, mask=mask, return_attention_weights=True, past_key_value=past_key_value
                        )
                        if len(attn_result) == 3:
                            # attention returned (output, weights, new_key_value)
                            attn_output, attn_weights, new_key_value = attn_result
                        else:
                            # fallback: attention doesn't support caching yet
                            attn_output, attn_weights = attn_result
                            new_key_value = None
                    else:
                        attn_result = self.attention.forward(norm1_x, norm1_x, norm1_x, mask=mask, past_key_value=past_key_value)
                        if isinstance(attn_result, tuple) and len(attn_result) == 2:
                            # attention returned (output, new_key_value)
                            attn_output, new_key_value = attn_result
                        else:
                            # fallback: attention doesn't support caching yet
                            attn_output = attn_result
                            new_key_value = None
                except TypeError:
                    # Attention layer doesn't support past_key_value yet - fall back to standard behavior
                    if return_attention_weights:
                        attn_output, attn_weights = self.attention.forward(
                            norm1_x, norm1_x, norm1_x, mask=mask, return_attention_weights=True
                        )
                    else:
                        attn_output = self.attention.forward(norm1_x, norm1_x, norm1_x, mask=mask)
                    new_key_value = None
            else:
                # Standard behavior (no caching)
                if return_attention_weights:
                    attn_output, attn_weights = self.attention.forward(
                        norm1_x, norm1_x, norm1_x, mask=mask, return_attention_weights=True
                    )
                else:
                    attn_output = self.attention.forward(norm1_x, norm1_x, norm1_x, mask=mask)
                new_key_value = None
            
            # Residual connection
            x = Tensor(x.data + attn_output.data)
            
            # Feed-forward with residual connection
            norm2_x = self.norm2(x)
            ffn_output = self.ffn.forward(norm2_x)
            
            # Residual connection
            output = Tensor(x.data + ffn_output.data)
            
        else:
            # Post-normalization: LayerNorm after attention/FFN (original transformer)
            
            # Self-attention with residual connection
            # Handle KV caching - try to pass past_key_value to attention if supported
            if past_key_value is not None:
                # Try to use KV caching - gracefully fall back if not supported
                try:
                    if return_attention_weights:
                        attn_result = self.attention.forward(
                            x, x, x, mask=mask, return_attention_weights=True, past_key_value=past_key_value
                        )
                        if len(attn_result) == 3:
                            # attention returned (output, weights, new_key_value)
                            attn_output, attn_weights, new_key_value = attn_result
                        else:
                            # fallback: attention doesn't support caching yet
                            attn_output, attn_weights = attn_result
                            new_key_value = None
                    else:
                        attn_result = self.attention.forward(x, x, x, mask=mask, past_key_value=past_key_value)
                        if isinstance(attn_result, tuple) and len(attn_result) == 2:
                            # attention returned (output, new_key_value)
                            attn_output, new_key_value = attn_result
                        else:
                            # fallback: attention doesn't support caching yet
                            attn_output = attn_result
                            new_key_value = None
                except TypeError:
                    # Attention layer doesn't support past_key_value yet - fall back to standard behavior
                    if return_attention_weights:
                        attn_output, attn_weights = self.attention.forward(
                            x, x, x, mask=mask, return_attention_weights=True
                        )
                    else:
                        attn_output = self.attention.forward(x, x, x, mask=mask)
                    new_key_value = None
            else:
                # Standard behavior (no caching)
                if return_attention_weights:
                    attn_output, attn_weights = self.attention.forward(
                        x, x, x, mask=mask, return_attention_weights=True
                    )
                else:
                    attn_output = self.attention.forward(x, x, x, mask=mask)
                new_key_value = None
            
            # Residual + LayerNorm
            attn_residual = Tensor(x.data + attn_output.data)
            norm1_output = self.norm1(attn_residual)
            
            # Feed-forward with residual connection
            ffn_output = self.ffn.forward(norm1_output)
            
            # Residual + LayerNorm
            ffn_residual = Tensor(norm1_output.data + ffn_output.data)
            output = self.norm2(ffn_residual)
        
        # Return appropriate tuple based on what was requested
        if past_key_value is not None:
            # KV caching is enabled
            if return_attention_weights:
                return output, attn_weights, new_key_value
            else:
                return output, new_key_value
        else:
            # Standard behavior (backward compatible)
            if return_attention_weights:
                return output, attn_weights
            else:
                return output
        ### END SOLUTION
    
    def __call__(self, x: Tensor, mask: Optional[Tensor] = None,
                 return_attention_weights: bool = False, past_key_value: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]]:
        """Make the class callable."""
        return self.forward(x, mask, return_attention_weights, past_key_value)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of transformer block components.
        
        This function is PROVIDED to show memory analysis.
        """
        # Get memory usage from components
        if hasattr(self.attention, 'get_memory_usage'):
            attention_memory = self.attention.get_memory_usage()['total_parameter_memory_mb']
        else:
            attention_memory = 0.0
        
        norm1_memory = self.norm1.get_memory_usage()['parameter_memory_mb']
        norm2_memory = self.norm2.get_memory_usage()['parameter_memory_mb']
        ffn_memory = self.ffn.get_memory_usage()['parameter_memory_mb']
        
        total_memory = attention_memory + norm1_memory + norm2_memory + ffn_memory
        total_params = len(self.parameters) if hasattr(self, 'parameters') else 0
        
        return {
            'total_memory_mb': total_memory,
            'attention_memory_mb': attention_memory,
            'norm_memory_mb': norm1_memory + norm2_memory,
            'ffn_memory_mb': ffn_memory,
            'total_parameters': sum(p.data.size for p in self.parameters) if hasattr(self, 'parameters') else 0,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'hidden_dim': self.hidden_dim,
            'pre_norm': self.pre_norm
        }

# %% [markdown]
"""
### 🧪 Test Your Transformer Block Implementation

Once you implement the TransformerBlock methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-transformer-block-immediate", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def test_unit_transformer_block():
    """Unit test for transformer block."""
    print("🔬 Unit Test: Transformer Block...")
    
    # Test configuration
    embed_dim = 256
    num_heads = 8
    hidden_dim = 1024
    transformer_block = TransformerBlock(
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        hidden_dim=hidden_dim,
        pre_norm=True
    )
    
    # Verify initialization
    assert transformer_block.embed_dim == embed_dim, "Should store embedding dimension"
    assert transformer_block.num_heads == num_heads, "Should store number of heads"
    assert transformer_block.hidden_dim == hidden_dim, "Should store hidden dimension"
    assert transformer_block.pre_norm == True, "Should store normalization type"
    
    # Verify components exist
    assert hasattr(transformer_block, 'attention'), "Should have attention layer"
    assert hasattr(transformer_block, 'norm1'), "Should have first norm layer"
    assert hasattr(transformer_block, 'norm2'), "Should have second norm layer"
    assert hasattr(transformer_block, 'ffn'), "Should have feed-forward network"
    
    # Test forward pass
    batch_size = 4
    seq_len = 16
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
    
    output = transformer_block.forward(x)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test with attention weights return
    output_with_attn, attn_weights = transformer_block.forward(x, return_attention_weights=True)
    
    assert output_with_attn.shape == expected_shape, "Output with attention should have correct shape"
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Expected attention shape {expected_attn_shape}, got {attn_weights.shape}"
    
    # Test with causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    causal_mask = 1 - causal_mask  # Convert to attention mask
    
    masked_output, masked_attn = transformer_block.forward(
        x, mask=Tensor(causal_mask), return_attention_weights=True
    )
    
    assert masked_output.shape == expected_shape, "Masked output should have correct shape"
    
    # Verify causal masking works
    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                assert np.all(masked_attn.data[:, head, i, j] < 1e-5), \
                    f"Position ({i},{j}) should be masked in head {head}"
    
    # Test residual connections by checking that output is different from pure attention
    # If we zero out the input, residual connections should preserve some information
    zero_input = Tensor(np.zeros((batch_size, seq_len, embed_dim)))
    zero_output = transformer_block.forward(zero_input)
    
    # Output should not be exactly zero due to biases and layer norm parameters
    assert not np.allclose(zero_output.data, 0), "Residual connections should prevent zero output"
    
    # Test post-normalization variant
    post_norm_block = TransformerBlock(
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        hidden_dim=hidden_dim,
        pre_norm=False
    )
    
    post_norm_output = post_norm_block.forward(x)
    assert post_norm_output.shape == expected_shape, "Post-norm should produce correct shape"
    
    # Pre-norm and post-norm should produce different outputs
    pre_norm_output = transformer_block.forward(x)
    assert not np.allclose(pre_norm_output.data, post_norm_output.data), \
        "Pre-norm and post-norm should produce different outputs"
    
    # Test callable interface
    output_callable = transformer_block(x)
    assert np.allclose(output_callable.data, output.data), "Callable interface should work"
    
    # Test different configurations
    for test_heads in [4, 16]:
        if embed_dim % test_heads == 0:
            test_block = TransformerBlock(embed_dim=embed_dim, num_heads=test_heads, hidden_dim=hidden_dim)
            test_output = test_block.forward(x)
            assert test_output.shape == expected_shape, f"Should work with {test_heads} heads"
    
    # Test memory usage calculation
    memory_stats = transformer_block.get_memory_usage()
    assert 'total_memory_mb' in memory_stats, "Should provide memory statistics"
    assert memory_stats['total_memory_mb'] > 0, "Should have positive memory usage"
    assert memory_stats['total_parameters'] > 0, "Should count parameters"
    
    print("✅ Transformer block tests passed!")
    print(f"✅ Pre-norm and post-norm architectures work correctly")
    print(f"✅ Residual connections preserve information flow")
    print(f"✅ Causal masking works across all attention heads")
    print(f"✅ Total parameters: {memory_stats['total_parameters']:,}")
    print(f"✅ Total memory: {memory_stats['total_memory_mb']:.2f}MB")

# Test function defined (called in main block)

# %% [markdown]
"""
## Complete Transformer Model

Finally, let's build a complete transformer model that can be used for language modeling tasks like text generation.
"""

# %% nbgrader={"grade": false, "grade_id": "transformer-model", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Transformer:
    """
    Complete transformer model for language processing.
    
    Stacks multiple transformer blocks with token embeddings and positional
    encoding to create a complete language model architecture.
    
    SUPPORTS KV CACHING (Module 19 integration):
    - Forward method accepts optional past_key_values parameter for caching
    - Generate method supports use_cache parameter for efficient generation
    - Returns new key-value pairs when caching is enabled
    - Backward compatible: works with or without caching
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, hidden_dim: int, max_seq_length: int = 1024,
                 dropout: float = 0.0, pre_norm: bool = True):
        """
        Initialize complete transformer model.
        
        TODO: Implement transformer model initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store model configuration
        2. Create token embedding layer
        3. Create positional encoding
        4. Create stack of transformer blocks
        5. Create output projection layer (for language modeling)
        6. Set up parameter tracking from all components
        
        LANGUAGE MODELING HEAD:
        Final linear layer that projects hidden states to vocabulary logits
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer blocks
            hidden_dim: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout rate
            pre_norm: Whether to use pre-normalization
        """
        ### BEGIN SOLUTION
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.pre_norm = pre_norm
        
        # Token embedding layer
        self.token_embedding = Embedding(vocab_size=vocab_size, embedding_dim=embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim=embed_dim, max_seq_length=max_seq_length)
        
        # Stack of transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                pre_norm=pre_norm
            )
            self.transformer_blocks.append(block)
        
        # Final layer normalization (for pre-norm architecture)
        if pre_norm:
            self.final_norm = LayerNorm(embed_dim)
        else:
            self.final_norm = None
        
        # Language modeling head (projects to vocabulary)
        xavier_bound = math.sqrt(6.0 / (embed_dim + vocab_size))
        self.lm_head = Tensor(np.random.uniform(-xavier_bound, xavier_bound, (embed_dim, vocab_size)))
        
        # Collect all parameters
        self.parameters = []
        if hasattr(self.token_embedding, 'parameters'):
            self.parameters.extend(self.token_embedding.parameters)
        
        for block in self.transformer_blocks:
            if hasattr(block, 'parameters'):
                self.parameters.extend(block.parameters)
        
        if self.final_norm:
            self.parameters.extend(self.final_norm.parameters)
        
        self.parameters.append(self.lm_head)
        ### END SOLUTION
    
    def forward(self, input_ids: Tensor, mask: Optional[Tensor] = None,
                return_attention_weights: bool = False, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None) -> Union[Tensor, Tuple[Tensor, List[Tensor]], Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tuple[Tensor, List[Tensor], List[Tuple[Tensor, Tensor]]]]:
        """
        Process input through complete transformer model.
        
        TODO: Implement transformer model forward pass.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Convert token IDs to embeddings
        2. Add positional encoding
        3. Process through all transformer blocks
        4. Apply final normalization (if pre-norm)
        5. Apply language modeling head
        6. Return logits (and optionally attention weights)
        
        Args:
            input_ids: Token indices with shape (batch_size, seq_len)
            mask: Optional attention mask
            return_attention_weights: Whether to return all attention weights
            past_key_values: Optional list of cached key-value pairs from previous forward pass
            
        Returns:
            Logits with shape (batch_size, seq_len, vocab_size)
            Optionally also list of attention weights from each layer
            Optionally also list of new key-value pairs for caching (if past_key_values provided)
        """
        ### BEGIN SOLUTION
        # Token embeddings
        embeddings = self.token_embedding.forward(input_ids)
        
        # Add positional encoding
        x = self.pos_encoding.forward(embeddings)
        
        # Process through transformer blocks
        all_attention_weights = []
        new_key_values = []
        
        for i, block in enumerate(self.transformer_blocks):
            # Get past key-value for this layer if available
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if past_key_values is not None:
                # KV caching enabled
                if return_attention_weights:
                    result = block.forward(x, mask=mask, return_attention_weights=True, past_key_value=past_key_value)
                    if len(result) == 3:
                        x, attn_weights, new_key_value = result
                        all_attention_weights.append(attn_weights)
                        new_key_values.append(new_key_value)
                    else:
                        # Fallback if block doesn't support KV caching yet
                        x, attn_weights = result
                        all_attention_weights.append(attn_weights)
                        new_key_values.append(None)
                else:
                    result = block.forward(x, mask=mask, past_key_value=past_key_value)
                    if isinstance(result, tuple) and len(result) == 2:
                        x, new_key_value = result
                        new_key_values.append(new_key_value)
                    else:
                        # Fallback if block doesn't support KV caching yet
                        x = result
                        new_key_values.append(None)
            else:
                # Standard behavior (backward compatible)
                if return_attention_weights:
                    x, attn_weights = block.forward(x, mask=mask, return_attention_weights=True)
                    all_attention_weights.append(attn_weights)
                else:
                    x = block.forward(x, mask=mask)
        
        # Final layer normalization (for pre-norm)
        if self.final_norm:
            x = self.final_norm.forward(x)
        
        # Language modeling head
        # x: (batch_size, seq_len, embed_dim)
        # lm_head: (embed_dim, vocab_size)
        # output: (batch_size, seq_len, vocab_size)
        
        batch_size, seq_len, embed_dim = x.shape
        x_reshaped = x.data.reshape(-1, embed_dim)  # (batch_size * seq_len, embed_dim)
        logits_reshaped = np.matmul(x_reshaped, self.lm_head.data)  # (batch_size * seq_len, vocab_size)
        logits = logits_reshaped.reshape(batch_size, seq_len, self.vocab_size)
        
        # Return appropriate tuple based on what was requested
        if past_key_values is not None:
            # KV caching is enabled
            if return_attention_weights:
                return Tensor(logits), all_attention_weights, new_key_values
            else:
                return Tensor(logits), new_key_values
        else:
            # Standard behavior (backward compatible)
            if return_attention_weights:
                return Tensor(logits), all_attention_weights
            else:
                return Tensor(logits)
        ### END SOLUTION
    
    def __call__(self, input_ids: Tensor, mask: Optional[Tensor] = None,
                 return_attention_weights: bool = False, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None) -> Union[Tensor, Tuple[Tensor, List[Tensor]], Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tuple[Tensor, List[Tensor], List[Tuple[Tensor, Tensor]]]]:
        """Make the class callable."""
        return self.forward(input_ids, mask, return_attention_weights, past_key_values)
    
    def generate(self, input_ids: Tensor, max_new_tokens: int = 50, 
                temperature: float = 1.0, use_cache: bool = False) -> Tensor:
        """
        Generate text autoregressively.
        
        This function is PROVIDED to show text generation capability.
        
        Args:
            input_ids: Input token IDs with shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (higher = more random)
            use_cache: Whether to use KV caching for faster generation
            
        Returns:
            Generated token IDs with shape (batch_size, original_seq_len + generated_tokens)
        """
        batch_size, current_seq_len = input_ids.shape
        
        if current_seq_len >= self.max_seq_length:
            raise ValueError(f"Input sequence length {current_seq_len} exceeds max {self.max_seq_length}")
        
        generated_ids = input_ids.data.copy()
        past_key_values = None  # Initialize cache for KV caching
        
        for step in range(max_new_tokens):
            if use_cache and step > 0:
                # For subsequent steps with caching, only process the last token
                current_input = Tensor(generated_ids[:, -1:])  # Only last token
                # No mask needed for single token
                current_mask = None
            else:
                # First step or no caching: process full sequence
                current_input = Tensor(generated_ids)
                # Create causal mask
                seq_len = generated_ids.shape[1]
                causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
                causal_mask = 1 - causal_mask
                current_mask = Tensor(causal_mask)
            
            # Forward pass with optional caching
            if use_cache:
                result = self.forward(current_input, mask=current_mask, past_key_values=past_key_values)
                if isinstance(result, tuple) and len(result) == 2:
                    logits, past_key_values = result
                else:
                    # Fallback if caching not fully implemented yet
                    logits = result
                    past_key_values = None
            else:
                logits = self.forward(current_input, mask=current_mask)
            
            # Get logits for last position
            last_logits = logits.data[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            last_logits = last_logits / temperature
            
            # Sample next token (using simple sampling)
            # Convert to probabilities
            exp_logits = np.exp(last_logits - np.max(last_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Sample from distribution
            next_tokens = []
            for i in range(batch_size):
                next_token = np.random.choice(self.vocab_size, p=probs[i])
                next_tokens.append(next_token)
            
            next_tokens = np.array(next_tokens).reshape(batch_size, 1)
            
            # Append to sequence
            generated_ids = np.concatenate([generated_ids, next_tokens], axis=1)
            
            # Stop if we reach max sequence length
            if generated_ids.shape[1] >= self.max_seq_length:
                break
        
        return Tensor(generated_ids)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Calculate memory usage of complete transformer model.
        
        This function is PROVIDED to show memory analysis.
        """
        # Token embedding memory
        if hasattr(self.token_embedding, 'get_memory_usage'):
            embedding_memory = self.token_embedding.get_memory_usage()['total_memory_mb']
        else:
            embedding_memory = self.vocab_size * self.embed_dim * 4 / (1024 * 1024)
        
        # Transformer blocks memory
        block_memory = 0
        if self.transformer_blocks:
            single_block_memory = self.transformer_blocks[0].get_memory_usage()['total_memory_mb']
            block_memory = single_block_memory * self.num_layers
        
        # Final norm memory
        final_norm_memory = 0
        if self.final_norm:
            final_norm_memory = self.final_norm.get_memory_usage()['parameter_memory_mb']
        
        # Language modeling head memory
        lm_head_memory = self.lm_head.data.nbytes / (1024 * 1024)
        
        total_memory = embedding_memory + block_memory + final_norm_memory + lm_head_memory
        total_params = sum(p.data.size for p in self.parameters) if hasattr(self, 'parameters') else 0
        
        return {
            'total_memory_mb': total_memory,
            'embedding_memory_mb': embedding_memory,
            'transformer_blocks_memory_mb': block_memory,
            'lm_head_memory_mb': lm_head_memory,
            'total_parameters': total_params,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'hidden_dim': self.hidden_dim
        }

# %% [markdown]
"""
### 🧪 Test Your Complete Transformer Implementation

Once you implement the Transformer methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-transformer-model-immediate", "locked": true, "points": 25, "schema_version": 3, "solution": false, "task": false}
def test_unit_transformer_model():
    """Unit test for complete transformer model."""
    print("🔬 Unit Test: Complete Transformer Model...")
    
    # Test configuration
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    hidden_dim = 512
    max_seq_length = 128
    
    transformer = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_length=max_seq_length,
        pre_norm=True
    )
    
    # Verify initialization
    assert transformer.vocab_size == vocab_size, "Should store vocabulary size"
    assert transformer.embed_dim == embed_dim, "Should store embedding dimension"
    assert transformer.num_layers == num_layers, "Should store number of layers"
    assert len(transformer.transformer_blocks) == num_layers, "Should create correct number of blocks"
    
    # Verify components exist
    assert hasattr(transformer, 'token_embedding'), "Should have token embedding"
    assert hasattr(transformer, 'pos_encoding'), "Should have positional encoding"
    assert hasattr(transformer, 'lm_head'), "Should have language modeling head"
    
    # Test forward pass with token IDs
    batch_size = 4
    seq_len = 32
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    input_tensor = Tensor(input_ids)
    
    logits = transformer.forward(input_tensor)
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    
    # Test with attention weights return
    logits_with_attn, all_attention_weights = transformer.forward(input_tensor, return_attention_weights=True)
    
    assert logits_with_attn.shape == expected_shape, "Logits with attention should have correct shape"
    assert len(all_attention_weights) == num_layers, f"Should return attention weights from {num_layers} layers"
    
    for i, attn_weights in enumerate(all_attention_weights):
        expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
        assert attn_weights.shape == expected_attn_shape, \
            f"Layer {i} attention should have shape {expected_attn_shape}, got {attn_weights.shape}"
    
    # Test with causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    causal_mask = 1 - causal_mask  # Convert to attention mask
    
    masked_logits, masked_attention = transformer.forward(
        input_tensor, mask=Tensor(causal_mask), return_attention_weights=True
    )
    
    assert masked_logits.shape == expected_shape, "Masked logits should have correct shape"
    
    # Verify causal masking propagates through all layers
    for layer_idx, attn_weights in enumerate(masked_attention):
        for head in range(num_heads):
            for i in range(seq_len):
                for j in range(i+1, seq_len):
                    assert np.all(attn_weights.data[:, head, i, j] < 1e-5), \
                        f"Layer {layer_idx}, head {head}: position ({i},{j}) should be masked"
    
    # Test callable interface
    logits_callable = transformer(input_tensor)
    assert np.allclose(logits_callable.data, logits.data), "Callable interface should work"
    
    # Test text generation capability
    print("  Testing text generation...")
    start_tokens = Tensor(np.random.randint(0, vocab_size, (2, 8)))  # 2 sequences, 8 tokens each
    generated = transformer.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    
    expected_gen_shape = (2, 18)  # 8 original + 10 new tokens
    assert generated.shape == expected_gen_shape, f"Generated shape should be {expected_gen_shape}, got {generated.shape}"
    
    # Verify original tokens are preserved
    assert np.array_equal(generated.data[:, :8], start_tokens.data), "Original tokens should be preserved"
    
    # Test different model configurations
    small_transformer = Transformer(
        vocab_size=500, embed_dim=128, num_heads=4, num_layers=2, hidden_dim=256
    )
    
    small_input = Tensor(np.random.randint(0, 500, (2, 16)))
    small_logits = small_transformer.forward(small_input)
    expected_small_shape = (2, 16, 500)
    assert small_logits.shape == expected_small_shape, "Small transformer should work"
    
    # Test pre-norm vs post-norm
    post_norm_transformer = Transformer(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=2, hidden_dim=hidden_dim, pre_norm=False
    )
    
    post_norm_logits = post_norm_transformer.forward(input_tensor)
    pre_norm_logits = Transformer(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=2, hidden_dim=hidden_dim, pre_norm=True
    ).forward(input_tensor)
    
    assert not np.allclose(post_norm_logits.data, pre_norm_logits.data), \
        "Pre-norm and post-norm should produce different outputs"
    
    # Test memory usage calculation
    memory_stats = transformer.get_memory_usage()
    assert 'total_memory_mb' in memory_stats, "Should provide memory statistics"
    assert memory_stats['total_memory_mb'] > 0, "Should have positive memory usage"
    assert memory_stats['total_parameters'] > 0, "Should count parameters"
    
    # Verify memory breakdown
    assert memory_stats['embedding_memory_mb'] > 0, "Should have embedding memory"
    assert memory_stats['transformer_blocks_memory_mb'] > 0, "Should have transformer block memory"
    assert memory_stats['lm_head_memory_mb'] > 0, "Should have language modeling head memory"
    
    print("✅ Complete transformer model tests passed!")
    print(f"✅ Forward pass produces correct logit shapes")
    print(f"✅ Causal masking works across all {num_layers} layers")
    print(f"✅ Text generation capability verified")
    print(f"✅ Total parameters: {memory_stats['total_parameters']:,}")
    print(f"✅ Total memory: {memory_stats['total_memory_mb']:.2f}MB")
    print(f"✅ Pre-norm and post-norm architectures work correctly")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🎯 ML Systems: Performance Analysis & Transformer Scaling

Now let's develop systems engineering skills by analyzing transformer performance and understanding how model depth and width affect memory usage and computational requirements.

### **Learning Outcome**: *"I understand how transformer architecture choices affect scalability, memory usage, and production deployment constraints"*
"""

# %% nbgrader={"grade": false, "grade_id": "transformer-profiler", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import time

class TransformerProfiler:
    """
    Performance profiling toolkit for transformer architectures.
    
    Helps ML engineers understand computational costs, memory scaling,
    and architectural trade-offs in transformer-based models.
    """
    
    def __init__(self):
        self.results = {}
    
    def measure_scaling_with_depth(self, base_config: Dict, layer_counts: List[int]) -> Dict:
        """
        Measure how transformer performance scales with number of layers.
        
        TODO: Implement transformer depth scaling measurement.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create transformers with different layer counts
        2. Measure memory usage and computation time for each
        3. Calculate scaling patterns (should be linear with depth)
        4. Analyze parameter growth and memory requirements
        5. Return comprehensive scaling analysis
        
        EXPECTED SCALING:
        - Parameters: Linear with depth
        - Memory: Linear with depth  
        - Computation: Linear with depth
        - Quality: Generally improves with depth (to a point)
        
        Args:
            base_config: Base transformer configuration
            layer_counts: List of layer counts to test
            
        Returns:
            Dictionary with scaling analysis results
        """
        ### BEGIN SOLUTION
        scaling_results = {}
        
        # Test input
        batch_size = 4
        seq_len = 32
        vocab_size = base_config['vocab_size']
        test_input = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
        
        for num_layers in layer_counts:
            # Create transformer with this depth
            transformer = Transformer(
                vocab_size=base_config['vocab_size'],
                embed_dim=base_config['embed_dim'],
                num_heads=base_config['num_heads'],
                num_layers=num_layers,
                hidden_dim=base_config['hidden_dim'],
                max_seq_length=base_config.get('max_seq_length', 128)
            )
            
            # Measure memory usage
            memory_stats = transformer.get_memory_usage()
            
            # Measure computation time
            start_time = time.time()
            logits = transformer.forward(test_input)
            end_time = time.time()
            
            computation_time_ms = (end_time - start_time) * 1000
            
            # Calculate throughput
            total_tokens = batch_size * seq_len
            tokens_per_second = total_tokens / (end_time - start_time) if end_time > start_time else 0
            
            scaling_results[num_layers] = {
                'num_layers': num_layers,
                'total_parameters': memory_stats['total_parameters'],
                'total_memory_mb': memory_stats['total_memory_mb'],
                'computation_time_ms': computation_time_ms,
                'tokens_per_second': tokens_per_second,
                'memory_per_layer_mb': memory_stats['transformer_blocks_memory_mb'] / num_layers if num_layers > 0 else 0,
                'parameters_per_layer': (memory_stats['total_parameters'] - 
                                       base_config['vocab_size'] * base_config['embed_dim'] * 2) // num_layers if num_layers > 0 else 0
            }
        
        return scaling_results
        ### END SOLUTION
    
    def analyze_width_vs_depth_tradeoffs(self, base_params: int, configurations: List[Dict]) -> Dict:
        """
        Compare different ways to allocate a fixed parameter budget.
        
        This function is PROVIDED to show parameter allocation analysis.
        """
        print(f"📊 WIDTH vs DEPTH TRADE-OFF ANALYSIS")
        print(f"Target parameter budget: ~{base_params:,} parameters")
        print("=" * 70)
        
        results = {}
        
        # Test input
        batch_size = 4
        seq_len = 32
        test_input = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)))
        
        print(f"{'Config':<15} {'Layers':<7} {'Embed':<6} {'Heads':<6} {'Hidden':<7} {'Params':<12} {'Time (ms)':<10} {'Memory'}")
        print("-" * 80)
        
        for i, config in enumerate(configurations):
            try:
                # Create transformer
                transformer = Transformer(
                    vocab_size=1000,  # Fixed vocab size
                    embed_dim=config['embed_dim'],
                    num_heads=config['num_heads'],
                    num_layers=config['num_layers'],
                    hidden_dim=config['hidden_dim'],
                    max_seq_length=128
                )
                
                # Get actual parameter count
                memory_stats = transformer.get_memory_usage()
                actual_params = memory_stats['total_parameters']
                
                # Measure performance
                start_time = time.time()
                logits = transformer.forward(test_input)
                computation_time = (time.time() - start_time) * 1000
                
                config_name = f"Config_{i+1}"
                results[config_name] = {
                    'config': config,
                    'actual_parameters': actual_params,
                    'computation_time_ms': computation_time,
                    'memory_mb': memory_stats['total_memory_mb'],
                    'parameter_efficiency': abs(actual_params - base_params) / base_params
                }
                
                print(f"{config_name:<15} {config['num_layers']:<7} {config['embed_dim']:<6} "
                      f"{config['num_heads']:<6} {config['hidden_dim']:<7} {actual_params:<12,} "
                      f"{computation_time:<10.2f} {memory_stats['total_memory_mb']:.1f}MB")
                
            except Exception as e:
                print(f"{config_name:<15} ERROR: {str(e)[:50]}")
        
        # Analysis
        print(f"\n💡 TRADE-OFF INSIGHTS:")
        print(f"   - Deeper models: Better at learning complex patterns, more sequential")
        print(f"   - Wider models: More parallelizable, can capture diverse features")
        print(f"   - More heads: Richer attention patterns, more computation")
        print(f"   - Hidden dimension: Affects FFN capacity, major parameter contributor")
        
        return results
    
    def simulate_production_scaling(self, model_sizes: List[str]) -> Dict:
        """
        Simulate memory and computation requirements for production model sizes.
        
        This function is PROVIDED to show production scaling analysis.
        """
        print(f"\n🏭 PRODUCTION MODEL SCALING SIMULATION")
        print("=" * 60)
        
        # Production model configurations (simplified)
        size_configs = {
            'Small': {'vocab_size': 50000, 'embed_dim': 512, 'num_heads': 8, 'num_layers': 6, 'hidden_dim': 2048},
            'Medium': {'vocab_size': 50000, 'embed_dim': 768, 'num_heads': 12, 'num_layers': 12, 'hidden_dim': 3072},
            'Large': {'vocab_size': 50000, 'embed_dim': 1024, 'num_heads': 16, 'num_layers': 24, 'hidden_dim': 4096},
            'XL': {'vocab_size': 50000, 'embed_dim': 1280, 'num_heads': 20, 'num_layers': 36, 'hidden_dim': 5120}
        }
        
        results = {}
        
        print(f"{'Model Size':<12} {'Parameters':<12} {'Memory (GB)':<12} {'Training GPU':<12} {'Inference'}")
        print("-" * 70)
        
        for size in model_sizes:
            if size not in size_configs:
                continue
                
            config = size_configs[size]
            
            # Estimate parameters
            # Embedding: vocab_size * embed_dim * 2 (input + output)
            embedding_params = config['vocab_size'] * config['embed_dim'] * 2
            
            # Per layer: 
            # - Attention: 4 * embed_dim^2 (Q, K, V, O projections)
            # - FFN: 2 * embed_dim * hidden_dim + embed_dim + hidden_dim (weights + biases)
            # - LayerNorm: 2 * embed_dim * 2 (two norms per layer)
            attention_params_per_layer = 4 * config['embed_dim'] ** 2
            ffn_params_per_layer = 2 * config['embed_dim'] * config['hidden_dim'] + config['embed_dim'] + config['hidden_dim']
            norm_params_per_layer = 4 * config['embed_dim']
            
            layer_params = attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
            total_params = embedding_params + layer_params * config['num_layers']
            
            # Estimate memory (parameters + activations + gradients for training)
            param_memory_gb = total_params * 4 / (1024**3)  # 4 bytes per float32
            
            # Training memory: parameters + gradients + optimizer states + activations
            training_memory_gb = param_memory_gb * 4  # Rough estimate (param + grad + 2x optimizer states)
            
            # Inference memory: just parameters + activations
            inference_memory_gb = param_memory_gb * 1.5  # Parameters + activation memory
            
            # GPU requirements (very rough estimates)
            if training_memory_gb < 24:
                training_gpu = "Single RTX 4090"
            elif training_memory_gb < 80:
                training_gpu = "Single A100"
            else:
                training_gpu = "Multi-GPU"
            
            if inference_memory_gb < 12:
                inference_req = "RTX 4060 Ti"
            elif inference_memory_gb < 24:
                inference_req = "RTX 4090"
            else:
                inference_req = "A100+"
            
            results[size] = {
                'config': config,
                'total_parameters': total_params,
                'training_memory_gb': training_memory_gb,
                'inference_memory_gb': inference_memory_gb,
                'training_gpu_req': training_gpu,
                'inference_gpu_req': inference_req
            }
            
            print(f"{size:<12} {total_params/1e6:.1f}M {training_memory_gb:.1f} {training_gpu:<12} {inference_req}")
        
        print(f"\n📈 SCALING OBSERVATIONS:")
        print(f"   - Model size grows super-linearly with dimension increases")
        print(f"   - Memory requirements dominate deployment decisions")
        print(f"   - Training requires 3-4x more memory than inference")
        print(f"   - Multi-GPU becomes necessary for large models")
        
        return results

def analyze_transformer_system_design():
    """
    Comprehensive analysis of transformer system design choices and trade-offs.
    
    This function is PROVIDED to show systems-level design thinking.
    """
    print("🏗️ TRANSFORMER SYSTEM DESIGN ANALYSIS")
    print("=" * 60)
    
    # Architecture decision analysis
    design_choices = {
        'Layer Normalization': {
            'Pre-norm': {'stability': 'High', 'training': 'Easier', 'performance': 'Good'},
            'Post-norm': {'stability': 'Lower', 'training': 'Harder', 'performance': 'Potentially better'}
        },
        'Attention Patterns': {
            'Full attention': {'complexity': 'O(N²)', 'quality': 'Best', 'scalability': 'Limited'},
            'Sparse attention': {'complexity': 'O(N√N)', 'quality': 'Good', 'scalability': 'Better'},
            'Linear attention': {'complexity': 'O(N)', 'quality': 'Reduced', 'scalability': 'Excellent'}
        },
        'Feed-Forward Size': {
            '2x embed_dim': {'parameters': 'Low', 'capacity': 'Limited', 'speed': 'Fast'},
            '4x embed_dim': {'parameters': 'Standard', 'capacity': 'Good', 'speed': 'Medium'},
            '8x embed_dim': {'parameters': 'High', 'capacity': 'High', 'speed': 'Slow'}
        }
    }
    
    print("🎯 ARCHITECTURAL DESIGN CHOICES:")
    for category, choices in design_choices.items():
        print(f"\n{category}:")
        for choice, properties in choices.items():
            prop_str = ", ".join([f"{k}: {v}" for k, v in properties.items()])
            print(f"   - {choice}: {prop_str}")
    
    # Memory scaling analysis
    print(f"\n📊 MEMORY SCALING PATTERNS:")
    print(f"Component breakdown for typical transformer:")
    print(f"   - Token embeddings: vocab_size × embed_dim parameters")
    print(f"   - Position encodings: 0 parameters (sinusoidal) or seq_len × embed_dim (learned)")
    print(f"   - Attention layers: 4 × embed_dim² parameters per layer")
    print(f"   - Feed-forward: 2 × embed_dim × hidden_dim parameters per layer")
    print(f"   - Layer normalization: 2 × embed_dim parameters per layer")
    print(f"   - Output projection: embed_dim × vocab_size parameters")
    
    print(f"\n🔧 OPTIMIZATION STRATEGIES:")
    optimization_techniques = [
        "Gradient checkpointing: Trade computation for memory",
        "Mixed precision training: Use FP16 for 2x memory reduction",
        "Parameter sharing: Share weights across layers",
        "Sparse attention: Reduce quadratic scaling",
        "Model parallelism: Distribute layers across GPUs",
        "Pipeline parallelism: Process different batch elements on different GPUs",
        "Activation checkpointing: Recompute activations instead of storing"
    ]
    
    for technique in optimization_techniques:
        print(f"   - {technique}")
    
    print(f"\n🎯 PRODUCTION DEPLOYMENT CONSIDERATIONS:")
    deployment_factors = [
        "Batch size: Larger batches improve GPU utilization but increase memory",
        "Sequence length: Quadratic impact on attention memory",
        "Model depth: Linear impact on memory and computation",
        "Model width: Quadratic impact on attention parameters",
        "Precision: FP32 vs FP16 vs INT8 trade-offs",
        "Hardware: GPU memory and compute capabilities",
        "Latency requirements: Real-time vs batch processing",
        "Throughput requirements: Tokens per second targets"
    ]
    
    for factor in deployment_factors:
        print(f"   - {factor}")

# %% [markdown]
"""
### 🧪 Test: Transformer Performance Analysis

Let's test our transformer profiler with realistic scaling scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "test-transformer-profiler", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_transformer_profiler():
    """Test transformer profiler with various scenarios."""
    print("🔬 Unit Test: Transformer Performance Profiler...")
    
    profiler = TransformerProfiler()
    
    # Test depth scaling measurement
    base_config = {
        'vocab_size': 500,
        'embed_dim': 128,
        'num_heads': 4,
        'hidden_dim': 256
    }
    
    layer_counts = [1, 2, 4]
    depth_results = profiler.measure_scaling_with_depth(base_config, layer_counts)
    
    # Verify depth scaling results
    assert len(depth_results) == len(layer_counts), f"Should test {len(layer_counts)} layer counts"
    
    for num_layers in layer_counts:
        assert num_layers in depth_results, f"Should include results for {num_layers} layers"
        result = depth_results[num_layers]
        
        # Verify required metrics
        required_keys = ['num_layers', 'total_parameters', 'total_memory_mb', 
                        'computation_time_ms', 'tokens_per_second']
        for key in required_keys:
            assert key in result, f"Missing metric: {key} for {num_layers} layers"
            assert isinstance(result[key], (int, float)), f"Invalid type for {key}"
        
        # Verify reasonable values
        assert result['num_layers'] == num_layers, "Should store correct layer count"
        assert result['total_parameters'] > 0, "Should have positive parameter count"
        assert result['total_memory_mb'] > 0, "Should have positive memory usage"
    
    # Test that parameters and memory scale roughly linearly with depth
    if len(layer_counts) >= 2:
        shallow = depth_results[layer_counts[0]]
        deep = depth_results[layer_counts[-1]]
        
        layer_ratio = deep['num_layers'] / shallow['num_layers']
        param_ratio = deep['total_parameters'] / shallow['total_parameters']
        memory_ratio = deep['total_memory_mb'] / shallow['total_memory_mb']
        
        # Allow some deviation due to fixed costs (embeddings, etc.)
        assert 1.0 < param_ratio < layer_ratio * 2, f"Parameters should scale sub-linearly, got {param_ratio:.2f}"
        assert 1.0 < memory_ratio < layer_ratio * 2, f"Memory should scale sub-linearly, got {memory_ratio:.2f}"
    
    print("✅ Depth scaling measurement test passed")
    
    # Test width vs depth analysis
    configurations = [
        {'embed_dim': 128, 'num_heads': 4, 'num_layers': 4, 'hidden_dim': 256},
        {'embed_dim': 256, 'num_heads': 8, 'num_layers': 2, 'hidden_dim': 512},
    ]
    
    width_depth_results = profiler.analyze_width_vs_depth_tradeoffs(100000, configurations)
    
    # Verify width vs depth results
    assert len(width_depth_results) > 0, "Should analyze at least one configuration"
    
    for config_name, result in width_depth_results.items():
        assert 'config' in result, "Should include configuration"
        assert 'actual_parameters' in result, "Should count actual parameters"
        assert 'computation_time_ms' in result, "Should measure computation time"
        assert result['actual_parameters'] > 0, "Should have positive parameter count"
    
    print("✅ Width vs depth analysis test passed")
    
    # Test production scaling simulation
    production_results = profiler.simulate_production_scaling(['Small', 'Medium'])
    
    # Verify production scaling results
    for size, result in production_results.items():
        assert 'config' in result, "Should include model configuration"
        assert 'total_parameters' in result, "Should estimate total parameters"
        assert 'training_memory_gb' in result, "Should estimate training memory"
        assert 'inference_memory_gb' in result, "Should estimate inference memory"
        
        # Verify reasonable scaling
        assert result['total_parameters'] > 1e6, "Should have millions of parameters"
        assert result['training_memory_gb'] > result['inference_memory_gb'], "Training should require more memory"
    
    print("✅ Production scaling simulation test passed")
    print("🎯 Transformer Profiler: All tests passed!")

# Test function defined (called in main block)

# %% [markdown]
"""
## Integration Testing: Complete Language Model Pipeline

Let's test the complete pipeline from tokenization through transformer processing:
"""

# %% nbgrader={"grade": false, "grade_id": "test-transformer-integration", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_complete_language_model_pipeline():
    """Test complete language model pipeline integration."""
    print("🧪 Integration Test: Complete Language Model Pipeline...")
    
    # Create a small but complete language model
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    hidden_dim = 512
    max_seq_length = 64
    
    print(f"  Creating transformer with {num_layers} layers, {embed_dim} dimensions...")
    transformer = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_length=max_seq_length
    )
    
    # Test 1: Basic text processing pipeline
    print("  Testing basic text processing pipeline...")
    batch_size = 4
    seq_len = 32
    
    # Simulate tokenized input
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    input_tensor = Tensor(input_ids)
    
    # Forward pass
    logits = transformer.forward(input_tensor)
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Test that logits are reasonable (not all zeros/inf/nan)
    assert not np.all(logits.data == 0), "Logits should not all be zero"
    assert not np.any(np.isinf(logits.data)), "Logits should not contain inf"
    assert not np.any(np.isnan(logits.data)), "Logits should not contain nan"
    
    print(f"    Forward pass successful: {logits.shape}")
    
    # Test 2: Language modeling with causal mask
    print("  Testing language modeling with causal attention...")
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    causal_mask = 1 - causal_mask  # Convert to attention mask
    
    masked_logits, all_attention = transformer.forward(
        input_tensor, mask=Tensor(causal_mask), return_attention_weights=True
    )
    
    assert len(all_attention) == num_layers, f"Should return attention from {num_layers} layers"
    
    # Verify causal masking works across all layers
    for layer_idx, attn_weights in enumerate(all_attention):
        # Check a few positions to ensure masking works
        for i in range(min(5, seq_len)):
            for j in range(i+1, min(i+5, seq_len)):
                future_attention = attn_weights.data[:, :, i, j]  # All heads, all batches
                assert np.all(future_attention < 1e-5), \
                    f"Layer {layer_idx}: future attention at ({i},{j}) should be ~0"
    
    print(f"    Causal masking verified across all layers")
    
    # Test 3: Text generation
    print("  Testing autoregressive text generation...")
    # Start with a shorter sequence for generation
    gen_start = Tensor(np.random.randint(0, vocab_size, (2, 8)))
    generated = transformer.generate(gen_start, max_new_tokens=8, temperature=1.0)
    
    expected_gen_shape = (2, 16)  # 8 start + 8 generated
    assert generated.shape == expected_gen_shape, f"Expected {expected_gen_shape}, got {generated.shape}"
    
    # Verify original tokens preserved
    assert np.array_equal(generated.data[:, :8], gen_start.data), "Should preserve original tokens"
    
    # Verify new tokens are valid
    new_tokens = generated.data[:, 8:]
    assert np.all(new_tokens >= 0), "Generated tokens should be >= 0"
    assert np.all(new_tokens < vocab_size), f"Generated tokens should be < {vocab_size}"
    
    print(f"    Generated {new_tokens.shape[1]} new tokens successfully")
    
    # Test 4: Different sequence lengths
    print("  Testing variable sequence lengths...")
    for test_seq_len in [16, 32, 48]:
        if test_seq_len > max_seq_length:
            continue
            
        test_input = Tensor(np.random.randint(0, vocab_size, (2, test_seq_len)))
        test_logits = transformer.forward(test_input)
        
        expected_test_shape = (2, test_seq_len, vocab_size)
        assert test_logits.shape == expected_test_shape, f"Failed for seq_len {test_seq_len}"
    
    print(f"    Variable sequence lengths work correctly")
    
    # Test 5: Memory usage analysis
    print("  Analyzing memory usage...")
    memory_stats = transformer.get_memory_usage()
    
    print(f"    Model parameters: {memory_stats['total_parameters']:,}")
    print(f"    Model memory: {memory_stats['total_memory_mb']:.1f}MB")
    print(f"    Embedding memory: {memory_stats['embedding_memory_mb']:.1f}MB")
    print(f"    Transformer blocks: {memory_stats['transformer_blocks_memory_mb']:.1f}MB")
    print(f"    LM head: {memory_stats['lm_head_memory_mb']:.1f}MB")
    
    # Verify memory breakdown makes sense
    component_memory = (memory_stats['embedding_memory_mb'] + 
                       memory_stats['transformer_blocks_memory_mb'] + 
                       memory_stats['lm_head_memory_mb'])
    
    # Allow small difference due to final norm layer
    memory_diff = abs(memory_stats['total_memory_mb'] - component_memory)
    assert memory_diff < 1.0, f"Memory breakdown doesn't add up: {memory_diff:.2f}MB difference"
    
    # Test 6: Performance characteristics
    print("  Testing performance characteristics...")
    
    # Time multiple forward passes
    num_iterations = 5
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = transformer.forward(input_tensor)
    
    total_time = time.time() - start_time
    avg_time_per_forward = total_time / num_iterations
    tokens_per_second = (batch_size * seq_len) / avg_time_per_forward
    
    print(f"    Average forward pass: {avg_time_per_forward*1000:.2f}ms")
    print(f"    Processing speed: {tokens_per_second:.0f} tokens/second")
    
    # Verify reasonable performance
    assert avg_time_per_forward < 1.0, "Forward pass should be < 1 second"
    assert tokens_per_second > 50, "Should process > 50 tokens/second"
    
    # Test 7: Gradient flow (simulated)
    print("  Testing gradient flow through layers...")
    
    # Create slightly different inputs to test sensitivity
    input_1 = Tensor(input_ids.copy())
    input_2 = Tensor(input_ids.copy())
    input_2.data[0, 0] = (input_2.data[0, 0] + 1) % vocab_size  # Change one token
    
    logits_1 = transformer.forward(input_1)
    logits_2 = transformer.forward(input_2)
    
    # Outputs should be different (model is sensitive to input changes)
    output_diff = np.mean(np.abs(logits_1.data - logits_2.data))
    assert output_diff > 1e-6, f"Model should be sensitive to input changes, diff: {output_diff}"
    
    # But not too different (model should be stable)
    assert output_diff < 100, f"Model should be stable, large diff: {output_diff}"
    
    print(f"    Model shows appropriate sensitivity to input changes")
    
    print("✅ Complete language model pipeline integration test passed!")
    print(f"✅ Forward pass, masking, generation, and performance verified")
    print(f"✅ Model processes {tokens_per_second:.0f} tokens/second")
    print(f"✅ Memory footprint: {memory_stats['total_memory_mb']:.1f}MB")

# Test function defined (called in main block)

# %% [markdown]
"""
## Main Execution Block

All transformer tests and demonstrations are run from here when the module is executed directly:
"""

# %% nbgrader={"grade": false, "grade_id": "transformers-main", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    # Run all unit tests
    test_unit_layer_norm()
    test_unit_feed_forward()
    test_unit_transformer_block()
    test_unit_transformer_model()
    test_transformer_profiler()
    test_complete_language_model_pipeline()
    
    print("\n" + "="*60)
    print("🔍 TRANSFORMER SYSTEMS ANALYSIS")
    print("="*60)
    
    # Performance analysis
    profiler = TransformerProfiler()
    
    # Test transformer scaling with different depths
    print("📈 TRANSFORMER DEPTH SCALING ANALYSIS:")
    base_config = {
        'vocab_size': 1000,
        'embed_dim': 256,
        'num_heads': 8,
        'hidden_dim': 1024
    }
    
    layer_counts = [2, 4, 8, 12]
    depth_results = profiler.measure_scaling_with_depth(base_config, layer_counts)
    
    # Analyze scaling patterns
    print(f"\n{'Layers':<7} {'Parameters':<12} {'Memory (MB)':<12} {'Time (ms)':<10} {'Tokens/sec':<10}")
    print("-" * 60)
    
    for num_layers in layer_counts:
        result = depth_results[num_layers]
        print(f"{num_layers:<7} {result['total_parameters']:<12,} {result['total_memory_mb']:<12.1f} "
              f"{result['computation_time_ms']:<10.2f} {result['tokens_per_second']:<10.0f}")
    
    # Width vs depth trade-off analysis
    print("\n" + "="*60)
    configurations = [
        {'embed_dim': 256, 'num_heads': 8, 'num_layers': 8, 'hidden_dim': 1024},  # Deep & narrow
        {'embed_dim': 512, 'num_heads': 16, 'num_layers': 4, 'hidden_dim': 2048}, # Wide & shallow
        {'embed_dim': 384, 'num_heads': 12, 'num_layers': 6, 'hidden_dim': 1536}, # Balanced
    ]
    
    width_depth_results = profiler.analyze_width_vs_depth_tradeoffs(2000000, configurations)
    
    # Production scaling simulation
    print("\n" + "="*60)
    production_results = profiler.simulate_production_scaling(['Small', 'Medium', 'Large'])
    
    # Systems design analysis
    print("\n" + "="*60)
    analyze_transformer_system_design()
    
    # Demonstrate realistic language model setup
    print("\n" + "="*60)
    print("🏗️ REALISTIC LANGUAGE MODEL DEMONSTRATION")
    print("="*60)
    
    # Create a realistic small language model
    vocab_size = 5000
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    hidden_dim = 2048
    max_seq_length = 256
    
    print(f"Language model configuration:")
    print(f"  Vocabulary: {vocab_size:,} tokens")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Transformer layers: {num_layers}")
    print(f"  Feed-forward dimension: {hidden_dim}")
    print(f"  Max sequence length: {max_seq_length}")
    
    # Create the model
    language_model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_length=max_seq_length,
        pre_norm=True
    )
    
    # Analyze model characteristics
    memory_stats = language_model.get_memory_usage()
    
    print(f"\nModel characteristics:")
    print(f"  Total parameters: {memory_stats['total_parameters']:,}")
    print(f"  Model size: {memory_stats['total_memory_mb']:.1f}MB")
    print(f"  Embedding table: {memory_stats['embedding_memory_mb']:.1f}MB ({memory_stats['embedding_memory_mb']/memory_stats['total_memory_mb']*100:.1f}%)")
    print(f"  Transformer layers: {memory_stats['transformer_blocks_memory_mb']:.1f}MB ({memory_stats['transformer_blocks_memory_mb']/memory_stats['total_memory_mb']*100:.1f}%)")
    print(f"  Output projection: {memory_stats['lm_head_memory_mb']:.1f}MB ({memory_stats['lm_head_memory_mb']/memory_stats['total_memory_mb']*100:.1f}%)")
    
    # Performance simulation
    batch_size = 8
    seq_len = 128
    test_input = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    
    start_time = time.time()
    logits = language_model.forward(test_input)
    forward_time = time.time() - start_time
    
    tokens_per_second = (batch_size * seq_len) / forward_time
    
    print(f"\nPerformance simulation:")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"  Forward pass time: {forward_time*1000:.2f}ms")
    print(f"  Throughput: {tokens_per_second:.0f} tokens/second")
    print(f"  Memory for batch: {logits.data.nbytes/(1024*1024):.1f}MB")
    
    # Text generation example
    print(f"\nText generation example:")
    start_sequence = Tensor(np.random.randint(0, vocab_size, (1, 10)))
    generated = language_model.generate(start_sequence, max_new_tokens=20, temperature=0.8)
    
    print(f"  Input sequence: {start_sequence.data[0].tolist()}")
    print(f"  Generated tokens: {generated.data[0, 10:].tolist()}")
    print(f"  Generation completed successfully")
    
    # Scaling predictions
    print(f"\nScaling analysis:")
    current_params = memory_stats['total_parameters']
    
    # Estimate for different scales
    scaling_factors = [2, 5, 10]
    for factor in scaling_factors:
        scaled_params = current_params * factor
        scaled_memory_gb = memory_stats['total_memory_mb'] * factor / 1024
        
        print(f"  {factor}x scale: {scaled_params/1e6:.0f}M params, ~{scaled_memory_gb:.1f}GB memory")
    
    print("\n" + "="*60)
    print("🎯 TRANSFORMERS MODULE COMPLETE!")
    print("="*60)
    print("All transformer tests passed!")
    print("Complete language model architecture implemented!")
    print("Ready for production deployment and optimization!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built complete transformer architectures, let's connect this work to broader ML systems challenges. These questions help you think critically about how transformer design choices affect production deployment and system performance.

Take time to reflect thoughtfully on each question - your insights will help you understand how transformer architectures connect to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Transformer Architecture Optimization and Resource Allocation

**Context**: Your transformer implementations demonstrate how layer depth, attention heads, and hidden dimensions affect model capacity and computational requirements. Production transformer systems must optimize these architectural choices within hardware constraints while maximizing model performance for specific tasks and deployment scenarios.

**Reflection Question**: Design a transformer architecture optimization strategy for deploying language models across diverse production scenarios: real-time chat (low latency), document processing (high throughput), and mobile inference (resource-constrained). How would you allocate a fixed parameter budget across depth, width, and attention heads to optimize for each scenario, implement architecture search strategies that consider hardware constraints, and design adaptive model scaling that adjusts to available computational resources? Consider the challenges of maintaining consistent model quality while optimizing for different performance metrics and deployment environments.

Think about: parameter budget allocation, architecture search strategies, hardware-aware optimization, and adaptive model scaling techniques.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-architecture-optimization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON TRANSFORMER ARCHITECTURE OPTIMIZATION:

TODO: Replace this text with your thoughtful response about transformer architecture optimization for diverse deployment scenarios.

Consider addressing:
- How would you allocate parameter budgets across depth, width, and attention heads for different scenarios?
- What architecture search strategies would you use to optimize within hardware constraints?
- How would you implement adaptive model scaling that adjusts to available resources?
- What approaches would you use to maintain model quality across different deployment environments?
- How would you balance latency, throughput, and resource constraints in architectural decisions?

Write a strategic analysis connecting your transformer implementations to real architecture optimization challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of transformer architecture trade-offs and optimization (3 points)
- Designs practical approaches to parameter allocation and architecture search (3 points)
- Addresses adaptive scaling and hardware-aware optimization (2 points)
- Shows systems thinking about production deployment optimization (2 points)
- Clear strategic reasoning with architecture optimization insights (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring strategic analysis of transformer architecture optimization
# Students should demonstrate understanding of architecture design and production deployment challenges
### END SOLUTION

# %% [markdown]
"""
### Question 2: Transformer Training and Inference System Design

**Context**: Your transformer implementation shows how layer normalization, residual connections, and feed-forward networks work together to enable training of deep models. Production transformer systems must optimize the training pipeline for efficiency while designing inference systems that handle diverse workloads with different latency and throughput requirements.

**Reflection Question**: Architect a transformer training and inference system that efficiently trains models with billions of parameters while serving diverse inference workloads with millisecond latency requirements. How would you design distributed training strategies that handle memory constraints and communication bottlenecks, implement efficient inference serving that optimizes for both batch and real-time processing, and manage model deployment across heterogeneous hardware environments? Consider the challenges of maintaining numerical stability during distributed training while achieving consistent inference performance across different deployment targets.

Think about: distributed training optimization, inference serving strategies, heterogeneous deployment, and training-inference consistency.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-training-inference-systems", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON TRANSFORMER TRAINING AND INFERENCE SYSTEM DESIGN:

TODO: Replace this text with your thoughtful response about transformer training and inference system architecture.

Consider addressing:
- How would you design distributed training for billion-parameter transformers with memory constraints?
- What strategies would you use for efficient inference serving with millisecond latency requirements?
- How would you manage model deployment across heterogeneous hardware environments?
- What approaches would you use to maintain numerical stability during distributed training?
- How would you ensure consistent inference performance across different deployment targets?

Write a system design analysis connecting your transformer implementation to large-scale training and serving challenges.

GRADING RUBRIC (Instructor Use):
- Shows understanding of distributed training and inference serving challenges (3 points)
- Designs practical approaches to memory management and latency optimization (3 points)
- Addresses heterogeneous deployment and numerical stability considerations (2 points)
- Demonstrates systems thinking about training-inference system coordination (2 points)
- Clear system design reasoning with scalability insights (bonus points for comprehensive system architecture)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring system design for transformer training and inference
# Students should demonstrate knowledge of distributed systems and production deployment architecture
### END SOLUTION

# %% [markdown]
"""
### Question 3: Transformer Optimization and Production Deployment

**Context**: Your complete transformer model demonstrates the integration of tokenization, embeddings, attention, and feed-forward components into a unified language processing system. Production transformer deployments must optimize the entire pipeline for efficiency while maintaining model quality and enabling continuous improvement through model updates and fine-tuning.

**Reflection Question**: Design a production transformer deployment system that optimizes the complete language processing pipeline while enabling continuous model improvement and adaptation. How would you implement end-to-end optimization that spans from tokenization through generation, design efficient model serving infrastructure that handles dynamic batching and request routing, and enable seamless model updates without service interruption? Consider the challenges of optimizing the entire pipeline holistically while maintaining modularity for individual component improvements and supporting diverse model variants and fine-tuned versions.

Think about: end-to-end pipeline optimization, model serving infrastructure, continuous deployment strategies, and modular system design.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-production-deployment", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON TRANSFORMER OPTIMIZATION AND PRODUCTION DEPLOYMENT:

TODO: Replace this text with your thoughtful response about transformer production deployment system design.

Consider addressing:
- How would you implement end-to-end optimization spanning tokenization through generation?
- What strategies would you use for efficient model serving with dynamic batching and request routing?
- How would you enable seamless model updates without service interruption?
- What approaches would you use to maintain pipeline modularity while optimizing holistically?
- How would you support diverse model variants and fine-tuned versions in production?

Write a deployment analysis connecting your transformer implementation to complete production system optimization.

GRADING RUBRIC (Instructor Use):
- Understands end-to-end optimization and production deployment challenges (3 points)
- Designs practical approaches to model serving and continuous deployment (3 points)
- Addresses modularity and system integration considerations (2 points)
- Shows systems thinking about holistic pipeline optimization (2 points)
- Clear deployment reasoning with production optimization insights (bonus points for innovative system design)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of production transformer deployment optimization
# Students should demonstrate knowledge of end-to-end system design and continuous deployment strategies
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Transformers

Congratulations! You have successfully implemented complete transformer architectures that power modern language models:

### ✅ What You Have Built
- **Layer Normalization**: Stable normalization for deep transformer training
- **Position-wise Feed-Forward**: Non-linear transformations applied to each sequence position
- **Transformer Blocks**: Complete transformer layers with attention, normalization, and residual connections
- **Complete Transformer**: Full language model with embeddings, multiple layers, and generation capability
- **Text Generation**: Autoregressive generation with proper causal masking
- **🆕 Performance Analysis**: Comprehensive scaling analysis and architectural optimization tools
- **🆕 Production Insights**: Understanding of real-world transformer deployment challenges

### ✅ Key Learning Outcomes
- **Understanding**: How transformer blocks enable powerful sequence modeling through attention and feed-forward layers
- **Implementation**: Built complete transformer architectures with proper layer organization and residual connections
- **Systems Insight**: How transformer depth affects memory usage, training efficiency, and model capacity
- **Performance Engineering**: Measured and analyzed transformer scaling characteristics and optimization opportunities
- **Production Context**: Understanding transformer deployment challenges and architectural trade-offs

### ✅ Technical Mastery
- **Layer Normalization**: Stabilizing deep network training with proper feature normalization
- **Residual Connections**: Enabling gradient flow through deep transformer architectures
- **Pre-norm vs Post-norm**: Understanding normalization placement effects on training stability
- **Parameter Scaling**: Understanding how transformer parameters scale with architectural choices
- **🆕 Generation Systems**: Autoregressive text generation with causal attention patterns

### ✅ Professional Skills Developed
- **Systems Architecture**: Designing complete transformer systems for production scale
- **Memory Engineering**: Understanding transformer memory scaling and optimization techniques
- **Performance Analysis**: Measuring and improving transformer computation and memory efficiency
- **Integration Design**: Building complete language processing pipelines from tokenization to generation

### ✅ Ready for Next Steps
Your transformer implementations provide the foundation for:
- **Advanced Language Models**: GPT, BERT, and other transformer-based architectures
- **Multi-modal Models**: Extending transformers to vision, audio, and other modalities
- **Production Optimization**: Memory optimization, distributed training, and efficient inference
- **🧠 AI Applications**: Real-world language processing applications and services

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **GPT Architecture**: Your transformer matches GPT's decoder-only architecture
- **BERT Components**: Layer normalization and attention mechanisms used in BERT
- **Production Optimization**: Understanding of memory scaling, batching, and generation optimization
- **Industry Applications**: Foundation for all modern language model deployments

### 🎯 The Complete Language Model
You have built the architecture that transformed AI:
- **Before**: RNNs and CNNs limited by sequential processing and local dependencies
- **After**: Transformers enable parallel processing and global attention across entire sequences

**Achievement Unlocked**: You now understand every component of modern language models from tokenization through generation!

Your complete transformer implementation provides the foundation for understanding and building modern AI systems. You've mastered the architecture that powers ChatGPT, GPT-4, BERT, and countless other AI applications.

From discrete tokens to continuous embeddings, from attention mechanisms to complete language generation - you've built the entire pipeline that enables machines to understand and generate human language.

**🏆 Congratulations on completing the complete transformer architecture implementation!**
"""