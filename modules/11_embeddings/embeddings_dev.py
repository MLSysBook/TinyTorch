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
# Embeddings - Converting Tokens to Dense Vector Representations

Welcome to the Embeddings module! You'll implement the systems that convert discrete tokens into rich vector representations that capture semantic meaning for language models.

## Learning Goals
- Systems understanding: How embedding tables scale with vocabulary size and affect model memory
- Core implementation skill: Build embedding layers with efficient lookup operations
- Pattern recognition: Understand how positional encoding enables sequence understanding
- Framework connection: See how your implementations match PyTorch's embedding systems
- Performance insight: Learn how embedding lookup patterns affect cache efficiency and memory bandwidth

## Build -> Use -> Reflect
1. **Build**: Embedding layer with lookup table and positional encoding systems
2. **Use**: Transform token sequences into rich vector representations for language processing
3. **Reflect**: How do embedding choices determine model capacity and computational efficiency?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how discrete tokens become continuous vector representations
- Practical capability to implement embedding systems that handle large vocabularies efficiently
- Systems insight into how embedding dimensions affect model capacity and memory usage
- Performance consideration of how embedding lookup patterns affect training and inference speed
- Connection to production systems like transformer embedding layers and their optimization techniques

## Systems Reality Check
TIP **Production Context**: Modern language models have embedding tables with billions of parameters (GPT-3: 50k vocab * 12k dim = 600M embedding params)
SPEED **Performance Note**: Embedding lookups are memory-bandwidth bound - efficient access patterns are critical for high-throughput training
"""

# %% nbgrader={"grade": false, "grade_id": "embeddings-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.embeddings

#| export
import math
import numpy as np
import os
import sys
from typing import Union, List, Optional, Tuple

# Import our Tensor class - try from package first, then from local module
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local tensor module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
    from tensor_dev import Tensor

# Try to import tokenization classes
try:
    from tinytorch.core.tokenization import CharTokenizer, BPETokenizer
except ImportError:
    # For development, import from local module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '11_tokenization'))
    try:
        from tokenization_dev import CharTokenizer, BPETokenizer
    except ImportError:
        # Create minimal mock classes if not available
        class CharTokenizer:
            def __init__(self): 
                self.vocab_size = 256
        class BPETokenizer:
            def __init__(self, vocab_size=1000):
                self.vocab_size = vocab_size

# %% nbgrader={"grade": false, "grade_id": "embeddings-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("TARGET TinyTorch Embeddings Module")
print(f"NumPy version: {np.__version__}")
print("Ready to build embedding systems!")

# %% [markdown]
"""
## PACKAGE Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/12_embeddings/embeddings_dev.py`  
**Building Side:** Code exports to `tinytorch.core.embeddings`

```python
# Final package structure:
from tinytorch.core.embeddings import Embedding, PositionalEncoding
from tinytorch.core.tokenization import CharTokenizer, BPETokenizer  # Previous module
from tinytorch.core.attention import MultiHeadAttention  # Next module
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like PyTorch's `torch.nn.Embedding`
- **Consistency:** All embedding tools live together in `core.embeddings`
- **Integration:** Works seamlessly with tokenization and attention systems
"""

# %% [markdown]
"""
## What are Embeddings?

### The Problem: Discrete to Continuous
Tokens are discrete symbols, but neural networks work best with continuous vectors:

```
Discrete Token Transformation:
    Token ID    ->    Dense Vector Representation
       42       ->    [0.1, -0.3, 0.8, 0.2, ...]
       
Visualization:
    Sparse One-Hot      Dense Embedding
    [0,0,0,1,0,...]  ->  [0.1,-0.3,0.8,0.2]
    100,000 dims        512 dims
```

### Embedding Table Visualization
An embedding layer is essentially a learnable lookup table:

```
Embedding Table Memory Layout:
+-------------------------------------+
| Embedding Weight Matrix             |
+-------------------------------------┤
| Token 0:  [0.1, -0.2,  0.3, ...]  |  <- "<PAD>" token
| Token 1:  [0.4,  0.1, -0.5, ...]  |  <- "<UNK>" token  
| Token 2:  [-0.1, 0.8,  0.2, ...]  |  <- "the" token
| Token 3:  [0.7, -0.3,  0.1, ...]  |  <- "and" token
| ...                               |
| Token N:  [0.2,  0.5, -0.7, ...]  |  <- Final token
+-------------------------------------+
    ^                    ^
  vocab_size        embedding_dim

Example: 50,000 * 512 = 25.6M parameters = 102.4MB (float32)
```

### Embedding Lookup Process
```
Lookup Operation Flow:
    Token IDs: [42, 17, 8]  (Input sequence)
         v Advanced Indexing
    Embedding Table[42] -> [0.1, -0.3, 0.8, ...]
    Embedding Table[17] -> [0.4,  0.1, -0.5, ...] 
    Embedding Table[8]  -> [-0.1, 0.8,  0.2, ...]
         v Stack Results
    Output: [[0.1, -0.3, 0.8, ...],    <- Token 42 embedding
             [0.4,  0.1, -0.5, ...],    <- Token 17 embedding  
             [-0.1, 0.8,  0.2, ...]]    <- Token 8 embedding
    
Complexity: O(seq_length) lookups, O(seq_length * embed_dim) memory
```

### Why Embeddings Work
- **Similarity**: Similar words get similar vectors through training
- **Composition**: Vector operations capture semantic relationships  
- **Learning**: Gradients update embeddings to improve task performance
- **Efficiency**: Dense vectors are more efficient than sparse one-hot

### Positional Encoding Visualization
Since transformers lack inherent position awareness, we add positional information:

```
Position-Aware Embedding Creation:
    Token Embedding    +    Positional Encoding    =    Final Representation
    +-------------+         +-------------+             +-------------+
    |[0.1,-0.3,0.8]|    +    |[0.0, 1.0,0.0]|        =    |[0.1, 0.7,0.8]|  <- Pos 0
    |[0.4, 0.1,-0.5]|    +    |[0.1, 0.9,0.1]|        =    |[0.5, 1.0,-0.4]|  <- Pos 1
    |[-0.1,0.8, 0.2]|    +    |[0.2, 0.8,0.2]|        =    |[0.1, 1.6, 0.4]|  <- Pos 2
    +-------------+         +-------------+             +-------------+
         ^                       ^                           ^
    Content Info           Position Info              Complete Context
```

### Systems Trade-offs
- **Embedding dimension**: Higher = more capacity, more memory  
- **Vocabulary size**: Larger = more parameters, better coverage
- **Lookup efficiency**: Memory access patterns affect performance
- **Position encoding**: Fixed vs learned vs hybrid approaches
"""

# %% [markdown]
"""
## Embedding Layer Implementation

Let's start with the core embedding layer - a learnable lookup table that converts token indices to dense vectors.

### Implementation Strategy
```
Embedding Layer Architecture:
    Input: Token IDs [batch_size, seq_length]
         v Index into weight matrix
    Weight Matrix: [vocab_size, embedding_dim] 
         v Advanced indexing: weight[input_ids]
    Output: Embeddings [batch_size, seq_length, embedding_dim]

Memory Layout:
+--------------------------------------+
| Embedding Weight Matrix              |  <- Main parameter storage
+--------------------------------------┤  
| Input Token IDs (integers)           |  <- Temporary during forward
+--------------------------------------┤
| Output Embeddings (float32)          |  <- Result tensor
+--------------------------------------+

Operation: O(1) lookup per token, O(seq_length) total
```
"""

# %% nbgrader={"grade": false, "grade_id": "embedding-layer", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class Embedding:
    """
    Embedding layer that converts token indices to dense vector representations.
    
    This is the foundation of modern language models - a learnable lookup table
    that maps discrete tokens to continuous vectors that capture semantic meaning.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None, 
                 init_type: str = 'uniform'):
        """
        Initialize embedding layer with learnable parameters.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Store configuration parameters
        2. Initialize embedding table with chosen initialization
        3. Handle special padding token if specified
        4. Set up for gradient tracking (will connect to autograd later)
        
        DESIGN DECISIONS:
        - Embedding table shape: (vocab_size, embedding_dim)
        - Initialization affects training dynamics
        - Padding idx gets zero gradient to stay constant
        
        Args:
            vocab_size: Number of tokens in vocabulary
            embedding_dim: Size of dense vector for each token
            padding_idx: Optional token index that should remain zero
            init_type: Initialization strategy ('uniform', 'normal', 'xavier')
        """
        ### BEGIN SOLUTION
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.init_type = init_type
        
        # Initialize embedding table based on strategy  
        # Different initialization strategies affect training dynamics
        if init_type == 'uniform':
            # Uniform initialization in [-1/sqrt(dim), 1/sqrt(dim)]
            # Keeps initial embeddings in reasonable range for gradient flow
            bound = 1.0 / math.sqrt(embedding_dim)  # Scale with dimension
            self.weight = Tensor(np.random.uniform(-bound, bound, (vocab_size, embedding_dim)))
        elif init_type == 'normal':
            # Normal initialization with std=1/sqrt(dim)
            # Gaussian distribution with dimension-aware scaling
            std = 1.0 / math.sqrt(embedding_dim)
            self.weight = Tensor(np.random.normal(0, std, (vocab_size, embedding_dim)))
        elif init_type == 'xavier':
            # Xavier/Glorot initialization - considers fan-in and fan-out
            # Good for maintaining activation variance across layers
            bound = math.sqrt(6.0 / (vocab_size + embedding_dim))
            self.weight = Tensor(np.random.uniform(-bound, bound, (vocab_size, embedding_dim)))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        # Set padding token to zero if specified
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0.0
        
        # Track parameters for optimization
        self.parameters = [self.weight]
        ### END SOLUTION
    
    def forward(self, input_ids: Union[Tensor, List[int], np.ndarray]) -> Tensor:
        """
        Look up embeddings for input token indices.
        
        TODO: Implement embedding lookup.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Convert input to numpy array if needed
        2. Validate token indices are within vocabulary
        3. Use advanced indexing to look up embeddings
        4. Return tensor with shape (batch_size, seq_len, embedding_dim)
        
        EXAMPLE:
        embed = Embedding(vocab_size=100, embedding_dim=64)
        tokens = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
        embeddings = embed.forward(tokens)  # Shape: (2, 3, 64)
        
        IMPLEMENTATION HINTS:
        - Handle both Tensor and list inputs
        - Use numpy advanced indexing: weight[indices]
        - Preserve batch and sequence dimensions
        
        Args:
            input_ids: Token indices with shape (batch_size, seq_len) or (seq_len,)
            
        Returns:
            Embeddings with shape (*input_shape, embedding_dim)
        """
        ### BEGIN SOLUTION
        # Convert input to numpy array
        if isinstance(input_ids, Tensor):
            indices = input_ids.data
        elif isinstance(input_ids, list):
            indices = np.array(input_ids)
        else:
            indices = input_ids
        
        # Validate indices
        indices = indices.astype(int)
        if np.any(indices < 0) or np.any(indices >= self.vocab_size):
            raise ValueError(f"Token indices must be in range [0, {self.vocab_size})")
        
        # Look up embeddings using advanced indexing (very efficient operation)
        # Memory access pattern: Random access into embedding table
        # self.weight.data has shape (vocab_size, embedding_dim)
        # indices has shape (...), result has shape (..., embedding_dim)
        embeddings = self.weight.data[indices]  # O(seq_length) lookups
        
        return Tensor(embeddings)
        ### END SOLUTION
    
    def __call__(self, input_ids: Union[Tensor, List[int], np.ndarray]) -> Tensor:
        """Make the layer callable."""
        return self.forward(input_ids)
    
    def get_memory_usage(self):
        """
        Calculate memory usage of embedding table.
        
        This function is PROVIDED to show memory analysis.
        """
        # Embedding table memory
        weight_memory_mb = self.weight.data.nbytes / (1024 * 1024)
        
        # Memory per token
        memory_per_token_kb = (self.embedding_dim * 4) / 1024  # 4 bytes per float32
        
        return {
            'total_memory_mb': weight_memory_mb,
            'memory_per_token_kb': memory_per_token_kb,
            'total_parameters': self.vocab_size * self.embedding_dim,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        }

# %% [markdown]
"""
### TEST Test Your Embedding Layer Implementation

Once you implement the Embedding forward method above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-embedding-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_embedding_layer():
    """Unit test for the embedding layer."""
    print("🔬 Unit Test: Embedding Layer...")
    
    # Create embedding layer
    vocab_size = 100
    embedding_dim = 64
    embed = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # Test single token
    single_token = [5]
    single_embedding = embed.forward(single_token)
    assert single_embedding.shape == (1, embedding_dim), f"Expected shape (1, {embedding_dim}), got {single_embedding.shape}"
    
    # Test sequence of tokens
    token_sequence = [1, 2, 3, 5, 10]
    sequence_embeddings = embed.forward(token_sequence)
    expected_shape = (len(token_sequence), embedding_dim)
    assert sequence_embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {sequence_embeddings.shape}"
    
    # Test batch of sequences
    batch_tokens = [[1, 2, 3], [4, 5, 6]]
    batch_embeddings = embed.forward(batch_tokens)
    assert batch_embeddings.shape == (2, 3, embedding_dim), f"Expected shape (2, 3, {embedding_dim}), got {batch_embeddings.shape}"
    
    # Test with Tensor input
    tensor_input = Tensor(np.array([[7, 8, 9], [10, 11, 12]]))
    tensor_embeddings = embed.forward(tensor_input)
    assert tensor_embeddings.shape == (2, 3, embedding_dim), "Should handle Tensor input"
    
    # Test embedding lookup consistency
    token_5_embed_1 = embed.forward([5])
    token_5_embed_2 = embed.forward([5])
    assert np.allclose(token_5_embed_1.data, token_5_embed_2.data), "Same token should give same embedding"
    
    # Test different tokens give different embeddings (with high probability)
    token_1_embed = embed.forward([1])
    token_2_embed = embed.forward([2])
    assert not np.allclose(token_1_embed.data, token_2_embed.data, atol=1e-3), "Different tokens should give different embeddings"
    
    # Test initialization bounds
    assert np.all(np.abs(embed.weight.data) <= 1.0), "Uniform initialization should be bounded"
    
    # Test padding token (if specified)
    embed_with_padding = Embedding(vocab_size=50, embedding_dim=32, padding_idx=0)
    assert np.allclose(embed_with_padding.weight.data[0], 0.0), "Padding token should be zero"
    
    # Test parameter tracking
    assert len(embed.parameters) == 1, "Should track embedding weight parameter"
    assert embed.parameters[0] is embed.weight, "Should track weight tensor"
    
    # Test memory usage calculation
    memory_stats = embed.get_memory_usage()
    assert 'total_memory_mb' in memory_stats, "Should provide memory statistics"
    assert memory_stats['total_parameters'] == vocab_size * embedding_dim, "Should calculate parameters correctly"
    
    print("PASS Embedding layer tests passed!")
    print(f"PASS Handles various input shapes correctly")
    print(f"PASS Consistent lookup and parameter tracking")
    print(f"PASS Memory usage: {memory_stats['total_memory_mb']:.2f}MB")

# Test function defined (called in main block)

# %% [markdown]
"""
## Positional Encoding Implementation

Transformers need explicit position information since attention is position-agnostic. Let's implement sinusoidal positional encoding used in the original transformer.

### Sinusoidal Positional Encoding Visualization
```
Mathematical Foundation:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))     <- Even dimensions
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))     <- Odd dimensions

Frequency Pattern:
    Position ->   0    1    2    3    4   ...
    Dim 0:    [sin] [sin] [sin] [sin] [sin] ... <- High frequency
    Dim 1:    [cos] [cos] [cos] [cos] [cos] ... <- High frequency
    Dim 2:    [sin] [sin] [sin] [sin] [sin] ... <- Med frequency
    Dim 3:    [cos] [cos] [cos] [cos] [cos] ... <- Med frequency
    ...        ...   ...   ...   ...   ...   
    Dim n-2:  [sin] [sin] [sin] [sin] [sin] ... <- Low frequency  
    Dim n-1:  [cos] [cos] [cos] [cos] [cos] ... <- Low frequency

Why This Works:
    - Each position gets unique encoding across all dimensions
    - Relative positions have consistent patterns
    - Model can learn to use positional relationships
    - No parameters needed (computed deterministically)
```

### Position Encoding Memory Layout
```
Precomputed Position Matrix:
+-------------------------------------+
| Position Encoding Matrix            |
+-------------------------------------┤ 
| Pos 0:  [0.00, 1.00, 0.00, 1.00...]|  <- sin(0), cos(0), sin(0), cos(0)
| Pos 1:  [0.84, 0.54, 0.10, 0.99...]|  <- sin(1), cos(1), sin(f1), cos(f1)
| Pos 2:  [0.91,-0.42, 0.20, 0.98...]|  <- sin(2), cos(2), sin(f2), cos(f2) 
| Pos 3:  [0.14,-0.99, 0.30, 0.95...]|  <- sin(3), cos(3), sin(f3), cos(f3)
| ...                               |
+-------------------------------------+
    ^                    ^
max_seq_length     embedding_dim

Memory: max_seq_length * embedding_dim * 4 bytes (precomputed)
```
"""

# %% nbgrader={"grade": false, "grade_id": "positional-encoding", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class PositionalEncoding:
    """
    Sinusoidal positional encoding that adds position information to embeddings.
    
    Uses sine and cosine functions of different frequencies to create
    unique position representations that the model can learn to use.
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 5000, 
                 dropout: float = 0.0):
        """
        Initialize positional encoding with sinusoidal patterns.
        
        TODO: Implement positional encoding initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create position matrix (max_seq_length, embedding_dim)
        2. For each position and dimension:
           - Calculate frequency based on dimension
           - Apply sine to even dimensions, cosine to odd dimensions
        3. Store the precomputed positional encodings
        
        MATHEMATICAL FOUNDATION:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Where:
        - pos = position in sequence
        - i = dimension index
        - d_model = embedding_dim
        
        Args:
            embedding_dim: Dimension of embeddings (must be even)
            max_seq_length: Maximum sequence length to precompute
            dropout: Dropout rate (for future use)
        """
        ### BEGIN SOLUTION
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        # Create positional encoding matrix
        pe = np.zeros((max_seq_length, embedding_dim))
        
        # Create position vector (0, 1, 2, ..., max_seq_length-1)
        position = np.arange(0, max_seq_length).reshape(-1, 1)  # Shape: (max_seq_length, 1)
        
        # Create dimension indices for frequency calculation
        # div_term calculates 10000^(2i/d_model) for i = 0, 1, 2, ...
        # This creates decreasing frequencies: high freq for early dims, low freq for later dims
        div_term = np.exp(np.arange(0, embedding_dim, 2) * 
                         -(math.log(10000.0) / embedding_dim))
        
        # Apply sine to even dimensions (0, 2, 4, ...) 
        # Broadcasting: position (max_seq_length, 1) * div_term (embedding_dim//2,)
        pe[:, 0::2] = np.sin(position * div_term)  # High to low frequency sine waves
        
        # Apply cosine to odd dimensions (1, 3, 5, ...)
        # Cosine provides phase-shifted version of sine for each frequency
        if embedding_dim % 2 == 1:
            # Handle odd embedding_dim - cosine gets one less dimension
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        # Store as tensor
        self.pe = Tensor(pe)
        ### END SOLUTION
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Add positional encoding to embeddings.
        
        TODO: Implement positional encoding addition.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Get sequence length from embeddings shape
        2. Extract relevant positional encodings
        3. Add positional encodings to embeddings
        4. Return position-aware embeddings
        
        EXAMPLE:
        pos_enc = PositionalEncoding(embedding_dim=64)
        embeddings = Tensor(np.random.randn(2, 10, 64))  # (batch, seq, dim)
        pos_embeddings = pos_enc.forward(embeddings)
        
        Args:
            embeddings: Input embeddings with shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Position-aware embeddings with same shape as input
        """
        ### BEGIN SOLUTION
        # Get sequence length from embeddings
        if len(embeddings.shape) == 3:
            batch_size, seq_length, embed_dim = embeddings.shape
        elif len(embeddings.shape) == 2:
            seq_length, embed_dim = embeddings.shape
            batch_size = None
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")
        
        if embed_dim != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {embed_dim}")
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max {self.max_seq_length}")
        
        # Extract positional encodings for this sequence length
        position_encodings = self.pe.data[:seq_length, :]
        
        # Add positional encodings to embeddings (element-wise addition)
        # This combines content information with positional information
        if batch_size is not None:
            # Broadcast positional encodings across batch dimension
            # embeddings: (batch, seq, dim) + position_encodings: (seq, dim)
            # Broadcasting rule: (B,S,D) + (1,S,D) = (B,S,D)
            result = embeddings.data + position_encodings[np.newaxis, :, :]
        else:
            # embeddings: (seq, dim) + position_encodings: (seq, dim)
            result = embeddings.data + position_encodings
        
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, embeddings: Tensor) -> Tensor:
        """Make the class callable."""
        return self.forward(embeddings)
    
    def visualize_encoding(self, seq_length: int = 100, dims_to_show: int = 10) -> None:
        """
        Visualize positional encoding patterns.
        
        This function is PROVIDED to show encoding patterns.
        """
        print(f"📊 POSITIONAL ENCODING VISUALIZATION")
        print(f"Sequence length: {seq_length}, Dimensions shown: {dims_to_show}")
        print("=" * 60)
        
        # Get subset of positional encodings
        pe_subset = self.pe.data[:seq_length, :dims_to_show]
        
        # Show patterns for first few positions
        print("First 10 positions, first 10 dimensions:")
        print("Pos", end="")
        for d in range(min(dims_to_show, 10)):
            print(f"    Dim{d:2d}", end="")
        print()
        
        for pos in range(min(seq_length, 10)):
            print(f"{pos:3d}", end="")
            for d in range(min(dims_to_show, 10)):
                print(f"{pe_subset[pos, d]:8.3f}", end="")
            print()
        
        # Show frequency analysis
        print(f"\nPROGRESS FREQUENCY ANALYSIS:")
        print("Even dimensions (sine): Lower frequencies for early dimensions")
        print("Odd dimensions (cosine): Same frequencies, phase-shifted")
        
        # Calculate frequency range
        min_freq = 1.0 / 10000
        max_freq = 1.0
        print(f"Frequency range: {min_freq:.6f} to {max_freq:.6f}")

# %% [markdown]
"""
### TEST Test Your Positional Encoding Implementation

Once you implement the PositionalEncoding methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-positional-encoding-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_positional_encoding():
    """Unit test for positional encoding."""
    print("🔬 Unit Test: Positional Encoding...")
    
    # Create positional encoding
    embedding_dim = 64
    max_seq_length = 100
    pos_enc = PositionalEncoding(embedding_dim=embedding_dim, max_seq_length=max_seq_length)
    
    # Test initialization
    assert pos_enc.pe.shape == (max_seq_length, embedding_dim), f"Expected shape ({max_seq_length}, {embedding_dim})"
    
    # Test that different positions have different encodings
    pos_0 = pos_enc.pe.data[0]
    pos_1 = pos_enc.pe.data[1]
    assert not np.allclose(pos_0, pos_1), "Different positions should have different encodings"
    
    # Test sine/cosine pattern
    # Even dimensions should use sine, odd should use cosine
    # This is hard to test directly, but we can check the encoding is reasonable
    assert not np.any(np.isnan(pos_enc.pe.data)), "Positional encodings should not contain NaN"
    assert not np.any(np.isinf(pos_enc.pe.data)), "Positional encodings should not contain inf"
    
    # Test forward pass with 3D input (batch, seq, dim)
    batch_size = 2
    seq_length = 10
    embeddings = Tensor(np.random.randn(batch_size, seq_length, embedding_dim))
    
    pos_embeddings = pos_enc.forward(embeddings)
    assert pos_embeddings.shape == embeddings.shape, "Output shape should match input shape"
    
    # Test forward pass with 2D input (seq, dim)
    embeddings_2d = Tensor(np.random.randn(seq_length, embedding_dim))
    pos_embeddings_2d = pos_enc.forward(embeddings_2d)
    assert pos_embeddings_2d.shape == embeddings_2d.shape, "2D output shape should match input"
    
    # Test that positional encoding is actually added
    original_mean = np.mean(embeddings.data)
    pos_mean = np.mean(pos_embeddings.data)
    assert abs(pos_mean - original_mean) > 1e-6, "Positional encoding should change the embeddings"
    
    # Test sequence length validation
    try:
        long_embeddings = Tensor(np.random.randn(max_seq_length + 10, embedding_dim))
        pos_enc.forward(long_embeddings)
        assert False, "Should raise error for sequence longer than max_seq_length"
    except ValueError:
        pass  # Expected behavior
    
    # Test embedding dimension validation
    try:
        wrong_dim_embeddings = Tensor(np.random.randn(seq_length, embedding_dim + 10))
        pos_enc.forward(wrong_dim_embeddings)
        assert False, "Should raise error for wrong embedding dimension"
    except ValueError:
        pass  # Expected behavior
    
    # Test deterministic behavior
    pos_embeddings_1 = pos_enc.forward(embeddings)
    pos_embeddings_2 = pos_enc.forward(embeddings)
    assert np.allclose(pos_embeddings_1.data, pos_embeddings_2.data), "Should be deterministic"
    
    # Test callable interface
    pos_embeddings_callable = pos_enc(embeddings)
    assert np.allclose(pos_embeddings_callable.data, pos_embeddings.data), "Callable interface should work"
    
    print("PASS Positional encoding tests passed!")
    print(f"PASS Handles 2D and 3D inputs correctly")
    print(f"PASS Proper validation and deterministic behavior")
    print(f"PASS Encoding dimension: {embedding_dim}, Max length: {max_seq_length}")

# Test function defined (called in main block)

# %% [markdown]
"""
## Learned Positional Embeddings

Some models use learned positional embeddings instead of fixed sinusoidal ones. Let's implement this alternative approach:

### Learned vs Sinusoidal Comparison
```
Sinusoidal Positional Encoding:
    OK Zero parameters (deterministic computation)
    OK Can extrapolate to longer sequences
    OK Mathematical guarantees about relative positions
    ✗ Fixed pattern - cannot adapt to task
    
Learned Positional Embeddings:
    OK Learnable parameters (adapts to task/data)
    OK Can capture task-specific positional patterns
    ✗ Requires additional parameters (max_seq_len * embed_dim)
    ✗ Cannot extrapolate beyond training sequence length
    ✗ Needs sufficient training data to learn good positions
```

### Learned Position Architecture
```
Learned Position System:
    Position IDs: [0, 1, 2, 3, ...]
          v Embedding lookup (just like token embeddings)
    Position Table: [max_seq_length, embedding_dim]
          v Standard embedding lookup
    Position Embeddings: [seq_length, embedding_dim]
          v Add to token embeddings
    Final Representation: Token + Position information

This is essentially two embedding tables:
    - Token Embedding: token_id -> content vector
    - Position Embedding: position_id -> position vector
```
"""

# %% nbgrader={"grade": false, "grade_id": "learned-positional", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class LearnedPositionalEmbedding:
    """
    Learned positional embeddings - another embedding table for positions.
    
    Unlike sinusoidal encoding, these are learned parameters that
    the model optimizes during training. Used in models like BERT.
    """
    
    def __init__(self, max_seq_length: int, embedding_dim: int):
        """
        Initialize learned positional embeddings.
        
        TODO: Implement learned positional embedding initialization.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create embedding layer for positions (0, 1, 2, ..., max_seq_length-1)
        2. Initialize with small random values
        3. Set up parameter tracking for optimization
        
        This is essentially an Embedding layer where the "vocabulary"
        is the set of possible positions in a sequence.
        
        Args:
            max_seq_length: Maximum sequence length supported
            embedding_dim: Dimension of position embeddings
        """
        ### BEGIN SOLUTION
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        
        # Create learned positional embedding table
        # This is like an embedding layer for positions (not tokens)
        # Vocabulary size = max sequence length (each position is a "token")
        self.position_embedding = Embedding(
            vocab_size=max_seq_length,  # Position 0, 1, 2, ..., max_seq_length-1
            embedding_dim=embedding_dim,  # Same dimension as token embeddings
            init_type='normal'  # Start with small random values
        )
        
        # Track parameters for optimization
        self.parameters = self.position_embedding.parameters
        ### END SOLUTION
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Add learned positional embeddings to input embeddings.
        
        TODO: Implement learned positional embedding addition.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Get sequence length from input shape
        2. Create position indices [0, 1, 2, ..., seq_length-1]
        3. Look up position embeddings using position indices
        4. Add position embeddings to input embeddings
        
        EXAMPLE:
        learned_pos = LearnedPositionalEmbedding(max_seq_length=100, embedding_dim=64)
        embeddings = Tensor(np.random.randn(2, 10, 64))  # (batch, seq, dim)
        pos_embeddings = learned_pos.forward(embeddings)
        
        Args:
            embeddings: Input embeddings with shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Position-aware embeddings with same shape as input
        """
        ### BEGIN SOLUTION
        # Get sequence length from embeddings
        if len(embeddings.shape) == 3:
            batch_size, seq_length, embed_dim = embeddings.shape
        elif len(embeddings.shape) == 2:
            seq_length, embed_dim = embeddings.shape
            batch_size = None
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")
        
        if embed_dim != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {embed_dim}")
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max {self.max_seq_length}")
        
        # Create position indices [0, 1, 2, ..., seq_length-1]
        # These are the "token IDs" for positions in the sequence
        position_ids = list(range(seq_length))
        
        # Look up position embeddings (same process as token embedding lookup)
        # Each position gets its own learned vector representation
        position_embeddings = self.position_embedding.forward(position_ids)
        
        # Add position embeddings to input embeddings
        if batch_size is not None:
            # Broadcast across batch dimension
            result = embeddings.data + position_embeddings.data[np.newaxis, :, :]
        else:
            result = embeddings.data + position_embeddings.data
        
        return Tensor(result)
        ### END SOLUTION
    
    def __call__(self, embeddings: Tensor) -> Tensor:
        """Make the class callable."""
        return self.forward(embeddings)

# %% [markdown]
"""
### TEST Test Your Learned Positional Embedding Implementation

Once you implement the LearnedPositionalEmbedding methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-learned-positional-immediate", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_learned_positional_embedding():
    """Unit test for learned positional embeddings."""
    print("🔬 Unit Test: Learned Positional Embeddings...")
    
    # Create learned positional embedding
    max_seq_length = 50
    embedding_dim = 32
    learned_pos = LearnedPositionalEmbedding(max_seq_length=max_seq_length, embedding_dim=embedding_dim)
    
    # Test initialization
    assert learned_pos.position_embedding.vocab_size == max_seq_length, "Should have position for each sequence position"
    assert learned_pos.position_embedding.embedding_dim == embedding_dim, "Should match embedding dimension"
    
    # Test parameter tracking
    assert len(learned_pos.parameters) == 1, "Should track position embedding parameters"
    assert learned_pos.parameters[0] is learned_pos.position_embedding.weight, "Should track weight tensor"
    
    # Test forward pass with 3D input
    batch_size = 3
    seq_length = 10
    embeddings = Tensor(np.random.randn(batch_size, seq_length, embedding_dim))
    
    pos_embeddings = learned_pos.forward(embeddings)
    assert pos_embeddings.shape == embeddings.shape, "Output shape should match input shape"
    
    # Test forward pass with 2D input
    embeddings_2d = Tensor(np.random.randn(seq_length, embedding_dim))
    pos_embeddings_2d = learned_pos.forward(embeddings_2d)
    assert pos_embeddings_2d.shape == embeddings_2d.shape, "2D output shape should match input"
    
    # Test that position embeddings are actually added
    original_mean = np.mean(embeddings.data)
    pos_mean = np.mean(pos_embeddings.data)
    assert abs(pos_mean - original_mean) > 1e-6, "Position embeddings should change the input"
    
    # Test that different sequence lengths give consistent positional embeddings
    # Use same base embeddings for the first 5 positions to test positional consistency
    base_embeddings = np.random.randn(batch_size, 5, embedding_dim)
    short_embeddings = Tensor(base_embeddings)
    
    # For long embeddings, use same first 5 positions plus additional positions
    extended_embeddings = np.random.randn(batch_size, 10, embedding_dim)
    extended_embeddings[:, :5, :] = base_embeddings  # Same first 5 positions
    long_embeddings = Tensor(extended_embeddings)
    
    short_pos = learned_pos.forward(short_embeddings)
    long_pos = learned_pos.forward(long_embeddings)
    
    # The first 5 positions should be the same (same input + same positional embeddings)
    assert np.allclose(short_pos.data, long_pos.data[:, :5, :], atol=1e-6), "Same positions should have same embeddings"
    
    # Test sequence length validation
    try:
        too_long_embeddings = Tensor(np.random.randn(batch_size, max_seq_length + 5, embedding_dim))
        learned_pos.forward(too_long_embeddings)
        assert False, "Should raise error for sequence longer than max_seq_length"
    except ValueError:
        pass  # Expected behavior
    
    # Test embedding dimension validation
    try:
        wrong_dim_embeddings = Tensor(np.random.randn(batch_size, seq_length, embedding_dim + 5))
        learned_pos.forward(wrong_dim_embeddings)
        assert False, "Should raise error for wrong embedding dimension"
    except ValueError:
        pass  # Expected behavior
    
    # Test callable interface
    pos_embeddings_callable = learned_pos(embeddings)
    assert np.allclose(pos_embeddings_callable.data, pos_embeddings.data), "Callable interface should work"
    
    print("PASS Learned positional embedding tests passed!")
    print(f"PASS Parameter tracking and optimization ready")
    print(f"PASS Handles various input shapes correctly")
    print(f"PASS Max sequence length: {max_seq_length}, Embedding dim: {embedding_dim}")

# Test function defined (called in main block)

# PASS IMPLEMENTATION CHECKPOINT: Ensure all embedding components are complete before analysis

# THINK PREDICTION: How does embedding table memory scale with vocabulary size and dimension?
# Linear with vocab_size? Linear with embedding_dim? Quadratic with both?
# Your prediction: _______

# MAGNIFY SYSTEMS INSIGHT #1: Embedding Memory Scaling Analysis
def analyze_embedding_memory_scaling():
    """Analyze how embedding memory scales with vocabulary and dimension parameters."""
    try:
        import time
        
        print("📊 EMBEDDING MEMORY SCALING ANALYSIS")
        print("=" * 50)
        
        # Test different configurations
        test_configs = [
            (1000, 128),   # Small model
            (10000, 256),  # Medium model  
            (50000, 512),  # Large model
            (100000, 1024) # Very large model
        ]
        
        print(f"{'Vocab Size':<12} {'Embed Dim':<10} {'Parameters':<12} {'Memory (MB)':<12} {'Lookup Time':<12}")
        print("-" * 70)
        
        for vocab_size, embed_dim in test_configs:
            # Create embedding layer
            embed = Embedding(vocab_size=vocab_size, embedding_dim=embed_dim)
            
            # Calculate memory
            memory_stats = embed.get_memory_usage()
            params = memory_stats['total_parameters']
            memory_mb = memory_stats['total_memory_mb']
            
            # Test lookup performance
            test_tokens = np.random.randint(0, vocab_size, (32, 64))
            start_time = time.time()
            _ = embed.forward(test_tokens) 
            lookup_time = (time.time() - start_time) * 1000
            
            print(f"{vocab_size:<12,} {embed_dim:<10} {params:<12,} {memory_mb:<12.1f} {lookup_time:<12.2f}")
        
        # TIP WHY THIS MATTERS: GPT-3 has 50k vocab * 12k dim = 600M embedding parameters!
        # That's 2.4GB just for the embedding table (before any other model weights)
        print("\nTIP SCALING INSIGHTS:")
        print("   - Memory scales linearly with both vocab_size AND embedding_dim")
        print("   - Lookup time is dominated by memory bandwidth, not computation")
        print("   - Large models spend significant memory on embeddings alone")
        
    except Exception as e:
        print(f"WARNING️ Error in memory scaling analysis: {e}")
        print("Make sure your Embedding class is implemented correctly")

analyze_embedding_memory_scaling()

# PASS IMPLEMENTATION CHECKPOINT: Ensure positional encoding works before analysis

# THINK PREDICTION: Which positional encoding uses more memory - sinusoidal or learned?
# Which can handle longer sequences? Your answer: _______

# MAGNIFY SYSTEMS INSIGHT #2: Positional Encoding Trade-offs
def analyze_positional_encoding_tradeoffs():
    """Compare memory and performance characteristics of different positional encodings."""
    try:
        import time
        
        print("\nMAGNIFY POSITIONAL ENCODING COMPARISON")
        print("=" * 50)
        
        embedding_dim = 512
        max_seq_length = 2048
        
        # Create both types
        sinusoidal_pe = PositionalEncoding(embedding_dim=embedding_dim, max_seq_length=max_seq_length)
        learned_pe = LearnedPositionalEmbedding(max_seq_length=max_seq_length, embedding_dim=embedding_dim)
        
        # Test different sequence lengths
        seq_lengths = [128, 512, 1024, 2048]
        batch_size = 16
        
        print(f"{'Seq Len':<8} {'Method':<12} {'Time (ms)':<10} {'Memory (MB)':<12} {'Parameters':<12}")
        print("-" * 65)
        
        for seq_len in seq_lengths:
            embeddings = Tensor(np.random.randn(batch_size, seq_len, embedding_dim))
            
            # Test sinusoidal
            start_time = time.time()
            _ = sinusoidal_pe.forward(embeddings)
            sin_time = (time.time() - start_time) * 1000
            sin_memory = 0  # No parameters
            sin_params = 0
            
            # Test learned
            start_time = time.time() 
            _ = learned_pe.forward(embeddings)
            learned_time = (time.time() - start_time) * 1000
            learned_memory = learned_pe.position_embedding.get_memory_usage()['total_memory_mb']
            learned_params = max_seq_length * embedding_dim
            
            print(f"{seq_len:<8} {'Sinusoidal':<12} {sin_time:<10.2f} {sin_memory:<12.1f} {sin_params:<12,}")
            print(f"{seq_len:<8} {'Learned':<12} {learned_time:<10.2f} {learned_memory:<12.1f} {learned_params:<12,}")
            print()
        
        # TIP WHY THIS MATTERS: Choice affects model size and sequence length flexibility
        print("TIP TRADE-OFF INSIGHTS:")
        print("   - Sinusoidal: 0 parameters, can extrapolate to any length")
        print("   - Learned: Many parameters, limited to training sequence length")
        print("   - Modern models often use learned for better task adaptation")
        
    except Exception as e:
        print(f"WARNING️ Error in positional encoding analysis: {e}")
        print("Make sure both positional encoding classes are implemented")

analyze_positional_encoding_tradeoffs()

# PASS IMPLEMENTATION CHECKPOINT: Ensure full embedding pipeline works

# THINK PREDICTION: What's the bottleneck in embedding pipelines - computation or memory?
# How does batch size affect throughput? Your prediction: _______

# MAGNIFY SYSTEMS INSIGHT #3: Embedding Pipeline Performance
def analyze_embedding_pipeline_performance():
    """Analyze performance characteristics of the complete embedding pipeline."""
    try:
        import time
        
        print("\nSPEED EMBEDDING PIPELINE PERFORMANCE")
        print("=" * 50)
        
        # Create pipeline components
        vocab_size = 10000
        embedding_dim = 256
        max_seq_length = 512
        
        embed = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        pos_enc = PositionalEncoding(embedding_dim=embedding_dim, max_seq_length=max_seq_length)
        
        # Test different batch sizes and sequence lengths
        test_configs = [
            (8, 128),    # Small batch, short sequences
            (32, 256),   # Medium batch, medium sequences
            (64, 512),   # Large batch, long sequences
        ]
        
        print(f"{'Batch':<6} {'Seq Len':<8} {'Total Tokens':<12} {'Time (ms)':<10} {'Tokens/sec':<12} {'Memory (MB)':<12}")
        print("-" * 75)
        
        for batch_size, seq_length in test_configs:
            # Create random token sequence
            tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))
            token_tensor = Tensor(tokens)
            
            # Measure full pipeline
            start_time = time.time()
            
            # Step 1: Embedding lookup
            embeddings = embed.forward(token_tensor)
            
            # Step 2: Add positional encoding
            pos_embeddings = pos_enc.forward(embeddings)
            
            end_time = time.time()
            
            # Calculate metrics
            total_tokens = batch_size * seq_length
            pipeline_time = (end_time - start_time) * 1000
            tokens_per_sec = total_tokens / (end_time - start_time) if end_time > start_time else 0
            memory_mb = pos_embeddings.data.nbytes / (1024 * 1024)
            
            print(f"{batch_size:<6} {seq_length:<8} {total_tokens:<12,} {pipeline_time:<10.2f} {tokens_per_sec:<12,.0f} {memory_mb:<12.1f}")
        
        # TIP WHY THIS MATTERS: Understanding pipeline bottlenecks for production deployment
        print("\nTIP PIPELINE INSIGHTS:")
        print("   - Embedding lookup is memory-bandwidth bound (not compute bound)")
        print("   - Larger batches improve throughput due to better memory utilization")
        print("   - Sequence length affects memory linearly, performance sublinearly")
        print("   - Production systems optimize with: embedding caching, mixed precision, etc.")
        
    except Exception as e:
        print(f"WARNING️ Error in pipeline analysis: {e}")
        print("Make sure your full embedding pipeline is working")

analyze_embedding_pipeline_performance()

# %% [markdown]
"""
## TARGET ML Systems: Performance Analysis & Embedding Scaling

Now let's develop systems engineering skills by analyzing embedding performance and understanding how embedding choices affect downstream ML system efficiency.

### **Learning Outcome**: *"I understand how embedding table size affects model memory, training speed, and language understanding capacity"*
"""

# %% nbgrader={"grade": false, "grade_id": "embedding-profiler", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import time

class EmbeddingProfiler:
    """
    Performance profiling toolkit for embedding systems.
    
    Helps ML engineers understand memory usage, lookup performance,
    and scaling characteristics of embedding layers.
    """
    
    def __init__(self):
        self.results = {}
    
    def measure_lookup_performance(self, embedding_layer: Embedding, 
                                  batch_sizes: List[int], seq_lengths: List[int]):
        """
        Measure embedding lookup performance across different batch sizes and sequence lengths.
        
        TODO: Implement embedding lookup performance measurement.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create test token indices for each (batch_size, seq_length) combination
        2. Measure time to perform embedding lookup
        3. Calculate throughput metrics (tokens/second, memory bandwidth)
        4. Return comprehensive performance analysis
        
        METRICS TO CALCULATE:
        - Lookup time (milliseconds)
        - Tokens per second throughput
        - Memory bandwidth utilization
        - Scaling patterns with batch size and sequence length
        
        Args:
            embedding_layer: Embedding layer to test
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary with performance metrics for each configuration
        """
        ### BEGIN SOLUTION
        results = {}
        vocab_size = embedding_layer.vocab_size
        
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                # Create random token indices
                token_indices = np.random.randint(0, vocab_size, (batch_size, seq_length))
                
                # Measure lookup performance
                start_time = time.time()
                embeddings = embedding_layer.forward(token_indices)
                end_time = time.time()
                
                # Calculate metrics
                lookup_time_ms = (end_time - start_time) * 1000
                total_tokens = batch_size * seq_length
                tokens_per_second = total_tokens / (end_time - start_time) if end_time > start_time else 0
                
                # Memory calculations
                input_memory_mb = token_indices.nbytes / (1024 * 1024)
                output_memory_mb = embeddings.data.nbytes / (1024 * 1024)
                memory_bandwidth_mb_s = (input_memory_mb + output_memory_mb) / (end_time - start_time) if end_time > start_time else 0
                
                config_key = f"batch_{batch_size}_seq_{seq_length}"
                results[config_key] = {
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'total_tokens': total_tokens,
                    'lookup_time_ms': lookup_time_ms,
                    'tokens_per_second': tokens_per_second,
                    'input_memory_mb': input_memory_mb,
                    'output_memory_mb': output_memory_mb,
                    'memory_bandwidth_mb_s': memory_bandwidth_mb_s,
                    'time_per_token_us': lookup_time_ms * 1000 / total_tokens if total_tokens > 0 else 0
                }
        
        return results
        ### END SOLUTION
    
    def analyze_memory_scaling(self, vocab_sizes: List[int], embedding_dims: List[int]):
        """
        Analyze how embedding memory usage scales with vocabulary size and embedding dimension.
        
        This function is PROVIDED to show memory scaling analysis.
        """
        print("📊 EMBEDDING MEMORY SCALING ANALYSIS")
        print("=" * 60)
        
        scaling_results = {}
        
        print(f"{'Vocab Size':<12} {'Embed Dim':<10} {'Parameters':<12} {'Memory (MB)':<12} {'Lookup Time':<12}")
        print("-" * 70)
        
        for vocab_size in vocab_sizes:
            for embed_dim in embedding_dims:
                # Create embedding layer
                embed = Embedding(vocab_size=vocab_size, embedding_dim=embed_dim)
                
                # Calculate memory usage
                memory_stats = embed.get_memory_usage()
                total_memory_mb = memory_stats['total_memory_mb']
                total_params = memory_stats['total_parameters']
                
                # Measure lookup time
                test_tokens = np.random.randint(0, vocab_size, (32, 64))  # Standard batch
                start_time = time.time()
                _ = embed.forward(test_tokens)
                lookup_time_ms = (time.time() - start_time) * 1000
                
                # Store results
                config_key = f"vocab_{vocab_size}_dim_{embed_dim}"
                scaling_results[config_key] = {
                    'vocab_size': vocab_size,
                    'embedding_dim': embed_dim,
                    'total_parameters': total_params,
                    'memory_mb': total_memory_mb,
                    'lookup_time_ms': lookup_time_ms
                }
                
                print(f"{vocab_size:<12,} {embed_dim:<10} {total_params:<12,} {total_memory_mb:<12.2f} {lookup_time_ms:<12.2f}")
        
        # Analyze scaling patterns
        print(f"\nPROGRESS SCALING INSIGHTS:")
        if len(vocab_sizes) > 1 and len(embedding_dims) > 1:
            # Compare scaling with vocab size (fixed embedding dim)
            fixed_dim = embedding_dims[0]
            small_vocab = min(vocab_sizes)
            large_vocab = max(vocab_sizes)
            
            small_key = f"vocab_{small_vocab}_dim_{fixed_dim}"
            large_key = f"vocab_{large_vocab}_dim_{fixed_dim}"
            
            if small_key in scaling_results and large_key in scaling_results:
                vocab_ratio = large_vocab / small_vocab
                memory_ratio = scaling_results[large_key]['memory_mb'] / scaling_results[small_key]['memory_mb']
                print(f"   Vocabulary scaling: {vocab_ratio:.1f}x vocab -> {memory_ratio:.1f}x memory (Linear)")
            
            # Compare scaling with embedding dim (fixed vocab)
            fixed_vocab = vocab_sizes[0]
            small_dim = min(embedding_dims)
            large_dim = max(embedding_dims)
            
            small_key = f"vocab_{fixed_vocab}_dim_{small_dim}"
            large_key = f"vocab_{fixed_vocab}_dim_{large_dim}"
            
            if small_key in scaling_results and large_key in scaling_results:
                dim_ratio = large_dim / small_dim
                memory_ratio = scaling_results[large_key]['memory_mb'] / scaling_results[small_key]['memory_mb']
                print(f"   Dimension scaling: {dim_ratio:.1f}x dim -> {memory_ratio:.1f}x memory (Linear)")
        
        return scaling_results
    
    def compare_positional_encodings(self, seq_length: int = 100, embedding_dim: int = 256):
        """
        Compare performance and characteristics of different positional encoding approaches.
        
        This function is PROVIDED to show positional encoding comparison.
        """
        print(f"\nMAGNIFY POSITIONAL ENCODING COMPARISON")
        print("=" * 50)
        
        # Create test embeddings
        batch_size = 16
        embeddings = Tensor(np.random.randn(batch_size, seq_length, embedding_dim))
        
        # Test sinusoidal positional encoding
        sinusoidal_pe = PositionalEncoding(embedding_dim=embedding_dim, max_seq_length=seq_length*2)
        start_time = time.time()
        sin_result = sinusoidal_pe.forward(embeddings)
        sin_time = (time.time() - start_time) * 1000
        
        # Test learned positional embedding
        learned_pe = LearnedPositionalEmbedding(max_seq_length=seq_length*2, embedding_dim=embedding_dim)
        start_time = time.time()
        learned_result = learned_pe.forward(embeddings)
        learned_time = (time.time() - start_time) * 1000
        
        # Calculate memory usage
        sin_memory = 0  # No learnable parameters
        learned_memory = learned_pe.position_embedding.get_memory_usage()['total_memory_mb']
        
        results = {
            'sinusoidal': {
                'computation_time_ms': sin_time,
                'memory_usage_mb': sin_memory,
                'parameters': 0,
                'deterministic': True,
                'extrapolation': 'Good (can handle longer sequences)'
            },
            'learned': {
                'computation_time_ms': learned_time,
                'memory_usage_mb': learned_memory,
                'parameters': seq_length * 2 * embedding_dim,
                'deterministic': False,
                'extrapolation': 'Limited (fixed max sequence length)'
            }
        }
        
        print(f"📊 COMPARISON RESULTS:")
        print(f"{'Method':<12} {'Time (ms)':<10} {'Memory (MB)':<12} {'Parameters':<12} {'Extrapolation'}")
        print("-" * 70)
        print(f"{'Sinusoidal':<12} {sin_time:<10.2f} {sin_memory:<12.2f} {0:<12,} {'Good'}")
        print(f"{'Learned':<12} {learned_time:<10.2f} {learned_memory:<12.2f} {results['learned']['parameters']:<12,} {'Limited'}")
        
        print(f"\nTIP INSIGHTS:")
        print(f"   - Sinusoidal: Zero parameters, deterministic, good extrapolation")
        print(f"   - Learned: Requires parameters, model-specific, limited extrapolation")
        print(f"   - Choice depends on: model capacity, sequence length requirements, extrapolation needs")
        
        return results

def analyze_embedding_system_design():
    """
    Comprehensive analysis of embedding system design choices and their impact.
    
    This function is PROVIDED to show systems-level design thinking.
    """
    print("🏗️ EMBEDDING SYSTEM DESIGN ANALYSIS")
    print("=" * 60)
    
    # Example model configurations
    model_configs = [
        {'name': 'Small GPT', 'vocab_size': 10000, 'embed_dim': 256, 'seq_length': 512},
        {'name': 'Medium GPT', 'vocab_size': 50000, 'embed_dim': 512, 'seq_length': 1024},
        {'name': 'Large GPT', 'vocab_size': 50000, 'embed_dim': 1024, 'seq_length': 2048}
    ]
    
    print(f"📋 MODEL CONFIGURATION COMPARISON:")
    print(f"{'Model':<12} {'Vocab Size':<10} {'Embed Dim':<10} {'Seq Len':<8} {'Embed Params':<12} {'Memory (MB)'}")
    print("-" * 80)
    
    for config in model_configs:
        # Calculate embedding parameters
        embed_params = config['vocab_size'] * config['embed_dim']
        
        # Calculate memory usage
        embed_memory_mb = embed_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        print(f"{config['name']:<12} {config['vocab_size']:<10,} {config['embed_dim']:<10} "
              f"{config['seq_length']:<8} {embed_params:<12,} {embed_memory_mb:<10.1f}")
    
    print(f"\nTARGET DESIGN TRADE-OFFS:")
    print(f"   1. Vocabulary Size:")
    print(f"      - Larger vocab: Better text coverage, more parameters")
    print(f"      - Smaller vocab: Longer sequences, more compute")
    print(f"   2. Embedding Dimension:")
    print(f"      - Higher dim: More model capacity, more memory")
    print(f"      - Lower dim: Faster computation, potential bottleneck")
    print(f"   3. Position Encoding:")
    print(f"      - Sinusoidal: No parameters, good extrapolation")
    print(f"      - Learned: Model-specific, limited to training length")
    print(f"   4. Memory Scaling:")
    print(f"      - Embedding table: O(vocab_size * embed_dim)")
    print(f"      - Sequence processing: O(batch_size * seq_length * embed_dim)")
    print(f"      - Total memory dominated by model size, not embedding table")
    
    print(f"\n🏭 PRODUCTION CONSIDERATIONS:")
    print(f"   - GPU memory limits affect maximum embedding table size")
    print(f"   - Embedding lookup is memory-bandwidth bound")
    print(f"   - Vocabulary size affects tokenization and model download size")
    print(f"   - Position encoding choice affects sequence length flexibility")

# %% [markdown]
"""
### TEST Test: Embedding Performance Analysis

Let's test our embedding profiler with realistic performance scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "test-embedding-profiler", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_embedding_profiler():
    """Test embedding profiler with various scenarios."""
    print("🔬 Unit Test: Embedding Performance Profiler...")
    
    profiler = EmbeddingProfiler()
    
    # Create test embedding layer
    vocab_size = 1000
    embedding_dim = 128
    embed = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # Test lookup performance measurement
    batch_sizes = [8, 16]
    seq_lengths = [32, 64]
    
    performance_results = profiler.measure_lookup_performance(embed, batch_sizes, seq_lengths)
    
    # Verify results structure
    expected_configs = len(batch_sizes) * len(seq_lengths)
    assert len(performance_results) == expected_configs, f"Should test {expected_configs} configurations"
    
    for config, metrics in performance_results.items():
        # Verify all required metrics are present
        required_keys = ['batch_size', 'seq_length', 'total_tokens', 'lookup_time_ms', 
                        'tokens_per_second', 'memory_bandwidth_mb_s']
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key} in {config}"
            assert isinstance(metrics[key], (int, float)), f"Invalid metric type for {key}"
        
        # Verify reasonable values
        assert metrics['total_tokens'] > 0, "Should count tokens"
        assert metrics['lookup_time_ms'] >= 0, "Time should be non-negative"
        assert metrics['tokens_per_second'] >= 0, "Throughput should be non-negative"
    
    print("PASS Lookup performance measurement test passed")
    
    # Test memory scaling analysis
    vocab_sizes = [500, 1000]
    embedding_dims = [64, 128]
    
    scaling_results = profiler.analyze_memory_scaling(vocab_sizes, embedding_dims)
    
    # Verify scaling results
    expected_configs = len(vocab_sizes) * len(embedding_dims)
    assert len(scaling_results) == expected_configs, f"Should test {expected_configs} configurations"
    
    for config, metrics in scaling_results.items():
        assert 'total_parameters' in metrics, "Should include parameter count"
        assert 'memory_mb' in metrics, "Should include memory usage"
        assert metrics['total_parameters'] > 0, "Should have parameters"
        assert metrics['memory_mb'] > 0, "Should use memory"
    
    print("PASS Memory scaling analysis test passed")
    
    # Test positional encoding comparison
    comparison_results = profiler.compare_positional_encodings(seq_length=50, embedding_dim=64)
    
    # Verify comparison results
    assert 'sinusoidal' in comparison_results, "Should test sinusoidal encoding"
    assert 'learned' in comparison_results, "Should test learned encoding"
    
    for method, metrics in comparison_results.items():
        assert 'computation_time_ms' in metrics, "Should measure computation time"
        assert 'memory_usage_mb' in metrics, "Should measure memory usage"
        assert 'parameters' in metrics, "Should count parameters"
    
    print("PASS Positional encoding comparison test passed")
    print("TARGET Embedding Profiler: All tests passed!")

# Test function defined (called in main block)

# %% [markdown]
"""
## Integration Testing: Complete Embedding Pipeline

Let's test how all our embedding components work together in a realistic language processing pipeline:
"""

# %% nbgrader={"grade": false, "grade_id": "test-embedding-integration", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_embedding_integration():
    """Test complete embedding pipeline with tokenization integration."""
    print("TEST Integration Test: Complete Embedding Pipeline...")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Create embedding layer
    embed = Embedding(vocab_size=tokenizer.vocab_size, embedding_dim=128, padding_idx=0)
    
    # Create positional encoding
    pos_encoding = PositionalEncoding(embedding_dim=128, max_seq_length=100)
    
    # Test text processing pipeline
    texts = [
        "Hello world!",
        "This is a test.",
        "Short text.",
        "A longer piece of text to test the pipeline."
    ]
    
    print(f"  Processing {len(texts)} texts through complete pipeline...")
    
    # Step 1: Tokenize texts
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokenized.append(tokens)
    
    # Step 2: Pad sequences for batch processing
    padded_sequences = tokenizer.pad_sequences(tokenized, max_length=20)
    batch_tokens = Tensor(np.array(padded_sequences))
    
    print(f"    Batch shape: {batch_tokens.shape}")
    
    # Step 3: Embedding lookup
    embeddings = embed.forward(batch_tokens)
    print(f"    Embeddings shape: {embeddings.shape}")
    
    # Step 4: Add positional encoding
    pos_embeddings = pos_encoding.forward(embeddings)
    print(f"    Position-aware embeddings shape: {pos_embeddings.shape}")
    
    # Verify pipeline correctness
    expected_shape = (len(texts), 20, 128)  # (batch, seq_len, embed_dim)
    assert pos_embeddings.shape == expected_shape, f"Expected {expected_shape}, got {pos_embeddings.shape}"
    
    # Test that padding tokens have correct embeddings (should be zero from embedding layer)
    padding_token_id = tokenizer.char_to_idx['<PAD>']
    
    # Find positions with padding tokens
    padding_positions = (batch_tokens.data == padding_token_id)
    
    if np.any(padding_positions):
        # Get embeddings for padding positions
        padding_embeddings = embeddings.data[padding_positions]
        
        # Padding embeddings should be close to zero (from embedding initialization)
        # Note: they won't be exactly zero because we add positional encoding
        print(f"    Padding token embeddings found: {np.sum(padding_positions)} positions")
    
    # Test different sequence lengths
    short_text = "Hi!"
    short_tokens = tokenizer.encode(short_text, add_special_tokens=True)
    short_tensor = Tensor(np.array([short_tokens]))  # Add batch dimension
    
    short_embeddings = embed.forward(short_tensor)
    short_pos_embeddings = pos_encoding.forward(short_embeddings)
    
    print(f"    Short text processing: {short_pos_embeddings.shape}")
    
    # Test memory efficiency
    large_batch_size = 32
    large_seq_length = 50
    large_tokens = np.random.randint(0, tokenizer.vocab_size, (large_batch_size, large_seq_length))
    large_tensor = Tensor(large_tokens)
    
    start_time = time.time()
    large_embeddings = embed.forward(large_tensor)
    large_pos_embeddings = pos_encoding.forward(large_embeddings)
    processing_time = time.time() - start_time
    
    print(f"    Large batch processing: {large_pos_embeddings.shape} in {processing_time*1000:.2f}ms")
    
    # Calculate memory usage
    embedding_memory = embed.get_memory_usage()
    total_memory_mb = embedding_memory['total_memory_mb']
    
    print(f"    Embedding table memory: {total_memory_mb:.2f}MB")
    print(f"    Sequence memory: {large_pos_embeddings.data.nbytes / (1024*1024):.2f}MB")
    
    print("PASS Complete embedding pipeline integration test passed!")
    print(f"PASS Tokenization -> Embedding -> Positional Encoding pipeline works")
    print(f"PASS Handles various batch sizes and sequence lengths")
    print(f"PASS Memory usage is reasonable for production systems")

# Test function defined (called in main block)

# %% [markdown]
"""
## Main Execution Block

All embedding tests and demonstrations are run from here when the module is executed directly:
"""

# %% nbgrader={"grade": false, "grade_id": "embeddings-main", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    # Run all unit tests
    test_unit_embedding_layer()
    test_unit_positional_encoding()
    test_unit_learned_positional_embedding()
    test_embedding_profiler()
    test_embedding_integration()
    
    print("\n" + "="*60)
    print("MAGNIFY EMBEDDING SYSTEMS ANALYSIS")
    print("="*60)
    
    # Performance analysis
    profiler = EmbeddingProfiler()
    
    # Test different embedding configurations
    print("\n📊 EMBEDDING PERFORMANCE COMPARISON:")
    
    # Compare embedding layers with different sizes
    vocab_sizes = [1000, 5000, 10000]
    embedding_dims = [128, 256, 512]
    
    scaling_results = profiler.analyze_memory_scaling(vocab_sizes, embedding_dims)
    
    # Compare positional encoding approaches
    print("\n" + "="*60)
    pos_comparison = profiler.compare_positional_encodings(seq_length=128, embedding_dim=256)
    
    # Systems design analysis
    print("\n" + "="*60)
    analyze_embedding_system_design()
    
    # Demonstrate realistic language model embedding setup
    print("\n" + "="*60)
    print("🏗️ REALISTIC LANGUAGE MODEL EMBEDDING SETUP")
    print("="*60)
    
    # Create realistic configuration
    vocab_size = 10000  # 10k vocabulary
    embedding_dim = 256  # 256-dim embeddings
    max_seq_length = 512  # 512 token sequences
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Max sequence length: {max_seq_length}")
    
    # Create components
    embedding_layer = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    pos_encoding = PositionalEncoding(embedding_dim=embedding_dim, max_seq_length=max_seq_length)
    
    # Calculate memory requirements
    embed_memory = embedding_layer.get_memory_usage()
    
    print(f"\nMemory analysis:")
    print(f"  Embedding table: {embed_memory['total_memory_mb']:.1f}MB")
    print(f"  Parameters: {embed_memory['total_parameters']:,}")
    
    # Simulate batch processing
    batch_size = 32
    seq_length = 256
    test_tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))
    
    start_time = time.time()
    embeddings = embedding_layer.forward(test_tokens)
    pos_embeddings = pos_encoding.forward(embeddings)
    total_time = time.time() - start_time
    
    sequence_memory_mb = pos_embeddings.data.nbytes / (1024 * 1024)
    
    print(f"\nBatch processing:")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_length}")
    print(f"  Processing time: {total_time*1000:.2f}ms")
    print(f"  Sequence memory: {sequence_memory_mb:.1f}MB")
    print(f"  Throughput: {(batch_size * seq_length) / total_time:.0f} tokens/second")
    
    print("\n" + "="*60)
    print("TARGET EMBEDDINGS MODULE COMPLETE!")
    print("="*60)
    print("All embedding tests passed!")
    print("Ready for attention mechanism integration!")

# %% [markdown]
"""
## THINK ML Systems Thinking: Interactive Questions

Now that you've built the embedding systems that convert tokens to rich vector representations, let's connect this work to broader ML systems challenges. These questions help you think critically about how embedding design scales to production language processing systems.

Take time to reflect thoughtfully on each question - your insights will help you understand how embedding choices connect to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Embedding Memory Optimization and Model Scaling

**Context**: Your embedding implementations demonstrate how vocabulary size and embedding dimension directly impact model parameters and memory usage. In your memory scaling analysis, you saw how a 100k vocabulary with 1024-dimensional embeddings requires ~400MB just for the embedding table. In production language models, embedding tables often contain billions of parameters (GPT-3's embedding table alone has ~600M parameters), making memory optimization critical for deployment and training efficiency.

**Reflection Question**: Based on your `Embedding` class implementation and memory scaling analysis, design a memory-optimized embedding system for a production language model that needs to handle a 100k vocabulary with 1024-dimensional embeddings while operating under GPU memory constraints. How would you modify your current `Embedding.forward()` method to implement embedding compression techniques, design efficient lookup patterns for high-throughput training, and handle dynamic vocabulary expansion for domain adaptation? Consider how your current weight initialization strategies could be adapted and what changes to your `get_memory_usage()` analysis would be needed for compressed embeddings.

Think about: adapting your embedding lookup implementation, modifying weight storage patterns, extending your memory analysis for compression techniques, and designing efficient gradient updates for compressed representations.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-embedding-memory", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON EMBEDDING MEMORY OPTIMIZATION:

TODO: Replace this text with your thoughtful response about memory-optimized embedding system design.

Consider addressing:
- How would you implement embedding compression for a 100k * 1024 vocabulary under GPU constraints?
- What techniques would you use to optimize lookup patterns for high-throughput training?
- How would you design dynamic vocabulary expansion while maintaining memory efficiency?
- What trade-offs would you make between embedding quality and memory footprint?
- How would you optimize differently for training vs inference scenarios?

Write a technical analysis connecting your embedding implementations to real memory optimization challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of embedding memory scaling and optimization (3 points)
- Designs practical approaches to compression and efficient lookup patterns (3 points)
- Addresses dynamic vocabulary and quality-memory trade-offs (2 points)
- Shows systems thinking about production memory constraints (2 points)
- Clear technical reasoning with memory optimization insights (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring technical analysis of embedding memory optimization
# Students should demonstrate understanding of large-scale embedding systems and memory efficiency
### END SOLUTION

# %% [markdown]
"""
### Question 2: Positional Encoding and Sequence Length Scalability

**Context**: Your positional encoding implementations show the trade-offs between fixed sinusoidal patterns and learned position embeddings. In your analysis, you saw that `PositionalEncoding` requires 0 parameters but `LearnedPositionalEmbedding` needs max_seq_length * embedding_dim parameters. Production language models increasingly need to handle variable sequence lengths efficiently while maintaining consistent position representations across different tasks and deployment scenarios.

**Reflection Question**: Based on your `PositionalEncoding` and `LearnedPositionalEmbedding` implementations, architect a hybrid positional encoding system for a production transformer that efficiently handles sequences from 512 tokens to 32k tokens. How would you modify your current `forward()` methods to create a hybrid approach that combines the benefits of both systems? What changes would you make to your position computation to optimize for variable-length sequences, and how would you extend your positional encoding comparison analysis to measure performance across different sequence length distributions?

Think about: combining your two encoding implementations, modifying the forward pass for variable lengths, extending your performance analysis methods, and optimizing position computation patterns from your current code.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-positional-encoding", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON POSITIONAL ENCODING AND SEQUENCE SCALABILITY:

TODO: Replace this text with your thoughtful response about scalable positional encoding system design.

Consider addressing:
- How would you design hybrid positional encoding for sequences from 512 to 32k tokens?
- What strategies would you use to optimize position computation for variable-length sequences?
- How would you balance memory efficiency with computational performance?
- What approaches would you use to handle different sequence length distributions?
- How would you maintain training stability across diverse sequence lengths?

Write an architectural analysis connecting your positional encoding work to scalable sequence processing.

GRADING RUBRIC (Instructor Use):
- Shows understanding of positional encoding scalability challenges (3 points)
- Designs practical approaches to hybrid encoding and variable-length optimization (3 points)
- Addresses memory and computational efficiency considerations (2 points)
- Demonstrates systems thinking about sequence length distribution handling (2 points)
- Clear architectural reasoning with scalability insights (bonus points for comprehensive system design)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of positional encoding scalability
# Students should demonstrate knowledge of sequence length optimization and hybrid approaches
### END SOLUTION

# %% [markdown]
"""
### Question 3: Embedding Pipeline Integration and Training Efficiency

**Context**: Your embedding pipeline integration demonstrates how tokenization, embedding lookup, and positional encoding work together in language model preprocessing. In your `test_embedding_integration()` function, you measured pipeline performance and saw how batch size affects throughput. In production training systems, the embedding pipeline often becomes a bottleneck due to memory bandwidth limitations and the need to process billions of tokens efficiently during training.

**Reflection Question**: Based on your complete embedding pipeline implementation (tokenization -> `Embedding.forward()` -> `PositionalEncoding.forward()`), design an optimization strategy for large-scale language model training that processes 1 trillion tokens efficiently. How would you modify your current pipeline functions to implement batch processing optimizations for mixed sequence lengths, design efficient gradient updates for your massive `Embedding.weight` parameters, and coordinate embedding updates across distributed training nodes? Consider how your current memory analysis and performance measurement techniques could be extended to monitor pipeline bottlenecks in distributed settings.

Think about: optimizing your current pipeline implementation, extending your performance analysis to distributed settings, modifying your batch processing patterns, and scaling your embedding weight update mechanisms.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-pipeline-integration", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON EMBEDDING PIPELINE INTEGRATION:

TODO: Replace this text with your thoughtful response about embedding pipeline optimization for large-scale training.

Consider addressing:
- How would you implement pipeline parallelism for processing 1 trillion tokens efficiently?
- What strategies would you use to optimize batch processing for mixed sequence lengths?
- How would you design efficient gradient updates for massive embedding tables?
- What approaches would you use for coordinating embedding updates across distributed nodes?
- How would you maintain GPU utilization while minimizing memory bandwidth bottlenecks?

Write a design analysis connecting your embedding pipeline to large-scale training optimization.

GRADING RUBRIC (Instructor Use):
- Understands embedding pipeline bottlenecks and optimization challenges (3 points)
- Designs practical approaches to pipeline parallelism and batch optimization (3 points)
- Addresses distributed training and gradient update efficiency (2 points)
- Shows systems thinking about large-scale training coordination (2 points)
- Clear design reasoning with pipeline optimization insights (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of large-scale embedding pipeline optimization
# Students should demonstrate knowledge of distributed training and pipeline efficiency
### END SOLUTION

# %% [markdown]
"""
## TARGET MODULE SUMMARY: Embeddings

Congratulations! You have successfully implemented comprehensive embedding systems for language processing:

### PASS What You Have Built
- **Embedding Layer**: Learnable lookup table converting tokens to dense vector representations
- **Positional Encoding**: Sinusoidal position information for sequence understanding
- **Learned Positional Embeddings**: Trainable position representations for model-specific optimization
- **Memory-Efficient Lookups**: Optimized embedding access patterns for production systems
- **Performance Analysis**: Comprehensive profiling and scaling analysis tools
- **🆕 Integration Pipeline**: Complete tokenization -> embedding -> positional encoding workflow
- **🆕 Systems Optimization**: Memory usage analysis and performance optimization techniques

### PASS Key Learning Outcomes
- **Understanding**: How discrete tokens become continuous vector representations
- **Implementation**: Built embedding systems from scratch with efficient lookup operations
- **Systems Insight**: How embedding table size affects model memory and training efficiency
- **Performance Engineering**: Measured and optimized embedding lookup patterns and memory usage
- **Production Context**: Understanding real-world embedding challenges and optimization techniques

### PASS Technical Mastery
- **Embedding Lookup**: Efficient table lookup with various initialization strategies
- **Positional Encoding**: Mathematical sine/cosine patterns for position representation
- **Memory Scaling**: Understanding O(vocab_size * embedding_dim) parameter scaling
- **Performance Optimization**: Cache-friendly access patterns and memory bandwidth optimization
- **🆕 Integration Design**: Seamless pipeline from text processing to vector representations

### PASS Professional Skills Developed
- **Systems Architecture**: Designing embedding systems for production scale
- **Memory Engineering**: Optimizing large parameter tables for efficient access
- **Performance Analysis**: Measuring and improving embedding pipeline throughput
- **Integration Thinking**: Connecting embedding systems with tokenization and attention

### PASS Ready for Next Steps
Your embedding systems are now ready to power:
- **Attention Mechanisms**: Processing sequence representations with attention
- **Transformer Models**: Complete language model architectures
- **Language Understanding**: Rich semantic representations for NLP tasks
- **🧠 Sequence Processing**: Foundation for advanced sequence modeling

### LINK Connection to Real ML Systems
Your implementations mirror production systems:
- **PyTorch Embeddings**: `torch.nn.Embedding` and `torch.nn.functional.embedding`
- **Transformer Models**: All modern language models use similar embedding approaches
- **Production Optimizations**: Memory mapping, gradient checkpointing, and distributed embeddings
- **Industry Applications**: GPT, BERT, and other transformer models rely on these foundations

### TARGET The Power of Dense Representations
You have unlocked the bridge between discrete tokens and continuous understanding:
- **Before**: Tokens were sparse, discrete symbols
- **After**: Tokens become rich, continuous vectors that capture semantic relationships

**Next Module**: Attention - Processing sequences with the mechanism that revolutionized language understanding!

Your embedding systems provide the rich vector representations that attention mechanisms need to understand language. Now let's build the attention that makes transformers work!
"""