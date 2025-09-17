"""
Attention mechanisms for TinyGPT transformer models.

Implements self-attention and multi-head attention using TinyTorch components.
"""

import numpy as np
import sys
import os

# Add TinyTorch to path for reusing components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import Softmax
except ImportError:
    print("⚠️ TinyTorch not available. Using mock implementations for development.")
    # Mock implementations for development
    class Tensor:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
            
        def __matmul__(self, other):
            if isinstance(other, Tensor):
                return Tensor(self.data @ other.data)
            return Tensor(self.data @ other)
            
        def transpose(self, axes=None):
            if axes is None:
                return Tensor(self.data.T)
            return Tensor(np.transpose(self.data, axes))
        
        def softmax(self, axis=-1):
            exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
            return Tensor(exp_data / np.sum(exp_data, axis=axis, keepdims=True))
            
        def __add__(self, other):
            if isinstance(other, Tensor):
                return Tensor(self.data + other.data)
            return Tensor(self.data + other)
            
        def __mul__(self, other):
            if isinstance(other, Tensor):
                return Tensor(self.data * other.data)
            return Tensor(self.data * other)
    
    class Dense:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Tensor(np.random.randn(in_features, out_features) * 0.1)
            self.bias = Tensor(np.zeros(out_features))
            
        def forward(self, x):
            return x @ self.weight + self.bias
    
    class Softmax:
        def forward(self, x):
            return x.softmax()


class MultiHeadAttention:
    """Multi-head self-attention mechanism using TinyTorch Dense layers."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            dropout: Dropout rate (not implemented yet)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V using TinyTorch Dense layers
        self.w_q = Dense(d_model, d_model)
        self.w_k = Dense(d_model, d_model)
        self.w_v = Dense(d_model, d_model)
        self.w_o = Dense(d_model, d_model)  # Output projection
        
        self.softmax = Softmax()
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                mask: Tensor = None) -> Tensor:
        """Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Attention output of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Reshape for TinyTorch Dense layers (expects 2D)
        query_2d = Tensor(query.data.reshape(-1, d_model))  # (batch_size * seq_len, d_model)
        key_2d = Tensor(key.data.reshape(-1, d_model))
        value_2d = Tensor(value.data.reshape(-1, d_model))
        
        # Linear projections
        Q_2d = self.w_q.forward(query_2d)  # (batch_size * seq_len, d_model)
        K_2d = self.w_k.forward(key_2d)
        V_2d = self.w_v.forward(value_2d)
        
        # Reshape back to 3D
        Q = Tensor(Q_2d.data.reshape(batch_size, seq_len, d_model))
        K = Tensor(K_2d.data.reshape(batch_size, seq_len, d_model))
        V = Tensor(V_2d.data.reshape(batch_size, seq_len, d_model))
        
        # Reshape for multi-head attention
        Q = self._reshape_for_attention(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self._reshape_for_attention(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self._reshape_for_attention(V)  # (batch_size, num_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = self._combine_heads(attention_output)
        
        # Final linear projection (reshape for Dense layer)
        batch_size, seq_len, d_model = attention_output.shape
        attention_2d = Tensor(attention_output.data.reshape(-1, d_model))
        output_2d = self.w_o.forward(attention_2d)
        output = Tensor(output_2d.data.reshape(batch_size, seq_len, d_model))
        
        return output
    
    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, d_model = x.shape
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        reshaped = Tensor(x.data.reshape(batch_size, seq_len, self.num_heads, self.d_k))
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return Tensor(reshaped.data.transpose(0, 2, 1, 3))
    
    def _combine_heads(self, x: Tensor) -> Tensor:
        """Combine attention heads back into single tensor."""
        batch_size, num_heads, seq_len, d_k = x.shape
        # Transpose back to (batch_size, seq_len, num_heads, d_k)
        transposed = Tensor(x.data.transpose(0, 2, 1, 3))
        # Reshape to (batch_size, seq_len, d_model)
        return Tensor(transposed.data.reshape(batch_size, seq_len, self.d_model))
    
    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, 
                                    mask: Tensor = None) -> Tensor:
        """Compute scaled dot-product attention."""
        # Compute attention scores
        # Q: (batch_size, num_heads, seq_len, d_k)
        # K: (batch_size, num_heads, seq_len, d_k)
        # Scores: (batch_size, num_heads, seq_len, seq_len)
        
        K_T = K.data.transpose(0, 1, 3, 2)  # Transpose K
        scores = Tensor(np.matmul(Q.data, K_T))  # QK^T using numpy matmul
        scores = scores * (1.0 / np.sqrt(self.d_k))  # Scale
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Apply softmax manually since TinyTorch Tensor doesn't have softmax
        # Subtract max for numerical stability
        scores_max = np.max(scores.data, axis=-1, keepdims=True)
        scores_shifted = scores.data - scores_max
        exp_scores = np.exp(scores_shifted)
        softmax_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attention_weights = Tensor(softmax_weights)
        
        # Apply attention to values
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        # V: (batch_size, num_heads, seq_len, d_k)
        # Output: (batch_size, num_heads, seq_len, d_k)
        output = Tensor(np.matmul(attention_weights.data, V.data))
        
        return output


class SelfAttention:
    """Simplified self-attention for easier understanding."""
    
    def __init__(self, d_model: int):
        """Initialize self-attention.
        
        Args:
            d_model: Model dimension
        """
        self.d_model = d_model
        self.scale = 1.0 / np.sqrt(d_model)
        
        # Single-head attention projections
        self.w_q = Dense(d_model, d_model)
        self.w_k = Dense(d_model, d_model)
        self.w_v = Dense(d_model, d_model)
        
        self.softmax = Softmax()
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Attention output of same shape as input
        """
        # Compute Q, K, V
        Q = self.w_q.forward(x)  # (batch_size, seq_len, d_model)
        K = self.w_k.forward(x)  # (batch_size, seq_len, d_model)
        V = self.w_v.forward(x)  # (batch_size, seq_len, d_model)
        
        # Compute attention scores
        scores = Q @ K.transpose((0, 2, 1))  # (batch_size, seq_len, seq_len)
        scores = scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Apply softmax
        attention_weights = scores.softmax(axis=-1)
        
        # Apply attention to values
        output = attention_weights @ V  # (batch_size, seq_len, d_model)
        
        return output


def create_causal_mask(seq_len: int) -> Tensor:
    """Create causal mask for preventing attention to future tokens.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    # Create lower triangular matrix (0 = attend, 1 = mask)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return Tensor(mask)


class PositionalEncoding:
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
        """
        self.d_model = d_model
        self.max_length = max_length
        
        # Create positional encoding matrix
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length).reshape(-1, 1)
        
        # Compute div_term for sinusoidal encoding
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        self.pe = Tensor(pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get positional encodings for this sequence length
        pos_encoding = Tensor(self.pe.data[:seq_len, :])
        
        # Add to input (broadcasting across batch dimension)
        return x + pos_encoding


if __name__ == "__main__":
    # Test attention mechanisms
    print("🧪 Testing TinyGPT Attention Mechanisms")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    
    # Create sample input
    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    print(f"Input shape: {x.shape}")
    
    # Test self-attention
    print("\n🎯 Self-Attention:")
    self_attn = SelfAttention(d_model)
    output = self_attn.forward(x)
    print(f"Output shape: {output.shape}")
    
    # Test multi-head attention
    print("\n🔀 Multi-Head Attention:")
    multi_head_attn = MultiHeadAttention(d_model, num_heads)
    output = multi_head_attn.forward(x, x, x)
    print(f"Output shape: {output.shape}")
    
    # Test causal mask
    print("\n🎭 Causal Mask:")
    mask = create_causal_mask(seq_len)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sample:\n{mask.data[:5, :5]}")
    
    # Test with causal mask
    masked_output = self_attn.forward(x, mask)
    print(f"Masked output shape: {masked_output.shape}")
    
    # Test positional encoding
    print("\n📍 Positional Encoding:")
    pos_encoding = PositionalEncoding(d_model, max_length=100)
    encoded_x = pos_encoding.forward(x)
    print(f"Encoded shape: {encoded_x.shape}")
    
    print("\n✅ Attention mechanism tests completed!")
    print("\n💡 Key insights:")
    print("   • Self-attention allows tokens to attend to each other")
    print("   • Multi-head attention captures different types of relationships")
    print("   • Causal masking prevents attention to future tokens")
    print("   • Positional encoding adds sequence order information")
    print("   • All components reuse TinyTorch Dense layers! 🎉")