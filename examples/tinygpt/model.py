"""
TinyGPT Model Implementation
A simple GPT-style transformer built entirely with TinyTorch components.
"""

import numpy as np
from typing import Optional, Tuple
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.activations import Softmax, ReLU

class MultiHeadAttention:
    """
    Multi-head attention mechanism - the core of transformers.
    Allows the model to attend to different positions simultaneously.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Query, Key, Value projections
        self.W_q = Dense(d_model, d_model, use_bias=False)
        self.W_k = Dense(d_model, d_model, use_bias=False)
        self.W_v = Dense(d_model, d_model, use_bias=False)
        self.W_o = Dense(d_model, d_model, use_bias=False)
        
        self.softmax = Softmax()
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Generate Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        # Split d_model into num_heads × d_k
        Q = self._reshape_for_heads(Q, batch_size, seq_len)
        K = self._reshape_for_heads(K, batch_size, seq_len)
        V = self._reshape_for_heads(V, batch_size, seq_len)
        
        # Compute attention scores
        scores = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back to (batch_size, seq_len, d_model)
        scores = self._reshape_from_heads(scores, batch_size, seq_len)
        
        # Final linear projection
        output = self.W_o(scores)
        
        return output
    
    def _reshape_for_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Reshape tensor for multi-head processing."""
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        x_data = x.data.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        x_data = x_data.transpose(1, 2)  # Move heads dimension
        return Tensor(x_data)
    
    def _reshape_from_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Reshape tensor back from multi-head processing."""
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x_data = x.data.transpose(1, 2)
        x_data = x_data.reshape(batch_size, seq_len, self.d_model)
        return Tensor(x_data)
    
    def _scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute scaled dot-product attention.
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
        """
        # Compute attention scores
        scores = np.matmul(Q.data, K.data.transpose(-2, -1))
        scores = scores / np.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = np.where(mask.data == 0, scores, -1e9)
        
        # Apply softmax
        attention_weights = self.softmax(Tensor(scores))
        
        # Apply attention to values
        output = np.matmul(attention_weights.data, V.data)
        
        return Tensor(output)


class LayerNorm:
    """
    Layer normalization - stabilizes training of deep networks.
    Normalizes across the feature dimension.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(d_model))  # Scale
        self.beta = Tensor(np.zeros(d_model))  # Shift
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        # Calculate mean and variance across last dimension
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma.data * x_norm + self.beta.data
        
        return Tensor(output)


class TransformerBlock:
    """
    A single transformer block consisting of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
        """
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        
        # Feed-forward network
        self.ff1 = Dense(d_model, d_ff)
        self.relu = ReLU()
        self.ff2 = Dense(d_ff, d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection
        attn_output = self.attention.forward(x, mask)
        x = Tensor(x.data + attn_output.data)  # Residual connection
        x = self.norm1.forward(x)
        
        # Feed-forward with residual connection
        ff_output = self.ff2(self.relu(self.ff1(x)))
        x = Tensor(x.data + ff_output.data)  # Residual connection
        x = self.norm2.forward(x)
        
        return x


class PositionalEncoding:
    """
    Positional encoding adds position information to embeddings.
    Uses sinusoidal functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        
        # Create div_term for sinusoidal pattern
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        self.pe = Tensor(pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.shape[1]
        # Add positional encoding (broadcast across batch)
        output = x.data + self.pe.data[:seq_len, :]
        return Tensor(output)


class TinyGPT:
    """
    TinyGPT - A minimal GPT implementation using TinyTorch.
    
    Architecture:
    1. Token embeddings
    2. Positional encoding
    3. Stack of transformer blocks
    4. Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 256
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embeddings
        self.embedding = Dense(vocab_size, d_model, use_bias=False)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, d_ff))
        
        # Output projection
        self.output_proj = Dense(d_model, vocab_size)
        
        print(f"🤖 TinyGPT initialized:")
        print(f"   Vocab: {vocab_size}, Model dim: {d_model}")
        print(f"   Heads: {num_heads}, Layers: {num_layers}")
        
    def forward(self, input_ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through TinyGPT.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert token IDs to one-hot vectors
        one_hot = np.zeros((batch_size, seq_len, self.vocab_size))
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = int(input_ids.data[b, s])
                if 0 <= token_id < self.vocab_size:
                    one_hot[b, s, token_id] = 1.0
        
        # Token embeddings
        x = self.embedding(Tensor(one_hot))
        
        # Add positional encoding
        x = self.pos_encoding.forward(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated token IDs
        """
        generated = prompt_ids.copy()
        
        for _ in range(max_length - len(prompt_ids)):
            # Create attention mask (causal)
            curr_len = len(generated)
            mask = create_causal_mask(curr_len)
            
            # Get model predictions
            input_tensor = Tensor(generated.reshape(1, -1))
            logits = self.forward(input_tensor, mask)
            
            # Get logits for last position
            last_logits = logits.data[0, -1, :]
            
            # Apply temperature
            last_logits = last_logits / temperature
            
            # Convert to probabilities
            probs = np.exp(last_logits) / np.sum(np.exp(last_logits))
            
            # Sample next token
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated = np.append(generated, next_token)
        
        return generated


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal attention mask to prevent attending to future tokens.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Mask tensor where 0 = allowed, 1 = masked
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask = 1 - mask  # Invert: 0 for allowed, 1 for masked
    return Tensor(mask)