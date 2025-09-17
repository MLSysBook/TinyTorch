"""
TinyGPT transformer models built on TinyTorch components.

Implements GPT-style autoregressive language models that maximize reuse
of TinyTorch layers while adding transformer-specific components.
"""

import numpy as np
import sys
import os

# Add TinyTorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.layers import Dense
    from tinytorch.core.activations import ReLU, Softmax
    # Don't import Sequential from TinyTorch - it doesn't handle 3D tensors
    TINYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ TinyTorch not available. Using mock implementations.")
    # Use mock implementations from attention.py
    from .attention import Tensor, Dense
    TINYTORCH_AVAILABLE = False
    
    class ReLU:
        def forward(self, x):
            return Tensor(np.maximum(0, x.data))
    
    class Softmax:
        def forward(self, x):
            return x.softmax()

# Custom Sequential that handles 3D tensors (works with or without TinyTorch)
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        # Handle 3D tensors by reshaping for Dense layers
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, d_model = original_shape
            x = Tensor(x.data.reshape(-1, d_model))
            
        for layer in self.layers:
            x = layer.forward(x)
            
        # Reshape back to original dimensions
        if len(original_shape) == 3:
            x = Tensor(x.data.reshape(batch_size, seq_len, -1))
            
        return x

from .attention import MultiHeadAttention, PositionalEncoding, create_causal_mask


class LayerNorm:
    """Layer normalization for transformer models."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """Initialize layer normalization.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters (simplified - would need proper gradient handling)
        self.gamma = Tensor(np.ones(d_model))
        self.beta = Tensor(np.zeros(d_model))
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        
        # Normalize
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = normalized * self.gamma.data + self.beta.data
        
        return Tensor(output)


class TransformerBlock:
    """Single transformer block with self-attention and feedforward network."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feedforward network dimension
            dropout: Dropout rate (not implemented)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feedforward network using TinyTorch Dense layers
        self.feedforward = Sequential([
            Dense(d_model, d_ff),
            ReLU(),
            Dense(d_ff, d_model)
        ])
        
        # Layer normalization
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention.forward(x, x, x, mask)
        x = self.ln1.forward(x + attn_output)  # Residual connection
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feedforward.forward(x)
        x = self.ln2.forward(x + ff_output)  # Residual connection
        
        return x


class TinyGPT:
    """TinyGPT: GPT-style transformer model using TinyTorch components."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = None, max_length: int = 1024,
                 dropout: float = 0.1):
        """Initialize TinyGPT model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feedforward dimension (default: 4 * d_model)
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_length = max_length
        self.dropout = dropout
        
        # Token embeddings using TinyTorch Dense layer
        self.token_embedding = Dense(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, self.d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_final = LayerNorm(d_model)
        
        # Output projection to vocabulary using TinyTorch Dense layer
        self.output_projection = Dense(d_model, vocab_size)
        
        print(f"🤖 TinyGPT initialized:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Model dim: {d_model}")
        print(f"   Heads: {num_heads}")
        print(f"   Layers: {num_layers}")
        print(f"   Parameters: ~{self.count_parameters():,}")
        
    def forward(self, input_ids: Tensor, use_cache: bool = False) -> Tensor:
        """Forward pass of TinyGPT.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            use_cache: Whether to use caching (not implemented)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert token indices to one-hot encoding for embedding
        # This is a simplified approach - in practice, we'd use proper embedding layers
        one_hot = np.zeros((batch_size, seq_len, self.vocab_size))
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = int(input_ids.data[b, s])
                if 0 <= token_id < self.vocab_size:
                    one_hot[b, s, token_id] = 1.0
        
        # Token embeddings (reshape for Dense layer)
        one_hot_2d = Tensor(one_hot.reshape(-1, self.vocab_size))  # (batch_size * seq_len, vocab_size)
        x_2d = self.token_embedding.forward(one_hot_2d)  # (batch_size * seq_len, d_model)
        x = Tensor(x_2d.data.reshape(batch_size, seq_len, self.d_model))  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding.forward(x)
        
        # Create causal mask
        mask = create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final layer norm
        x = self.ln_final.forward(x)
        
        # Project to vocabulary (reshape for Dense layer)
        x_2d = Tensor(x.data.reshape(-1, self.d_model))  # (batch_size * seq_len, d_model)
        logits_2d = self.output_projection.forward(x_2d)  # (batch_size * seq_len, vocab_size)
        logits = Tensor(logits_2d.data.reshape(batch_size, seq_len, self.vocab_size))  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def generate(self, input_ids: Tensor, max_new_tokens: int = 50, 
                temperature: float = 1.0, do_sample: bool = True) -> Tensor:
        """Generate text autoregressively.
        
        Args:
            input_ids: Starting token indices of shape (1, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token sequence including input
        """
        generated = input_ids.data.copy()
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(Tensor(generated))
            
            # Get logits for last token
            next_token_logits = logits.data[0, -1, :]  # (vocab_size,)
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample next token
            if do_sample:
                # Softmax to get probabilities
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                next_token = np.random.choice(len(probs), p=probs)
            else:
                # Greedy decoding
                next_token = np.argmax(next_token_logits)
            
            # Append to sequence
            generated = np.concatenate([
                generated,
                np.array([[next_token]])
            ], axis=1)
            
            # Stop if we hit maximum length
            if generated.shape[1] >= self.max_length:
                break
        
        return Tensor(generated)
    
    def count_parameters(self) -> int:
        """Estimate number of parameters in the model."""
        params = 0
        
        # Token embedding: vocab_size * d_model
        params += self.vocab_size * self.d_model
        
        # Each transformer block
        for _ in range(self.num_layers):
            # Multi-head attention: 4 * d_model * d_model (Q, K, V, O projections)
            params += 4 * self.d_model * self.d_model
            
            # Feedforward: d_model * d_ff + d_ff * d_model
            params += 2 * self.d_model * self.d_ff
            
            # Layer norms: 2 * 2 * d_model (gamma and beta for each)
            params += 4 * self.d_model
        
        # Final layer norm: 2 * d_model
        params += 2 * self.d_model
        
        # Output projection: d_model * vocab_size
        params += self.d_model * self.vocab_size
        
        return params


class SimpleLM:
    """Simplified language model for testing and comparison."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, d_hidden: int = 256):
        """Initialize simple language model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Embedding dimension
            d_hidden: Hidden layer dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # Simple feedforward network using TinyTorch components
        self.embedding = Dense(vocab_size, d_model)
        self.hidden = Dense(d_model, d_hidden)
        self.activation = ReLU()
        self.output = Dense(d_hidden, vocab_size)
        
        print(f"🔤 Simple LM initialized: {vocab_size} vocab, {d_model} dim")
        
    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass of simple language model."""
        batch_size, seq_len = input_ids.shape
        
        # Convert to one-hot
        one_hot = np.zeros((batch_size, seq_len, self.vocab_size))
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = int(input_ids.data[b, s])
                if 0 <= token_id < self.vocab_size:
                    one_hot[b, s, token_id] = 1.0
        
        # Simple feedforward (reshape for Dense layers)
        one_hot_2d = Tensor(one_hot.reshape(-1, self.vocab_size))
        x = self.embedding.forward(one_hot_2d)
        x = self.hidden.forward(x)
        x = self.activation.forward(x)
        logits_2d = self.output.forward(x)
        logits = Tensor(logits_2d.data.reshape(batch_size, seq_len, self.vocab_size))
        
        return logits


if __name__ == "__main__":
    # Test TinyGPT models
    print("🧪 Testing TinyGPT Models")
    print("=" * 50)
    
    # Model parameters
    vocab_size = 50
    d_model = 64
    num_heads = 4
    num_layers = 2
    seq_len = 10
    batch_size = 2
    
    # Create sample input (token indices)
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample tokens: {input_ids.data[0, :5]}")
    
    # Test TinyGPT
    print("\n🤖 TinyGPT:")
    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_length=128
    )
    
    # Forward pass
    logits = model.forward(input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits sample: {logits.data[0, 0, :5]}")
    
    # Test generation
    print("\n📝 Text Generation:")
    start_tokens = Tensor(np.array([[1, 2, 3]]))  # Start with tokens 1, 2, 3
    generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated.data[0]}")
    
    # Test simple LM for comparison
    print("\n🔤 Simple LM (for comparison):")
    simple_model = SimpleLM(vocab_size=vocab_size, d_model=d_model)
    simple_logits = simple_model.forward(input_ids)
    print(f"Simple LM logits shape: {simple_logits.shape}")
    
    # Compare model sizes
    print("\n📊 Model Comparison:")
    print(f"TinyGPT parameters: ~{model.count_parameters():,}")
    simple_params = vocab_size * d_model + d_model * 256 + 256 * vocab_size
    print(f"Simple LM parameters: ~{simple_params:,}")
    print(f"TinyGPT is {model.count_parameters() / simple_params:.1f}x larger")
    
    print("\n✅ Model tests completed!")
    print("\n💡 Key insights:")
    print("   • TinyGPT successfully reuses TinyTorch Dense layers")
    print("   • Transformer architecture much more powerful than simple LM")
    print("   • Self-attention enables long-range dependencies")
    print("   • Autoregressive generation works out of the box")
    print("   • 🎉 Vision and language models share the same foundation!")