"""
Simple GPT model for CodeBot milestone - bypasses LayerNorm gradient bug.

This is a workaround for the milestone until core Tensor operations
(subtraction, mean) are fixed to maintain gradient flow.
"""

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.attention import MultiHeadAttention  
from tinytorch.core.activations import GELU
from tinytorch.text.embeddings import Embedding


class SimpleGPT:
    """
    Simplified GPT without LayerNorm (workaround for gradient flow bugs).
    
    Architecture:
    - Token + Position embeddings
    - N transformer blocks (attention + MLP, NO LayerNorm)
    - Output projection to vocabulary
    
    Note: This is a temporary solution for the milestone. The full GPT
    with LayerNorm requires fixes to core Tensor subtraction/mean operations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        mlp_ratio: int = 4
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.position_embedding = Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks (simplified - no LayerNorm)
        self.blocks = []
        for _ in range(num_layers):
            block = {
                'attention': MultiHeadAttention(embed_dim, num_heads),
                'mlp_fc1': Linear(embed_dim, embed_dim * mlp_ratio),
                'mlp_gelu': GELU(),  # Use tinytorch's GELU
                'mlp_fc2': Linear(embed_dim * mlp_ratio, embed_dim),
            }
            self.blocks.append(block)
        
        # Output projection
        self.lm_head = Linear(embed_dim, vocab_size)
    
    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass through simplified GPT.
        
        Args:
            tokens: Token indices, shape (batch_size, seq_len)
            
        Returns:
            logits: Predictions, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        
        # Embeddings
        token_emb = self.token_embedding.forward(tokens)
        positions = Tensor(np.arange(seq_len).reshape(1, seq_len))
        pos_emb = self.position_embedding.forward(positions)
        x = token_emb + pos_emb  # (batch, seq, embed)
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention with residual
            attn_out = block['attention'].forward(x)
            x = x + attn_out  # Residual connection
            
            # MLP with residual
            mlp_out = block['mlp_fc1'].forward(x)
            mlp_out = block['mlp_gelu'].forward(mlp_out)  # Activation
            mlp_out = block['mlp_fc2'].forward(mlp_out)
            x = x + mlp_out  # Residual connection
        
        # Project to vocabulary
        logits = self.lm_head.forward(x)
        return logits
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.token_embedding.parameters())
        params.extend(self.position_embedding.parameters())
        
        for block in self.blocks:
            params.extend(block['attention'].parameters())
            params.extend(block['mlp_fc1'].parameters())
            params.extend(block['mlp_fc2'].parameters())
        
        params.extend(self.lm_head.parameters())
        return params

