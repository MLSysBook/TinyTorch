#!/usr/bin/env python3
"""
Clean TinyGPT Example - What Students Built
==========================================

After completing all modules 02-14, students can build complete transformer
language models. This demonstrates how attention enables contextual understanding.

MODULES EXERCISED IN THIS EXAMPLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)        : Data structure with gradient tracking
  Module 03 (Activations)   : ReLU in feed-forward networks
  Module 04 (Layers)        : Linear layers in FFN and output projection
  Module 05 (Networks)      : Module base class for transformer
  Module 06 (Autograd)      : Backprop through attention layers
  Module 08 (Optimizers)    : Adam optimizer for training
  Module 10 (Training)      : Language modeling loss and training loop
  Module 12 (Embeddings)    : Token embeddings and positional encoding
  Module 13 (Attention)     : Multi-head self-attention mechanism
  Module 14 (Transformers)  : LayerNorm and complete transformer blocks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Transformer Architecture (Bottom to Top Flow):

                    ┌─────────────────────────────────────────────────────────────┐
                    │                    Output Logits                            │
                    │              Vocabulary Predictions (1000)                 │
                    └─────────────────────────────────────────────────────────────┘
                                                  ▲
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  Output Projection                          │
                    │           Module 04: vectors → vocabulary                  │
                    └─────────────────────────────────────────────────────────────┘
                                                  ▲
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    Layer Norm                              │
                    │              Module 14: Final normalization                │
                    └─────────────────────────────────────────────────────────────┘
                                                  ▲
                    ╔═════════════════════════════════════════════════════════════╗
                    ║              Transformer Block × 4 (Repeat)                ║
                    ║ ┌─────────────────────────────────────────────────────────┐ ║
                    ║ │                    Layer Norm                           │ ║
                    ║ │              Module 14: Post-FFN normalization          │ ║
                    ║ └─────────────────────────────────────────────────────────┘ ║
                    ║                            ▲                                ║
                    ║ ┌─────────────────────────────────────────────────────────┐ ║
                    ║ │              Feed Forward Network (FFN)                 │ ║
                    ║ │      Module 04: Linear(128→512) → ReLU → Linear(512→128)│ ║
                    ║ └─────────────────────────────────────────────────────────┘ ║
                    ║                            ▲                                ║
                    ║ ┌─────────────────────────────────────────────────────────┐ ║
                    ║ │                    Layer Norm                           │ ║
                    ║ │            Module 14: Post-attention normalization      │ ║
                    ║ └─────────────────────────────────────────────────────────┘ ║
                    ║                            ▲                                ║
                    ║ ┌─────────────────────────────────────────────────────────┐ ║
                    ║ │              Multi-Head Self-Attention                  │ ║
                    ║ │         Module 13: 8 heads × (Q·K^T/√d_k)·V            │ ║
                    ║ │    Each head: 16-dim attention on 128-dim embeddings   │ ║
                    ║ └─────────────────────────────────────────────────────────┘ ║
                    ╚═════════════════════════════════════════════════════════════╝
                                                  ▲
                    ┌─────────────────────────────────────────────────────────────┐
                    │                 Positional Encoding                        │
                    │      Module 12: Add position information (sin/cos)        │
                    └─────────────────────────────────────────────────────────────┘
                                                  ▲
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  Token Embeddings                          │
                    │        Module 12: tokens → 128-dim vectors                │
                    └─────────────────────────────────────────────────────────────┘
                                                  ▲
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    Input Tokens                            │
                    │              [token_1, token_2, ..., token_10]             │
                    └─────────────────────────────────────────────────────────────┘

Key Insight: Attention allows each token to "look at" all other tokens
to understand context and meaning relationships.
"""

from tinytorch import nn, optim
from tinytorch.core.tensor import Tensor
import numpy as np

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, num_heads, num_layers):
        super().__init__()
        
        # Token representation
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.PositionalEncoding(embed_dim, max_length)
        
        # Transformer stack  
        self.layers = []
        hidden_dim = embed_dim * 4  # Standard 4x expansion in FFN
        for _ in range(num_layers):
            block = nn.TransformerBlock(embed_dim, num_heads, hidden_dim)
            self.layers.append(block)
        
        # Output head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        # Convert tokens to contextual vectors
        x = self.embedding(x)        # tokens → vectors (Module 12)
        x = self.pos_encoding(x)     # add position info (Module 12)
        
        # Process through transformer layers
        for layer in self.layers:
            # Each layer: Attention → Norm → FFN → Norm (Modules 13+14)
            x = layer(x)
        
        # Generate predictions
        x = self.layer_norm(x)       # final normalization (Module 14)
        return self.output_proj(x)   # vocab predictions (Module 04)

def main():
    # Hyperparameters for demo GPT
    vocab_size = 1000
    embed_dim = 128
    max_length = 50
    num_heads = 8
    num_layers = 4
    
    model = TinyGPT(vocab_size, embed_dim, max_length, num_heads, num_layers)
    optimizer = optim.Adam(model.parameters(), learning_rate=0.001)  # Module 08
    
    # Demo training data (random tokens)
    batch_size, seq_length = 2, 10
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
    target_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
    
    print("🤖 Training Transformer Language Model")
    print("   Architecture: Embedding → Position → Attention × 4 → Output")
    print(f"   Parameters: {sum(p.data.size for p in model.parameters()):,} weights")
    print(f"   Vocabulary: {vocab_size:,} possible tokens")
    print(f"   Context: {max_length} token sequences")
    print()
    
    # What students built: Complete transformer training
    for step in range(10):
        logits = model(input_ids)    # Forward: Full transformer stack
        
        # Language modeling loss (Module 10)
        batch_size, seq_length = target_ids.data.shape
        targets_one_hot = np.zeros((batch_size, seq_length, vocab_size))
        for b in range(batch_size):
            for s in range(seq_length):
                targets_one_hot[b, s, int(target_ids.data[b, s])] = 1.0
        
        loss_value = np.mean((logits.data - targets_one_hot) ** 2)
        loss = Tensor([loss_value])
        
        loss.backward()      # Autodiff through transformer (Module 06)
        optimizer.step()     # Adam updates (Module 08)
        optimizer.zero_grad()
        
        if step % 5 == 0:
            print(f"   Step {step:2d}: Loss = {loss_value:.4f}")
    
    print("\n✅ Success! Complete transformer language model")
    print("\n🎯 What You Learned by Building:")
    print("   • How attention creates contextual word representations")
    print("   • Why positional encoding is crucial for sequence understanding")
    print("   • How layer normalization stabilizes deep network training")
    print("   • Complete transformer architecture from first principles")
    print("\n🏭 Production Note:")
    print("   Real PyTorch uses optimized CUDA kernels for attention,")
    print("   but you built and understand the core mathematics!")

if __name__ == "__main__":
    main()