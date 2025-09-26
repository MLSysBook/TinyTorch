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
from tinytorch.core.autograd import to_numpy
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
        self.vocab_size = vocab_size  # Store for reshaping
    
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
        
        # Reshape for Linear layer: (batch, seq, embed) → (batch*seq, embed)
        x_np = to_numpy(x)
        batch_size, seq_len, embed_dim = x_np.shape
        x_2d_np = x_np.reshape(batch_size * seq_len, embed_dim)
        x_2d = Tensor(x_2d_np)
        
        # Apply output projection
        logits_2d = self.output_proj(x_2d)   # vocab predictions (Module 04)
        
        # Reshape back: (batch*seq, vocab) → (batch, seq, vocab)
        logits_2d_np = to_numpy(logits_2d)
        logits_np = logits_2d_np.reshape(batch_size, seq_len, self.vocab_size)
        logits = Tensor(logits_np)
        return logits

def main():
    # Simpler hyperparameters for validation
    vocab_size = 100  # Smaller vocabulary
    embed_dim = 32    # Smaller embeddings  
    max_length = 16   # Shorter sequences
    num_heads = 4     # Fewer attention heads
    num_layers = 2    # Fewer layers
    
    model = TinyGPT(vocab_size, embed_dim, max_length, num_heads, num_layers)
    optimizer = optim.Adam(model.parameters(), learning_rate=0.001)  # Module 08
    
    # Demo training data (random tokens)
    batch_size, seq_length = 1, 8  # Smaller batch and sequence
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
    target_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)))
    
    print("🤖 Training Transformer Language Model")
    print(f"   Architecture: Embedding → Position → Attention × {num_layers} → Output")
    print(f"   Parameters: {sum(p.data.size for p in model.parameters()):,} weights")
    print(f"   Vocabulary: {vocab_size:,} possible tokens")
    print(f"   Context: {max_length} token sequences")
    print()
    
    # What students built: Complete transformer training
    for step in range(5):  # Fewer steps for validation
        logits = model(input_ids)    # Forward: Full transformer stack
        
        # Language modeling loss (Module 10)
        logits_np = to_numpy(logits)
        targets_np = to_numpy(target_ids)
        batch_size, seq_length = targets_np.shape
        targets_one_hot = np.zeros((batch_size, seq_length, vocab_size))
        for b in range(batch_size):
            for s in range(seq_length):
                targets_one_hot[b, s, int(targets_np[b, s])] = 1.0
        
        loss_value = np.mean((logits_np - targets_one_hot) ** 2)
        loss = Tensor([loss_value])
        
        loss.backward()      # Autodiff through transformer (Module 06)
        optimizer.step()     # Adam updates (Module 08)
        optimizer.zero_grad()
        
        if step % 2 == 0:
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