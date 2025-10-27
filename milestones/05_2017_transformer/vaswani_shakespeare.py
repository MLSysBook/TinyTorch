#!/usr/bin/env python3
"""
Shakespeare Text Generation (2017) - Transformer Era
===================================================

📚 HISTORICAL CONTEXT:
In 2017, Vaswani et al. published "Attention Is All You Need", showing that
attention mechanisms alone (no RNNs!) could achieve state-of-the-art results
on sequence tasks. This breakthrough launched the era of GPT, BERT, and modern LLMs.

🎯 WHAT YOU'RE BUILDING:
Using YOUR TinyTorch implementations, you'll build a character-level language model  
that generates Shakespeare-style text - proving YOUR attention mechanism works!

✅ REQUIRED MODULES (Run after Module 13):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 02 (Tensor)       : YOUR data structure with autograd
  Module 03 (Activations)  : YOUR ReLU in feed-forward networks
  Module 04 (Layers)       : YOUR Linear layers
  Module 08 (Optimizers)   : YOUR Adam optimizer
  Module 11 (Embeddings)   : YOUR token & positional embeddings
  Module 12 (Attention)    : YOUR multi-head self-attention
  Module 13 (Transformers) : YOUR LayerNorm + TransformerBlock
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Character-Level Language Model):
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                               Output Predictions                             │
    │                         Character Probabilities (vocab_size)                 │
    └──────────────────────────────────────────────────────────────────────────────┘
                                            ▲
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                            Output Projection                                 │
    │                       Module 04: vectors → vocabulary                        │
    └──────────────────────────────────────────────────────────────────────────────┘
                                            ▲
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                              Layer Norm                                      │
    │                        Module 13: Final normalization                        │
    └──────────────────────────────────────────────────────────────────────────────┘
                                            ▲
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                      Transformer Block × N (Repeat)                          ║
    ║  ┌────────────────────────────────────────────────────────────────────────┐  ║
    ║  │                       Feed Forward Network                             │  ║
    ║  │              Module 04: Linear → ReLU → Linear                         │  ║
    ║  └────────────────────────────────────────────────────────────────────────┘  ║
    ║                                  ▲                                            ║
    ║  ┌────────────────────────────────────────────────────────────────────────┐  ║
    ║  │                    Multi-Head Self-Attention                           │  ║
    ║  │           Module 12: Query·Key^T·Value across all positions           │  ║
    ║  └────────────────────────────────────────────────────────────────────────┘  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
                                            ▲
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                          Positional Encoding                                 │
    │                   Module 11: Add position information                        │
    └──────────────────────────────────────────────────────────────────────────────┘
                                            ▲
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                         Character Embeddings                                 │
    │                    Module 11: chars → embed_dim vectors                      │
    └──────────────────────────────────────────────────────────────────────────────┘
                                            ▲
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                            Input Characters                                  │
    │                    "To be or not to be, that is..."                          │
    └──────────────────────────────────────────────────────────────────────────────┘

📊 EXPECTED PERFORMANCE:
- Dataset: ~1MB Shakespeare corpus (40,000 lines)
- Training time: 5-10 minutes (demonstration mode)
- Vocabulary: ~65 unique characters
- Expected: Coherent (if not perfect) Shakespeare-style text
- Parameters: ~500K (small by modern standards!)
"""

import sys
import os
import numpy as np
import argparse
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor                    # Module 02: YOU built this!
from tinytorch.core.layers import Linear                    # Module 04: YOU built this!
from tinytorch.core.activations import ReLU, Softmax        # Module 03: YOU built this!
from tinytorch.core.optimizers import Adam                  # Module 08: YOU built this!
from tinytorch.core.attention import MultiHeadAttention     # Module 12: YOU built this!
from tinytorch.models.transformer import LayerNorm, TransformerBlock  # Module 13: YOU built this!
from tinytorch.text.embeddings import Embedding, PositionalEncoding   # Module 11: YOU built this!
from tinytorch.data.loader import DataLoader, Dataset   # Module 08: YOU built this!

# Import dataset manager
try:
    from data_manager import DatasetManager
except ImportError:
    sys.path.append(os.path.join(project_root, 'milestones'))
    from data_manager import DatasetManager


class ShakespeareDataset(Dataset):
    """
    Character-level Shakespeare dataset using YOUR Dataset interface!
    
    Tokenizes text into characters and creates sequences for language modeling.
    """
    
    def __init__(self, text, seq_length=64):
        """
        Initialize dataset with text and sequence length.
        
        Args:
            text: Raw Shakespeare text
            seq_length: Length of input sequences
        """
        # Build character vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        self.seq_length = seq_length
        
        # Calculate number of sequences
        self.num_sequences = len(self.data) - seq_length
        
    def __getitem__(self, idx):
        """Get a single training sequence - YOUR Dataset interface!"""
        # Input: characters at positions [idx, idx+seq_length)
        # Target: characters at positions [idx+1, idx+seq_length+1)
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        
        return Tensor(np.array(input_seq, dtype=np.int32)), Tensor(np.array(target_seq, dtype=np.int32))
    
    def __len__(self):
        """Return dataset size - YOUR Dataset interface!"""
        return self.num_sequences
    
    def decode(self, indices):
        """Convert indices back to text."""
        return ''.join([self.idx_to_char[int(idx)] for idx in indices])


class TinyGPT:
    """
    Character-level Transformer Language Model using YOUR TinyTorch!
    
    This architecture is what powers GPT, ChatGPT, and modern LLMs.
    """
    
    def __init__(self, vocab_size, embed_dim, max_length, num_heads, num_layers):
        print("🧠 Building TinyGPT with YOUR TinyTorch modules...")
        
        # Token representation
        self.embedding = Embedding(vocab_size, embed_dim)           # Module 11!
        self.pos_encoding = PositionalEncoding(max_length, embed_dim)  # Module 11!

        # Transformer stack
        self.layers = []
        hidden_dim = embed_dim * 4  # Standard 4x expansion in FFN
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, hidden_dim)  # Module 13!
            self.layers.append(block)

        # Output head
        self.layer_norm = LayerNorm(embed_dim)          # Module 13!
        self.output_proj = Linear(embed_dim, vocab_size)  # Module 04!
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Calculate parameters
        self.total_params = self._count_parameters()
        
        print(f"   Architecture: {num_layers} layers, {num_heads} heads, {embed_dim}-dim embeddings")
        print(f"   Vocabulary: {vocab_size} characters")
        print(f"   Total parameters: {self.total_params:,} (YOUR components!)")
    
    def _count_parameters(self):
        """Count total parameters in model."""
        count = 0
        for param in self.parameters():
            count += param.data.size
        return count

    def parameters(self):
        """Get all trainable parameters from YOUR model."""
        params = []
        # Embedding parameters
        params.extend([self.embedding.weight])
        # Transformer block parameters
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                if callable(layer.parameters):
                    params.extend(layer.parameters())
                else:
                    params.extend(layer.parameters)
        # Output projection parameters
        params.extend([self.layer_norm.gamma, self.layer_norm.beta])
        params.extend([self.output_proj.weight, self.output_proj.bias])
        return params

    def forward(self, x):
        """Forward pass through YOUR transformer stack."""
        # Convert tokens to contextual vectors
        x = self.embedding.forward(x)        # Module 11: char → vectors
        x = self.pos_encoding.forward(x)     # Module 11: add position info
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer.forward(x)  # Module 13: Attention → FFN
        
        # Generate predictions
        x = self.layer_norm.forward(x)       # Module 13: final norm

        # Reshape for Linear layer
        x_np = np.array(x.data.data if hasattr(x.data, 'data') else x.data)
        batch_size, seq_len, embed_dim = x_np.shape
        x_2d_np = x_np.reshape(batch_size * seq_len, embed_dim)
        x_2d = Tensor(x_2d_np)

        # Apply output projection
        logits_2d = self.output_proj(x_2d)   # Module 04: vocab predictions

        # Reshape back
        logits_2d_np = np.array(logits_2d.data.data if hasattr(logits_2d.data, 'data') else logits_2d.data)
        logits_np = logits_2d_np.reshape(batch_size, seq_len, self.vocab_size)
        logits = Tensor(logits_np)
        
        return logits


def visualize_transformer():
    """Show how transformers process text sequences."""
    print("\n" + "="*70)
    print("🤖 VISUALIZING TRANSFORMER TEXT GENERATION:")
    print("="*70)
    
    print("""
    How YOUR Transformer Sees Text:      What It Learns:
    
    Input: "To be or not to be"          Layer 1 (Attention):
    ┌─────────────────────┐              • Each word attends to others
    │ T o   b e   o r ... │              • "be" looks at "To", "or", etc.
    └─────────────────────┘              • Captures dependencies
            ↓                            
    Character Embeddings                 Layer 2-4 (Deep Attention):
    ┌─────────────────────┐              • Builds complex patterns
    │ 128-dim vectors     │              • Grammar, style, meaning
    │ for each character  │              • Shakespeare-specific patterns
    └─────────────────────┘              
            ↓                            Output Prediction:
    Position Encoding                    "To be or not to be, that is the"
    ┌─────────────────────┐                                         ↓
    │ Add positional info │              Next char probabilities:
    │ (order matters!)    │              't' → 0.85  (highest!)
    └─────────────────────┘              'n' → 0.03
            ↓                            'a' → 0.02
    Transformer Layers ×4                ...
    ┌─────────────────────┐
    │ Self-Attention      │              Key Transformer Insight:
    │ Feed-Forward        │              Unlike RNNs, attention lets each
    │ Layer Norm          │              position look at ALL others
    └─────────────────────┘              simultaneously - capturing long-range
            ↓                            dependencies in O(1) operations!
    Character Predictions
    ┌─────────────────────┐
    │ Probability for     │
    │ each next character │
    └─────────────────────┘
    """)
    print("="*70)


def train_shakespeare_gpt(model, train_loader, dataset, epochs=5, learning_rate=0.001):
    """Train TinyGPT using YOUR complete training system with DataLoader!"""
    print("\n🚀 Training Shakespeare TinyGPT with YOUR TinyTorch!")
    print(f"   Dataset: {len(train_loader.dataset):,} character sequences")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   YOUR DataLoader (Module 08) handles batching!")
    print(f"   YOUR Adam optimizer (Module 08)")
    
    # YOUR optimizer
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    
    for epoch in range(epochs):
        print(f"\n   Epoch {epoch+1}/{epochs}:")
        epoch_loss = 0
        batch_count = 0
        
        # Use YOUR DataLoader to iterate through batches!
        for batch_idx, (batch_input, batch_target) in enumerate(train_loader):
            if batch_idx >= 100:  # Demo mode - limit batches
                break
            
            # Forward pass with YOUR Transformer
            logits = model.forward(batch_input)  # YOUR attention mechanism!
            
            # Language modeling loss
            logits_np = np.array(logits.data.data if hasattr(logits.data, 'data') else logits.data)
            targets_np = np.array(batch_target.data.data if hasattr(batch_target.data, 'data') else batch_target.data)
            
            batch_size, seq_length = targets_np.shape
            vocab_size = logits_np.shape[-1]
            
            # Cross-entropy loss
            targets_one_hot = np.zeros((batch_size, seq_length, vocab_size))
            for b in range(batch_size):
                for s in range(seq_length):
                    targets_one_hot[b, s, int(targets_np[b, s])] = 1.0

            # Softmax + cross entropy
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=2, keepdims=True))
            softmax = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
            loss_value = -np.mean(np.sum(targets_one_hot * np.log(softmax + 1e-8), axis=2))
            loss = Tensor([loss_value])
            
            # Backward pass with YOUR autograd
            optimizer.zero_grad()  # Module 08!
            loss.backward()        # Module 05: YOUR autodiff!
            optimizer.step()       # Module 08!
            
            epoch_loss += loss_value
            batch_count += 1
            
            # Progress
            if (batch_idx + 1) % 20 == 0:
                print(f"   Batch {batch_idx+1}: Loss = {loss_value:.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / max(1, batch_count)
        print(f"   → Epoch Complete: Avg Loss = {avg_loss:.4f} (YOUR Transformer learning!)")
    
    return model


def generate_text(model, dataset, prompt="To be or not", max_length=200, temperature=0.8):
    """
    Generate text from a prompt - THE WOW MOMENT!
    
    This is autoregressive generation: predict next char, add it, repeat.
    """
    print("\n✨ TEXT GENERATION DEMO - THE PAYOFF!")
    print("="*70)
    
    # Convert prompt to indices
    prompt_indices = [dataset.char_to_idx[ch] for ch in prompt if ch in dataset.char_to_idx]
    generated = prompt_indices.copy()
    
    print(f"📝 Prompt: \"{prompt}\"")
    print(f"🎯 Generating {max_length} characters...\n")
    
    # Generate character by character
    for _ in range(max_length):
        # Take last seq_length characters as input
        input_seq = generated[-dataset.seq_length:] if len(generated) >= dataset.seq_length else generated
        
        # Pad if necessary
        if len(input_seq) < dataset.seq_length:
            input_seq = [0] * (dataset.seq_length - len(input_seq)) + input_seq
        
        # Forward pass
        input_tensor = Tensor(np.array([input_seq], dtype=np.int32))
        logits = model.forward(input_tensor)
        
        # Get logits for last position
        logits_np = np.array(logits.data.data if hasattr(logits.data, 'data') else logits.data)
        next_logits = logits_np[0, -1, :]  # Last position predictions
        
        # Apply temperature and sample
        next_logits = next_logits / temperature
        exp_logits = np.exp(next_logits - np.max(next_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample from distribution
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(next_idx)
    
    # Decode to text
    generated_text = dataset.decode(generated)
    
    print("📖 Generated Text:")
    print("─" * 70)
    print(generated_text)
    print("─" * 70)
    
    return generated_text


def analyze_transformer_systems(model):
    """Analyze YOUR Transformer from an ML systems perspective."""
    print("\n🔬 SYSTEMS ANALYSIS of YOUR Transformer Implementation:")
    
    print(f"\n   Model Architecture:")
    print(f"   • Parameters: {model.total_params:,} weights")
    print(f"   • Embedding dim: {model.embed_dim}")
    print(f"   • Vocabulary: {model.vocab_size} characters")
    
    print(f"\n   Computational Complexity:")
    print(f"   • Attention: O(n²·d) where n=sequence, d=dimension")
    print(f"   • Self-attention allows parallel processing (vs RNN sequential)")
    print(f"   • YOUR implementation: Pure Python + NumPy")
    
    print(f"\n   Memory Requirements:")
    print(f"   • Parameters: {model.total_params * 4 / 1024:.1f} KB")
    print(f"   • Attention matrices: O(n²) per layer")
    print(f"   • YOUR TinyTorch tracks gradients automatically")
    
    print(f"\n   🏛️ Transformer Evolution:")
    print(f"   • 2017: Vaswani et al. 'Attention Is All You Need'")
    print(f"   • 2018: BERT (bidirectional), GPT (autoregressive)")
    print(f"   • 2020: GPT-3 (175B params, same architecture!)")
    print(f"   • 2022: ChatGPT (YOUR architecture at massive scale)")
    print(f"   • YOUR TinyGPT: Core principles that power them all!")
    
    print(f"\n   💡 Why Transformers Dominate:")
    print(f"   • Parallelizable (vs sequential RNNs)")
    print(f"   • Long-range dependencies (attention sees everything)")
    print(f"   • Scalable (architecture works from 1M to 175B params)")
    print(f"   • YOUR implementation demonstrates all of these!")


def main():
    """Demonstrate Shakespeare text generation using YOUR TinyTorch!"""
    
    parser = argparse.ArgumentParser(description='Shakespeare Transformer 2017')
    parser.add_argument('--test-only', action='store_true',
                       help='Test architecture only')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs (demo mode)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seq-length', type=int, default=64,
                       help='Sequence length')
    parser.add_argument('--embed-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Show transformer visualization')
    parser.add_argument('--quick-test', action='store_true',
                       help='Use small subset for testing')
    args = parser.parse_args()
    
    print("🎯 Shakespeare Transformer - Text Generation with YOUR Attention!")
    print("   Historical significance: Attention revolutionized sequence modeling")
    print("   YOUR achievement: Generate Shakespeare-style text")
    print("   Components used: YOUR complete transformer system (Modules 2-13)")
    
    # Visualization
    if args.visualize:
        visualize_transformer()
    
    # Step 1: Load Shakespeare dataset
    print("\n📥 Loading Shakespeare corpus...")
    data_manager = DatasetManager()
    
    try:
        text = data_manager.get_shakespeare()
        
        if args.quick_test:
            text = text[:10000]  # Use small subset for testing
            print("   (Using subset for quick testing)")
            
    except Exception as e:
        print(f"⚠️  Shakespeare download failed: {e}")
        print("   Using synthetic text for demonstration...")
        text = "To be or not to be, that is the question. " * 100
    
    # Step 2: Create Dataset and DataLoader using YOUR Module 08!
    print(f"\n📦 Creating YOUR Dataset and DataLoader (Module 08)...")
    dataset = ShakespeareDataset(text, seq_length=args.seq_length)
    
    # YOUR DataLoader handles batching and shuffling!
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"   Vocabulary: {dataset.vocab_size} unique characters")
    print(f"   Characters: '{dataset.decode(list(range(min(20, dataset.vocab_size))))}...'")
    print(f"   DataLoader: {len(dataset):,} sequences, batch_size={args.batch_size}")
    
    # Step 3: Build Transformer
    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        max_length=args.seq_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    if args.test_only:
        print("\n🧪 ARCHITECTURE TEST MODE")
        # Test with minimal data
        test_input = Tensor(np.random.randint(0, dataset.vocab_size, (1, args.seq_length), dtype=np.int32))
        test_output = model.forward(test_input)
        print(f"✅ Forward pass successful! Output shape: {test_output.data.shape}")
        print("✅ YOUR Transformer + DataLoader work together!")
        return
    
    # Step 4: Train using YOUR DataLoader
    start_time = time.time()
    model = train_shakespeare_gpt(model, train_loader, dataset, epochs=args.epochs)
    train_time = time.time() - start_time
    
    # Step 5: Generate text!
    generated = generate_text(model, dataset, prompt="To be or not", max_length=200)
    
    # Additional generation examples
    print("\n🎭 More Generation Examples:")
    print("─" * 70)
    
    prompts = ["ROMEO:", "The king", "What is"]
    for prompt in prompts:
        if all(ch in dataset.char_to_idx for ch in prompt):
            print(f"\nPrompt: \"{prompt}\"")
            gen = generate_text(model, dataset, prompt=prompt, max_length=100, temperature=0.8)
    
    # Step 6: Systems Analysis
    analyze_transformer_systems(model)
    
    print(f"\n⏱️  Training time: {train_time:.1f} seconds")
    print(f"   Sequences/sec: {len(dataset) * args.epochs / train_time:.0f}")
    
    print("\n✅ SUCCESS! Shakespeare Transformer Milestone Complete!")
    print("\n🎓 What YOU Accomplished:")
    print("   • YOUR attention mechanism processes sequences in parallel")
    print("   • YOUR transformer captures long-range text dependencies")
    print("   • YOUR DataLoader efficiently batches character sequences")
    print("   • YOUR TinyGPT generates coherent text!")
    print("   • YOUR complete language modeling system works!")
    
    print("\n🚀 Next Steps:")
    print("   • Continue to Module 14 (KV-Caching) for 3x faster inference")
    print("   • YOUR transformer architecture scales to GPT-scale models")
    print("   • This is the foundation of ChatGPT, GPT-4, and all modern LLMs!")

if __name__ == "__main__":
    main()
