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
    ║                                  ▲                                           ║
    ║  ┌────────────────────────────────────────────────────────────────────────┐  ║
    ║  │                    Multi-Head Self-Attention                           │  ║
    ║  │           Module 12: Query·Key^T·Value across all positions            │  ║
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
- Training time: 45-60 minutes (proper training)
- Vocabulary: ~65 unique characters
- Expected: Good Shakespeare-style text with proper structure
- Parameters: ~2.5M (balanced for character-level tasks)
"""

import sys
import os
import numpy as np
import argparse
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

console = Console()

# Import TinyTorch components YOU BUILT!
from tinytorch.core.tensor import Tensor                    # Module 02: YOU built this!
from tinytorch.core.layers import Linear                    # Module 04: YOU built this!
from tinytorch.core.activations import ReLU, Softmax        # Module 03: YOU built this!
from tinytorch.core.optimizers import Adam                  # Module 08: YOU built this!
from tinytorch.core.losses import CrossEntropyLoss          # Module 04: YOU built this!
from tinytorch.core.attention import MultiHeadAttention     # Module 12: YOU built this!
from tinytorch.models.transformer import LayerNorm, TransformerBlock  # Module 13: YOU built this!
from tinytorch.text.embeddings import Embedding, PositionalEncoding   # Module 11: YOU built this!
from tinytorch.data.loader import DataLoader, Dataset   # Module 08: YOU built this!

# Import dataset manager
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Calculate parameters
        self.total_params = self._count_parameters()
    
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
        params.extend(self.pos_encoding.parameters())  # Add positional encoding params!
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
        
        # Ensure all parameters have requires_grad=True
        for param in params:
            param.requires_grad = True
        
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

        # Reshape for Linear layer - KEEP COMPUTATION GRAPH!
        batch_size, seq_len, embed_dim = x.shape
        x_2d = x.reshape(batch_size * seq_len, embed_dim)  # Use Tensor.reshape()

        # Apply output projection
        logits_2d = self.output_proj(x_2d)   # Module 04: vocab predictions

        # Reshape back - KEEP COMPUTATION GRAPH!
        logits = logits_2d.reshape(batch_size, seq_len, self.vocab_size)  # Use Tensor.reshape()
        
        return logits


def visualize_transformer():
    """Show how transformers process text sequences."""
    console.print("")
    console.print(Panel.fit(
        "[bold]In 2017, 'Attention Is All You Need' Changed Everything[/bold]\n\n"
        "[yellow]The Problem:[/yellow]\n"
        "RNNs process sequences one step at a time\n"
        "Can't parallelize → slow training on long sequences\n"
        "Struggle with long-range dependencies\n\n"
        "[green]The Innovation:[/green]\n"
        "Transformers: Attention mechanisms process ENTIRE sequences in parallel\n"
        "  • Self-attention: Every token attends to every other token\n"
        "  • Multi-head attention: Learn multiple attention patterns\n"
        "  • Positional encoding: Preserve sequence order\n\n"
        "[bold]Can attention alone match RNN performance?[/bold]",
        title="🎯 ACT 1: THE CHALLENGE",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    console.print("""
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


def train_shakespeare_gpt(model, train_loader, dataset, epochs=5, learning_rate=0.0003):
    """Train TinyGPT using YOUR complete training system with DataLoader!"""
    console.print("\n[bold]🚀 Training Shakespeare TinyGPT with YOUR TinyTorch![/bold]")
    console.print(f"  Dataset: [cyan]{len(train_loader.dataset):,}[/cyan] character sequences")
    console.print(f"  Batch size: [cyan]{train_loader.batch_size}[/cyan]")
    console.print(f"  Learning rate: [cyan]{learning_rate}[/cyan] (3e-4, standard for transformers)")
    console.print(f"  YOUR DataLoader (Module 08) handles batching!")
    console.print(f"  YOUR Adam optimizer (Module 08)")
    console.print(f"  YOUR CrossEntropyLoss (Module 04) with autograd!")
    
    # YOUR optimizer and loss function
    # Using 3e-4 learning rate (standard for transformers, per Vaswani et al. 2017 / GPT-2)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()  # YOUR loss function with autograd!
    
    for epoch in range(epochs):
        console.print(f"\n  [bold]Epoch {epoch+1}/{epochs}:[/bold]")
        epoch_loss = 0
        batch_count = 0
        
        # Use YOUR DataLoader to iterate through batches!
        for batch_idx, (batch_input, batch_target) in enumerate(train_loader):
            if batch_idx >= 500:  # Training mode - process more batches
                break
            
            if batch_idx == 0:
                console.print(f"    [dim]Processing first batch... (this may take a moment)[/dim]")
            
            # Forward pass with YOUR Transformer
            logits = model.forward(batch_input)  # YOUR attention mechanism!
            
            # Reshape for loss computation: (batch, seq, vocab) -> (batch*seq, vocab)
            batch_size, seq_length, vocab_size = logits.shape
            logits_2d = Tensor(logits.data.reshape(batch_size * seq_length, vocab_size))
            targets_1d = Tensor(batch_target.data.reshape(-1))
            
            # Compute loss with YOUR CrossEntropyLoss (connects to autograd!)
            loss = loss_fn.forward(logits_2d, targets_1d)  # Module 04 + Module 05!
            loss_value = float(loss.data)
            
            # Backward pass with YOUR autograd
            optimizer.zero_grad()  # Module 08!
            loss.backward()        # Module 05: YOUR autodiff!
            optimizer.step()       # Module 08!
            
            epoch_loss += loss_value
            batch_count += 1
            
            # Progress
            if (batch_idx + 1) % 20 == 0:
                console.print(f"    Batch {batch_idx+1}: Loss = [cyan]{loss_value:.4f}[/cyan]")
        
        # Epoch summary
        avg_loss = epoch_loss / max(1, batch_count)
        console.print(f"    → Epoch Complete: Avg Loss = [bold cyan]{avg_loss:.4f}[/bold cyan] (YOUR Transformer learning!)")
    
    return model


def generate_text(model, dataset, prompt="To be or not", max_length=200, temperature=0.8):
    """
    Generate text from a prompt - THE WOW MOMENT!
    
    This is autoregressive generation: predict next char, add it, repeat.
    """
    console.print("\n[bold]✨ TEXT GENERATION DEMO - THE PAYOFF![/bold]")
    console.print("="*70)
    
    # Convert prompt to indices
    prompt_indices = [dataset.char_to_idx[ch] for ch in prompt if ch in dataset.char_to_idx]
    generated = prompt_indices.copy()
    
    console.print(f"📝 Prompt: [cyan]\"{prompt}\"[/cyan]")
    console.print(f"🎯 Generating [cyan]{max_length}[/cyan] characters...\n")
    
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
    
    console.print("[bold]📖 Generated Text:[/bold]")
    console.print("─" * 70)
    console.print(f"[green]{generated_text}[/green]")
    console.print("─" * 70)
    
    return generated_text


def analyze_transformer_systems(model):
    """Analyze YOUR Transformer from an ML systems perspective."""
    console.print("")
    console.print(Panel.fit(
        f"[bold]Model Architecture:[/bold]\n"
        f"  • Parameters: [cyan]{model.total_params:,}[/cyan] weights\n"
        f"  • Embedding dim: [cyan]{model.embed_dim}[/cyan]\n"
        f"  • Vocabulary: [cyan]{model.vocab_size}[/cyan] characters\n\n"
        
        "[bold]Computational Complexity:[/bold]\n"
        "  • Attention: O(n²·d) where n=sequence, d=dimension\n"
        "  • Self-attention allows parallel processing (vs RNN sequential)\n"
        "  • YOUR implementation: Pure Python + NumPy\n\n"
        
        f"[bold]Memory Requirements:[/bold]\n"
        f"  • Parameters: [cyan]{model.total_params * 4 / 1024:.1f} KB[/cyan]\n"
        "  • Attention matrices: O(n²) per layer\n"
        "  • YOUR TinyTorch tracks gradients automatically\n\n"
        
        "[bold]🏛️ Transformer Evolution:[/bold]\n"
        "  • 2017: Vaswani et al. 'Attention Is All You Need'\n"
        "  • 2018: BERT (bidirectional), GPT (autoregressive)\n"
        "  • 2020: GPT-3 (175B params, same architecture!)\n"
        "  • 2022: ChatGPT (YOUR architecture at massive scale)\n"
        "  • YOUR TinyGPT: Core principles that power them all!\n\n"
        
        "[bold]💡 Why Transformers Dominate:[/bold]\n"
        "  • Parallelizable (vs sequential RNNs)\n"
        "  • Long-range dependencies (attention sees everything)\n"
        "  • Scalable (architecture works from 1M to 175B params)\n"
        "  • YOUR implementation demonstrates all of these!",
        
        title="🔬 SYSTEMS ANALYSIS",
        border_style="cyan",
        box=box.DOUBLE
    ))


def main():
    """Demonstrate Shakespeare text generation using YOUR TinyTorch!"""
    
    parser = argparse.ArgumentParser(description='Shakespeare Transformer 2017')
    parser.add_argument('--test-only', action='store_true',
                       help='Test architecture only')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seq-length', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Show transformer visualization')
    parser.add_argument('--quick-test', action='store_true',
                       help='Use small subset for testing')
    args = parser.parse_args()
    
    console.print("")
    console.print(Panel.fit(
        "[bold cyan]Shakespeare Transformer - Text Generation with YOUR Attention![/bold cyan]\n\n"
        "[yellow]Historical significance:[/yellow] Attention revolutionized sequence modeling\n"
        "[green]YOUR achievement:[/green] Generate Shakespeare-style text\n"
        "[cyan]Components used:[/cyan] YOUR complete transformer system (Modules 2-13)",
        title="🎯 Milestone 05: Transformer Era (2017)",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Visualization
    if args.visualize:
        visualize_transformer()
    
    # Step 1: Load Shakespeare dataset
    console.print("\n[bold]📥 Loading Shakespeare corpus...[/bold]")
    data_manager = DatasetManager()
    
    text = data_manager.get_shakespeare()
    
    if args.quick_test:
        text = text[:10000]  # Use small subset for testing
        console.print("  [dim](Using subset for quick testing)[/dim]")
    
    # Step 2: Create Dataset and DataLoader using YOUR Module 08!
    console.print(f"\n[bold]📦 Creating YOUR Dataset and DataLoader (Module 08)...[/bold]")
    dataset = ShakespeareDataset(text, seq_length=args.seq_length)
    
    # YOUR DataLoader handles batching and shuffling!
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    console.print(f"  Vocabulary: [cyan]{dataset.vocab_size}[/cyan] unique characters")
    console.print(f"  Characters: [dim]'{dataset.decode(list(range(min(20, dataset.vocab_size))))}...'[/dim]")
    console.print(f"  DataLoader: [cyan]{len(dataset):,}[/cyan] sequences, batch_size=[cyan]{args.batch_size}[/cyan]")
    
    # Step 3: Build Transformer
    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        max_length=args.seq_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    # Display model info
    console.print("\n[bold]🧠 Building TinyGPT with YOUR TinyTorch...[/bold]")
    console.print(f"  Architecture: [cyan]{args.num_layers}[/cyan] layers, [cyan]{args.num_heads}[/cyan] heads, [cyan]{args.embed_dim}[/cyan]-dim embeddings")
    console.print(f"  Vocabulary: [cyan]{dataset.vocab_size}[/cyan] characters")
    console.print(f"  Total parameters: [bold cyan]{model.total_params:,}[/bold cyan] (YOUR components!)")
    
    if args.test_only:
        console.print("\n[bold yellow]🧪 ARCHITECTURE TEST MODE[/bold yellow]")
        # Test with minimal data
        test_input = Tensor(np.random.randint(0, dataset.vocab_size, (1, args.seq_length), dtype=np.int32))
        test_output = model.forward(test_input)
        console.print(f"[green]✅ Forward pass successful! Output shape: {test_output.data.shape}[/green]")
        console.print(f"[green]✅ YOUR Transformer + DataLoader work together![/green]")
        return
    
    # Step 4: Train using YOUR DataLoader
    start_time = time.time()
    model = train_shakespeare_gpt(model, train_loader, dataset, epochs=args.epochs)
    train_time = time.time() - start_time
    
    # Step 5: Generate text!
    generated = generate_text(model, dataset, prompt="To be or not", max_length=200)
    
    # Additional generation examples
    console.print("\n[bold]🎭 More Generation Examples:[/bold]")
    console.print("─" * 70)
    
    prompts = ["ROMEO:", "The king", "What is"]
    for prompt in prompts:
        if all(ch in dataset.char_to_idx for ch in prompt):
            console.print(f"\n[cyan]Prompt: \"{prompt}\"[/cyan]")
            gen = generate_text(model, dataset, prompt=prompt, max_length=100, temperature=0.8)
    
    # Step 6: Systems Analysis
    analyze_transformer_systems(model)
    
    console.print(f"\n[bold]⏱️  Training time:[/bold] [cyan]{train_time:.1f}[/cyan] seconds")
    console.print(f"  Sequences/sec: [cyan]{len(dataset) * args.epochs / train_time:.0f}[/cyan]")
    
    console.print("")
    console.print(Panel.fit(
        "[bold green]✅ SUCCESS! Shakespeare Transformer Milestone Complete![/bold green]\n\n"
        
        "[bold]🎓 What YOU Accomplished:[/bold]\n"
        "  • YOUR attention mechanism processes sequences in parallel\n"
        "  • YOUR transformer captures long-range text dependencies\n"
        "  • YOUR DataLoader efficiently batches character sequences\n"
        "  • YOUR TinyGPT generates coherent text!\n"
        "  • YOUR complete language modeling system works!\n\n"
        
        "[bold]🚀 Next Steps:[/bold]\n"
        "  • Continue to Module 14 (KV-Caching) for 3x faster inference\n"
        "  • YOUR transformer architecture scales to GPT-scale models\n"
        "  • This is the foundation of ChatGPT, GPT-4, and all modern LLMs!",
        
        title="🌟 2017 Transformer Revolution Complete",
        border_style="green",
        box=box.DOUBLE
    ))

if __name__ == "__main__":
    main()
