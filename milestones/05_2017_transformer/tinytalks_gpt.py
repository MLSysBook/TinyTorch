#!/usr/bin/env python3
"""
TinyTalks Q&A Generation (2017) - Transformer Era
==================================================

📚 HISTORICAL CONTEXT:
In 2017, Vaswani et al. published "Attention Is All You Need", showing that
attention mechanisms alone (no RNNs!) could achieve state-of-the-art results
on sequence tasks. This breakthrough launched the era of GPT, BERT, and modern LLMs.

🎯 WHAT YOU'RE BUILDING:
Using YOUR TinyTorch implementations, you'll build a character-level conversational
model that learns to answer questions - proving YOUR attention mechanism works!

TinyTalks is PERFECT for learning:
- Small dataset (17.5 KB) = 3-5 minute training!
- Clear Q&A format (easy to verify learning)
- Progressive difficulty (5 levels)
- Instant gratification: Watch your transformer learn to chat!

✅ REQUIRED MODULES (Run after Module 13):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Module 01 (Tensor)        : YOUR data structure with autograd
  Module 02 (Activations)   : YOUR ReLU and GELU activations
  Module 03 (Layers)        : YOUR Linear layers
  Module 04 (Losses)        : YOUR CrossEntropyLoss
  Module 05 (Autograd)      : YOUR automatic differentiation
  Module 06 (Optimizers)    : YOUR Adam optimizer
  Module 08 (DataLoader)    : YOUR data batching
  Module 10 (Tokenization)  : YOUR CharTokenizer for text→numbers
  Module 11 (Embeddings)    : YOUR token & positional embeddings
  Module 12 (Attention)     : YOUR multi-head self-attention
  Module 13 (Transformers)  : YOUR LayerNorm + TransformerBlock + GPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏗️ ARCHITECTURE (Character-Level Q&A Model):
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                               Output Predictions                             │
    │                         Character Probabilities (vocab_size)                 │
    └──────────────────────────────────────────────────────────────────────────────┘
                                            ▲
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                            Output Projection                                 │
    │                       Module 03: vectors → vocabulary                        │
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
    ║  │              Module 03: Linear → GELU → Linear                         │  ║
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
    │                    "Q: What color is the sky? A:"                            │
    └──────────────────────────────────────────────────────────────────────────────┘

📊 EXPECTED PERFORMANCE:
- Dataset: 17.5 KB TinyTalks (301 Q&A pairs, 5 difficulty levels)
- Training time: 3-5 minutes (instant gratification!)
- Vocabulary: ~68 unique characters (simple English Q&A)
- Expected: 70-80% accuracy on Level 1-2 questions after training
- Parameters: ~1.2M (perfect size for fast learning on small data)

💡 WHAT TO WATCH FOR:
- Epoch 1-3: Model learns Q&A structure ("A:" follows "Q:")
- Epoch 4-7: Starts giving sensible (if incorrect) answers
- Epoch 8-12: 50-60% accuracy on simple questions
- Epoch 13-20: 70-80% accuracy, proper grammar
- Success = "Wow, my transformer actually learned to answer questions!"
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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

console = Console()


def print_banner():
    """Print a beautiful banner for the milestone"""
    banner_text = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║            🤖 TinyTalks Q&A Bot Training (2017)                  ║
║                   Transformer Architecture                       ║
║                                                                  ║
║  "Your first transformer learning to answer questions!"         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner_text, border_style="bright_blue", box=box.DOUBLE))


def filter_by_levels(text, levels):
    """
    Filter TinyTalks dataset to only include specified difficulty levels.
    
    Levels are marked in the original generation as:
    L1: Greetings (47 pairs)
    L2: Facts (82 pairs)
    L3: Math (45 pairs)
    L4: Reasoning (87 pairs)
    L5: Context (40 pairs)
    
    For simplicity, we filter by common patterns:
    L1: Hello, Hi, What is your name, etc.
    L2: What color, How many, etc.
    L3: What is X plus/minus, etc.
    """
    if levels is None or levels == [1, 2, 3, 4, 5]:
        return text  # Use full dataset
    
    # Parse Q&A pairs
    pairs = []
    blocks = text.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) == 2 and lines[0].startswith('Q:') and lines[1].startswith('A:'):
            q = lines[0][3:].strip()
            a = lines[1][3:].strip()
            
            # Classify level (heuristic)
            level = 5  # default
            q_lower = q.lower()
            
            if any(word in q_lower for word in ['hello', 'hi', 'hey', 'goodbye', 'bye', 'name', 'who are you', 'what are you']):
                level = 1
            elif any(word in q_lower for word in ['color', 'legs', 'days', 'months', 'sound', 'capital']):
                level = 2
            elif any(word in q_lower for word in ['plus', 'minus', 'times', 'divided', 'equals']):
                level = 3
            elif any(word in q_lower for word in ['use', 'where do', 'what do', 'happens if', 'need to']):
                level = 4
            
            if level in levels:
                pairs.append(f"Q: {q}\nA: {a}")
    
    filtered_text = '\n\n'.join(pairs)
    console.print(f"[yellow]📊 Filtered to Level(s) {levels}:[/yellow]")
    console.print(f"    Q&A pairs: {len(pairs)}")
    console.print(f"    Characters: {len(filtered_text)}")
    
    return filtered_text


class TinyTalksDataset:
    """
    Character-level dataset for TinyTalks Q&A.
    
    Creates sequences of characters for autoregressive language modeling:
    - Input: "Q: What color is the sky? A: The sk"
    - Target: ": What color is the sky? A: The sky"
    
    The model learns to predict the next character given previous characters,
    naturally learning the Q&A pattern.
    """
    
    def __init__(self, text, seq_length=64, levels=None):
        """
        Args:
            text: Full text string (Q&A pairs)
            seq_length: Length of input sequences
            levels: List of difficulty levels to include (1-5), None = all
        """
        from tinytorch.text.tokenization import CharTokenizer
        
        self.seq_length = seq_length
        
        # Filter by levels if specified
        if levels:
            text = filter_by_levels(text, levels)
        
        # Store original text for testing
        self.text = text
        
        # Build character vocabulary using CharTokenizer
        self.tokenizer = CharTokenizer()
        self.tokenizer.build_vocab([text])
        
        # Encode entire text
        self.data = self.tokenizer.encode(text)
        
        console.print(f"[green]✓[/green] Dataset initialized:")
        console.print(f"    Total characters: {len(text)}")
        console.print(f"    Vocabulary size: {self.tokenizer.vocab_size}")
        console.print(f"    Sequence length: {seq_length}")
        console.print(f"    Total sequences: {len(self)}")
    
    def __len__(self):
        """Number of possible sequences"""
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        """
        Get one training example.
        
        Returns:
            input_seq: Characters [idx : idx+seq_length]
            target_seq: Characters [idx+1 : idx+seq_length+1] (shifted by 1)
        """
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return input_seq, target_seq
    
    def decode(self, indices):
        """Decode token indices back to text"""
        return self.tokenizer.decode(indices)


class TinyGPT:
    """
    Character-level GPT model for TinyTalks Q&A.
    
    This is a simplified GPT architecture:
    1. Token embeddings (convert characters to vectors)
    2. Positional encodings (add position information)
    3. N transformer blocks (self-attention + feed-forward)
    4. Output projection (vectors back to character probabilities)
    
    Built entirely from YOUR TinyTorch modules!
    """
    
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, num_heads=4, 
                 max_seq_len=64, dropout=0.1):
        """
        Args:
            vocab_size: Number of unique characters
            embed_dim: Dimension of embeddings and hidden states
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads per block
            max_seq_len: Maximum sequence length
            dropout: Dropout probability (for training)
        """
        from tinytorch.core.tensor import Tensor
        from tinytorch.text.embeddings import Embedding, PositionalEncoding
        from tinytorch.models.transformer import LayerNorm, TransformerBlock
        from tinytorch.core.layers import Linear
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # 1. Token embeddings: char_id → embed_dim vector
        self.token_embedding = Embedding(vocab_size, embed_dim)
        
        # 2. Positional encoding: add position information
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        
        # 3. Transformer blocks (stacked)
        self.blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4,  # FFN hidden_dim = 4 * embed_dim
                dropout_prob=dropout
            )
            self.blocks.append(block)
        
        # 4. Final layer normalization
        self.ln_f = LayerNorm(embed_dim)
        
        # 5. Output projection: embed_dim → vocab_size
        self.output_proj = Linear(embed_dim, vocab_size)
        
        console.print(f"[green]✓[/green] TinyGPT model initialized:")
        console.print(f"    Vocabulary: {vocab_size}")
        console.print(f"    Embedding dim: {embed_dim}")
        console.print(f"    Layers: {num_layers}")
        console.print(f"    Heads: {num_heads}")
        console.print(f"    Max sequence: {max_seq_len}")
        
        # Count parameters
        total_params = self.count_parameters()
        console.print(f"    [bold]Total parameters: {total_params:,}[/bold]")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, seq_len) with token indices
        
        Returns:
            logits: Output tensor of shape (batch, seq_len, vocab_size)
        """
        from tinytorch.core.tensor import Tensor
        
        # 1. Token embeddings: (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.token_embedding.forward(x)
        
        # 2. Add positional encoding
        x = self.pos_encoding.forward(x)
        
        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # 4. Final layer norm
        x = self.ln_f.forward(x)
        
        # 5. Project to vocabulary: (batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)
        logits = self.output_proj.forward(x)
        
        return logits
    
    def parameters(self):
        """Get all trainable parameters"""
        params = []
        
        # Token embeddings
        params.extend(self.token_embedding.parameters())
        
        # Positional encoding (learnable parameters)
        params.extend(self.pos_encoding.parameters())
        
        # Transformer blocks
        for block in self.blocks:
            params.extend(block.parameters())
        
        # Final layer norm
        params.extend(self.ln_f.parameters())
        
        # Output projection
        params.extend(self.output_proj.parameters())
        
        # Ensure all require gradients
        for param in params:
            param.requires_grad = True
        
        return params
    
    def count_parameters(self):
        """Count total trainable parameters"""
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total
    
    def generate(self, tokenizer, prompt="Q:", max_new_tokens=100, temperature=1.0):
        """
        Generate text autoregressively.
        
        Args:
            tokenizer: CharTokenizer for encoding/decoding
            prompt: Starting text
            max_new_tokens: How many characters to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated text string
        """
        from tinytorch.core.tensor import Tensor
        
        # Encode prompt
        indices = tokenizer.encode(prompt)
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Get last max_seq_len tokens (context window)
            context = indices[-self.max_seq_len:]
            
            # Prepare input: (1, seq_len)
            x_input = Tensor(np.array([context]))
            
            # Forward pass
            logits = self.forward(x_input)
            
            # Get logits for last position: (vocab_size,)
            last_logits = logits.data[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(last_logits - np.max(last_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Sample from distribution
            next_idx = np.random.choice(len(probs), p=probs)
            
            # Append to sequence
            indices.append(next_idx)
            
            # Stop if we generate newline after "A:"
            if len(indices) > 3 and tokenizer.decode(indices[-3:]) == "\n\nQ":
                break
        
        return tokenizer.decode(indices)


def test_model_predictions(model, dataset, test_prompts=None):
    """Test model on specific prompts and show predictions"""
    if test_prompts is None:
        test_prompts = ["Q: Hello!", "Q: What is your name?", "Q: Hi!"]
    
    console.print("\n[bold yellow]🧪 Testing Live Predictions:[/bold yellow]")
    for prompt in test_prompts:
        try:
            full_prompt = prompt + "\nA:"
            response = model.generate(dataset.tokenizer, prompt=full_prompt, max_new_tokens=30, temperature=0.5)
            
            # Extract just the answer
            if "\nA:" in response:
                answer = response.split("\nA:")[1].split("\n")[0].strip()
            else:
                answer = response[len(full_prompt):].strip()
            
            console.print(f"  {prompt}")
            console.print(f"  → [cyan]{answer}[/cyan]")
        except Exception as e:
            console.print(f"  {prompt} → [red]Error: {str(e)[:50]}[/red]")


def train_tinytalks_gpt(model, dataset, optimizer, criterion, epochs=20, batch_size=32, 
                        log_interval=50, test_prompts=None):
    """
    Train the TinyGPT model on TinyTalks dataset.
    
    Training loop:
    1. Sample random batch of sequences
    2. Forward pass: predict next character for each position
    3. Compute cross-entropy loss
    4. Backward pass: compute gradients
    5. Update parameters with Adam
    6. Periodically test on sample questions to show learning
    
    Args:
        model: TinyGPT instance
        dataset: TinyTalksDataset instance
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        epochs: Number of training epochs
        batch_size: Number of sequences per batch
        log_interval: Print loss every N batches
        test_prompts: Optional list of questions to test during training
    """
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.autograd import enable_autograd
    
    # Enable autograd
    enable_autograd()
    
    console.print("\n[bold cyan]Starting Training...[/bold cyan]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Dataset size: {len(dataset)} sequences")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Calculate batches per epoch
        batches_per_epoch = min(500, len(dataset) // batch_size)
        
        for batch_idx in range(batches_per_epoch):
            # Sample random batch
            batch_indices = np.random.randint(0, len(dataset), size=batch_size)
            
            batch_inputs = []
            batch_targets = []
            
            for idx in batch_indices:
                input_seq, target_seq = dataset[int(idx)]
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
            
            # Convert to tensors: (batch, seq_len)
            batch_input = Tensor(np.array(batch_inputs))
            batch_target = Tensor(np.array(batch_targets))
            
            # Forward pass
            logits = model(batch_input)
            
            # Reshape for loss computation: (batch, seq, vocab) → (batch*seq, vocab)
            # IMPORTANT: Use Tensor.reshape() to preserve computation graph!
            batch_size_actual, seq_length, vocab_size = logits.shape
            logits_2d = logits.reshape(batch_size_actual * seq_length, vocab_size)
            targets_1d = batch_target.reshape(-1)
            
            # Compute loss
            loss = criterion.forward(logits_2d, targets_1d)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Track loss
            batch_loss = float(loss.data)
            epoch_loss += batch_loss
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                avg_loss = epoch_loss / num_batches
                elapsed = time.time() - start_time
                console.print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx+1}/{batches_per_epoch} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start
        console.print(
            f"[green]✓[/green] Epoch {epoch+1}/{epochs} complete | "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Test model every 5 epochs to show learning progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            test_model_predictions(model, dataset, test_prompts)
    
    total_time = time.time() - start_time
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Total time: {total_time/60:.2f} minutes")


def demo_questions(model, tokenizer):
    """
    Demonstrate the model answering questions.
    
    Shows how well the model learned from TinyTalks by asking
    various questions from different difficulty levels.
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🤖 TinyBot Demo: Ask Me Questions![/bold cyan]")
    console.print("=" * 70)
    
    # Test questions from different levels
    test_questions = [
        "Q: Hello!",
        "Q: What is your name?",
        "Q: What color is the sky?",
        "Q: How many legs does a dog have?",
        "Q: What is 2 plus 3?",
        "Q: What do you use a pen for?",
    ]
    
    for question in test_questions:
        console.print(f"\n[yellow]{question}[/yellow]")
        
        # Generate answer
        response = model.generate(tokenizer, prompt=question + "\nA:", max_new_tokens=50, temperature=0.8)
        
        # Extract just the answer part
        if "\nA:" in response:
            answer = response.split("\nA:")[1].split("\n")[0].strip()
            console.print(f"[green]A: {answer}[/green]")
        else:
            console.print(f"[dim]{response}[/dim]")
    
    console.print("\n" + "=" * 70)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train TinyGPT on TinyTalks Q&A')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--seq-length', type=int, default=64, help='Sequence length (default: 64)')
    parser.add_argument('--embed-dim', type=int, default=96, help='Embedding dimension (default: 96, ~500K params)')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers (default: 4)')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads (default: 4)')
    parser.add_argument('--levels', type=str, default=None, help='Difficulty levels to train on (e.g. "1" or "1,2"). Default: all levels')
    args = parser.parse_args()
    
    # Parse levels argument
    if args.levels:
        levels = [int(l.strip()) for l in args.levels.split(',')]
    else:
        levels = None
    
    print_banner()
    
    # Import TinyTorch components
    console.print("\n[bold]Importing TinyTorch components...[/bold]")
    try:
        from tinytorch.core.tensor import Tensor
        from tinytorch.core.optimizers import Adam
        from tinytorch.core.losses import CrossEntropyLoss
        from tinytorch.text.tokenization import CharTokenizer
        console.print("[green]✓[/green] All modules imported successfully!")
    except ImportError as e:
        console.print(f"[red]✗[/red] Import error: {e}")
        console.print("\nMake sure you have completed all required modules:")
        console.print("  - Module 01 (Tensor)")
        console.print("  - Module 02 (Activations)")
        console.print("  - Module 03 (Layers)")
        console.print("  - Module 04 (Losses)")
        console.print("  - Module 05 (Autograd)")
        console.print("  - Module 06 (Optimizers)")
        console.print("  - Module 10 (Tokenization)")
        console.print("  - Module 11 (Embeddings)")
        console.print("  - Module 12 (Attention)")
        console.print("  - Module 13 (Transformers)")
        return
    
    # Load TinyTalks dataset
    console.print("\n[bold]Loading TinyTalks dataset...[/bold]")
    dataset_path = os.path.join(project_root, "datasets", "tinytalks", "splits", "train.txt")
    
    if not os.path.exists(dataset_path):
        console.print(f"[red]✗[/red] Dataset not found: {dataset_path}")
        console.print("\nPlease generate the dataset first:")
        console.print("  python datasets/tinytalks/scripts/generate_tinytalks.py")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    console.print(f"[green]✓[/green] Loaded dataset from: {os.path.basename(dataset_path)}")
    console.print(f"    File size: {len(text)} characters")
    
    # Create dataset with level filtering
    dataset = TinyTalksDataset(text, seq_length=args.seq_length, levels=levels)
    
    # Set test prompts based on levels
    if levels and 1 in levels:
        test_prompts = ["Q: Hello!", "Q: What is your name?", "Q: Hi!"]
    elif levels and 2 in levels:
        test_prompts = ["Q: What color is the sky?", "Q: How many legs does a dog have?"]
    elif levels and 3 in levels:
        test_prompts = ["Q: What is 2 plus 3?", "Q: What is 5 minus 2?"]
    else:
        test_prompts = ["Q: Hello!", "Q: What is your name?", "Q: What color is the sky?"]
    
    # Initialize model
    console.print("\n[bold]Initializing TinyGPT model...[/bold]")
    model = TinyGPT(
        vocab_size=dataset.tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_length,
        dropout=0.1
    )
    
    # Initialize optimizer and loss
    console.print("\n[bold]Initializing training components...[/bold]")
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()
    console.print(f"[green]✓[/green] Optimizer: Adam (lr={args.lr})")
    console.print(f"[green]✓[/green] Loss: CrossEntropyLoss")
    
    # Print configuration
    table = Table(title="Training Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    dataset_desc = f"TinyTalks Level(s) {levels}" if levels else "TinyTalks (All Levels)"
    table.add_row("Dataset", dataset_desc)
    table.add_row("Vocabulary Size", str(dataset.tokenizer.vocab_size))
    table.add_row("Model Parameters", f"{model.count_parameters():,}")
    table.add_row("Epochs", str(args.epochs))
    table.add_row("Batch Size", str(args.batch_size))
    table.add_row("Learning Rate", str(args.lr))
    table.add_row("Sequence Length", str(args.seq_length))
    table.add_row("Embedding Dim", str(args.embed_dim))
    table.add_row("Layers", str(args.num_layers))
    table.add_row("Attention Heads", str(args.num_heads))
    table.add_row("Expected Time", "3-5 minutes")
    
    console.print(table)
    
    # Train model
    train_tinytalks_gpt(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_interval=50,
        test_prompts=test_prompts
    )
    
    # Demo Q&A
    demo_questions(model, dataset.tokenizer)
    
    # Success message
    console.print("\n[bold green]🎉 Congratulations![/bold green]")
    console.print("You've successfully trained a transformer to answer questions!")
    console.print("\nYou used:")
    console.print("  ✓ YOUR Tensor implementation (Module 01)")
    console.print("  ✓ YOUR Activations (Module 02)")
    console.print("  ✓ YOUR Linear layers (Module 03)")
    console.print("  ✓ YOUR CrossEntropyLoss (Module 04)")
    console.print("  ✓ YOUR Autograd system (Module 05)")
    console.print("  ✓ YOUR Adam optimizer (Module 06)")
    console.print("  ✓ YOUR CharTokenizer (Module 10)")
    console.print("  ✓ YOUR Embeddings (Module 11)")
    console.print("  ✓ YOUR Multi-Head Attention (Module 12)")
    console.print("  ✓ YOUR Transformer blocks (Module 13)")
    console.print("\n[bold]This is the foundation of ChatGPT, built by YOU from scratch![/bold]")


if __name__ == "__main__":
    main()

