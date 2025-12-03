#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🗣️ TINYTALKS: Your First Language Model                   ║
║              Watch YOUR Transformer Complete Simple Phrases                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

After proving attention works (sequence reversal), let's see YOUR transformer 
complete phrases - just like a tiny GPT!

🎯 THE TASK: Phrase Completion
    Input:  "hel"    → Output: "hello"
    Input:  "good"   → Output: "good day"  
    Input:  "how ar" → Output: "how are you"

This is simpler than full generation but shows the same principle:
YOUR attention looks at all previous characters to predict the next one!

✅ REQUIRED MODULES:
  Module 01-03: Tensor, Activations, Layers
  Module 06: Optimizers (Adam)
  Module 11: Embeddings
  Module 12: Attention  
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()


def main():
    # ========================================================================
    # WELCOME
    # ========================================================================
    
    console.print(Panel(
        "[bold magenta]╔═══════════════════════════════╗[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]🗣️ TINYTALKS                  [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]   Phrase Completion Demo     [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta]                               [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] YOUR Transformer completes   [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] simple phrases!              [bold magenta]║[/bold magenta]\n"
        "[bold magenta]╚═══════════════════════════════╝[/bold magenta]",
        border_style="bright_magenta"
    ))
    
    # ========================================================================
    # IMPORT YOUR IMPLEMENTATIONS
    # ========================================================================
    
    console.print("\n[bold cyan]📦 Loading YOUR TinyTorch...[/bold cyan]\n")
    
    try:
        from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss
        from tinytorch.core.optimizers import Adam
        from tinytorch.text.embeddings import Embedding, PositionalEncoding
        from tinytorch.core.attention import MultiHeadAttention
        from tinytorch.models.transformer import LayerNorm
        
        console.print("  [green]✓[/green] All YOUR implementations loaded!")
        
    except ImportError as e:
        console.print(f"[red]Import Error: {e}[/red]")
        return 1
    
    # ========================================================================
    # SIMPLE PHRASE PAIRS
    # ========================================================================
    
    console.print(Panel(
        "[bold cyan]📚 Training Task: Complete the Phrase[/bold cyan]\n\n"
        "We train on simple prompt → completion pairs:\n\n"
        "  [dim]Prompt[/dim]     [yellow]→[/yellow]  [dim]Complete[/dim]\n"
        "  [cyan]'hel'[/cyan]      [yellow]→[/yellow]  [green]'hello'[/green]\n"
        "  [cyan]'goo'[/cyan]      [yellow]→[/yellow]  [green]'good'[/green]\n"
        "  [cyan]'wor'[/cyan]      [yellow]→[/yellow]  [green]'world'[/green]\n"
        "  [cyan]'nic'[/cyan]      [yellow]→[/yellow]  [green]'nice'[/green]\n\n"
        "[dim]Just like GPT completes your sentences![/dim]",
        border_style="cyan"
    ))
    
    # Training data: (prefix, full_word)
    # Keep it simple - just complete words
    training_pairs = [
        ("hel", "hello"),
        ("goo", "good"),
        ("wor", "world"),
        ("nic", "nice"),
        ("the", "there"),
        ("mor", "morning"),
        ("how", "how are"),
        ("mee", "meet"),
    ]
    
    # Build vocabulary from all characters
    all_chars = set()
    for prefix, full in training_pairs:
        all_chars.update(prefix)
        all_chars.update(full)
    all_chars.add(' ')  # Space
    all_chars.add('_')  # Padding
    
    chars = sorted(list(all_chars))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    console.print(f"  [green]✓[/green] Vocabulary: {vocab_size} characters\n")
    
    # ========================================================================
    # BUILD TINY TRANSFORMER
    # ========================================================================
    
    console.print(Panel(
        "[bold cyan]🏗️ Building Phrase Completion Model[/bold cyan]\n\n"
        "Using YOUR implementations from Modules 11-12",
        border_style="cyan"
    ))
    
    # Config - keep small
    embed_dim = 32
    num_heads = 2
    max_len = 16
    
    # Build model
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(max_len, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    # Collect parameters
    all_params = []
    all_params.extend(embedding.parameters())
    all_params.extend(attention.parameters())
    all_params.extend(ln.parameters())
    all_params.extend(output_proj.parameters())
    
    param_count = sum(p.data.size for p in all_params)
    console.print(f"  [green]✓[/green] Model: {param_count:,} parameters\n")
    
    def forward(tokens):
        """Simple forward: embed → position → attention → output."""
        x = embedding(tokens)
        x = pos_encoding(x)
        attn_out = attention(x)
        # Residual
        x = Tensor(x.data + attn_out.data, requires_grad=True)
        x = ln(x)
        logits = output_proj(x)
        return logits
    
    # ========================================================================
    # TRAINING: Learn to complete each character
    # ========================================================================
    
    console.print(Panel(
        "[bold yellow]🏋️ Training: Next Character Prediction[/bold yellow]\n\n"
        "For 'hello': h→e, he→l, hel→l, hell→o",
        border_style="yellow"
    ))
    
    optimizer = Adam(all_params, lr=0.02)
    loss_fn = CrossEntropyLoss()
    
    def encode(text, length):
        """Encode text to indices, pad to length."""
        text = text + '_' * (length - len(text))
        return [char_to_idx.get(c, 0) for c in text[:length]]
    
    # Create training examples
    # For input "hel", predict the next char "l" (to form "hell")
    # Input positions matter: we predict at the LAST input position
    train_inputs = []
    train_targets = []
    train_lengths = []  # Length of input (position to predict at)
    
    for prefix, full in training_pairs:
        # For each character we need to complete
        for i in range(len(prefix), len(full)):
            # Input is everything before position i
            inp_text = full[:i]
            inp = encode(inp_text, max_len)
            
            # Target is the character at position i
            target = char_to_idx.get(full[i], 0)
            
            train_inputs.append(inp)
            train_targets.append(target)
            train_lengths.append(len(inp_text))  # Predict after last input char
    
    X = Tensor(np.array(train_inputs))
    y = np.array(train_targets)
    lengths = np.array(train_lengths)
    
    console.print(f"  [dim]Training examples: {len(train_inputs)}[/dim]")
    
    # Training - more epochs for small dataset
    num_epochs = 300
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True
    ) as progress:
        task = progress.add_task("Training...", total=num_epochs)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(len(train_inputs)):
                # Forward
                inp = Tensor(X.data[i:i+1])
                logits = forward(inp)
                
                # Predict at the position AFTER the last input char
                # For input "hel" (length 3), predict at position 2 (0-indexed)
                pos = lengths[i] - 1
                pred = Tensor(logits.data[0, pos:pos+1, :])
                target = Tensor(np.array([y[i]]))
                
                loss = loss_fn(pred, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += float(loss.data)
            
            progress.advance(task)
    
    console.print("  [green]✓[/green] Training complete!\n")
    
    # ========================================================================
    # DEMO: Complete phrases!
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🎉 PHRASE COMPLETION DEMO[/bold green]\n\n"
        "Watch YOUR transformer complete phrases!",
        border_style="green"
    ))
    
    def complete(prefix, max_chars=10):
        """Complete a phrase character by character."""
        text = prefix
        
        console.print(f"\n  [bold cyan]Prefix:[/bold cyan] [yellow]{prefix}[/yellow]")
        console.print(f"  [bold cyan]Completing:[/bold cyan] [green]{prefix}[/green]", end="")
        
        for _ in range(max_chars):
            # Encode current text
            inp = Tensor(np.array([encode(text, max_len)]))
            
            # Forward
            logits = forward(inp)
            
            # Get prediction at current position
            pos = min(len(text) - 1, max_len - 2)
            next_probs = logits.data[0, pos, :]
            
            # Softmax + argmax
            probs = np.exp(next_probs - np.max(next_probs))
            probs = probs / probs.sum()
            next_idx = np.argmax(probs)
            next_char = idx_to_char[next_idx]
            
            # Stop on padding or space after some chars
            if next_char == '_':
                break
            if next_char == ' ' and len(text) > len(prefix) + 3:
                console.print(f"[green]{next_char}[/green]", end="")
                text += next_char
                break
                
            console.print(f"[green]{next_char}[/green]", end="")
            text += next_char
            time.sleep(0.1)
        
        console.print()
        return text
    
    # Test completions
    test_prefixes = ["hel", "goo", "wor", "how"]
    
    for prefix in test_prefixes:
        complete(prefix)
        time.sleep(0.2)
    
    # ========================================================================
    # SUCCESS
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🏆 TINYTALKS COMPLETE![/bold green]\n\n"
        "[green]YOUR transformer completed phrases![/green]\n\n"
        "[bold]How it works:[/bold]\n"
        "  1. [cyan]Embedding[/cyan]: Characters → Vectors\n"
        "  2. [cyan]Position[/cyan]: Add position info\n"
        "  3. [cyan]Attention[/cyan]: Look at ALL previous chars\n"
        "  4. [cyan]Predict[/cyan]: What char comes next?\n\n"
        "[dim]This is exactly how GPT works - just at a tiny scale![/dim]\n\n"
        "[bold]🎓 You've built a working language model![/bold]",
        title="🗣️ TinyTalks",
        border_style="bright_green",
        box=box.DOUBLE,
        padding=(1, 2)
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
