#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🗣️ TINYTALKS: Your First Language Model                   ║
║              Watch YOUR Transformer Generate Text Character by Character     ║
╚══════════════════════════════════════════════════════════════════════════════╝

After proving attention works (sequence reversal), let's see YOUR transformer 
actually GENERATE text like a real language model!

🎯 WHAT YOU'LL SEE:
1. A tiny transformer trained on simple phrases
2. Text generated character by character 
3. YOUR attention mechanism deciding what comes next
4. The magic of autoregressive generation!

✅ REQUIRED MODULES:
  Module 01-03: Tensor, Activations, Layers
  Module 06: Optimizers (Adam)
  Module 11: Embeddings
  Module 12: Attention  
  Module 13: Transformer

🏗️ HOW GENERATION WORKS:

    Prompt: "hello"
    
    Step 1: Process "hello" → Predict " "
    Step 2: Process "hello " → Predict "w"
    Step 3: Process "hello w" → Predict "o"
    Step 4: Process "hello wo" → Predict "r"
    ...
    
    Output: "hello world"

Each step uses YOUR attention to look at ALL previous characters!
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
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()


def main():
    # ========================================================================
    # WELCOME
    # ========================================================================
    
    console.print(Panel(
        "[bold magenta]╔═══════════════════════════════╗[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]🗣️ TINYTALKS                  [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] [bold]   Your First Language Model  [/bold][bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta]                               [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] Watch YOUR Transformer       [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] generate text, one character [bold magenta]║[/bold magenta]\n"
        "[bold magenta]║[/bold magenta] at a time!                   [bold magenta]║[/bold magenta]\n"
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
    # TRAINING DATA: Simple Phrases
    # ========================================================================
    
    console.print(Panel(
        "[bold cyan]📚 Training Data: Simple Phrases[/bold cyan]\n\n"
        "We'll train on tiny phrases so you can see generation work quickly:\n\n"
        "  [yellow]'hello world'[/yellow]\n"
        "  [yellow]'hi there'[/yellow]\n"
        "  [yellow]'good day'[/yellow]\n"
        "  [yellow]'nice to meet you'[/yellow]\n"
        "  [yellow]'how are you'[/yellow]\n\n"
        "[dim]Training takes ~30 seconds[/dim]",
        border_style="cyan"
    ))
    
    # Training phrases
    phrases = [
        "hello world",
        "hi there",
        "good day",
        "nice to meet you",
        "how are you",
        "hello there",
        "good morning",
        "have a nice day",
    ]
    
    # Build vocabulary
    all_chars = set()
    for phrase in phrases:
        all_chars.update(phrase)
    all_chars.add('<')  # Start token
    all_chars.add('>')  # End token
    all_chars.add('_')  # Padding
    
    chars = sorted(list(all_chars))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    console.print(f"  [green]✓[/green] Vocabulary: {vocab_size} characters")
    console.print(f"  [dim]Characters: {' '.join(chars)}[/dim]\n")
    
    # ========================================================================
    # BUILD TINY TRANSFORMER
    # ========================================================================
    
    console.print(Panel(
        "[bold cyan]🏗️ Building TinyTalks Transformer[/bold cyan]\n\n"
        "Using YOUR implementations:\n"
        "  • Embedding (Module 11)\n"
        "  • PositionalEncoding (Module 11)\n"
        "  • MultiHeadAttention (Module 12)\n"
        "  • Linear, ReLU (Modules 02-03)",
        border_style="cyan"
    ))
    
    # Model config
    embed_dim = 64
    num_heads = 4
    max_len = 32
    
    # Build model components
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(max_len, embed_dim)  # (max_seq_len, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    ln1 = LayerNorm(embed_dim)
    ff1 = Linear(embed_dim, embed_dim * 2)
    relu = ReLU()
    ff2 = Linear(embed_dim * 2, embed_dim)
    ln2 = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    # Collect all parameters
    all_params = []
    all_params.extend(embedding.parameters())
    all_params.extend(attention.parameters())
    all_params.extend(ln1.parameters())
    all_params.extend(ff1.parameters())
    all_params.extend(ff2.parameters())
    all_params.extend(ln2.parameters())
    all_params.extend(output_proj.parameters())
    
    param_count = sum(p.data.size for p in all_params)
    console.print(f"  [green]✓[/green] Model built: {param_count:,} parameters\n")
    
    def forward(tokens):
        """Forward pass through transformer."""
        x = embedding(tokens)
        x = pos_encoding(x)
        
        # Self-attention
        attn_out = attention(x)
        x = Tensor(x.data + attn_out.data, requires_grad=True)
        x = ln1(x)
        
        # Feed-forward
        ff_out = ff1(x)
        ff_out = relu(ff_out)
        ff_out = ff2(ff_out)
        x = Tensor(x.data + ff_out.data, requires_grad=True)
        x = ln2(x)
        
        # Output
        logits = output_proj(x)
        return logits
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    console.print(Panel(
        "[bold yellow]🏋️ Training YOUR Transformer[/bold yellow]\n\n"
        "Teaching it to predict the next character...",
        border_style="yellow"
    ))
    
    optimizer = Adam(all_params, lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    # Prepare training data
    def encode(text):
        return [char_to_idx[c] for c in text]
    
    def prepare_sequences(phrases, seq_len):
        """Create fixed-length sequences for training."""
        inputs = []
        targets = []
        
        for phrase in phrases:
            # Pad phrase to fixed length
            text = phrase + ' ' * (seq_len - len(phrase))
            text = text[:seq_len]
            
            # Input is text, target is shifted by 1
            inp_ids = encode(text)
            tgt_ids = encode(text[1:] + ' ')  # Shifted target
            
            inputs.append(inp_ids)
            targets.append(tgt_ids)
        
        return Tensor(np.array(inputs)), np.array(targets)
    
    seq_len = 16  # Fixed sequence length
    X_train, y_train = prepare_sequences(phrases, seq_len)
    
    # Training loop
    num_epochs = 100
    losses = []
    
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
            
            # Process each phrase
            for i in range(len(phrases)):
                inp = Tensor(X_train.data[i:i+1])
                tgt = y_train[i]
                
                # Forward
                logits = forward(inp)
                
                # Loss on each position
                for pos in range(min(len(phrases[i]), seq_len - 1)):
                    pos_logits = Tensor(logits.data[0, pos:pos+1, :])
                    pos_target = Tensor(np.array([tgt[pos]]))
                    loss = loss_fn(pos_logits, pos_target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += float(loss.data)
            
            losses.append(total_loss)
            progress.advance(task)
    
    console.print("  [green]✓[/green] Training complete!\n")
    
    # ========================================================================
    # GENERATION DEMO
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🎉 TINYTALKS GENERATION DEMO[/bold green]\n\n"
        "Watch YOUR transformer generate text character by character!",
        border_style="green"
    ))
    
    def generate(prompt, max_new=20):
        """Generate text autoregressively."""
        text = '<' + prompt
        generated = prompt
        
        console.print(f"\n  [bold cyan]Prompt:[/bold cyan] [yellow]{prompt}[/yellow]")
        console.print(f"  [bold cyan]Generating:[/bold cyan] ", end="")
        
        for _ in range(max_new):
            # Encode current text
            padded = text + '_' * (max_len - len(text))
            tokens = Tensor(np.array([encode(padded[:max_len])]))
            
            # Forward pass
            logits = forward(tokens)
            
            # Get next token prediction (last position of current text)
            pos = min(len(text) - 1, max_len - 1)
            next_logits = logits.data[0, pos, :]
            
            # Sample (or argmax)
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / probs.sum()
            next_idx = np.argmax(probs)
            next_char = idx_to_char[next_idx]
            
            if next_char == '>':
                break
            if next_char == '_':
                break
                
            text += next_char
            generated += next_char
            
            # Print character with slight delay for effect
            console.print(f"[green]{next_char}[/green]", end="")
            time.sleep(0.05)
        
        console.print()
        return generated
    
    # Generate from different prompts
    prompts = ["hel", "hi", "goo", "how"]
    
    for prompt in prompts:
        generate(prompt)
        time.sleep(0.3)
    
    # ========================================================================
    # SUCCESS
    # ========================================================================
    
    console.print(Panel(
        "[bold green]🏆 TINYTALKS COMPLETE![/bold green]\n\n"
        "[green]YOUR transformer just generated text![/green]\n\n"
        "[bold]What happened:[/bold]\n"
        "  1. YOUR Embedding converted chars → vectors\n"
        "  2. YOUR PositionalEncoding added position info\n"
        "  3. YOUR Attention looked at all previous chars\n"
        "  4. YOUR Linear layers predicted next char\n"
        "  5. Repeat until done!\n\n"
        "[cyan]This is exactly how GPT works - just bigger![/cyan]\n\n"
        "[bold]🎓 You've built a working language model![/bold]",
        title="🗣️ TinyTalks Success",
        border_style="bright_green",
        box=box.DOUBLE,
        padding=(1, 2)
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

