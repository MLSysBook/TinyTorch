#!/usr/bin/env python3
"""
Diagnostic tests for TinyTalks - debug why the model isn't learning.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_imports():
    """Test that all imports work."""
    console.print("\n[bold cyan]Test 1: Imports[/bold cyan]")
    try:
        from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss
        from tinytorch.core.optimizers import Adam
        from tinytorch.text.embeddings import Embedding, PositionalEncoding
        from tinytorch.core.attention import MultiHeadAttention
        from tinytorch.models.transformer import LayerNorm
        console.print("  [green]✓[/green] All imports successful")
        return True
    except ImportError as e:
        console.print(f"  [red]✗[/red] Import failed: {e}")
        return False


def test_embedding_gradients():
    """Test that gradients flow through embeddings."""
    console.print("\n[bold cyan]Test 2: Embedding Gradients[/bold cyan]")
    
    from tinytorch import Tensor
    from tinytorch.text.embeddings import Embedding
    
    embed = Embedding(10, 8)
    tokens = Tensor(np.array([[1, 2, 3]]))
    
    out = embed(tokens)
    console.print(f"  Output shape: {out.shape}")
    
    # Create a simple loss
    loss = out.sum()
    loss.backward()
    
    # Check if embedding weights have gradients
    has_grad = embed.weight.grad is not None
    if has_grad:
        grad_norm = np.linalg.norm(embed.weight.grad)
        console.print(f"  [green]✓[/green] Embedding has gradients (norm: {grad_norm:.4f})")
    else:
        console.print(f"  [red]✗[/red] Embedding has NO gradients!")
    
    return has_grad


def test_attention_gradients():
    """Test that gradients flow through attention."""
    console.print("\n[bold cyan]Test 3: Attention Gradients[/bold cyan]")
    
    from tinytorch import Tensor
    from tinytorch.core.attention import MultiHeadAttention
    
    attn = MultiHeadAttention(embed_dim=16, num_heads=2)
    x = Tensor(np.random.randn(1, 4, 16).astype(np.float32), requires_grad=True)
    
    out = attn(x)
    console.print(f"  Output shape: {out.shape}")
    
    loss = out.sum()
    loss.backward()
    
    # Check if attention parameters have gradients
    params = attn.parameters()
    all_have_grad = all(p.grad is not None for p in params)
    
    if all_have_grad:
        console.print(f"  [green]✓[/green] All attention params have gradients")
    else:
        console.print(f"  [red]✗[/red] Some attention params have NO gradients!")
        for i, p in enumerate(params):
            status = "✓" if p.grad is not None else "✗"
            console.print(f"    {status} Param {i}: shape {p.shape}")
    
    # Check input gradient
    if x.grad is not None:
        console.print(f"  [green]✓[/green] Input received gradients")
    else:
        console.print(f"  [red]✗[/red] Input has NO gradients!")
    
    return all_have_grad


def test_full_forward_backward():
    """Test full forward-backward through the model."""
    console.print("\n[bold cyan]Test 4: Full Forward-Backward[/bold cyan]")
    
    from tinytorch import Tensor, Linear, CrossEntropyLoss
    from tinytorch.core.optimizers import Adam
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.core.attention import MultiHeadAttention
    from tinytorch.models.transformer import LayerNorm
    
    vocab_size = 10
    embed_dim = 16
    num_heads = 2
    max_len = 8
    
    # Build model
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(max_len, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    all_params = []
    all_params.extend(embedding.parameters())
    all_params.extend(attention.parameters())
    all_params.extend(ln.parameters())
    all_params.extend(output_proj.parameters())
    
    console.print(f"  Total params: {len(all_params)}")
    
    # Forward - following the pattern from attention_proof.py
    tokens = Tensor(np.array([[1, 2, 3, 0, 0, 0, 0, 0]]))
    
    x = embedding(tokens)
    console.print(f"  After embedding: {x.shape}")
    
    x = pos_encoding(x)
    console.print(f"  After pos_encoding: {x.shape}")
    
    attn_out = attention(x)
    console.print(f"  After attention: {attn_out.shape}")
    
    # ISSUE: x + attn_out creates new tensor without grad tracking
    # The working code in attention_proof uses: x = self.ln1(x + attn_out)
    # Let's check if that works by using layernorm directly
    x = ln(x + attn_out)  # This should work if Tensor.__add__ connects to graph
    console.print(f"  After layernorm: {x.shape}")
    
    # Reshape for output projection (like attention_proof does)
    batch, seq, embed = x.shape
    x_2d = x.reshape(batch * seq, embed)
    logits_2d = output_proj(x_2d)
    logits = logits_2d.reshape(batch, seq, vocab_size)
    console.print(f"  Logits: {logits.shape}")
    
    # Compute loss using sum (maintains graph)
    loss = logits.sum()
    console.print(f"  Loss (sum): {float(loss.data):.4f}")
    
    # Backward
    optimizer = Adam(all_params, lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grads_exist = []
    for i, p in enumerate(all_params):
        has_grad = p.grad is not None
        grads_exist.append(has_grad)
        if not has_grad:
            console.print(f"  [red]✗[/red] Param {i} (shape {p.shape}) has NO gradient")
    
    if all(grads_exist):
        console.print(f"  [green]✓[/green] All {len(all_params)} params have gradients")
    else:
        console.print(f"  [red]✗[/red] {sum(not g for g in grads_exist)}/{len(all_params)} params missing gradients")
    
    return all(grads_exist)


def test_loss_decreases():
    """Test that loss actually decreases with training."""
    console.print("\n[bold cyan]Test 5: Loss Decreases[/bold cyan]")
    
    from tinytorch import Tensor, Linear, CrossEntropyLoss
    from tinytorch.core.optimizers import Adam
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.core.attention import MultiHeadAttention
    from tinytorch.models.transformer import LayerNorm
    
    vocab_size = 10
    embed_dim = 16
    num_heads = 2
    max_len = 8
    
    # Build model
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(max_len, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    all_params = []
    all_params.extend(embedding.parameters())
    all_params.extend(attention.parameters())
    all_params.extend(ln.parameters())
    all_params.extend(output_proj.parameters())
    
    optimizer = Adam(all_params, lr=0.05)
    loss_fn = CrossEntropyLoss()
    
    def forward(tokens):
        x = embedding(tokens)
        x = pos_encoding(x)
        attn_out = attention(x)
        x = Tensor(x.data + attn_out.data, requires_grad=True)
        x = ln(x)
        return output_proj(x)
    
    # Simple training: input [1,2,3], target next char = 4
    tokens = Tensor(np.array([[1, 2, 3, 0, 0, 0, 0, 0]]))
    target_idx = 4
    
    losses = []
    for step in range(50):
        logits = forward(tokens)
        pred = Tensor(logits.data[0, 2:3, :])  # Predict at position 2
        target = Tensor(np.array([target_idx]))
        
        loss = loss_fn(pred, target)
        losses.append(float(loss.data))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    console.print(f"  Initial loss: {losses[0]:.4f}")
    console.print(f"  Final loss: {losses[-1]:.4f}")
    console.print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")
    
    if losses[-1] < losses[0]:
        console.print(f"  [green]✓[/green] Loss decreased!")
        
        # Check prediction
        logits = forward(tokens)
        pred_idx = np.argmax(logits.data[0, 2, :])
        if pred_idx == target_idx:
            console.print(f"  [green]✓[/green] Correct prediction: {pred_idx} == {target_idx}")
        else:
            console.print(f"  [yellow]~[/yellow] Prediction: {pred_idx}, Target: {target_idx}")
        
        return True
    else:
        console.print(f"  [red]✗[/red] Loss did NOT decrease!")
        return False


def main():
    console.print(Panel(
        "[bold]🔬 TinyTalks Diagnostic Tests[/bold]\n\n"
        "Testing gradient flow and learning",
        border_style="cyan"
    ))
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Embedding Gradients", test_embedding_gradients()))
    results.append(("Attention Gradients", test_attention_gradients()))
    results.append(("Full Forward-Backward", test_full_forward_backward()))
    results.append(("Loss Decreases", test_loss_decreases()))
    
    console.print("\n" + "=" * 50)
    console.print("[bold]Summary:[/bold]")
    for name, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        console.print("\n[bold green]All tests passed! Model should be able to learn.[/bold green]")
    else:
        console.print("\n[bold red]Some tests failed - investigate the failures above.[/bold red]")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

