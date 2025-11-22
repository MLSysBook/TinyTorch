#!/usr/bin/env python3
"""
Debug script for sequence reversal milestone.

This script systematically tests each component to find what's broken.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())

from tinytorch import Tensor, Linear, ReLU, CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.text.embeddings import Embedding, PositionalEncoding
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.models.transformer import LayerNorm

from rich.console import Console
from rich.panel import Panel

console = Console()

def test_embedding_layer():
    """Test that embedding layer works correctly."""
    console.print("\n[bold cyan]Test 1: Embedding Layer[/bold cyan]")
    
    vocab_size = 10
    embed_dim = 32
    seq_len = 6
    
    # Create embedding
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(seq_len, embed_dim)
    
    # Create input
    x = Tensor(np.array([[1, 2, 3, 4, 5, 6]]))  # (1, 6)
    
    # Embed
    embedded = embedding(x)  # Should be (1, 6, 32)
    console.print(f"  Input shape: {x.shape}")
    console.print(f"  Embedded shape: {embedded.shape}")
    console.print(f"  Expected: (1, 6, 32)")
    
    # Add positional encoding
    pos_embedded = pos_encoding(embedded)
    console.print(f"  After pos encoding: {pos_embedded.shape}")
    
    # Check gradient flow
    loss = pos_embedded.sum()
    loss.backward()
    
    has_grad = embedding.weight.grad is not None
    grad_nonzero = np.any(embedding.weight.grad.data) if has_grad else False
    
    console.print(f"  Embedding has gradient: {has_grad}")
    console.print(f"  Gradient is non-zero: {grad_nonzero}")
    
    if pos_embedded.shape == (1, 6, 32) and has_grad and grad_nonzero:
        console.print("  [green]✓ Embedding layer works![/green]")
        return True
    else:
        console.print("  [red]✗ Embedding layer has issues[/red]")
        return False


def test_attention_layer():
    """Test that attention mechanism works."""
    console.print("\n[bold cyan]Test 2: Attention Layer[/bold cyan]")
    
    embed_dim = 32
    num_heads = 4
    seq_len = 6
    
    # Create attention
    attention = MultiHeadAttention(embed_dim, num_heads)
    
    # Create input (batch=1, seq=6, embed=32)
    x = Tensor(np.random.randn(1, seq_len, embed_dim))
    
    console.print(f"  Input shape: {x.shape}")
    
    # Forward
    attn_out = attention.forward(x, mask=None)
    console.print(f"  Attention output shape: {attn_out.shape}")
    console.print(f"  Expected: (1, 6, 32)")
    
    # Check gradient flow
    loss = attn_out.sum()
    loss.backward()
    
    params = attention.parameters()
    has_grads = all(p.grad is not None for p in params)
    grads_nonzero = all(np.any(p.grad.data) for p in params) if has_grads else False
    
    console.print(f"  All params have gradients: {has_grads}")
    console.print(f"  All gradients non-zero: {grads_nonzero}")
    console.print(f"  Number of parameters: {len(params)}")
    
    if attn_out.shape == (1, 6, 32) and has_grads:
        console.print("  [green]✓ Attention layer works![/green]")
        return True
    else:
        console.print("  [red]✗ Attention layer has issues[/red]")
        return False


def test_ffn_layer():
    """Test feed-forward network."""
    console.print("\n[bold cyan]Test 3: Feed-Forward Network[/bold cyan]")
    
    embed_dim = 32
    
    fc1 = Linear(embed_dim, embed_dim * 2)
    relu = ReLU()
    fc2 = Linear(embed_dim * 2, embed_dim)
    
    # Input
    x = Tensor(np.random.randn(1, 6, embed_dim))
    
    # Forward
    h = fc1(x)
    h = relu(h)
    out = fc2(h)
    
    console.print(f"  Input shape: {x.shape}")
    console.print(f"  Output shape: {out.shape}")
    console.print(f"  Expected: (1, 6, 32)")
    
    # Gradient flow
    loss = out.sum()
    loss.backward()
    
    params = [fc1.weight, fc1.bias, fc2.weight, fc2.bias]
    has_grads = all(p.grad is not None for p in params)
    
    console.print(f"  All params have gradients: {has_grads}")
    
    if out.shape == (1, 6, 32) and has_grads:
        console.print("  [green]✓ FFN works![/green]")
        return True
    else:
        console.print("  [red]✗ FFN has issues[/red]")
        return False


def test_residual_connection():
    """Test that residual connections preserve computation graph."""
    console.print("\n[bold cyan]Test 4: Residual Connections[/bold cyan]")
    
    embed_dim = 32
    
    # Create layers
    attention = MultiHeadAttention(embed_dim, 4)
    ln = LayerNorm(embed_dim)
    
    # Input
    x = Tensor(np.random.randn(1, 6, embed_dim))
    x.requires_grad = True
    
    # Residual connection
    attn_out = attention.forward(x, mask=None)
    residual = x + attn_out  # This should preserve graph
    out = ln(residual)
    
    console.print(f"  Output shape: {out.shape}")
    
    # Gradient flow
    loss = out.sum()
    loss.backward()
    
    has_x_grad = x.grad is not None
    has_attn_grads = all(p.grad is not None for p in attention.parameters())
    has_ln_grads = all(p.grad is not None for p in ln.parameters())
    
    console.print(f"  Input has gradient: {has_x_grad}")
    console.print(f"  Attention has gradients: {has_attn_grads}")
    console.print(f"  LayerNorm has gradients: {has_ln_grads}")
    
    if has_x_grad and has_attn_grads and has_ln_grads:
        console.print("  [green]✓ Residual connection preserves gradients![/green]")
        return True
    else:
        console.print("  [red]✗ Residual connection breaks gradients[/red]")
        return False


def test_full_forward_pass():
    """Test full forward pass through transformer."""
    console.print("\n[bold cyan]Test 5: Full Forward Pass[/bold cyan]")
    
    # Import by loading the file directly (can't import modules starting with numbers)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "attention_proof", 
        "milestones/05_2017_transformer/00_vaswani_attention_proof.py"
    )
    attention_proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention_proof)
    ReversalTransformer = attention_proof.ReversalTransformer
    
    # Create model
    model = ReversalTransformer(vocab_size=10, embed_dim=32, num_heads=4, seq_len=6)
    
    # Set requires_grad
    for param in model.parameters():
        param.requires_grad = True
    
    # Input
    x = Tensor(np.array([[1, 2, 3, 4, 5, 6]]))
    
    console.print(f"  Input shape: {x.shape}")
    
    # Forward
    logits = model(x)
    
    console.print(f"  Output shape: {logits.shape}")
    console.print(f"  Expected: (1, 6, 10)")
    
    # Loss
    target = Tensor(np.array([[6, 5, 4, 3, 2, 1]]))
    loss_fn = CrossEntropyLoss()
    
    logits_2d = logits.reshape(-1, 10)
    target_1d = target.reshape(-1)
    loss = loss_fn(logits_2d, target_1d)
    
    console.print(f"  Loss value: {loss.data:.4f}")
    console.print(f"  Loss has grad_fn: {loss._grad_fn is not None}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = len(model.parameters())
    
    console.print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    
    if logits.shape == (1, 6, 10) and params_with_grad == total_params:
        console.print("  [green]✓ Full forward/backward pass works![/green]")
        return True
    else:
        console.print("  [red]✗ Full pass has issues[/red]")
        return False


def test_training_step():
    """Test that one training step actually updates weights."""
    console.print("\n[bold cyan]Test 6: Training Step Updates Weights[/bold cyan]")
    
    # Import by loading the file directly (can't import modules starting with numbers)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "attention_proof", 
        "milestones/05_2017_transformer/00_vaswani_attention_proof.py"
    )
    attention_proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention_proof)
    ReversalTransformer = attention_proof.ReversalTransformer
    
    # Create model
    model = ReversalTransformer(vocab_size=10, embed_dim=32, num_heads=4, seq_len=6)
    
    # Set requires_grad
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_fn = CrossEntropyLoss()
    
    # Save initial weights
    initial_weights = {}
    for i, param in enumerate(model.parameters()):
        initial_weights[i] = param.data.copy()
    
    # Training step
    x = Tensor(np.array([[1, 2, 3, 4, 5, 6]]))
    target = Tensor(np.array([[6, 5, 4, 3, 2, 1]]))
    
    logits = model(x)
    logits_2d = logits.reshape(-1, 10)
    target_1d = target.reshape(-1)
    loss = loss_fn(logits_2d, target_1d)
    
    console.print(f"  Initial loss: {loss.data:.4f}")
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Check if weights changed
    weights_changed = 0
    for i, param in enumerate(model.parameters()):
        if not np.allclose(param.data, initial_weights[i], atol=1e-6):
            weights_changed += 1
    
    console.print(f"  Weights changed: {weights_changed}/{len(model.parameters())}")
    
    if weights_changed == len(model.parameters()):
        console.print("  [green]✓ All weights updated![/green]")
        return True
    else:
        console.print(f"  [yellow]⚠ Only {weights_changed} weights updated[/yellow]")
        return False


def main():
    console.print(Panel.fit(
        "[bold]Sequence Reversal Debug Suite[/bold]\n"
        "Testing each component systematically",
        border_style="cyan"
    ))
    
    results = {
        "Embedding Layer": test_embedding_layer(),
        "Attention Layer": test_attention_layer(),
        "FFN Layer": test_ffn_layer(),
        "Residual Connections": test_residual_connection(),
        "Full Forward Pass": test_full_forward_pass(),
        "Training Step": test_training_step()
    }
    
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold]Summary[/bold]",
        border_style="green"
    ))
    
    for test_name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {status} - {test_name}")
    
    all_passed = all(results.values())
    if all_passed:
        console.print("\n[bold green]All tests passed! The issue might be hyperparameters.[/bold green]")
    else:
        console.print("\n[bold red]Some tests failed! Fix these components first.[/bold red]")
    
    console.print("="*70)


if __name__ == "__main__":
    main()

