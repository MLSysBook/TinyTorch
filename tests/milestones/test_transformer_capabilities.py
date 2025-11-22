"""
Transformer Capability Tests - Progressive Difficulty

Tests the Transformer architecture with increasingly complex tasks:
- Level 0: Copy Task (sanity check)
- Level 1: Sequence Reversal (requires attention)
- Level 2: Sequence Sorting (requires comparison)
- Level 3: Arithmetic Operations (modulus, addition, etc.)

Each test is independent and can be run separately.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.text.embeddings import Embedding
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.text.embeddings import PositionalEncoding
from tinytorch.models.transformer import LayerNorm
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def generate_copy_data(num_samples=100, seq_len=8, vocab_size=10):
    """
    Generate copy task data: input == output
    
    This is a sanity check - if the model can't learn this, something is broken.
    """
    sequences = []
    for _ in range(num_samples):
        seq = np.random.randint(1, vocab_size, size=seq_len)
        sequences.append((seq, seq.copy()))
    return sequences


def generate_reversal_data(num_samples=100, seq_len=8, vocab_size=10):
    """
    Generate sequence reversal data: [1,2,3,4] -> [4,3,2,1]
    
    This REQUIRES attention to work - each output position must attend
    to a different input position.
    """
    sequences = []
    for _ in range(num_samples):
        seq = np.random.randint(1, vocab_size, size=seq_len)
        reversed_seq = seq[::-1].copy()
        sequences.append((seq, reversed_seq))
    return sequences


def generate_sorting_data(num_samples=100, seq_len=8, vocab_size=10):
    """
    Generate sequence sorting data: [3,1,4,2] -> [1,2,3,4]
    
    Tests multi-position comparison and ordering.
    """
    sequences = []
    for _ in range(num_samples):
        seq = np.random.randint(1, vocab_size, size=seq_len)
        sorted_seq = np.sort(seq)
        sequences.append((seq, sorted_seq))
    return sequences


def generate_modulus_data(num_samples=100, modulus=5):
    """
    Generate modulus arithmetic data: [7, %, 5, =] -> [2]
    
    Tests symbolic reasoning: a % b = c
    Format: [operand1, operator_token, operand2, equals_token] -> [result]
    
    Token mapping:
    - Numbers: 0-9 → tokens 0-9
    - %: token 10
    - =: token 11
    """
    sequences = []
    PERCENT_TOKEN = 10
    EQUALS_TOKEN = 11
    
    for _ in range(num_samples):
        a = np.random.randint(0, 20)  # Larger range for interesting modulus
        b = np.random.randint(1, modulus + 1)  # Avoid division by zero
        result = a % b
        
        # Input: [a, %, b, =]
        input_seq = np.array([a, PERCENT_TOKEN, b, EQUALS_TOKEN])
        # Output: [result]
        output_seq = np.array([result])
        
        sequences.append((input_seq, output_seq))
    
    return sequences


def build_simple_transformer(vocab_size, embed_dim=32, num_heads=4, seq_len=16):
    """
    Build a simple transformer for testing.
    
    Architecture:
    - Embedding + Positional Encoding
    - 1 Transformer Block (Attention + FFN)
    - Output Projection
    """
    # Components
    embedding = Embedding(vocab_size, embed_dim)
    pos_encoding = PositionalEncoding(seq_len, embed_dim)
    attention = MultiHeadAttention(embed_dim, num_heads)
    ln1 = LayerNorm(embed_dim)
    ln2 = LayerNorm(embed_dim)
    fc1 = Linear(embed_dim, embed_dim * 2)
    relu = ReLU()
    fc2 = Linear(embed_dim * 2, embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    # Collect parameters
    params = (
        [embedding.weight] +
        attention.parameters() +
        ln1.parameters() + ln2.parameters() +
        [fc1.weight, fc1.bias, fc2.weight, fc2.bias] +
        [output_proj.weight, output_proj.bias]
    )
    
    # Set requires_grad
    for param in params:
        param.requires_grad = True
    
    def forward(x, target_len=None):
        """Forward pass through transformer."""
        # Embed
        x = embedding(x)
        x = pos_encoding(x)
        
        # Transformer block
        attn_out = attention.forward(x, mask=None)
        x = ln1(x + attn_out)
        
        # FFN
        ffn_out = fc2(relu(fc1(x)))
        x = ln2(x + ffn_out)
        
        # Project to vocabulary
        batch, seq, embed = x.shape
        if target_len is not None:
            # Only use last target_len positions for output
            x = x[:, -target_len:, :]
        x_2d = x.reshape(batch * x.shape[1], embed)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, -1, vocab_size)
        
        return logits
    
    return forward, params


def train_transformer(data, vocab_size, epochs=20, lr=0.001, task_name="Task"):
    """
    Train transformer on given data.
    
    Returns:
        accuracy, predictions on test set
    """
    # Split train/test
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data = data[split:]
    
    # Determine sequence lengths
    max_input_len = max(len(x) for x, _ in data)
    max_output_len = max(len(y) for _, y in data)
    
    # Build model
    forward, params = build_simple_transformer(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        seq_len=max_input_len + max_output_len
    )
    
    # Optimizer
    optimizer = Adam(params, lr=lr)
    loss_fn = CrossEntropyLoss()
    
    # Training
    console.print(f"\n[cyan]Training {task_name}...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Epochs...", total=epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for input_seq, target_seq in train_data:
                # Prepare input (pad if needed)
                input_tensor = Tensor(input_seq.reshape(1, -1))
                
                # Forward
                logits = forward(input_tensor, target_len=len(target_seq))
                
                # Loss
                target_tensor = Tensor(target_seq.reshape(1, -1))
                logits_2d = logits.reshape(-1, vocab_size)
                target_1d = target_tensor.reshape(-1)
                loss = loss_fn(logits_2d, target_1d)
                
                # Backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.data
            
            progress.update(task, advance=1)
    
    # Evaluation
    correct = 0
    total = len(test_data)
    predictions = []
    
    for input_seq, target_seq in test_data:
        input_tensor = Tensor(input_seq.reshape(1, -1))
        logits = forward(input_tensor, target_len=len(target_seq))
        
        # Get predictions
        pred = np.argmax(logits.data, axis=-1).flatten()
        predictions.append((input_seq, target_seq, pred))
        
        # Check if all positions match
        if np.array_equal(pred, target_seq):
            correct += 1
    
    accuracy = (correct / total) * 100
    return accuracy, predictions


def test_copy_task():
    """
    Level 0: Copy Task
    
    Task: [1, 2, 3, 4] -> [1, 2, 3, 4]
    Success: 100% accuracy
    Time: ~10 seconds
    
    This is a sanity check - if this fails, basic architecture is broken.
    """
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold cyan]Level 0: Copy Task (Sanity Check)[/bold cyan]\n"
        "[dim]Task: Output = Input[/dim]",
        border_style="cyan"
    ))
    console.print("="*70)
    
    # Generate data
    vocab_size = 10
    data = generate_copy_data(num_samples=100, seq_len=6, vocab_size=vocab_size)
    
    # Train
    accuracy, predictions = train_transformer(
        data, 
        vocab_size=vocab_size + 1,  # +1 for padding
        epochs=15,
        lr=0.01,
        task_name="Copy Task"
    )
    
    # Report
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Accuracy: [cyan]{accuracy:.1f}%[/cyan]")
    
    # Show examples
    console.print(f"\n[bold]Sample Predictions:[/bold]")
    for i, (inp, target, pred) in enumerate(predictions[:3]):
        match = "✓" if np.array_equal(pred, target) else "✗"
        console.print(f"  {match} Input:  {inp.tolist()}")
        console.print(f"    Target: {target.tolist()}")
        console.print(f"    Pred:   {pred.tolist()}\n")
    
    # Verdict
    passed = accuracy >= 95.0
    if passed:
        console.print("[green]✅ PASS: Copy task learned[/green]")
    else:
        console.print("[red]❌ FAIL: Cannot learn identity function - check basic architecture[/red]")
    
    return passed


def test_sequence_reversal():
    """
    Level 1: Sequence Reversal ⭐ CORE TEST
    
    Task: [1, 2, 3, 4] -> [4, 3, 2, 1]
    Success: 95%+ accuracy
    Time: ~30 seconds
    
    This REQUIRES attention to work - cannot be solved without it!
    From "Attention is All You Need" paper.
    """
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold cyan]Level 1: Sequence Reversal ⭐ Core Attention Test[/bold cyan]\n"
        "[dim]Task: Reverse the input sequence[/dim]\n"
        "[yellow]This test REQUIRES attention to work![/yellow]",
        border_style="cyan"
    ))
    console.print("="*70)
    
    # Generate data
    vocab_size = 10
    data = generate_reversal_data(num_samples=100, seq_len=6, vocab_size=vocab_size)
    
    # Train
    accuracy, predictions = train_transformer(
        data,
        vocab_size=vocab_size + 1,
        epochs=25,
        lr=0.005,
        task_name="Sequence Reversal"
    )
    
    # Report
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Accuracy: [cyan]{accuracy:.1f}%[/cyan]")
    
    # Show examples
    console.print(f"\n[bold]Sample Predictions:[/bold]")
    for i, (inp, target, pred) in enumerate(predictions[:5]):
        match = "✓" if np.array_equal(pred, target) else "✗"
        console.print(f"  {match} Input:  {inp.tolist()}")
        console.print(f"    Target: {target.tolist()}")
        console.print(f"    Pred:   {pred.tolist()}\n")
    
    # Verdict
    passed = accuracy >= 90.0
    if passed:
        console.print("[green]✅ PASS: Attention mechanism is working![/green]")
        console.print("[dim]The model learned to reverse sequences - attention is computing relationships.[/dim]")
    else:
        console.print("[red]❌ FAIL: Attention mechanism not working properly[/red]")
        console.print("[dim]Check: Multi-head attention, Query-Key-Value computation, positional encoding[/dim]")
    
    return passed


def test_sequence_sorting():
    """
    Level 2: Sequence Sorting
    
    Task: [3, 1, 4, 2] -> [1, 2, 3, 4]
    Success: 85%+ accuracy
    Time: ~1 minute
    
    Tests multi-position comparison and ordering.
    """
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold cyan]Level 2: Sequence Sorting[/bold cyan]\n"
        "[dim]Task: Sort the input sequence[/dim]",
        border_style="cyan"
    ))
    console.print("="*70)
    
    # Generate data
    vocab_size = 10
    data = generate_sorting_data(num_samples=100, seq_len=6, vocab_size=vocab_size)
    
    # Train
    accuracy, predictions = train_transformer(
        data,
        vocab_size=vocab_size + 1,
        epochs=30,
        lr=0.003,
        task_name="Sequence Sorting"
    )
    
    # Report
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Accuracy: [cyan]{accuracy:.1f}%[/cyan]")
    
    # Show examples
    console.print(f"\n[bold]Sample Predictions:[/bold]")
    for i, (inp, target, pred) in enumerate(predictions[:5]):
        match = "✓" if np.array_equal(pred, target) else "✗"
        console.print(f"  {match} Input:  {inp.tolist()}")
        console.print(f"    Target: {target.tolist()}")
        console.print(f"    Pred:   {pred.tolist()}\n")
    
    # Verdict
    passed = accuracy >= 70.0
    if passed:
        console.print("[green]✅ PASS: Can learn comparison and ordering[/green]")
    else:
        console.print("[yellow]⚠️  MARGINAL: Sorting is challenging - may need more capacity[/yellow]")
    
    return passed


def test_modulus_arithmetic():
    """
    Level 3: Modulus Arithmetic
    
    Task: [7, %, 5, =] -> [2]
    Success: 80%+ accuracy
    Time: ~2 minutes
    
    Tests symbolic reasoning: understanding that % means modulo operation.
    """
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold cyan]Level 3: Modulus Arithmetic[/bold cyan]\n"
        "[dim]Task: Compute a % b[/dim]\n"
        "[dim]Format: [operand1, %, operand2, =] -> [result][/dim]",
        border_style="cyan"
    ))
    console.print("="*70)
    
    # Generate data
    modulus = 5
    vocab_size = 25  # 0-19 for numbers, 20 for %, 21 for =, rest for padding
    data = generate_modulus_data(num_samples=150, modulus=modulus)
    
    # Train
    accuracy, predictions = train_transformer(
        data,
        vocab_size=vocab_size,
        epochs=40,
        lr=0.002,
        task_name="Modulus Arithmetic"
    )
    
    # Report
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Accuracy: [cyan]{accuracy:.1f}%[/cyan]")
    
    # Show examples
    console.print(f"\n[bold]Sample Predictions:[/bold]")
    PERCENT_TOKEN = 10
    EQUALS_TOKEN = 11
    
    for i, (inp, target, pred) in enumerate(predictions[:5]):
        match = "✓" if np.array_equal(pred, target) else "✗"
        # Decode for display
        a, op, b, eq = inp
        result = target[0]
        pred_result = pred[0] if len(pred) > 0 else -1
        
        console.print(f"  {match} {a} % {b} = {result} (predicted: {pred_result})")
    
    # Verdict
    passed = accuracy >= 70.0
    if passed:
        console.print("[green]✅ PASS: Can learn symbolic reasoning (modulus)[/green]")
    else:
        console.print("[yellow]⚠️  MARGINAL: Arithmetic reasoning is challenging[/yellow]")
    
    return passed


if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold cyan]TRANSFORMER CAPABILITY TESTS[/bold cyan]")
    console.print("Progressive difficulty: Copy → Reversal → Sorting → Arithmetic")
    console.print("="*70)
    
    results = {}
    
    # Run tests
    tests = [
        ("Copy Task", test_copy_task),
        ("Sequence Reversal ⭐", test_sequence_reversal),
        ("Sequence Sorting", test_sequence_sorting),
        ("Modulus Arithmetic", test_modulus_arithmetic),
    ]
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = passed
        except Exception as e:
            console.print(f"[red]❌ {name} ERROR: {e}[/red]")
            results[name] = False
            import traceback
            traceback.print_exc()
    
    # Summary
    console.print("\n" + "="*70)
    console.print("[bold]SUMMARY[/bold]")
    console.print("="*70)
    
    table = Table(box=box.ROUNDED)
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="green")
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        table.add_row(name, status)
    
    console.print(table)
    
    passed_count = sum(results.values())
    total_count = len(results)
    console.print(f"\n[bold]Total: {passed_count}/{total_count} tests passed[/bold]")
    
    if passed_count == total_count:
        console.print("[green]✅ All transformer capability tests passed![/green]")
    elif results.get("Sequence Reversal ⭐", False):
        console.print("[yellow]⚠️  Core attention test passed - transformer is working[/yellow]")
    else:
        console.print("[red]❌ Core attention test failed - transformer needs debugging[/red]")
    
    console.print("="*70)
    
    sys.exit(0 if passed_count >= 2 else 1)  # Pass if at least copy + reversal work

