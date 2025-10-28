#!/usr/bin/env python3
"""
Progressive Test Suite for TinyGPT Learning

Tests transformer learning from absolute simplest to complex:
0. Memorize single sequence (MUST work)
1. Pattern completion (A B A → B)
2. Copy task (COPY: X → X)
3. Simple arithmetic (2+3 → 5)
4. TinyTalks greetings

This helps identify EXACTLY where learning breaks down.
"""

import sys
import os
import numpy as np
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def run_test_0_memorize_sequence():
    """
    TEST 0: Memorize Single Sequence
    
    The ABSOLUTE simplest test. Can the model memorize ONE sequence?
    "HELLO WORLD" repeated many times.
    
    If this fails, there's a fundamental bug in:
    - Forward pass
    - Loss computation
    - Backward pass
    - Parameter updates
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]TEST 0: Single Sequence Memorization[/bold cyan]")
    console.print("=" * 70)
    console.print("Task: Memorize 'HELLO WORLD' (repeated 100 times)")
    console.print("Expected: Loss should drop to near 0")
    console.print("Why: If this fails, autograd/optimizer is broken\n")
    
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.core.autograd import enable_autograd
    from tinytorch.text.tokenization import CharTokenizer
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.models.transformer import TransformerBlock, LayerNorm
    from tinytorch.core.layers import Linear
    
    enable_autograd()
    
    # Super simple data: just repeat "HELLO WORLD"
    text = "HELLO WORLD " * 100
    
    # Tokenize
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([text])
    data = tokenizer.encode(text)
    
    console.print(f"Data length: {len(data)} tokens")
    console.print(f"Vocabulary: {tokenizer.vocab_size} chars")
    console.print(f"Unique text: '{text[:50]}...'\n")
    
    # Tiny model
    vocab_size = tokenizer.vocab_size
    embed_dim = 32
    seq_len = 16
    
    # Build minimal model
    embedding = Embedding(vocab_size, embed_dim)
    pos_enc = PositionalEncoding(seq_len, embed_dim)
    transformer = TransformerBlock(embed_dim, num_heads=2, mlp_ratio=2, dropout_prob=0.1)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    params = []
    params.extend(embedding.parameters())
    params.extend(pos_enc.parameters())
    params.extend(transformer.parameters())
    params.extend(ln.parameters())
    params.extend(output_proj.parameters())
    
    for p in params:
        p.requires_grad = True
    
    console.print(f"Model: {len(params)} parameter tensors")
    console.print(f"Embed dim: {embed_dim}, Seq len: {seq_len}\n")
    
    # Train
    optimizer = Adam(params, lr=0.01)
    criterion = CrossEntropyLoss()
    
    console.print("[yellow]Training (10 steps)...[/yellow]")
    console.print("[dim]Watching for: loss decrease, gradient flow, parameter updates[/dim]\n")
    
    initial_loss = None
    final_loss = None
    
    for step in range(10):
        # Random sequence
        start = np.random.randint(0, len(data) - seq_len - 1)
        input_seq = data[start:start+seq_len]
        target_seq = data[start+1:start+seq_len+1]
        
        console.print(f"[dim]Step {step+1}:[/dim]", end=" ")
        
        # Forward
        x = Tensor(np.array([input_seq]))
        y = Tensor(np.array([target_seq]))
        
        console.print(f"input shape={x.shape}", end=" ")
        
        # Through model
        x = embedding(x)
        console.print(f"embed_out={x.shape}", end=" ")
        
        x = pos_enc(x)
        console.print(f"pos_out={x.shape}", end=" ")
        
        x = transformer(x)
        console.print(f"trans_out={x.shape}", end=" ")
        
        x = ln(x)
        console.print(f"ln_out={x.shape}", end=" ")
        
        # Reshape
        batch, seq, dim = x.shape
        x_2d = x.reshape(batch * seq, dim)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, vocab_size)
        
        console.print(f"logits={logits.shape}", end=" ")
        
        # Loss
        logits_flat = logits.reshape(batch * seq, vocab_size)
        targets_flat = y.reshape(-1)
        
        console.print(f"logits_flat={logits_flat.shape} targets_flat={targets_flat.shape}", end=" ")
        
        loss = criterion(logits_flat, targets_flat)
        
        loss_val = float(loss.data)
        console.print(f"loss={loss_val:.4f}", end=" ")
        
        # Check if loss has grad_fn
        has_grad_fn = hasattr(loss, '_grad_fn') and loss._grad_fn is not None
        console.print(f"has_grad_fn={has_grad_fn}", end=" ")
        
        # Backward
        optimizer.zero_grad()
        
        console.print("backward...", end=" ")
        loss.backward()
        
        # Check if params got gradients
        params_with_grad = sum(1 for p in params if p.grad is not None and np.any(p.grad != 0))
        console.print(f"params_w_grad={params_with_grad}/{len(params)}", end=" ")
        
        optimizer.step()
        console.print("updated")
        
        if step == 0:
            initial_loss = loss_val
            console.print(f"  [yellow]→ Initial loss: {initial_loss:.4f}[/yellow]")
        if step == 9:
            final_loss = loss_val
        
        if step % 2 == 0 and step > 0:
            console.print(f"  [cyan]→ Loss so far: {loss_val:.4f}[/cyan]")
    
    # Result
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Initial loss: {initial_loss:.4f}")
    console.print(f"  Final loss: {final_loss:.4f}")
    console.print(f"  Decrease: {initial_loss - final_loss:.4f}")
    
    if final_loss < initial_loss * 0.8:
        console.print(f"  [green]✓ PASS: Loss decreased significantly[/green]")
        return True
    else:
        console.print(f"  [red]✗ FAIL: Loss didn't decrease enough[/red]")
        console.print(f"  [red]→ Bug in: autograd, optimizer, or forward pass[/red]")
        return False


def run_test_1_pattern_completion():
    """
    TEST 1: Pattern Completion
    
    Can it learn: "A B A B A B" → next is "A"
                  "1 2 1 2 1 2" → next is "1"
    
    Tests: Can model learn simple repeating patterns?
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]TEST 1: Pattern Completion[/bold cyan]")
    console.print("=" * 70)
    console.print("Task: Learn repeating patterns (ABAB... → A, 1212... → 1)")
    console.print("Expected: Predict next token correctly after training")
    console.print("Why: Tests if attention can learn simple sequences\n")
    
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.models.transformer import TransformerBlock, LayerNorm
    from tinytorch.core.layers import Linear
    
    # Create pattern data
    patterns = [
        "A B A B A B A B A B ",
        "1 2 1 2 1 2 1 2 1 2 ",
        "X Y X Y X Y X Y X Y ",
    ]
    
    text = "".join(patterns * 50)  # Repeat 50 times
    
    console.print(f"Data: {len(text)} chars")
    console.print(f"Patterns: ABAB, 1212, XYXY")
    console.print(f"Sample: '{text[:40]}...'\n")
    
    # Tokenize
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = np.array([char_to_idx[ch] for ch in text])
    
    console.print(f"Vocab: {vocab_size} chars: {repr(''.join(chars))}\n")
    
    # Build tiny model
    embed_dim = 32
    num_heads = 2
    seq_len = 8
    
    embedding = Embedding(vocab_size, embed_dim)
    pos_enc = PositionalEncoding(max_seq_len=seq_len, embed_dim=embed_dim)
    transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2, dropout_prob=0.1)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    params = [embedding.weight] + pos_enc.parameters() + transformer.parameters() + ln.parameters() + output_proj.parameters()
    
    # Set requires_grad
    for p in params:
        p.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    criterion = CrossEntropyLoss()
    
    console.print(f"[yellow]Training (30 steps on patterns)...[/yellow]")
    
    initial_loss = None
    final_loss = None
    
    for step in range(30):
        start = np.random.randint(0, len(data) - seq_len - 1)
        input_seq = data[start:start+seq_len]
        target_seq = data[start+1:start+seq_len+1]
        
        x = Tensor(np.array([input_seq]))
        y = Tensor(np.array([target_seq]))
        
        x = embedding(x)
        x = pos_enc(x)
        x = transformer(x)
        x = ln(x)
        
        batch, seq, dim = x.shape
        x_2d = x.reshape(batch * seq, dim)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, vocab_size)
        
        logits_flat = logits.reshape(batch * seq, vocab_size)
        targets_flat = y.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = float(loss.data)
        if step == 0:
            initial_loss = loss_val
        if step == 29:
            final_loss = loss_val
        
        if step % 10 == 0 or step == 29:
            console.print(f"  Step {step+1}: Loss = {loss_val:.4f}")
    
    decrease = initial_loss - final_loss
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Initial: {initial_loss:.4f}")
    console.print(f"  Final: {final_loss:.4f}")
    console.print(f"  Decrease: {decrease:.4f}")
    
    if decrease > 0.5:
        console.print(f"  [green]✓ PASS: Loss decreased significantly[/green]")
        return True
    else:
        console.print(f"  [red]✗ FAIL: Loss didn't decrease enough[/red]")
        return False


def run_test_2_copy_task():
    """
    TEST 2: Copy Task
    
    Input: "COPY: hello"
    Output: "hello"
    
    Classic transformer test from research papers.
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]TEST 2: Copy Task[/bold cyan]")
    console.print("=" * 70)
    console.print("Task: COPY: X → X (reproduce input)")
    console.print("Expected: Model learns to copy the input text")
    console.print("Why: Classic test of attention mechanism\n")
    
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.models.transformer import TransformerBlock, LayerNorm
    from tinytorch.core.layers import Linear
    
    # Create copy task data
    words = ["hello", "world", "test", "copy", "learn", "task"]
    examples = []
    for word in words:
        examples.append(f"COPY:{word}={word} ")
    
    text = "".join(examples * 50)  # Repeat
    
    console.print(f"Data: {len(text)} chars")
    console.print(f"Examples: COPY:hello=hello, COPY:world=world")
    console.print(f"Sample: '{text[:50]}...'\n")
    
    # Tokenize
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    data = np.array([char_to_idx[ch] for ch in text])
    
    console.print(f"Vocab: {vocab_size} chars\n")
    
    # Build model
    embed_dim = 32
    num_heads = 2
    seq_len = 16
    
    embedding = Embedding(vocab_size, embed_dim)
    pos_enc = PositionalEncoding(max_seq_len=seq_len, embed_dim=embed_dim)
    transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2, dropout_prob=0.1)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    params = [embedding.weight] + pos_enc.parameters() + transformer.parameters() + ln.parameters() + output_proj.parameters()
    for p in params:
        p.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    criterion = CrossEntropyLoss()
    
    console.print(f"[yellow]Training (40 steps on copy task)...[/yellow]")
    
    initial_loss = None
    final_loss = None
    
    for step in range(40):
        start = np.random.randint(0, len(data) - seq_len - 1)
        input_seq = data[start:start+seq_len]
        target_seq = data[start+1:start+seq_len+1]
        
        x = Tensor(np.array([input_seq]))
        y = Tensor(np.array([target_seq]))
        
        x = embedding(x)
        x = pos_enc(x)
        x = transformer(x)
        x = ln(x)
        
        batch, seq, dim = x.shape
        x_2d = x.reshape(batch * seq, dim)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, vocab_size)
        
        logits_flat = logits.reshape(batch * seq, vocab_size)
        targets_flat = y.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = float(loss.data)
        if step == 0:
            initial_loss = loss_val
        if step == 39:
            final_loss = loss_val
        
        if step % 10 == 0 or step == 39:
            console.print(f"  Step {step+1}: Loss = {loss_val:.4f}")
    
    decrease = initial_loss - final_loss
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Initial: {initial_loss:.4f}")
    console.print(f"  Final: {final_loss:.4f}")
    console.print(f"  Decrease: {decrease:.4f}")
    
    if decrease > 0.5:
        console.print(f"  [green]✓ PASS: Loss decreased[/green]")
        return True
    else:
        console.print(f"  [red]✗ FAIL: Loss didn't decrease enough[/red]")
        return False


def run_test_3_simple_arithmetic():
    """
    TEST 3: Simple Arithmetic
    
    2+3=5
    1+1=2
    5-2=3
    
    Tests: Can model learn simple rules?
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]TEST 3: Simple Arithmetic[/bold cyan]")
    console.print("=" * 70)
    console.print("Task: 2+3=5, 1+1=2, etc. (single digit)")
    console.print("Expected: Correct answers after training")
    console.print("Why: Tests reasoning ability\n")
    
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.models.transformer import TransformerBlock, LayerNorm
    from tinytorch.core.layers import Linear
    
    # Create arithmetic data
    examples = []
    for a in range(1, 6):
        for b in range(1, 6):
            examples.append(f"{a}+{b}={a+b} ")
    
    text = "".join(examples * 30)  # Repeat
    
    console.print(f"Data: {len(text)} chars")
    console.print(f"Examples: 1+1=2, 2+3=5, 4+5=9")
    console.print(f"Sample: '{text[:40]}...'\n")
    
    # Tokenize
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    data = np.array([char_to_idx[ch] for ch in text])
    
    console.print(f"Vocab: {vocab_size} chars\n")
    
    # Build model
    embed_dim = 48
    num_heads = 3
    seq_len = 12
    
    embedding = Embedding(vocab_size, embed_dim)
    pos_enc = PositionalEncoding(max_seq_len=seq_len, embed_dim=embed_dim)
    transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2, dropout_prob=0.1)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    params = [embedding.weight] + pos_enc.parameters() + transformer.parameters() + ln.parameters() + output_proj.parameters()
    for p in params:
        p.requires_grad = True
    
    optimizer = Adam(params, lr=0.01)
    criterion = CrossEntropyLoss()
    
    console.print(f"[yellow]Training (50 steps on arithmetic)...[/yellow]")
    
    initial_loss = None
    final_loss = None
    
    for step in range(50):
        start = np.random.randint(0, len(data) - seq_len - 1)
        input_seq = data[start:start+seq_len]
        target_seq = data[start+1:start+seq_len+1]
        
        x = Tensor(np.array([input_seq]))
        y = Tensor(np.array([target_seq]))
        
        x = embedding(x)
        x = pos_enc(x)
        x = transformer(x)
        x = ln(x)
        
        batch, seq, dim = x.shape
        x_2d = x.reshape(batch * seq, dim)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, vocab_size)
        
        logits_flat = logits.reshape(batch * seq, vocab_size)
        targets_flat = y.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = float(loss.data)
        if step == 0:
            initial_loss = loss_val
        if step == 49:
            final_loss = loss_val
        
        if step % 10 == 0 or step == 49:
            console.print(f"  Step {step+1}: Loss = {loss_val:.4f}")
    
    decrease = initial_loss - final_loss
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Initial: {initial_loss:.4f}")
    console.print(f"  Final: {final_loss:.4f}")
    console.print(f"  Decrease: {decrease:.4f}")
    
    if decrease > 0.3:
        console.print(f"  [green]✓ PASS: Loss decreased[/green]")
        console.print(f"  [dim](arithmetic is harder, so lower threshold)[/dim]")
        return True
    else:
        console.print(f"  [red]✗ FAIL: Loss didn't decrease enough[/red]")
        return False


def run_test_4_tinytalks_level1():
    """
    TEST 4: TinyTalks Level 1
    
    Q: Hello!
    A: Hi there!
    
    The actual task we want to solve.
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]TEST 4: TinyTalks Level 1[/bold cyan]")
    console.print("=" * 70)
    console.print("Task: Learn greeting Q&A pairs from TinyTalks")
    console.print("Expected: Can respond to greetings")
    console.print("Why: The actual milestone goal\n")
    
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss
    from tinytorch.text.embeddings import Embedding, PositionalEncoding
    from tinytorch.models.transformer import TransformerBlock, LayerNorm
    from tinytorch.core.layers import Linear
    
    # Load TinyTalks Level 1 data
    try:
        with open("datasets/tinytalks/splits/train.txt", "r") as f:
            full_text = f.read()
        
        # Heuristic: Level 1 = very short Q&A (< 40 chars each)
        lines = full_text.split('\n')
        level_1_text = []
        for i in range(0, len(lines) - 1, 3):  # Q, A, blank
            if i+1 < len(lines):
                q_line = lines[i]
                a_line = lines[i+1]
                if q_line.startswith('Q:') and a_line.startswith('A:'):
                    if len(q_line) < 40 and len(a_line) < 40:
                        level_1_text.append(q_line + '\n' + a_line + '\n\n')
        
        if not level_1_text:
            console.print("[red]No Level 1 data found, using first 10 Q&A[/red]")
            level_1_text = [full_text[:500]]
        
        text = "".join(level_1_text[:10])  # First 10 simple Q&A
        
        console.print(f"Data: {len(text)} chars (Level 1 greetings)")
        console.print(f"Sample:\n{text[:100]}...\n")
        
    except FileNotFoundError:
        console.print("[red]TinyTalks not found, skipping Test 4[/red]")
        return None
    
    # Tokenize
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    data = np.array([char_to_idx[ch] for ch in text])
    
    console.print(f"Vocab: {vocab_size} chars\n")
    
    # Build model (slightly larger for Q&A)
    embed_dim = 64
    num_heads = 4
    seq_len = 32
    
    embedding = Embedding(vocab_size, embed_dim)
    pos_enc = PositionalEncoding(max_seq_len=seq_len, embed_dim=embed_dim)
    transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2, dropout_prob=0.1)
    ln = LayerNorm(embed_dim)
    output_proj = Linear(embed_dim, vocab_size)
    
    params = [embedding.weight] + pos_enc.parameters() + transformer.parameters() + ln.parameters() + output_proj.parameters()
    for p in params:
        p.requires_grad = True
    
    optimizer = Adam(params, lr=0.005)  # Lower LR for Q&A
    criterion = CrossEntropyLoss()
    
    console.print(f"[yellow]Training (100 steps on TinyTalks Level 1)...[/yellow]")
    
    initial_loss = None
    final_loss = None
    
    for step in range(100):
        if len(data) < seq_len + 1:
            console.print("[red]Dataset too small[/red]")
            return None
        
        start = np.random.randint(0, len(data) - seq_len - 1)
        input_seq = data[start:start+seq_len]
        target_seq = data[start+1:start+seq_len+1]
        
        x = Tensor(np.array([input_seq]))
        y = Tensor(np.array([target_seq]))
        
        x = embedding(x)
        x = pos_enc(x)
        x = transformer(x)
        x = ln(x)
        
        batch, seq, dim = x.shape
        x_2d = x.reshape(batch * seq, dim)
        logits_2d = output_proj(x_2d)
        logits = logits_2d.reshape(batch, seq, vocab_size)
        
        logits_flat = logits.reshape(batch * seq, vocab_size)
        targets_flat = y.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = float(loss.data)
        if step == 0:
            initial_loss = loss_val
        if step == 99:
            final_loss = loss_val
        
        if step % 20 == 0 or step == 99:
            console.print(f"  Step {step+1}: Loss = {loss_val:.4f}")
    
    decrease = initial_loss - final_loss
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Initial: {initial_loss:.4f}")
    console.print(f"  Final: {final_loss:.4f}")
    console.print(f"  Decrease: {decrease:.4f}")
    
    if decrease > 0.3:
        console.print(f"  [green]✓ PASS: Model is learning TinyTalks![/green]")
        console.print(f"  [cyan]→ Now train full model with tinytalks_gpt.py[/cyan]")
        return True
    else:
        console.print(f"  [yellow]⚠ PARTIAL: Some learning, may need more steps[/yellow]")
        return False


def main():
    """Run all tests in sequence"""
    console.print("\n")
    console.print(Panel(
        "[bold cyan]TinyGPT Learning Diagnostic Suite[/bold cyan]\n\n"
        "Progressive tests from simplest to complex:\n"
        "  0. Single sequence memorization (MUST work)\n"
        "  1. Pattern completion (A B A → B)\n"
        "  2. Copy task (COPY: X → X)\n"
        "  3. Simple arithmetic (2+3 → 5)\n"
        "  4. TinyTalks greetings (Q&A)\n\n"
        "[yellow]This identifies EXACTLY where learning breaks down[/yellow]",
        title="🔬 Diagnostic Tests",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    results = {}
    
    # Run tests
    try:
        results[0] = run_test_0_memorize_sequence()
    except Exception as e:
        console.print(f"\n[red]Test 0 crashed: {str(e)}[/red]")
        results[0] = False
    
    # Only run next tests if previous passed
    if results.get(0):
        results[1] = run_test_1_pattern_completion()
        results[2] = run_test_2_copy_task()
        results[3] = run_test_3_simple_arithmetic()
        results[4] = run_test_4_tinytalks_level1()
    
    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold]Test Summary:[/bold]")
    console.print("=" * 70)
    
    for test_num, result in results.items():
        if result is True:
            console.print(f"  Test {test_num}: [green]✓ PASS[/green]")
        elif result is False:
            console.print(f"  Test {test_num}: [red]✗ FAIL[/red]")
        else:
            console.print(f"  Test {test_num}: [yellow]○ TODO[/yellow]")
    
    console.print("\n" + "=" * 70)
    
    if results.get(0) is False:
        console.print("[bold red]CRITICAL: Test 0 failed![/bold red]")
        console.print("The transformer cannot even memorize a single sequence.")
        console.print("This indicates a fundamental bug in:")
        console.print("  - Forward pass computation")
        console.print("  - Autograd backward pass")
        console.print("  - Optimizer parameter updates")
        console.print("  - Loss computation")


if __name__ == "__main__":
    main()

