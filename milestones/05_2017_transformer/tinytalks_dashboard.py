"""
TinyTalks Interactive Dashboard - Watch Learning Happen Live!
=============================================================

A beautiful, educational dashboard showing a transformer learn to chat.

Students see:
- Live training metrics
- Responses improving from gibberish to coherent
- Real-time checkpoints with before/after comparison
- Visual feedback on what's correct vs incorrect
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
from tinytorch.core.optimizers import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.models.transformer import GPT
from tinytalks_dataset import create_tinytalks_dataset, get_dataset_stats

enable_autograd()

# Rich CLI imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import box
from rich.text import Text

console = Console()

# ============================================================================
# Tokenization (same as tinytalks_chatbot.py)
# ============================================================================

def create_tokenizer(conversations):
    """Create character-level tokenizer with special tokens."""
    all_text = ' '.join([q + ' ' + a for q, a in conversations])
    all_chars = sorted(set(all_text))
    
    special_tokens = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<SEP>': 2,
        '<EOS>': 3,
    }
    
    char_to_idx = {**special_tokens}
    idx_to_char = {v: k for k, v in special_tokens.items()}
    
    for idx, char in enumerate(all_chars, start=len(special_tokens)):
        char_to_idx[char] = idx
        idx_to_char[idx] = char
    
    return char_to_idx, idx_to_char


def encode_conversation(question, answer, char_to_idx, max_len=80):
    """Encode Q&A pair as: <SOS> question <SEP> answer <EOS> <PAD>..."""
    tokens = [char_to_idx['<SOS>']]
    
    for c in question:
        tokens.append(char_to_idx.get(c, 0))
    
    tokens.append(char_to_idx['<SEP>'])
    
    for c in answer:
        tokens.append(char_to_idx.get(c, 0))
    
    tokens.append(char_to_idx['<EOS>'])
    
    if len(tokens) < max_len:
        tokens = tokens + [char_to_idx['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return tokens


def decode_tokens(tokens, idx_to_char):
    """Decode tokens to string."""
    chars = []
    for t in tokens:
        if t == 0 or t == 1:  # PAD or SOS
            continue
        elif t == 2:  # SEP
            continue
        elif t == 3:  # EOS
            break
        else:
            chars.append(idx_to_char.get(t, '?'))
    return ''.join(chars)


def generate_response(model, question, char_to_idx, idx_to_char, max_len=50):
    """Generate response to a question."""
    tokens = [char_to_idx['<SOS>']]
    for c in question:
        tokens.append(char_to_idx.get(c, 0))
    tokens.append(char_to_idx['<SEP>'])
    
    generated_tokens = []
    for _ in range(max_len):
        input_tokens = tokens + generated_tokens
        while len(input_tokens) < 80:
            input_tokens.append(char_to_idx['<PAD>'])
        input_tokens = input_tokens[:80]
        
        x = Tensor(np.array([input_tokens], dtype=np.int32), requires_grad=False)
        logits = model.forward(x)
        
        next_pos = len(tokens) + len(generated_tokens) - 1
        if next_pos < logits.shape[1]:
            next_logits = logits.data[0, next_pos, :]
            next_token = int(np.argmax(next_logits))
            
            if next_token == char_to_idx['<EOS>'] or next_token == char_to_idx['<PAD>']:
                break
            
            generated_tokens.append(next_token)
        else:
            break
    
    response = decode_tokens(generated_tokens, idx_to_char)
    return response


# ============================================================================
# Dashboard Components
# ============================================================================

def create_welcome_panel():
    """Create the welcome panel."""
    return Panel.fit(
        "[bold cyan]🤖 TINYTALKS - Watch a Transformer Learn to Chat![/bold cyan]\n\n"
        "[dim]You're about to see AI learning happen in real-time.\n"
        "The model starts knowing nothing - just random noise.\n"
        "Every training step makes it slightly smarter.\n"
        "Watch responses improve from gibberish to coherent conversation![/dim]\n\n"
        "[bold]Training Duration:[/bold] 10-15 minutes\n"
        "[bold]Checkpoints:[/bold] Every ~2 minutes\n"
        "[bold]What to watch:[/bold] Loss ↓ = Better responses ✓",
        title="🎓 Educational AI Training Demo",
        border_style="cyan",
        box=box.DOUBLE
    )


def create_metrics_table(step, loss, elapsed, steps_per_sec):
    """Create current training metrics table."""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")
    
    table.add_row("Step", f"{step:,}")
    table.add_row("Loss", f"{loss:.4f}")
    table.add_row("Time", f"{int(elapsed/60)}m {int(elapsed%60)}s")
    table.add_row("Speed", f"{steps_per_sec:.1f} steps/sec")
    
    return table


def create_checkpoint_comparison(checkpoint_num, step, loss, test_results, expected_answers):
    """Create a checkpoint panel showing test results."""
    
    # Count correct
    correct = 0
    for (q, actual), expected in zip(test_results, expected_answers):
        if actual.strip().lower() == expected.strip().lower():
            correct += 1
    
    accuracy = (correct / len(test_results)) * 100
    
    # Create results table
    table = Table(
        title=f"Checkpoint {checkpoint_num} - Step {step:,} | Loss: {loss:.4f} | Accuracy: {accuracy:.0f}%",
        box=box.ROUNDED,
        show_header=True
    )
    table.add_column("Question", style="cyan", width=22)
    table.add_column("Model Response", style="white", width=28)
    table.add_column("Status", justify="center", width=8)
    
    for (question, actual), expected in zip(test_results, expected_answers):
        # Determine if correct
        is_correct = actual.strip().lower() == expected.strip().lower()
        is_close = expected.strip().lower() in actual.strip().lower() or actual.strip().lower() in expected.strip().lower()
        
        # Color code and emoji
        if is_correct:
            status = "[green]✓ Perfect[/green]"
            response_style = "green"
        elif is_close:
            status = "[yellow]≈ Close[/yellow]"
            response_style = "yellow"
        elif len(actual.strip()) > 0:
            status = "[red]✗ Wrong[/red]"
            response_style = "red"
        else:
            status = "[dim]- Empty[/dim]"
            response_style = "dim"
        
        # Truncate long responses
        display_response = actual[:26] + "..." if len(actual) > 26 else actual
        
        table.add_row(
            question,
            f"[{response_style}]{display_response}[/{response_style}]",
            status
        )
    
    return table


def create_progress_panel(step, total_steps, checkpoint_num, total_checkpoints):
    """Create progress indicators panel."""
    step_progress = (step / total_steps) * 100 if total_steps > 0 else 0
    checkpoint_progress = (checkpoint_num / total_checkpoints) * 100 if total_checkpoints > 0 else 0
    
    # Progress bars (ASCII style)
    step_bar_filled = int(step_progress / 2.5)  # 40 chars max
    step_bar = "[" + "=" * step_bar_filled + " " * (40 - step_bar_filled) + "]"
    
    checkpoint_bar_filled = int(checkpoint_progress / 2.5)
    checkpoint_bar = "[" + "=" * checkpoint_bar_filled + " " * (40 - checkpoint_bar_filled) + "]"
    
    text = (
        f"[bold]Training Progress:[/bold]\n"
        f"{step_bar} {step_progress:.1f}% ({step}/{total_steps} steps)\n\n"
        f"[bold]Checkpoints:[/bold]\n"
        f"{checkpoint_bar} {checkpoint_progress:.1f}% ({checkpoint_num}/{total_checkpoints} completed)"
    )
    
    return Panel(text, title="📊 Progress", border_style="blue")


# ============================================================================
# Training with Dashboard
# ============================================================================

def train_with_dashboard(model, optimizer, loss_fn, train_data, test_questions, expected_answers,
                        char_to_idx, idx_to_char, max_time_minutes=10, checkpoint_interval_steps=1500):
    """
    Train with beautiful dashboard showing live progress.
    """
    max_time_seconds = max_time_minutes * 60
    
    console.clear()
    console.print(create_welcome_panel())
    console.print()
    
    input("[bold cyan]Press ENTER to start training...[/bold cyan]")
    console.clear()
    
    # Training setup
    start_time = time.time()
    losses = []
    step = 0
    checkpoint_num = 0
    
    # Calculate expected checkpoints
    estimated_total_steps = int(max_time_seconds * 12)  # ~12 steps/sec
    total_checkpoints = estimated_total_steps // checkpoint_interval_steps
    
    # Initial evaluation
    console.print("\n[bold]📊 CHECKPOINT 0: Initial Model (Untrained)[/bold]\n")
    initial_results = [(q, generate_response(model, q, char_to_idx, idx_to_char)) for q in test_questions]
    console.print(create_checkpoint_comparison(0, 0, 999.9, initial_results, expected_answers))
    console.print()
    
    console.print("[dim]Starting training... Watch the responses improve![/dim]\n")
    time.sleep(2)
    
    next_checkpoint = checkpoint_interval_steps
    last_print_time = time.time()
    
    # Training loop
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time_seconds:
            break
        
        # Training step
        tokens = train_data[np.random.randint(len(train_data))]
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        
        x = Tensor(np.array([input_seq], dtype=np.int32), requires_grad=False)
        y_true = Tensor(np.array([target_seq], dtype=np.int32), requires_grad=False)
        
        logits = model.forward(x)
        
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = y_true.reshape(batch_size * seq_len)
        loss = loss_fn.forward(logits_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        
        for param in model.parameters():
            if param.grad is not None:
                np.clip(param.grad, -1.0, 1.0, out=param.grad)
        
        optimizer.step()
        
        losses.append(loss.data.item())
        step += 1
        
        # Print progress every 5 seconds
        if time.time() - last_print_time >= 5.0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            steps_per_sec = step / elapsed
            console.print(
                f"[dim]Step {step:5d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {int(elapsed/60)}m{int(elapsed%60):02d}s | "
                f"Speed: {steps_per_sec:.1f} steps/sec[/dim]"
            )
            last_print_time = time.time()
        
        # Checkpoint evaluation
        if step >= next_checkpoint:
            checkpoint_num += 1
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            
            console.print("\n" + "="*70)
            console.print(f"[bold yellow]⏸️  CHECKPOINT {checkpoint_num}[/bold yellow]")
            console.print(f"[dim]Pausing training to evaluate... (Step {step:,})[/dim]\n")
            
            # Evaluate
            current_results = [(q, generate_response(model, q, char_to_idx, idx_to_char)) for q in test_questions]
            
            # Show results
            console.print(create_checkpoint_comparison(checkpoint_num, step, avg_loss, current_results, expected_answers))
            console.print()
            
            # Show progress
            console.print(create_progress_panel(step, estimated_total_steps, checkpoint_num, total_checkpoints))
            console.print()
            
            console.print("[dim]Continuing training...[/dim]\n")
            next_checkpoint += checkpoint_interval_steps
            time.sleep(1)
    
    # Final results
    final_elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    initial_loss = np.mean(losses[:10])
    improvement = (1 - final_loss / initial_loss) * 100
    
    console.print("\n" + "="*70)
    console.print("[bold green]🎉 TRAINING COMPLETE![/bold green]\n")
    
    # Final evaluation
    final_results = [(q, generate_response(model, q, char_to_idx, idx_to_char)) for q in test_questions]
    console.print(create_checkpoint_comparison("FINAL", step, final_loss, final_results, expected_answers))
    console.print()
    
    # Summary table
    summary = Table(title="Training Summary", box=box.DOUBLE, show_header=True)
    summary.add_column("Metric", style="cyan", width=30)
    summary.add_column("Value", style="green bold", width=30)
    
    summary.add_row("Total Training Time", f"{final_elapsed/60:.1f} minutes")
    summary.add_row("Total Steps", f"{step:,}")
    summary.add_row("Steps/Second", f"{step/final_elapsed:.1f}")
    summary.add_row("Initial Loss", f"{initial_loss:.4f}")
    summary.add_row("Final Loss", f"{final_loss:.4f}")
    summary.add_row("Improvement", f"{improvement:.1f}%")
    summary.add_row("Checkpoints Evaluated", f"{checkpoint_num}")
    
    console.print(summary)
    console.print()
    
    # Count perfect responses for milestone card
    correct = sum(1 for (q, actual), expected in zip(final_results, expected_answers) 
                  if actual.strip().lower() == expected.strip().lower())
    accuracy = (correct / len(test_questions)) * 100
    
    return losses, step, accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    # Dataset
    conversations = create_tinytalks_dataset()
    char_to_idx, idx_to_char = create_tokenizer(conversations)
    vocab_size = len(idx_to_char)
    
    # Encode
    max_seq_len = 80
    train_data = [encode_conversation(q, a, char_to_idx, max_seq_len) for q, a in conversations]
    
    # Test questions and expected answers
    test_questions = [
        "Hi",
        "How are you",
        "What is your name",
        "What is the sky",
        "Is grass green",
        "What is 1 plus 1",
        "Are you happy"
    ]
    
    expected_answers = [
        "Hello! How can I help you?",
        "I am doing well, thanks!",
        "I am TinyBot",
        "The sky is blue",
        "Yes, grass is green",
        "1 plus 1 equals 2",
        "Yes, I am happy"
    ]
    
    # Model
    config = {
        'vocab_size': vocab_size,
        'embed_dim': 16,
        'num_layers': 1,
        'num_heads': 2,
        'max_seq_len': max_seq_len,
    }
    
    model = GPT(**config)
    num_params = sum(np.prod(p.shape) for p in model.parameters())
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Train with dashboard
    train_time = 15  # 15 minutes for better results
    checkpoint_interval = 2000  # Every ~2.5 minutes
    
    console.print(Panel.fit(
        f"[bold]Model:[/bold] {num_params:,} parameters (ultra-tiny!)\n"
        f"[bold]Training Time:[/bold] {train_time} minutes\n"
        f"[bold]Checkpoints:[/bold] Every {checkpoint_interval} steps (~2 min)\n"
        f"[bold]Test Questions:[/bold] {len(test_questions)} questions\n\n"
        f"[dim]Watch loss decrease and responses improve![/dim]",
        title="⚙️ Configuration",
        border_style="blue"
    ))
    
    losses, total_steps, final_accuracy = train_with_dashboard(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        test_questions=test_questions,
        expected_answers=expected_answers,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        max_time_minutes=train_time,
        checkpoint_interval_steps=checkpoint_interval
    )
    
    # Calculate metrics for milestone card
    loss_improvement = (1 - np.mean(losses[-100:]) / np.mean(losses[:10])) * 100
    
    # Milestone completion card
    console.print()
    if final_accuracy >= 50 and loss_improvement >= 80:
        console.print(Panel.fit(
            "[bold green]🎉 Congratulations! You've Built a Working Chatbot![/bold green]\n\n"
            
            f"Final accuracy: [bold]{final_accuracy:.0f}%[/bold] | "
            f"Loss improved: [bold]{loss_improvement:.1f}%[/bold]\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "[bold]💡 What YOU Just Accomplished:[/bold]\n"
            "  ✓ Built a TRANSFORMER (2017 Vaswani et al)\n"
            "  ✓ Trained with attention mechanism from scratch\n"
            "  ✓ Watched AI learn language patterns in real-time\n"
            "  ✓ Demonstrated gradient descent on complex architectures\n"
            f"  ✓ Trained {total_steps:,} steps in {train_time} minutes!\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "[bold]🎓 Why This Matters:[/bold]\n"
            "  This is the SAME architecture behind ChatGPT, GPT-4, and BERT.\n"
            "  You just witnessed the magic of:\n"
            "  • Self-attention (learning relationships between words)\n"
            "  • Position encoding (understanding word order)\n"
            "  • Autoregressive generation (predicting next token)\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "[bold]📌 The Key Insight:[/bold]\n"
            "  You saw responses evolve from gibberish to coherent:\n"
            "    Checkpoint 0: Random noise\n"
            "    Checkpoint 1: Recognizable words\n"
            "    Checkpoint 2: Partial sentences\n"
            "    Final: Perfect responses!\n"
            "  \n"
            "  [yellow]Scale it up:[/yellow] Same process, more data, more params →\n"
            "  You get GPT-4 (175B params, trained for weeks)!\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "[bold]🚀 What You Can Do Now:[/bold]\n"
            "• Experiment with different architectures (layers, heads)\n"
            "• Try longer training (15-20 minutes for better results)\n"
            "• Add more conversation patterns to the dataset\n"
            "• Scale up the model (more parameters = better learning)\n\n"
            
            "[bold cyan]You've mastered the foundation of modern AI! 🌟[/bold cyan]",
            
            title="🌟 2017 Transformer Complete - Milestone 05",
            border_style="green",
            box=box.DOUBLE
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️  Training Complete - Needs More Time[/bold yellow]\n\n"
            f"Current accuracy: {final_accuracy:.0f}% | Loss improved: {loss_improvement:.1f}%\n\n"
            "Your transformer is learning but needs more training time.\n\n"
            "[bold]What to try:[/bold]\n"
            "• Train for 15-20 minutes instead of 10\n"
            "• Use a slightly bigger model (2 layers, 24 dims)\n"
            "• Add more data repetition for reinforcement\n\n"
            "[dim]The attention mechanism is working - it just needs more steps to converge!\n"
            "Even partial success shows the transformer learned patterns.[/dim]",
            title="🔄 Learning in Progress",
            border_style="yellow",
            box=box.DOUBLE
        ))


if __name__ == "__main__":
    main()

