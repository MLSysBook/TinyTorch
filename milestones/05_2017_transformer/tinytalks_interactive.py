"""
TinyTalks Interactive Learning Dashboard
=========================================

Watch a chatbot learn in real-time!

Students can see:
- Loss decreasing over time
- Responses improving from gibberish to coherent
- Learning progress at multiple checkpoints
- Interactive control (pause/continue)
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

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better visualization: pip install rich")

# ============================================================================
# Tokenization (copied from tinytalks_chatbot.py)
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
# Interactive Training with Checkpoints
# ============================================================================

def evaluate_at_checkpoint(model, test_questions, char_to_idx, idx_to_char):
    """Evaluate model on test questions."""
    results = []
    for question in test_questions:
        response = generate_response(model, question, char_to_idx, idx_to_char)
        results.append((question, response))
    return results


def show_checkpoint_panel(checkpoint_num, step, loss, results, prev_results=None):
    """Show checkpoint results in a nice panel."""
    if RICH_AVAILABLE:
        console = Console()
        
        # Header
        console.print()
        console.print("=" * 70, style="bold cyan")
        console.print(f"CHECKPOINT {checkpoint_num} - Step {step:,} | Loss: {loss:.4f}", 
                     style="bold yellow", justify="center")
        console.print("=" * 70, style="bold cyan")
        console.print()
        
        # Show responses
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Question", style="cyan", width=25)
        table.add_column("Response", style="green", width=35)
        if prev_results:
            table.add_column("Previous", style="dim", width=10)
        
        for i, (question, response) in enumerate(results):
            if prev_results and i < len(prev_results):
                prev_response = prev_results[i][1]
                improved = "📈" if len(response) > len(prev_response) else "📉"
                table.add_row(question, response, improved)
            else:
                table.add_row(question, response)
        
        console.print(table)
        console.print()
    else:
        # Fallback to simple print
        print()
        print("=" * 70)
        print(f"CHECKPOINT {checkpoint_num} - Step {step:,} | Loss: {loss:.4f}")
        print("=" * 70)
        print()
        for question, response in results:
            print(f"Q: {question}")
            print(f"A: {response}")
            print()


def train_interactive(model, optimizer, loss_fn, train_data, test_questions, 
                     char_to_idx, idx_to_char, max_time_minutes=15, 
                     checkpoint_steps=1000, auto_continue_seconds=10):
    """
    Train with interactive checkpoints.
    
    Args:
        checkpoint_steps: Pause every N steps to show results
        auto_continue_seconds: Auto-continue after N seconds (0 = wait for ENTER)
    """
    max_time_seconds = max_time_minutes * 60
    
    print("=" * 70)
    print(f"INTERACTIVE TRAINING - {max_time_minutes} MINUTES")
    print("=" * 70)
    print(f"Dataset: {len(train_data)} conversations")
    print(f"Checkpoints: Every {checkpoint_steps} steps")
    print(f"Auto-continue: {auto_continue_seconds}s (or press ENTER)")
    print("=" * 70)
    print()
    print("Watch the model learn from gibberish to coherent responses!")
    print()
    
    # Initial evaluation (before training)
    print("Evaluating initial model (untrained)...")
    initial_results = evaluate_at_checkpoint(model, test_questions, char_to_idx, idx_to_char)
    show_checkpoint_panel(0, 0, 999.9, initial_results)
    
    if auto_continue_seconds > 0:
        print(f"Starting training in {auto_continue_seconds} seconds (or press ENTER)...")
        time.sleep(auto_continue_seconds)
    elif auto_continue_seconds == 0:
        print("Starting training immediately...")
        time.sleep(0.5)
    else:
        input("Press ENTER to start training...")
    
    print()
    print("Training started...")
    print()
    
    start_time = time.time()
    losses = []
    step = 0
    checkpoint_num = 1
    prev_results = initial_results
    
    next_checkpoint = checkpoint_steps
    
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
        
        # Show progress every 100 steps
        if step % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            print(f"[{int(elapsed):4d}s] Step {step:5d} | Loss: {avg_loss:.4f}")
        
        # Checkpoint evaluation
        if step >= next_checkpoint:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            
            print()
            print(f"Evaluating at step {step}...")
            current_results = evaluate_at_checkpoint(model, test_questions, char_to_idx, idx_to_char)
            
            show_checkpoint_panel(checkpoint_num, step, avg_loss, current_results, prev_results)
            
            prev_results = current_results
            checkpoint_num += 1
            next_checkpoint += checkpoint_steps
            
            # Interactive pause
            if auto_continue_seconds > 0:
                print(f"Continuing in {auto_continue_seconds}s (or press ENTER)...")
                time.sleep(auto_continue_seconds)
            elif auto_continue_seconds == 0:
                print("Continuing immediately...")
                time.sleep(0.5)
            else:
                input("Press ENTER to continue training...")
            
            print()
            print("Training resumed...")
            print()
    
    # Final results
    final_elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    initial_loss = np.mean(losses[:10])
    improvement = (1 - final_loss / initial_loss) * 100
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total time: {final_elapsed:.1f}s ({final_elapsed/60:.1f} minutes)")
    print(f"Total steps: {step:,}")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print()
    
    # Final evaluation
    print("Final evaluation...")
    final_results = evaluate_at_checkpoint(model, test_questions, char_to_idx, idx_to_char)
    show_checkpoint_panel("FINAL", step, final_loss, final_results, prev_results)
    
    return losses, step


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("TINYTALKS INTERACTIVE LEARNING DASHBOARD")
    print("=" * 70)
    print()
    print("Watch a transformer learn to chat in real-time!")
    print("You'll see responses improve from gibberish to coherent answers.")
    print()
    
    # Dataset
    conversations = create_tinytalks_dataset()
    stats = get_dataset_stats()
    
    print(f"Dataset: {stats['total_examples']} examples ({stats['unique_examples']} unique)")
    print()
    
    # Tokenizer
    char_to_idx, idx_to_char = create_tokenizer(conversations)
    vocab_size = len(idx_to_char)
    
    # Encode
    max_seq_len = 80
    train_data = [encode_conversation(q, a, char_to_idx, max_seq_len) for q, a in conversations]
    
    # Test questions for checkpoints
    test_questions = [
        "Hi",
        "How are you",
        "What is your name",
        "What is the sky",
        "Is grass green",
    ]
    
    # Model: Ultra-tiny for speed
    config = {
        'vocab_size': vocab_size,
        'embed_dim': 16,
        'num_layers': 1,
        'num_heads': 2,
        'max_seq_len': max_seq_len,
    }
    
    model = GPT(**config)
    num_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"Model: {num_params:,} parameters")
    print()
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Settings
    train_time = 5  # minutes (shorter for demo)
    checkpoint_steps = 1000  # Evaluate every 1000 steps (~1-2 minutes)
    auto_continue = 0  # Auto-continue immediately (0 = no wait for demo)
    
    print(f"Training for {train_time} minutes")
    print(f"Checkpoints every {checkpoint_steps} steps")
    print()
    
    # Train with interactive checkpoints
    losses, total_steps = train_interactive(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        test_questions=test_questions,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        max_time_minutes=train_time,
        checkpoint_steps=checkpoint_steps,
        auto_continue_seconds=auto_continue
    )
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("You just watched a transformer learn from scratch!")
    print(f"✓ {total_steps:,} training steps")
    print(f"✓ {len(losses)} loss values")
    print(f"✓ {(1 - np.mean(losses[-100:])/np.mean(losses[:10]))*100:.1f}% improvement")
    print()
    print("Key takeaway: Loss decrease = Better responses!")
    print()


if __name__ == "__main__":
    main()

