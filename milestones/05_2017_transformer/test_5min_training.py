"""
Milestone 05 - 5-Minute Training Test
======================================

GOAL: Train the best possible transformer in exactly 5 minutes.

We'll optimize for:
- Maximum learning in 5 minutes
- Clear progress visualization
- Actual generation testing
- Student-friendly output

This will show what's realistically achievable in a classroom demo.
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

enable_autograd()

# ============================================================================
# Dataset: Mix of memorization + patterns
# ============================================================================

def create_dataset():
    """Create a diverse but simple dataset."""
    sequences = [
        # Easy memorization
        "AAAA", "BBBB", "CCCC", "1111", "2222",
        # Simple sequences
        "ABCD", "EFGH", "IJKL", "MNOP", "QRST",
        "1234", "5678", "9012",
        # Patterns (with repetition for learning)
        "AB", "CD", "EF", "GH",
        "12", "34", "56", "78",
    ] * 3  # Triple the dataset for better learning
    return sequences


def create_tokenizer(sequences):
    """Simple character tokenizer."""
    all_chars = sorted(set(''.join(sequences)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(all_chars)}
    idx_to_char = {idx + 1: char for idx, char in enumerate(all_chars)}
    char_to_idx['<PAD>'] = 0
    idx_to_char[0] = '<PAD>'
    return char_to_idx, idx_to_char


def encode(seq, char_to_idx, max_len=10):
    """Encode and pad sequence."""
    tokens = [char_to_idx.get(c, 0) for c in seq]
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def decode(tokens, idx_to_char):
    """Decode tokens to string."""
    return ''.join([idx_to_char.get(t, '') for t in tokens if t != 0])


# ============================================================================
# Training with 5-minute time limit
# ============================================================================

def train_5_minutes(model, optimizer, loss_fn, train_data, max_time_seconds=300):
    """
    Train for exactly 5 minutes, show progress throughout.
    """
    print("=" * 70)
    print("TRAINING FOR 5 MINUTES")
    print("=" * 70)
    print(f"Dataset: {len(train_data)} sequences")
    print(f"Time limit: {max_time_seconds}s ({max_time_seconds/60:.1f} minutes)")
    print()
    
    start_time = time.time()
    losses = []
    step = 0
    
    # Progress checkpoints at 1, 2, 3, 4, 5 minutes
    checkpoints = [60, 120, 180, 240, 300]
    checkpoint_idx = 0
    
    print("Training started...")
    print()
    
    while True:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= max_time_seconds:
            break
        
        # Sample random sequence
        tokens = train_data[np.random.randint(len(train_data))]
        
        # Next token prediction
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        
        x = Tensor(np.array([input_seq], dtype=np.int32), requires_grad=False)
        y_true = Tensor(np.array([target_seq], dtype=np.int32), requires_grad=False)
        
        # Forward
        logits = model.forward(x)
        
        # Loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = y_true.reshape(batch_size * seq_len)
        loss = loss_fn.forward(logits_flat, targets_flat)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in model.parameters():
            if param.grad is not None:
                np.clip(param.grad, -1.0, 1.0, out=param.grad)
        
        # Update
        optimizer.step()
        
        losses.append(loss.data.item())
        step += 1
        
        # Show progress at checkpoints
        if checkpoint_idx < len(checkpoints) and elapsed >= checkpoints[checkpoint_idx]:
            avg_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
            steps_per_sec = step / elapsed
            print(f"[{int(elapsed):3d}s] Step {step:4d} | Loss: {avg_loss:.4f} | Speed: {steps_per_sec:.2f} steps/sec")
            checkpoint_idx += 1
        
        # Also show every 50 steps if we're going fast
        if step % 50 == 0:
            if checkpoint_idx == 0 or elapsed < checkpoints[0]:  # Only if we haven't hit first checkpoint
                avg_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
                print(f"[{int(elapsed):3d}s] Step {step:4d} | Loss: {avg_loss:.4f}")
    
    final_elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    initial_loss = np.mean(losses[:10])
    improvement = (1 - final_loss / initial_loss) * 100
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {final_elapsed:.1f}s ({final_elapsed/60:.2f} minutes)")
    print(f"Total steps: {step}")
    print(f"Steps/second: {step/final_elapsed:.2f}")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print()
    
    return losses, step


# ============================================================================
# Testing
# ============================================================================

def test_generation(model, test_sequences, char_to_idx, idx_to_char):
    """Test generation quality."""
    print("=" * 70)
    print("TESTING GENERATION")
    print("=" * 70)
    print()
    
    correct = 0
    total = len(test_sequences)
    
    for seq in test_sequences[:15]:  # Test first 15
        tokens = encode(seq, char_to_idx, max_len=10)
        
        # Get predictions
        x = Tensor(np.array([tokens[:-1]], dtype=np.int32), requires_grad=False)
        logits = model.forward(x)
        
        # Predict each position
        predicted_tokens = []
        for i in range(logits.shape[1]):
            pred = int(np.argmax(logits.data[0, i, :]))
            predicted_tokens.append(pred)
        
        # Compare
        expected = tokens[1:]
        match = all(e == p for e, p in zip(expected, predicted_tokens) if e != 0)
        
        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        expected_str = decode(expected, idx_to_char)
        predicted_str = decode(predicted_tokens, idx_to_char)
        
        print(f"{status} Input: {seq[:6]:8s} → Expected: {expected_str:8s} | Got: {predicted_str:8s}")
    
    accuracy = (correct / 15) * 100  # Out of 15 tested
    print()
    print(f"Accuracy: {correct}/15 ({accuracy:.1f}%)")
    print()
    
    return accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("MILESTONE 05 - 5-MINUTE TRAINING TEST")
    print("=" * 70)
    print()
    print("Let's find out what we can learn in exactly 5 minutes!")
    print()
    
    # Dataset
    sequences = create_dataset()
    char_to_idx, idx_to_char = create_tokenizer(sequences)
    vocab_size = len(idx_to_char)
    
    print(f"Dataset: {len(sequences)} sequences (with repetition)")
    print(f"Unique sequences: {len(set(sequences))}")
    print(f"Vocabulary: {vocab_size} tokens")
    print()
    
    # Encode
    train_data = [encode(seq, char_to_idx, max_len=10) for seq in sequences]
    
    # Model: Ultra-tiny for maximum steps in 5 mins
    # Goal: <1s per step → ~300+ steps in 5 mins
    # Strategy: Minimize params for speed
    config = {
        'vocab_size': vocab_size,
        'embed_dim': 16,      # Very small
        'num_layers': 1,      # Just 1 layer!
        'num_heads': 2,       # 2 heads
        'max_seq_len': 10,
    }
    
    print("Model configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()
    
    model = GPT(**config)
    num_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Train for 5 minutes
    print("Starting 5-minute training run...")
    print("(Progress will be shown every minute)")
    print()
    
    losses, total_steps = train_5_minutes(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        max_time_seconds=300  # 5 minutes
    )
    
    # Test
    print("Testing what the model learned...")
    print()
    accuracy = test_generation(model, sequences, char_to_idx, idx_to_char)
    
    # Final summary
    print("=" * 70)
    print("5-MINUTE TRAINING SUMMARY")
    print("=" * 70)
    print(f"✓ Model: {num_params:,} parameters")
    print(f"✓ Steps completed: {total_steps}")
    print(f"✓ Loss: {np.mean(losses[:10]):.4f} → {np.mean(losses[-100:]):.4f}")
    print(f"✓ Improvement: {(1 - np.mean(losses[-100:])/np.mean(losses[:10]))*100:.1f}%")
    print(f"✓ Accuracy: {accuracy:.1f}%")
    print()
    
    if accuracy >= 60:
        print("🎉 EXCELLENT! Model learned well in 5 minutes!")
    elif accuracy >= 40:
        print("✓ GOOD! Model is learning, could use more training.")
    elif accuracy >= 20:
        print("⚠️  FAIR: Model is learning but needs optimization.")
    else:
        print("⚠️  Model needs more training time or tuning.")
    print()


if __name__ == "__main__":
    main()

