"""
Milestone 05 - Level 1: Transformer Memorization Test
======================================================

SIMPLEST POSSIBLE TRANSFORMER TEST:
Can the transformer memorize and reproduce simple sequences?

Task: Given "ABCD", predict "BCDE"
      Given "1234", predict "2345"

Expected: 
- Train in < 2 minutes
- Loss should drop from ~3.0 to < 0.1
- Should perfectly predict next character

This validates:
✓ Transformer architecture works
✓ Attention mechanism works
✓ Gradient flow works
✓ Training loop works
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
# Level 1: Simple Memorization Dataset
# ============================================================================

def create_memorization_dataset():
    """
    Create ultra-simple sequences to memorize:
    - Alphabet sequences: ABCD, EFGH, etc.
    - Number sequences: 1234, 5678, etc.
    - Pattern sequences: AAAA, BBBB, etc.
    """
    sequences = [
        # Alphabet
        "ABCDE",
        "FGHIJ",
        "KLMNO",
        "PQRST",
        "UVWXY",
        # Numbers
        "12345",
        "67890",
        # Patterns
        "AAAAA",
        "BBBBB",
        "CCCCC",
        # Mixed
        "A1B2C",
        "X9Y8Z",
    ]
    return sequences


def create_simple_tokenizer(sequences):
    """Create character-level tokenizer for sequences."""
    # Get all unique characters
    all_chars = sorted(set(''.join(sequences)))
    
    # Create mappings (0 is reserved for padding)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(all_chars)}
    idx_to_char = {idx + 1: char for idx, char in enumerate(all_chars)}
    char_to_idx['<PAD>'] = 0
    idx_to_char[0] = '<PAD>'
    
    return char_to_idx, idx_to_char


def encode_sequence(seq, char_to_idx, max_len=8):
    """Encode sequence to token IDs."""
    tokens = [char_to_idx.get(c, 0) for c in seq]
    # Pad to max_len
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def decode_sequence(tokens, idx_to_char):
    """Decode token IDs to string."""
    chars = [idx_to_char.get(t, '') for t in tokens if t != 0]
    return ''.join(chars)


# ============================================================================
# Training
# ============================================================================

def train_memorization(model, optimizer, loss_fn, train_data, vocab_size, max_steps=200):
    """
    Train transformer to memorize sequences.
    Target: < 2 minutes, loss < 0.1
    """
    print("=" * 70)
    print("TRAINING LEVEL 1: MEMORIZATION")
    print("=" * 70)
    print(f"Dataset: {len(train_data)} sequences")
    print(f"Vocab size: {vocab_size}")
    print(f"Max steps: {max_steps}")
    print(f"Target: Loss < 0.1 in < 2 minutes")
    print()
    
    start_time = time.time()
    losses = []
    
    for step in range(max_steps):
        # Sample random sequence
        tokens = train_data[np.random.randint(len(train_data))]
        
        # Input: all but last token
        # Target: all but first token (next token prediction)
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        
        # Convert to tensors
        x = Tensor(np.array([input_seq], dtype=np.int32), requires_grad=False)
        y_true = Tensor(np.array([target_seq], dtype=np.int32), requires_grad=False)
        
        # Forward pass
        logits = model.forward(x)
        
        # Compute loss
        batch_size, seq_len, vocab_size_out = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size_out)
        targets_flat = y_true.reshape(batch_size * seq_len)
        loss = loss_fn.forward(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in model.parameters():
            if param.grad is not None:
                np.clip(param.grad, -1.0, 1.0, out=param.grad)
        
        # Update
        optimizer.step()
        
        losses.append(loss.data.item())
        
        # Progress every 50 steps
        if step % 50 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{max_steps} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            
            # Early stopping
            if avg_loss < 0.2:
                print(f"\n✓ Target reached! Loss < 0.2 at step {step}")
                break
    
    elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:])
    initial_loss = np.mean(losses[:10])
    improvement = (1 - final_loss / initial_loss) * 100
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print()
    
    return losses


# ============================================================================
# Testing
# ============================================================================

def test_memorization(model, test_sequences, char_to_idx, idx_to_char):
    """
    Test if model can reproduce memorized sequences.
    """
    print("=" * 70)
    print("TESTING LEVEL 1: MEMORIZATION")
    print("=" * 70)
    print()
    
    correct = 0
    total = len(test_sequences)
    
    for seq in test_sequences:
        # Encode
        tokens = encode_sequence(seq, char_to_idx, max_len=8)
        
        # Get model predictions
        x = Tensor(np.array([tokens[:-1]], dtype=np.int32), requires_grad=False)
        logits = model.forward(x)
        
        # Decode predictions (greedy)
        predicted_tokens = []
        for i in range(logits.shape[1]):
            next_token = int(np.argmax(logits.data[0, i, :]))
            predicted_tokens.append(next_token)
        
        # Compare
        expected = tokens[1:]  # Target sequence
        predicted = predicted_tokens
        
        # Check if match (ignoring padding)
        match = True
        for exp, pred in zip(expected, predicted):
            if exp == 0:  # Padding, stop checking
                break
            if exp != pred:
                match = False
                break
        
        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        # Decode for display
        expected_str = decode_sequence(expected, idx_to_char)
        predicted_str = decode_sequence(predicted, idx_to_char)
        
        print(f"{status} Input: {seq[:4]:8s} → Expected: {expected_str:8s} | Got: {predicted_str:8s}")
    
    accuracy = (correct / total) * 100
    print()
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print()
    
    if accuracy >= 90:
        print("✓ LEVEL 1 PASSED: Transformer can memorize sequences!")
    else:
        print("✗ LEVEL 1 FAILED: Needs more training or debugging")
    
    return accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("MILESTONE 05 - LEVEL 1: TRANSFORMER MEMORIZATION TEST")
    print("=" * 70)
    print()
    print("Goal: Train transformer to memorize simple sequences in < 2 minutes")
    print()
    
    # Create dataset
    sequences = create_memorization_dataset()
    char_to_idx, idx_to_char = create_simple_tokenizer(sequences)
    vocab_size = len(idx_to_char)
    
    print(f"Dataset: {len(sequences)} sequences")
    print(f"Vocabulary: {vocab_size} tokens")
    print(f"Example: {sequences[0]} → {encode_sequence(sequences[0], char_to_idx)}")
    print()
    
    # Encode all sequences
    train_data = [encode_sequence(seq, char_to_idx, max_len=8) for seq in sequences]
    
    # Create ULTRA-tiny model for speed
    config = {
        'vocab_size': vocab_size,
        'embed_dim': 16,      # Super tiny!
        'num_layers': 1,      # Just 1 layer
        'num_heads': 2,       # 2 heads
        'max_seq_len': 8,     # Short sequences
    }
    
    print("Model configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()
    
    model = GPT(**config)
    num_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = CrossEntropyLoss()
    
    # Train
    print("Starting training...")
    print()
    losses = train_memorization(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        vocab_size=vocab_size,
        max_steps=200  # Reduced for speed (ultra-tiny model)
    )
    
    # Test
    print("Starting testing...")
    print()
    accuracy = test_memorization(model, sequences, char_to_idx, idx_to_char)
    
    # Summary
    print("=" * 70)
    print("LEVEL 1 SUMMARY")
    print("=" * 70)
    print(f"✓ Training: {len(losses)} steps")
    print(f"✓ Loss: {np.mean(losses[:10]):.4f} → {np.mean(losses[-100:]):.4f}")
    print(f"✓ Accuracy: {accuracy:.1f}%")
    print()
    
    if accuracy >= 90:
        print("🎉 LEVEL 1 COMPLETE! Ready for Level 2: Pattern Completion")
    else:
        print("⚠️  LEVEL 1 INCOMPLETE: Needs debugging")
    print()


if __name__ == "__main__":
    main()

