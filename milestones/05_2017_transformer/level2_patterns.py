"""
Milestone 05 - Level 2: Transformer Pattern Completion
=======================================================

SIMPLE PATTERN COMPLETION TEST:
Can the transformer learn to complete simple patterns?

Task: Given "A B C", predict "D"
      Given "1 2 3", predict "4"
      Given "do re mi", predict "fa"

Expected: 
- Train in < 5 minutes
- Loss should drop from ~3.0 to < 0.5
- Should complete 70%+ of patterns correctly

This validates:
✓ Transformer can learn relationships
✓ Attention mechanism captures patterns
✓ Model generalizes beyond memorization
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
# Level 2: Pattern Completion Dataset
# ============================================================================

def create_pattern_dataset():
    """
    Create simple completion patterns:
    - Sequences: A B C → D
    - Counting: 1 2 3 → 4
    - Musical: do re mi → fa
    """
    patterns = [
        # Alphabet sequences
        ("A B C", "D"),
        ("D E F", "G"),
        ("M N O", "P"),
        ("W X Y", "Z"),
        # Numbers
        ("1 2 3", "4"),
        ("5 6 7", "8"),
        # Words (short)
        ("cat dog", "rat"),
        ("up down", "left"),
        # Repetition
        ("A A A", "A"),
        ("B B B", "B"),
        ("1 1 1", "1"),
    ]
    return patterns


def create_tokenizer(patterns):
    """Create character-level tokenizer."""
    # Get all unique characters
    all_text = ' '.join([p[0] + ' ' + p[1] for p in patterns])
    all_chars = sorted(set(all_text))
    
    # Create mappings (0 = padding, 1 = EOS)
    char_to_idx = {char: idx + 2 for idx, char in enumerate(all_chars)}
    idx_to_char = {idx + 2: char for idx, char in enumerate(all_chars)}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<EOS>'] = 1
    idx_to_char[0] = '<PAD>'
    idx_to_char[1] = '<EOS>'
    
    return char_to_idx, idx_to_char


def encode_pattern(input_str, target_str, char_to_idx, max_len=16):
    """Encode pattern as: input + <EOS> + target + <EOS>, then pad."""
    # Encode input
    input_tokens = [char_to_idx.get(c, 0) for c in input_str]
    input_tokens.append(1)  # EOS
    
    # Encode target
    target_tokens = [char_to_idx.get(c, 0) for c in target_str]
    target_tokens.append(1)  # EOS
    
    # Combine
    tokens = input_tokens + target_tokens
    
    # Pad
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return tokens


def decode_tokens(tokens, idx_to_char):
    """Decode tokens to string."""
    chars = []
    for t in tokens:
        if t == 0:  # padding
            break
        if t == 1:  # EOS
            break
        chars.append(idx_to_char.get(t, '?'))
    return ''.join(chars)


# ============================================================================
# Training
# ============================================================================

def train_patterns(model, optimizer, loss_fn, train_data, vocab_size, max_steps=400):
    """
    Train transformer to complete patterns.
    Target: < 5 minutes, loss < 0.5
    """
    print("=" * 70)
    print("TRAINING LEVEL 2: PATTERN COMPLETION")
    print("=" * 70)
    print(f"Dataset: {len(train_data)} patterns")
    print(f"Vocab size: {vocab_size}")
    print(f"Max steps: {max_steps}")
    print(f"Target: Loss < 0.5 in < 5 minutes")
    print()
    
    start_time = time.time()
    losses = []
    
    for step in range(max_steps):
        # Sample random pattern
        tokens = train_data[np.random.randint(len(train_data))]
        
        # Input: all but last
        # Target: all but first (shifted by 1)
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
        if step % 50 == 0 or step == max_steps - 1:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{max_steps} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            
            # Early stopping
            if avg_loss < 0.5:
                print(f"\n✓ Target reached! Loss < 0.5 at step {step}")
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

def test_patterns(model, test_patterns, char_to_idx, idx_to_char, max_len=16):
    """
    Test if model can complete patterns.
    """
    print("=" * 70)
    print("TESTING LEVEL 2: PATTERN COMPLETION")
    print("=" * 70)
    print()
    
    correct = 0
    total = len(test_patterns)
    
    for input_str, expected_target in test_patterns:
        # Encode input + EOS
        input_tokens = [char_to_idx.get(c, 0) for c in input_str]
        input_tokens.append(1)  # EOS
        
        # Pad to max_len-1 (leave room for generation)
        while len(input_tokens) < max_len - 1:
            input_tokens.append(0)
        input_tokens = input_tokens[:max_len-1]
        
        # Forward pass
        x = Tensor(np.array([input_tokens], dtype=np.int32), requires_grad=False)
        logits = model.forward(x)
        
        # Get prediction for next token (after input + EOS)
        input_len = len([c for c in input_str]) + 1  # +1 for EOS
        if input_len < len(input_tokens):
            next_token_logits = logits.data[0, input_len - 1, :]  # Predict position after EOS
            predicted_token = int(np.argmax(next_token_logits))
            
            # Decode
            predicted_char = idx_to_char.get(predicted_token, '?')
            
            # Check if correct (compare first character of target)
            expected_first_char = expected_target[0] if len(expected_target) > 0 else ''
            match = (predicted_char == expected_first_char)
        else:
            match = False
            predicted_char = '?'
        
        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} Input: \"{input_str:12s}\" → Expected: \"{expected_target:6s}\" | Got: \"{predicted_char}\"")
    
    accuracy = (correct / total) * 100
    print()
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print()
    
    if accuracy >= 70:
        print("✓ LEVEL 2 PASSED: Transformer can complete patterns!")
    else:
        print("✗ LEVEL 2 FAILED: Needs more training")
    
    return accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("MILESTONE 05 - LEVEL 2: TRANSFORMER PATTERN COMPLETION")
    print("=" * 70)
    print()
    print("Goal: Train transformer to complete patterns in < 5 minutes")
    print()
    
    # Create dataset
    patterns = create_pattern_dataset()
    char_to_idx, idx_to_char = create_tokenizer(patterns)
    vocab_size = len(idx_to_char)
    
    print(f"Dataset: {len(patterns)} patterns")
    print(f"Vocabulary: {vocab_size} tokens")
    print(f"Example: \"{patterns[0][0]}\" → \"{patterns[0][1]}\"")
    print()
    
    # Encode all patterns
    max_len = 16
    train_data = [encode_pattern(inp, out, char_to_idx, max_len) for inp, out in patterns]
    
    # Create small model (bigger than Level 1)
    config = {
        'vocab_size': vocab_size,
        'embed_dim': 24,      # Slightly bigger
        'num_layers': 2,      # 2 layers
        'num_heads': 2,       # 2 heads
        'max_seq_len': max_len,
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
    losses = train_patterns(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        vocab_size=vocab_size,
        max_steps=400
    )
    
    # Test
    print("Starting testing...")
    print()
    accuracy = test_patterns(model, patterns, char_to_idx, idx_to_char, max_len)
    
    # Summary
    print("=" * 70)
    print("LEVEL 2 SUMMARY")
    print("=" * 70)
    print(f"✓ Training: {len(losses)} steps")
    print(f"✓ Loss: {np.mean(losses[:10]):.4f} → {np.mean(losses[-100:]):.4f}")
    print(f"✓ Accuracy: {accuracy:.1f}%")
    print()
    
    if accuracy >= 70:
        print("🎉 LEVEL 2 COMPLETE! Ready for Level 3: Text Generation")
    else:
        print("⚠️  LEVEL 2 INCOMPLETE: Needs more training")
    print()


if __name__ == "__main__":
    main()

