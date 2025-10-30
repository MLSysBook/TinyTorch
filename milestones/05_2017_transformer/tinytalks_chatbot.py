"""
TinyTalks Chatbot - Train a Simple Conversational AI in 10-15 Minutes
======================================================================

A minimal but functional chatbot trained on simple Q&A pairs.

Goal: Show that transformers can learn conversational patterns quickly!
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

# ============================================================================
# Tokenization
# ============================================================================

def create_tokenizer(conversations):
    """Create character-level tokenizer with special tokens."""
    # Get all unique characters
    all_text = ' '.join([q + ' ' + a for q, a in conversations])
    all_chars = sorted(set(all_text))
    
    # Special tokens
    special_tokens = {
        '<PAD>': 0,
        '<SOS>': 1,  # Start of sequence
        '<SEP>': 2,  # Separator between Q and A
        '<EOS>': 3,  # End of sequence
    }
    
    # Character mappings
    char_to_idx = {**special_tokens}
    idx_to_char = {v: k for k, v in special_tokens.items()}
    
    for idx, char in enumerate(all_chars, start=len(special_tokens)):
        char_to_idx[char] = idx
        idx_to_char[idx] = char
    
    return char_to_idx, idx_to_char


def encode_conversation(question, answer, char_to_idx, max_len=80):
    """
    Encode Q&A pair as: <SOS> question <SEP> answer <EOS> <PAD>...
    
    Example:
    Q: "Hi"
    A: "Hello"
    → [<SOS>, H, i, <SEP>, H, e, l, l, o, <EOS>, <PAD>, ...]
    """
    # Build sequence
    tokens = [char_to_idx['<SOS>']]
    
    # Add question
    for c in question:
        tokens.append(char_to_idx.get(c, 0))
    
    # Add separator
    tokens.append(char_to_idx['<SEP>'])
    
    # Add answer
    for c in answer:
        tokens.append(char_to_idx.get(c, 0))
    
    # Add EOS
    tokens.append(char_to_idx['<EOS>'])
    
    # Pad
    if len(tokens) < max_len:
        tokens = tokens + [char_to_idx['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return tokens


def decode_tokens(tokens, idx_to_char, stop_at_eos=True):
    """Decode tokens to string."""
    chars = []
    for t in tokens:
        if t == 0:  # PAD
            if stop_at_eos:
                break
        elif t == 1:  # SOS
            continue
        elif t == 2:  # SEP
            chars.append(' | ')
        elif t == 3:  # EOS
            if stop_at_eos:
                break
        else:
            chars.append(idx_to_char.get(t, '?'))
    return ''.join(chars)


# ============================================================================
# Training
# ============================================================================

def train_chatbot(model, optimizer, loss_fn, train_data, max_time_minutes=10):
    """
    Train TinyTalks chatbot.
    """
    max_time_seconds = max_time_minutes * 60
    
    print("=" * 70)
    print(f"TRAINING TINYTALKS CHATBOT FOR {max_time_minutes} MINUTES")
    print("=" * 70)
    print(f"Dataset: {len(train_data)} conversations")
    print(f"Time limit: {max_time_seconds}s ({max_time_minutes} minutes)")
    print()
    
    start_time = time.time()
    losses = []
    step = 0
    
    # Progress checkpoints every 2 minutes
    checkpoint_interval = 120  # 2 minutes
    next_checkpoint = checkpoint_interval
    
    print("Training started...")
    print()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time_seconds:
            break
        
        # Sample random conversation
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
        if elapsed >= next_checkpoint:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            steps_per_sec = step / elapsed
            mins = int(elapsed / 60)
            print(f"[{mins:2d} min] Step {step:5d} | Loss: {avg_loss:.4f} | Speed: {steps_per_sec:.1f} steps/sec")
            next_checkpoint += checkpoint_interval
        
        # Also show every 500 steps for early progress
        if step % 500 == 0 and step <= 2000:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            print(f"[{int(elapsed):4d}s] Step {step:5d} | Loss: {avg_loss:.4f}")
    
    final_elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    initial_loss = np.mean(losses[:10])
    improvement = (1 - final_loss / initial_loss) * 100
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {final_elapsed:.1f}s ({final_elapsed/60:.1f} minutes)")
    print(f"Total steps: {step:,}")
    print(f"Steps/second: {step/final_elapsed:.1f}")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print()
    
    return losses, step


# ============================================================================
# Generation / Chat
# ============================================================================

def generate_response(model, question, char_to_idx, idx_to_char, max_len=50):
    """
    Generate response to a question.
    
    Process:
    1. Encode: <SOS> question <SEP>
    2. Generate tokens until <EOS> or max_len
    3. Decode generated tokens
    """
    # Encode question
    tokens = [char_to_idx['<SOS>']]
    for c in question:
        tokens.append(char_to_idx.get(c, 0))
    tokens.append(char_to_idx['<SEP>'])
    
    # Generate response
    generated_tokens = []
    for _ in range(max_len):
        # Pad input to model's expected length
        input_tokens = tokens + generated_tokens
        while len(input_tokens) < 80:  # Match training max_len
            input_tokens.append(char_to_idx['<PAD>'])
        input_tokens = input_tokens[:80]
        
        # Forward pass
        x = Tensor(np.array([input_tokens], dtype=np.int32), requires_grad=False)
        logits = model.forward(x)
        
        # Get next token (position after current sequence)
        next_pos = len(tokens) + len(generated_tokens) - 1
        if next_pos < logits.shape[1]:
            next_logits = logits.data[0, next_pos, :]
            next_token = int(np.argmax(next_logits))
            
            # Stop at EOS or PAD
            if next_token == char_to_idx['<EOS>'] or next_token == char_to_idx['<PAD>']:
                break
            
            generated_tokens.append(next_token)
        else:
            break
    
    # Decode generated response
    response = decode_tokens(generated_tokens, idx_to_char, stop_at_eos=False)
    return response


def test_chatbot(model, test_questions, char_to_idx, idx_to_char):
    """Test chatbot on sample questions."""
    print("=" * 70)
    print("TESTING CHATBOT")
    print("=" * 70)
    print()
    
    for question in test_questions:
        response = generate_response(model, question, char_to_idx, idx_to_char)
        print(f"Q: {question}")
        print(f"A: {response}")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("TINYTALKS CHATBOT - 10-15 MINUTE TRAINING")
    print("=" * 70)
    print()
    
    # Load dataset
    conversations = create_tinytalks_dataset()
    stats = get_dataset_stats()
    
    print(f"Dataset: {stats['total_examples']} examples ({stats['unique_examples']} unique)")
    print(f"Repetition: {stats['repetition_factor']:.1f}x for better learning")
    print(f"Avg lengths: Q={stats['avg_question_len']:.1f} chars, A={stats['avg_answer_len']:.1f} chars")
    print()
    
    # Create tokenizer
    char_to_idx, idx_to_char = create_tokenizer(conversations)
    vocab_size = len(idx_to_char)
    print(f"Vocabulary: {vocab_size} tokens (including special tokens)")
    print()
    
    # Encode dataset
    max_seq_len = 80
    train_data = [encode_conversation(q, a, char_to_idx, max_seq_len) for q, a in conversations]
    
    # Model: Ultra-tiny for speed (learned from 5-min test!)
    # Target: ~20-30 steps/sec with longer sequences
    # In 10 mins (600s): ~12,000-18,000 steps
    config = {
        'vocab_size': vocab_size,
        'embed_dim': 16,      # Keep it tiny!
        'num_layers': 1,      # Just 1 layer
        'num_heads': 2,       # 2 heads
        'max_seq_len': max_seq_len,
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
    
    # Train for 15 minutes (adjustable)
    train_time = 15  # minutes
    print(f"Training for {train_time} minutes...")
    print()
    
    losses, total_steps = train_chatbot(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        max_time_minutes=train_time
    )
    
    # Test with sample questions
    test_questions = [
        "Hi",
        "How are you",
        "What is your name",
        "What is the sky",
        "Is grass green",
        "What is 1 plus 1",
        "Are you happy",
        "Bye",
    ]
    
    print("Testing chatbot responses...")
    print()
    test_chatbot(model, test_questions, char_to_idx, idx_to_char)
    
    # Summary
    print("=" * 70)
    print("TINYTALKS SUMMARY")
    print("=" * 70)
    print(f"✓ Model: {num_params:,} parameters")
    print(f"✓ Training: {train_time} minutes, {total_steps:,} steps")
    print(f"✓ Loss: {np.mean(losses[:10]):.4f} → {np.mean(losses[-100:]):.4f}")
    print(f"✓ Improvement: {(1 - np.mean(losses[-100:])/np.mean(losses[:10]))*100:.1f}%")
    print()
    print("Try it yourself:")
    print("  1. Ask simple questions from the training set")
    print("  2. The model should generate learned responses")
    print("  3. Experiment with model size and training time!")
    print()


if __name__ == "__main__":
    main()

