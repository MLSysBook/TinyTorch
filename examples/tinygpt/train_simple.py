#!/usr/bin/env python3
"""
Simple TinyGPT Training Example
Train a small language model on a simple repetitive pattern to verify it works.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.tinygpt import CharTokenizer, TinyGPT, LanguageModelTrainer

def train_simple_pattern():
    """Train TinyGPT on a simple repetitive pattern to verify learning."""
    
    print("🤖 TinyGPT Simple Pattern Training")
    print("=" * 50)
    
    # Create a simple repetitive text that should be easy to learn
    # This pattern is highly predictable: abc repeats
    simple_text = "abcabcabcabcabcabcabcabcabcabc" * 10  # 300 chars of "abc" pattern
    print(f"📝 Training text: '{simple_text[:30]}...' ({len(simple_text)} chars)")
    print(f"   Pattern: 'abc' repeated {len(simple_text)//3} times")
    
    # Create tokenizer
    print("\n🔤 Creating tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.fit(simple_text)
    print(f"   Vocabulary: {tokenizer.vocab}")
    print(f"   Vocab size: {len(tokenizer.vocab)}")
    
    # Create a tiny model (small enough to overfit quickly)
    print("\n🧠 Creating TinyGPT model...")
    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=32,     # Very small
        num_heads=2,      # Minimal heads
        num_layers=1,     # Single layer
        max_seq_len=12    # Short sequences
    )
    print(f"   Model parameters: ~{sum(np.prod(p.shape) for p in [
        model.embed.weight, model.pos_encoding.pe,
        model.head.weight if hasattr(model, 'head') else np.zeros(1)
    ]):,}")
    
    # Create trainer
    print("\n🎓 Setting up trainer...")
    trainer = LanguageModelTrainer(model, tokenizer)
    
    # Train with many epochs on this simple pattern
    print("\n🚀 Training on simple pattern...")
    history = trainer.train(
        text=simple_text,
        epochs=50,        # Many epochs to ensure learning
        batch_size=2,
        seq_length=9,     # Multiple of 3 for clean patterns
        learning_rate=0.01,
        val_split=0.1,
        verbose=True
    )
    
    # Test generation
    print("\n📝 Testing generation after training:")
    
    test_prompts = ["a", "ab", "abc", "abca", "b", "c"]
    for prompt in test_prompts:
        generated = trainer.generate(
            prompt=prompt,
            max_length=12,
            temperature=0.5
        )
        print(f"   '{prompt}' → '{generated}'")
        
        # Check if it learned the pattern
        expected_continuation = ("abc" * 10)[len(prompt):len(prompt)+12]
        if generated[len(prompt):].startswith(expected_continuation[:3]):
            print(f"      ✅ Learned pattern!")
        else:
            print(f"      ❌ Expected to continue with '{expected_continuation[:6]}...'")
    
    # Analyze training
    print("\n📈 Training Analysis:")
    if len(history['train_loss']) > 0:
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
        
        if final_loss < initial_loss * 0.5:
            print("   ✅ Model is learning!")
        else:
            print("   ⚠️ Model may not be learning effectively")
    
    return model, tokenizer, history

def test_memorization():
    """Test if TinyGPT can memorize a very short sequence."""
    print("\n🧪 Testing Memorization on Tiny Sequence")
    print("=" * 50)
    
    # Even simpler: can it memorize "hello"?
    tiny_text = "hello" * 20  # 100 chars of "hello"
    
    print(f"📝 Memorization text: '{tiny_text[:25]}...' ({len(tiny_text)} chars)")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(tiny_text)
    print(f"🔤 Vocabulary: {tokenizer.vocab}")
    
    # Create tiny model
    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=16,    # Tiny
        num_heads=1,     # Minimal
        num_layers=1,    # Single layer
        max_seq_len=10
    )
    
    # Train
    trainer = LanguageModelTrainer(model, tokenizer)
    
    print("\n🚀 Training for memorization...")
    history = trainer.train(
        text=tiny_text,
        epochs=100,       # Lots of epochs
        batch_size=1,     # Small batch
        seq_length=5,     # Length of "hello"
        learning_rate=0.1,  # Higher LR for faster learning
        val_split=0.0,    # No validation, pure memorization
        verbose=False
    )
    
    # Show progress every 10 epochs
    for i in range(0, len(history['train_loss']), 10):
        if i < len(history['train_loss']):
            print(f"   Epoch {i+1}: Loss = {history['train_loss'][i]:.4f}")
    
    # Final test
    print("\n📝 Memorization test:")
    test_prompts = ["h", "he", "hel", "hell"]
    for prompt in test_prompts:
        generated = trainer.generate(prompt, max_length=10, temperature=0.1)
        print(f"   '{prompt}' → '{generated}'")
        if "hello" in generated:
            print(f"      ✅ Memorized!")
    
    return model, tokenizer, history

if __name__ == "__main__":
    print("🔥 TinyGPT Simple Training Examples")
    print("Testing if TinyGPT can learn basic patterns...\n")
    
    # Test 1: Simple pattern
    model1, tok1, hist1 = train_simple_pattern()
    
    # Test 2: Memorization
    model2, tok2, hist2 = test_memorization()
    
    print("\n✨ Testing complete!")
    print("If the models learned their patterns, TinyGPT is working correctly.")
    print("If not, there may be issues with the gradient flow or loss computation.")