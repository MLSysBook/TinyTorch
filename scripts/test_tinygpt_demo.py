#!/usr/bin/env python3
"""
Test TinyGPT package demo to see if text generation works
"""

import sys
import time
import tinytorch.tinygpt as tgpt

def test_tinygpt_demo():
    """Test if TinyGPT can generate text as a packaged demo"""
    print("🤖 TinyGPT Package Demo Test")
    print("=" * 50)
    
    # Simple Shakespeare text for testing
    text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them."""
    
    print(f"📚 Training text: {len(text)} characters")
    
    try:
        # Create tokenizer
        print("\n🔤 Creating tokenizer...")
        tokenizer = tgpt.CharTokenizer(vocab_size=50)
        tokenizer.fit(text)
        vocab_size = tokenizer.get_vocab_size()
        print(f"   Vocabulary size: {vocab_size}")
        
        # Create model
        print("\n🧠 Creating TinyGPT model...")
        model = tgpt.TinyGPT(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_length=128,
            dropout=0.1
        )
        print(f"   Model parameters: {model.count_parameters():,}")
        
        # Create trainer
        print("\n🎓 Creating trainer...")
        trainer = tgpt.LanguageModelTrainer(model, tokenizer)
        
        # Test generation BEFORE training (should be random)
        print("\n📝 Pre-training generation test:")
        prompt = "To be"
        generated = trainer.generate_text(prompt, max_length=20, temperature=1.0)
        print(f"   '{prompt}' → '{generated}'")
        
        # Quick training test
        print("\n🚀 Quick training test (1 epoch)...")
        history = trainer.fit(
            text=text,
            epochs=1,
            seq_length=16,
            batch_size=2,
            val_split=0.2,
            verbose=True
        )
        
        # Test generation AFTER training
        print("\n📝 Post-training generation test:")
        for temp in [0.3, 0.7, 1.0]:
            generated = trainer.generate_text(prompt, max_length=30, temperature=temp)
            print(f"   '{prompt}' (T={temp}) → '{generated}'")
        
        print("\n✅ TinyGPT package demo successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tinygpt_demo()
    sys.exit(0 if success else 1)