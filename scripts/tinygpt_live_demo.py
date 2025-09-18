#!/usr/bin/env python3
"""
TinyGPT Live Typing Demo - Shows text generation character by character
Like watching a real AI think and type!
"""

import sys
import time
import tinytorch.tinygpt as tgpt

def typewriter_effect(text, delay=0.05):
    """Print text with typewriter effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # New line at end

def live_generation_demo():
    """Demo TinyGPT with live character-by-character generation"""
    print("🤖 TinyGPT Live Generation Demo")
    print("=" * 60)
    print("Watch TinyGPT learn and generate Shakespeare-style text!")
    print()
    
    # Extended Shakespeare for better learning
    shakespeare_text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course, untrimmed;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st,
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee."""
    
    print(f"📚 Shakespeare corpus: {len(shakespeare_text):,} characters")
    print(f"   {len(shakespeare_text.split())} words from Hamlet & Sonnet 18")
    print()
    
    # Setup phase with typewriter effect
    typewriter_effect("🔤 Creating character tokenizer...")
    tokenizer = tgpt.CharTokenizer(vocab_size=100)
    tokenizer.fit(shakespeare_text)
    vocab_size = tokenizer.get_vocab_size()
    print(f"   ✅ Vocabulary: {vocab_size} unique characters")
    print()
    
    typewriter_effect("🧠 Building TinyGPT neural network...")
    model = tgpt.TinyGPT(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        max_length=200,
        dropout=0.1
    )
    print(f"   ✅ Model: {model.count_parameters():,} parameters")
    print(f"   ✅ Architecture: {3} transformer layers, {8} attention heads")
    print()
    
    typewriter_effect("🎓 Initializing training system...")
    trainer = tgpt.LanguageModelTrainer(model, tokenizer)
    print()
    
    # Pre-training generation with live typing
    print("📝 BEFORE TRAINING - Random Neural Noise:")
    print("-" * 50)
    prompts = ["To be", "Shall I", "When in"]
    
    for prompt in prompts:
        print(f"🎯 Prompt: '{prompt}'")
        print("🤖 TinyGPT: ", end='', flush=True)
        
        # Generate text
        generated = trainer.generate_text(prompt, max_length=25, temperature=1.0)
        generated_part = generated[len(prompt):]
        
        # Type out the generated part character by character
        typewriter_effect(generated_part, delay=0.08)
        print()
    
    # Training phase with progress
    print("🚀 TRAINING PHASE - Learning Shakespeare...")
    print("=" * 50)
    
    typewriter_effect("Feeding Shakespeare into neural networks...")
    print("⚡ Processing language patterns...")
    time.sleep(0.5)
    print("🔄 Optimizing attention weights...")
    time.sleep(0.5)
    print("🧮 Computing gradients...")
    time.sleep(0.5)
    
    # Actual training
    start_time = time.time()
    history = trainer.fit(
        text=shakespeare_text,
        epochs=3,
        seq_length=32,
        batch_size=4,
        val_split=0.2,
        verbose=True
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ Training complete in {training_time:.1f} seconds!")
    print(f"   Final accuracy: {history['val_accuracy'][-1]:.1%}")
    print()
    
    # Post-training generation with dramatic effect
    print("📝 AFTER TRAINING - Shakespearean AI:")
    print("=" * 50)
    
    generation_prompts = [
        "To be, or not to",
        "Shall I compare thee",
        "When in eternal",
        "The slings and arrows",
        "But thy eternal"
    ]
    
    for i, prompt in enumerate(generation_prompts, 1):
        print(f"🎭 Generation {i}/5")
        print(f"🎯 Prompt: '{prompt}'")
        print("🤖 TinyGPT: ", end='', flush=True)
        
        # Generate with different temperatures for variety
        temp = [0.3, 0.5, 0.7, 0.9, 1.0][i-1]
        generated = trainer.generate_text(prompt, max_length=40, temperature=temp)
        generated_part = generated[len(prompt):]
        
        # Live typing effect - slower and more dramatic
        typewriter_effect(generated_part, delay=0.1)
        print(f"   (temperature: {temp})")
        print()
        
        # Small pause between generations
        time.sleep(0.5)
    
    # Finale
    print("🎉 FINALE - Continuous Generation:")
    print("=" * 50)
    print("🤖 TinyGPT composing original Shakespeare-style text...")
    print()
    
    print("🎭 ", end='', flush=True)
    final_poem = trainer.generate_text("To be", max_length=80, temperature=0.6)
    typewriter_effect(final_poem, delay=0.08)
    
    print()
    print("✨ TinyGPT Demo Complete!")
    print(f"🏆 Achievements:")
    print(f"   • Built complete GPT from {model.count_parameters():,} parameters")
    print(f"   • Learned Shakespeare in {training_time:.1f} seconds")
    print(f"   • Generated original text with {vocab_size} character vocabulary")
    print(f"   • Demonstrated autoregressive language modeling")
    print()
    print("🔥 This entire AI was built from scratch using only TinyTorch!")

if __name__ == "__main__":
    try:
        live_generation_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)