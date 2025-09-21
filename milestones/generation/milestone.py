#!/usr/bin/env python3
"""
Generation Milestone: Python Code Generation with TinyGPT
Generates Python functions from natural language using YOUR TinyTorch transformer.

This demonstrates your complete language model - attention mechanisms, 
embeddings, and autoregressive generation.
"""

import tinytorch
from tinytorch.core import Tensor
from tinytorch.models import TinyGPT
from tinytorch.core.attention import MultiHeadAttention
from tinytorch.data import CodeDataset, Tokenizer
import numpy as np

# Initialize tokenizer and load dataset
print("Loading Python code dataset...")
tokenizer = Tokenizer(vocab_size=10000)
dataset = CodeDataset(tokenizer=tokenizer)

# Example prompts for code generation
prompts = [
    "def fibonacci(n):",
    "def reverse_string(s):",
    "def find_prime_numbers(limit):",
    "def bubble_sort(arr):",
    "def binary_search(arr, target):"
]

# Build TinyGPT model
class CodeGenerator:
    """TinyGPT for Python code generation."""
    
    def __init__(self, vocab_size=10000, d_model=512, n_heads=8, n_layers=6):
        self.model = TinyGPT(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=512
        )
        self.tokenizer = tokenizer
        
    def load_pretrained(self, checkpoint_path='generation_weights.npz'):
        """Load pre-trained weights for code generation."""
        self.model.load_checkpoint(checkpoint_path)
        print("✅ Loaded pre-trained TinyGPT weights")
    
    def generate(self, prompt, max_length=100, temperature=0.8):
        """Generate code from a prompt."""
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = Tensor(np.array([tokens]))
        
        generated = tokens.copy()
        
        # Autoregressive generation
        for _ in range(max_length):
            # Forward pass
            outputs = self.model.forward(input_ids)
            
            # Get next token probabilities
            next_token_logits = outputs.data[0, -1, :] / temperature
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
            
            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            
            # Stop if we generate end token
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Update input
            input_ids = Tensor(np.array([generated]))
        
        # Decode to text
        return self.tokenizer.decode(generated)
    
    def beam_search(self, prompt, beam_width=3, max_length=100):
        """Generate code using beam search for better quality."""
        # More sophisticated generation with beam search
        tokens = self.tokenizer.encode(prompt)
        
        # Initialize beams
        beams = [(tokens, 0.0)]  # (sequence, score)
        
        for _ in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                input_ids = Tensor(np.array([seq]))
                outputs = self.model.forward(input_ids)
                
                # Get top-k next tokens
                logits = outputs.data[0, -1, :]
                top_k_indices = np.argsort(logits)[-beam_width:]
                
                for token_id in top_k_indices:
                    new_seq = seq + [token_id]
                    new_score = score + logits[token_id]
                    new_beams.append((new_seq, new_score))
            
            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
            
            # Check if all beams ended
            if all(seq[-1] == self.tokenizer.eos_token_id for seq, _ in beams):
                break
        
        # Return best sequence
        best_seq, _ = beams[0]
        return self.tokenizer.decode(best_seq)

# Create and load model
generator = CodeGenerator()
generator.load_pretrained()

print("\n🤖 Generating Python code with YOUR TinyGPT...")
print("=" * 50)

# Generate code for each prompt
for i, prompt in enumerate(prompts, 1):
    print(f"\n📝 Prompt {i}: {prompt}")
    print("-" * 40)
    
    # Generate with sampling
    generated_code = generator.generate(prompt, max_length=150)
    print("Generated (sampling):")
    print(generated_code)
    
    # Generate with beam search for comparison
    beam_code = generator.beam_search(prompt, beam_width=3, max_length=150)
    print("\nGenerated (beam search):")
    print(beam_code)

# Interactive demo
print("\n" + "=" * 50)
print("🎮 INTERACTIVE MODE")
print("Enter a Python function signature to generate code!")
print("(Type 'quit' to exit)")

while True:
    user_prompt = input("\n> ")
    if user_prompt.lower() == 'quit':
        break
    
    print("\nGenerating...")
    code = generator.generate(user_prompt, max_length=200)
    print(code)

print("\n🎯 GENERATION MILESTONE COMPLETE!")
print("YOUR TinyGPT generates Python code from natural language!")
print("You've built the foundation of AI code assistants!")

print("\n📦 Modules Used:")
print("  • tinytorch.models.TinyGPT - Complete transformer architecture")
print("  • tinytorch.core.attention - Multi-head attention mechanism")
print("  • tinytorch.data.{CodeDataset, Tokenizer} - NLP pipeline")
print("  • All 16 TinyTorch modules working together!")

print("\n🚀 What You've Built:")
print("  ✅ Transformer architecture with attention")
print("  ✅ Autoregressive text generation")
print("  ✅ Beam search for quality output")
print("  ✅ Complete language model from scratch!")

print("\n💡 Real-World Impact:")
print("This technology powers GitHub Copilot, ChatGPT, and the future of programming!")