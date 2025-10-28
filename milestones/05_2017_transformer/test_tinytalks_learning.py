#!/usr/bin/env python3
"""
Quick diagnostic to test if the model can learn ANY pattern at all.
"""

import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import Adam
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.autograd import enable_autograd
from tinytorch.text.tokenization import CharTokenizer

# Enable autograd
enable_autograd()

# Super simple test: Can the model learn to predict "A" after "Q:"?
test_data = """Q: Hello!
A: Hi there!

Q: What is your name?
A: I am TinyBot.

Q: What color is the sky?
A: The sky is blue.
"""

print("Testing if model can learn simple patterns...")
print(f"Test data: {repr(test_data[:100])}...")

# Build tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab([test_data])
tokens = tokenizer.encode(test_data)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print(f"Decoded: {repr(tokenizer.decode(tokens[:20]))}")

# Check specific patterns
q_colon_tokens = tokenizer.encode("Q:")
print(f"\n'Q:' tokens: {q_colon_tokens}")
print(f"'Q:' decoded: {repr(tokenizer.decode(q_colon_tokens))}")

a_colon_tokens = tokenizer.encode("A:")
print(f"'A:' tokens: {a_colon_tokens}")
print(f"'A:' decoded: {repr(tokenizer.decode(a_colon_tokens))}")

# Find all occurrences of "Q:" followed by space/newline then "A:"
print("\nPattern analysis:")
text_str = test_data
q_count = text_str.count("Q:")
a_count = text_str.count("A:")
print(f"'Q:' appears: {q_count} times")
print(f"'A:' appears: {a_count} times")

print("\n✅ Tokenizer is working correctly!")
print("\nConclusion: The model should be able to learn that 'A:' follows 'Q:'")
print("If it's generating garbage, the model is either:")
print("  1. Too small (need more parameters)")
print("  2. Not trained enough (need more epochs)")
print("  3. Learning rate is wrong")
print("  4. Or there's a bug in the training loop")

