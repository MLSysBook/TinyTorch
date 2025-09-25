# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
"""
# Tokenization - Text Processing for Language Models

Welcome to the Tokenization module! You'll implement the fundamental text processing systems that convert raw text into numerical sequences that neural networks can understand.

## Learning Goals
- Systems understanding: How tokenization affects model performance, memory usage, and computational efficiency
- Core implementation skill: Build character and subword tokenizers from scratch
- Pattern recognition: Understand how tokenization choices impact model capacity and training dynamics
- Framework connection: See how your implementations match production tokenization systems
- Performance insight: Learn how tokenization throughput affects training pipeline efficiency

## Build → Use → Reflect
1. **Build**: Character tokenizer and basic BPE (Byte Pair Encoding) implementation
2. **Use**: Process real text and observe how different tokenization strategies affect sequence length
3. **Reflect**: How does tokenization choice determine model efficiency and language understanding?

## What You'll Achieve
By the end of this module, you'll understand:
- Deep technical understanding of how text becomes numbers that models can process
- Practical capability to implement tokenizers that handle real text data efficiently
- Systems insight into how vocabulary size affects memory usage and model performance
- Performance consideration of how tokenization speed affects overall training throughput
- Connection to production systems like GPT's tokenizers and their design trade-offs

## Systems Reality Check
💡 **Production Context**: Modern language models use sophisticated tokenizers (GPT's tiktoken, SentencePiece) - your implementation reveals the algorithmic foundations
⚡ **Performance Note**: Tokenization can become a bottleneck in training pipelines - efficient string processing is critical for high-throughput training
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| default_exp core.tokenization

#| export
import os
import sys
import re
import json
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict

# Import our Tensor class - try from package first, then from local module
try:
    from tinytorch.core.tensor import Tensor
except ImportError:
    # For development, import from local tensor module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_tensor'))
    from tensor_dev import Tensor

# %% nbgrader={"grade": false, "grade_id": "tokenization-welcome", "locked": false, "schema_version": 3, "solution": false, "task": false}
print("🔤 TinyTorch Tokenization Module")
print("Ready to build text processing systems!")

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/source/11_tokenization/tokenization_dev.py`  
**Building Side:** Code exports to `tinytorch.core.tokenization`

```python
# Final package structure:
from tinytorch.core.tokenization import CharTokenizer, BPETokenizer
from tinytorch.core.tensor import Tensor  # Foundation
from tinytorch.core.embeddings import Embedding  # Next module
```

**Why this matters:**
- **Learning:** Focused modules for deep understanding
- **Production:** Proper organization like Hugging Face's tokenizers
- **Consistency:** All tokenization tools live together in `core.tokenization`
- **Integration:** Works seamlessly with embeddings and language models
"""

# %% [markdown]
"""
## What is Tokenization?

### The Problem: Text to Numbers
Neural networks work with numbers, but we want to process text:
```
"Hello world!" → [15496, 995, 0]  # Numbers the model can understand
```

### Tokenization Strategies

**Character-level tokenization:**
- "Hello" → ['H', 'e', 'l', 'l', 'o'] → [72, 101, 108, 108, 111]
- Small vocabulary (~256 characters)
- Long sequences (every character is a token)

**Subword tokenization (BPE):**
- "Hello" → ['Hel', 'lo'] → [1234, 5678]
- Medium vocabulary (~50k subwords)
- Moderate sequences (chunks of characters)

**Word-level tokenization:**
- "Hello world!" → ['Hello', 'world', '!'] → [15496, 995, 33]
- Large vocabulary (~100k+ words)
- Short sequences (each word is a token)

### Systems Trade-offs
- **Vocabulary size** affects model parameters (embedding table size)
- **Sequence length** affects memory usage (O(N²) attention scaling)
- **Tokenization speed** affects training throughput
"""

# %% [markdown]
"""
## Character Tokenizer Implementation

Let's start with the simplest tokenizer: character-level. Every character becomes a token.
"""

# %% nbgrader={"grade": false, "grade_id": "char-tokenizer", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class CharTokenizer:
    """
    Character-level tokenizer that converts text to character tokens.
    
    Simple but effective for understanding tokenization fundamentals.
    Used in character-level language models and as baseline for comparison.
    """
    
    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        """
        Initialize character tokenizer with optional special tokens.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Initialize character-to-index and index-to-character mappings
        2. Add standard special tokens (PAD, UNK, BOS, EOS)
        3. Build vocabulary from printable ASCII characters
        4. Add any additional special tokens provided
        
        DESIGN DECISIONS:
        - Use ASCII characters (32-126) for basic English text
        - Reserve indices 0-3 for special tokens
        - Build bidirectional mappings for efficiency
        
        Args:
            special_tokens: Optional dict of special token name -> index
        """
        ### BEGIN SOLUTION
        # Initialize mappings
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Standard special tokens
        default_special = {
            '<PAD>': 0,   # Padding token
            '<UNK>': 1,   # Unknown token
            '<BOS>': 2,   # Beginning of sequence
            '<EOS>': 3    # End of sequence
        }
        
        # Merge with user-provided special tokens
        if special_tokens is None:
            special_tokens = {}
        all_special = {**default_special, **special_tokens}
        
        # Add special tokens first
        for token, idx in all_special.items():
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
            self.vocab_size = max(self.vocab_size, idx + 1)
        
        # Add printable ASCII characters (space to ~)
        next_idx = self.vocab_size
        for i in range(32, 127):  # ASCII printable characters
            char = chr(i)
            if char not in self.char_to_idx:
                self.char_to_idx[char] = next_idx
                self.idx_to_char[next_idx] = char
                next_idx += 1
        
        self.vocab_size = next_idx
        ### END SOLUTION
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to list of token indices.
        
        TODO: Implement text encoding.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Optionally add beginning-of-sequence token
        2. Convert each character to its index
        3. Handle unknown characters with UNK token
        4. Optionally add end-of-sequence token
        5. Return list of integers
        
        EXAMPLE:
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("Hi!")
        # Returns: [2, 72, 105, 33, 3] (BOS, H, i, !, EOS)
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token indices
        """
        ### BEGIN SOLUTION
        tokens = []
        
        # Add beginning of sequence token
        if add_special_tokens:
            tokens.append(self.char_to_idx['<BOS>'])
        
        # Convert each character
        for char in text:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                # Unknown character - use UNK token
                tokens.append(self.char_to_idx['<UNK>'])
        
        # Add end of sequence token
        if add_special_tokens:
            tokens.append(self.char_to_idx['<EOS>'])
        
        return tokens
        ### END SOLUTION
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert list of token indices back to text.
        
        TODO: Implement token decoding.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Convert each token index to its character
        2. Optionally skip special tokens (PAD, UNK, BOS, EOS)
        3. Join characters into string
        4. Return decoded text
        
        EXAMPLE:
        tokenizer = CharTokenizer()
        text = tokenizer.decode([2, 72, 105, 33, 3])
        # Returns: "Hi!" (BOS and EOS removed)
        
        Args:
            tokens: List of token indices
            skip_special_tokens: Whether to exclude special tokens
            
        Returns:
            Decoded text string
        """
        ### BEGIN SOLUTION
        special_tokens = {'<PAD>', '<UNK>', '<BOS>', '<EOS>'}
        chars = []
        
        for token_idx in tokens:
            if token_idx in self.idx_to_char:
                char = self.idx_to_char[token_idx]
                # Skip special tokens if requested
                if skip_special_tokens and char in special_tokens:
                    continue
                chars.append(char)
            else:
                # Unknown token index
                if not skip_special_tokens:
                    chars.append('<UNK>')
        
        return ''.join(chars)
        ### END SOLUTION
    
    def pad_sequences(self, sequences: List[List[int]], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Pad sequences to uniform length for batch processing.
        
        This function is PROVIDED to show padding implementation.
        Essential for creating batches of text data.
        """
        if not sequences:
            return []
        
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        pad_token = self.char_to_idx['<PAD>']
        padded = []
        
        for sequence in sequences:
            if len(sequence) >= max_length:
                # Truncate if too long
                padded.append(sequence[:max_length])
            else:
                # Pad if too short
                padding_needed = max_length - len(sequence)
                padded_sequence = sequence + [pad_token] * padding_needed
                padded.append(padded_sequence)
        
        return padded

# %% [markdown]
"""
### 🧪 Test Your Character Tokenizer Implementation

Once you implement the CharTokenizer encode and decode methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-char-tokenizer-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_char_tokenizer():
    """Unit test for the character tokenizer."""
    print("🔬 Unit Test: Character Tokenizer...")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Test basic encoding
    text = "Hi!"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    expected_chars = ['H', 'i', '!']
    
    assert len(tokens) == len(expected_chars), f"Expected {len(expected_chars)} tokens, got {len(tokens)}"
    
    # Test decoding
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    
    # Test with special tokens
    tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
    assert len(tokens_with_special) == len(tokens) + 2, "Should add BOS and EOS tokens"
    assert tokens_with_special[0] == tokenizer.char_to_idx['<BOS>'], "First token should be BOS"
    assert tokens_with_special[-1] == tokenizer.char_to_idx['<EOS>'], "Last token should be EOS"
    
    # Test vocabulary size (4 special + 95 ASCII = 99 total)
    assert tokenizer.vocab_size >= 99, "Should have at least 99 tokens (4 special + 95 ASCII)"
    
    # Test unknown character handling
    unknown_tokens = tokenizer.encode("🚀", add_special_tokens=False)  # Emoji not in ASCII
    assert unknown_tokens[0] == tokenizer.char_to_idx['<UNK>'], "Should use UNK token for unknown chars"
    
    # Test padding
    sequences = [[1, 2, 3], [4, 5]]
    padded = tokenizer.pad_sequences(sequences, max_length=4)
    assert len(padded[0]) == 4, "First sequence should be padded to length 4"
    assert len(padded[1]) == 4, "Second sequence should be padded to length 4"
    assert padded[1][-1] == tokenizer.char_to_idx['<PAD>'], "Should use PAD token for padding"
    
    print("✅ Character tokenizer tests passed!")
    print(f"✅ Vocabulary size: {tokenizer.vocab_size}")
    print(f"✅ Encode/decode cycle works correctly")
    print(f"✅ Special tokens handled properly")
    print(f"✅ Padding functionality works")

# Test function defined (called in main block)

# %% [markdown]
"""
## Basic BPE (Byte Pair Encoding) Tokenizer

Now let's implement a simplified version of BPE, the subword tokenization algorithm used in GPT and many modern language models.

### BPE Algorithm Overview:
1. Start with character-level tokenization
2. Find the most frequent pair of adjacent tokens
3. Merge this pair into a new token
4. Repeat until desired vocabulary size reached

This creates subword units that balance vocabulary size and sequence length.
"""

# %% nbgrader={"grade": false, "grade_id": "bpe-tokenizer", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
class BPETokenizer:
    """
    Basic Byte Pair Encoding (BPE) tokenizer implementation.
    
    Learns subword units by iteratively merging the most frequent
    character pairs. This creates a vocabulary that balances
    sequence length and vocabulary size.
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (includes special tokens)
        """
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.merges = []  # List of (pair, new_token) merges learned during training
        self.trained = False
        
        # Initialize with special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for i, token in enumerate(special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token
    
    def _get_word_tokens(self, text: str) -> List[List[str]]:
        """
        Convert text to list of words, where each word is a list of characters.
        
        This function is PROVIDED to handle text preprocessing.
        """
        # Simple whitespace tokenization, then character splitting
        words = text.lower().split()
        word_tokens = []
        
        for word in words:
            # Add end-of-word marker to distinguish word boundaries
            word_chars = list(word) + ['</w>']
            word_tokens.append(word_chars)
        
        return word_tokens
    
    def _get_pair_counts(self, word_tokens: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent token pairs.
        
        TODO: Implement pair counting.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Initialize empty count dictionary
        2. For each word (list of tokens):
           - For each adjacent pair of tokens
           - Count how many times this pair appears
        3. Return dictionary of (token1, token2) -> count
        
        EXAMPLE:
        word_tokens = [['h', 'e', 'l', 'l', 'o', '</w>'], ['h', 'i', '</w>']]
        pairs = _get_pair_counts(word_tokens)
        # Returns: {('h', 'e'): 1, ('e', 'l'): 1, ('l', 'l'): 1, ('l', 'o'): 1, ('o', '</w>'): 1, ('h', 'i'): 1, ('i', '</w>'): 1}
        
        Args:
            word_tokens: List of words, each word is list of tokens
            
        Returns:
            Dictionary mapping token pairs to their counts
        """
        ### BEGIN SOLUTION
        pair_counts = defaultdict(int)
        
        for word in word_tokens:
            # Count adjacent pairs in this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1
        
        return dict(pair_counts)
        ### END SOLUTION
    
    def _merge_pair(self, word_tokens: List[List[str]], pair: Tuple[str, str], new_token: str) -> List[List[str]]:
        """
        Replace all occurrences of a token pair with a new merged token.
        
        TODO: Implement pair merging.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Create new list to store updated words
        2. For each word:
           - Scan through tokens looking for the target pair
           - When found, replace pair with new_token
           - Continue until no more pairs in this word
        3. Return updated word tokens
        
        EXAMPLE:
        word_tokens = [['h', 'e', 'l', 'l', 'o', '</w>']]
        pair = ('l', 'l')
        new_token = 'll'
        result = _merge_pair(word_tokens, pair, new_token)
        # Returns: [['h', 'e', 'll', 'o', '</w>']]
        
        Args:
            word_tokens: List of words (each word is list of tokens)
            pair: The token pair to merge
            new_token: The new token to replace the pair
            
        Returns:
            Updated word tokens with pairs merged
        """
        ### BEGIN SOLUTION
        updated_words = []
        
        for word in word_tokens:
            new_word = []
            i = 0
            
            while i < len(word):
                # Check if current position has the target pair
                if (i < len(word) - 1 and 
                    word[i] == pair[0] and 
                    word[i + 1] == pair[1]):
                    # Found the pair - replace with merged token
                    new_word.append(new_token)
                    i += 2  # Skip both tokens in the pair
                else:
                    # No pair match - keep current token
                    new_word.append(word[i])
                    i += 1
            
            updated_words.append(new_word)
        
        return updated_words
        ### END SOLUTION
    
    def train(self, texts: List[str]) -> None:
        """
        Train BPE tokenizer on a corpus of texts.
        
        This function is PROVIDED to show the complete BPE training algorithm.
        Students implement the helper functions above.
        """
        print(f"Training BPE tokenizer (target vocab size: {self.vocab_size})...")
        
        # Step 1: Convert texts to word tokens (character level initially)
        all_word_tokens = []
        for text in texts:
            word_tokens = self._get_word_tokens(text)
            all_word_tokens.extend(word_tokens)
        
        # Step 2: Build initial character vocabulary
        all_chars = set()
        for word in all_word_tokens:
            all_chars.update(word)
        
        # Add characters to vocabulary (after special tokens)
        next_idx = len(self.char_to_idx)
        for char in sorted(all_chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = next_idx
                self.idx_to_char[next_idx] = char
                next_idx += 1
        
        # Step 3: Iteratively merge most frequent pairs
        current_word_tokens = all_word_tokens
        
        while len(self.char_to_idx) < self.vocab_size:
            # Count all adjacent pairs
            pair_counts = self._get_pair_counts(current_word_tokens)
            
            if not pair_counts:
                print("No more pairs to merge!")
                break
            
            # Find most frequent pair
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            most_frequent_count = pair_counts[most_frequent_pair]
            
            if most_frequent_count < 2:
                print("No pairs occur more than once - stopping merge process")
                break
            
            # Create new merged token
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            
            # Add to vocabulary
            self.char_to_idx[new_token] = len(self.char_to_idx)
            self.idx_to_char[len(self.idx_to_char)] = new_token
            
            # Record this merge for later encoding
            self.merges.append((most_frequent_pair, new_token))
            
            # Apply merge to all words
            current_word_tokens = self._merge_pair(current_word_tokens, most_frequent_pair, new_token)
            
            if len(self.char_to_idx) % 100 == 0:
                print(f"  Vocabulary size: {len(self.char_to_idx)}, Last merge: {most_frequent_pair} -> '{new_token}' (count: {most_frequent_count})")
        
        self.trained = True
        print(f"Training complete! Final vocabulary size: {len(self.char_to_idx)}")
        print(f"Learned {len(self.merges)} merges")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text using trained BPE tokenizer.
        
        This function is PROVIDED to show BPE encoding process.
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before encoding!")
        
        # Convert to word tokens (character level initially)
        word_tokens = self._get_word_tokens(text)
        
        # Apply all learned merges in order
        for pair, new_token in self.merges:
            word_tokens = self._merge_pair(word_tokens, pair, new_token)
        
        # Convert tokens to indices
        tokens = []
        if add_special_tokens:
            tokens.append(self.char_to_idx['<BOS>'])
        
        for word in word_tokens:
            for token in word:
                if token in self.char_to_idx:
                    tokens.append(self.char_to_idx[token])
                else:
                    tokens.append(self.char_to_idx['<UNK>'])
        
        if add_special_tokens:
            tokens.append(self.char_to_idx['<EOS>'])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode tokens back to text.
        
        This function is PROVIDED to show BPE decoding process.
        """
        special_tokens = {'<PAD>', '<UNK>', '<BOS>', '<EOS>'}
        token_strings = []
        
        for token_idx in tokens:
            if token_idx in self.idx_to_char:
                token_str = self.idx_to_char[token_idx]
                if skip_special_tokens and token_str in special_tokens:
                    continue
                token_strings.append(token_str)
        
        # Join tokens and handle word boundaries
        result = ''.join(token_strings)
        result = result.replace('</w>', ' ')  # Replace end-of-word markers with spaces
        
        return result.strip()

# %% [markdown]
"""
### 🧪 Test Your BPE Implementation

Once you implement the BPE helper methods above, run this cell to test it:
"""

# %% nbgrader={"grade": true, "grade_id": "test-bpe-tokenizer-immediate", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_unit_bpe_tokenizer():
    """Unit test for the BPE tokenizer."""
    print("🔬 Unit Test: BPE Tokenizer...")
    
    # Create BPE tokenizer
    bpe = BPETokenizer(vocab_size=50)  # Small vocab for testing
    
    # Test training data
    training_texts = [
        "hello world hello",
        "world hello world",
        "hello hello world world"
    ]
    
    # Test training
    bpe.train(training_texts)
    
    # Verify training completed
    assert bpe.trained, "Tokenizer should be marked as trained"
    assert len(bpe.char_to_idx) >= 10, "Should have reasonable vocabulary size"
    assert len(bpe.merges) > 0, "Should have learned some merges"
    
    # Test encoding
    test_text = "hello world"
    tokens = bpe.encode(test_text, add_special_tokens=False)
    assert len(tokens) > 0, "Should produce some tokens"
    assert all(isinstance(t, int) for t in tokens), "All tokens should be integers"
    
    # Test decoding
    decoded = bpe.decode(tokens, skip_special_tokens=True)
    # Should be similar to original (might have different spacing due to </w> markers)
    assert "hello" in decoded.lower(), "Should contain 'hello'"
    assert "world" in decoded.lower(), "Should contain 'world'"
    
    # Test with special tokens
    tokens_with_special = bpe.encode(test_text, add_special_tokens=True)
    assert len(tokens_with_special) == len(tokens) + 2, "Should add BOS and EOS"
    assert tokens_with_special[0] == bpe.char_to_idx['<BOS>'], "First should be BOS"
    assert tokens_with_special[-1] == bpe.char_to_idx['<EOS>'], "Last should be EOS"
    
    # Test helper functions
    word_tokens = [['h', 'e', 'l', 'l', 'o']]
    pair_counts = bpe._get_pair_counts(word_tokens)
    assert ('l', 'l') in pair_counts, "Should find the 'll' pair"
    assert pair_counts[('l', 'l')] == 1, "Should count 'll' pair once"
    
    # Test merge function
    merged = bpe._merge_pair(word_tokens, ('l', 'l'), 'll')
    assert 'll' in merged[0], "Should contain merged token 'll'"
    assert merged[0].count('l') == 1, "Should have only one 'l' left after merge"
    
    print("✅ BPE tokenizer tests passed!")
    print(f"✅ Trained vocabulary size: {len(bpe.char_to_idx)}")
    print(f"✅ Learned {len(bpe.merges)} merges")
    print(f"✅ Encode/decode cycle works")

# Test function defined (called in main block)

# %% [markdown]
"""
## 🎯 ML Systems: Performance Analysis & Tokenization Efficiency

Now let's develop systems engineering skills by analyzing tokenization performance and understanding how tokenization choices affect downstream ML system efficiency.

### **Learning Outcome**: *"I understand how tokenization affects model memory, training speed, and language understanding"*
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-profiler", "locked": false, "schema_version": 3, "solution": true, "task": false}
#| export
import time

class TokenizationProfiler:
    """
    Performance profiling toolkit for tokenization systems.
    
    Helps ML engineers understand computational costs and optimize
    text processing pipelines for production deployment.
    """
    
    def __init__(self):
        self.results = {}
    
    def measure_tokenization_speed(self, tokenizer, texts: List[str], tokenizer_name: str) -> Dict:
        """
        Measure tokenization throughput and efficiency.
        
        TODO: Implement tokenization speed measurement.
        
        STEP-BY-STEP IMPLEMENTATION:
        1. Record start time
        2. Tokenize all texts
        3. Record end time and calculate metrics
        4. Calculate tokens per second, characters per second
        5. Return comprehensive performance metrics
        
        METRICS TO CALCULATE:
        - Total time (seconds)
        - Texts per second
        - Characters per second
        - Average tokens per text
        - Average sequence length
        
        Args:
            tokenizer: Tokenizer instance (CharTokenizer or BPETokenizer)
            texts: List of texts to tokenize
            tokenizer_name: Name for reporting
            
        Returns:
            Dictionary with performance metrics
        """
        ### BEGIN SOLUTION
        start_time = time.time()
        
        # Tokenize all texts
        all_tokens = []
        total_chars = 0
        
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.append(tokens)
            total_chars += len(text)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        total_texts = len(texts)
        total_tokens = sum(len(tokens) for tokens in all_tokens)
        
        metrics = {
            'tokenizer_name': tokenizer_name,
            'total_time_sec': total_time,
            'total_texts': total_texts,
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'texts_per_second': total_texts / total_time if total_time > 0 else 0,
            'chars_per_second': total_chars / total_time if total_time > 0 else 0,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'avg_tokens_per_text': total_tokens / total_texts if total_texts > 0 else 0,
            'avg_sequence_length': total_tokens / total_texts if total_texts > 0 else 0,
            'compression_ratio': total_chars / total_tokens if total_tokens > 0 else 0
        }
        
        return metrics
        ### END SOLUTION
    
    def compare_tokenizers(self, texts: List[str]) -> Dict:
        """
        Compare performance of different tokenization strategies.
        
        This function is PROVIDED to show comprehensive comparison.
        """
        print("🔍 TOKENIZER COMPARISON")
        print("=" * 50)
        
        # Create tokenizers
        char_tokenizer = CharTokenizer()
        
        # Train small BPE tokenizer
        bpe_tokenizer = BPETokenizer(vocab_size=200)
        bpe_tokenizer.train(texts[:10])  # Train on subset for speed
        
        tokenizers = [
            (char_tokenizer, "Character"),
            (bpe_tokenizer, "BPE")
        ]
        
        results = {}
        
        # Test each tokenizer
        for tokenizer, name in tokenizers:
            metrics = self.measure_tokenization_speed(tokenizer, texts, name)
            results[name] = metrics
            
            print(f"\n📊 {name} Tokenizer:")
            print(f"   Speed: {metrics['texts_per_second']:.1f} texts/sec")
            print(f"   Throughput: {metrics['chars_per_second']:.0f} chars/sec")
            print(f"   Avg sequence length: {metrics['avg_sequence_length']:.1f} tokens")
            print(f"   Compression ratio: {metrics['compression_ratio']:.2f} chars/token")
            print(f"   Vocabulary size: {tokenizer.vocab_size}")
        
        return results
    
    def analyze_memory_scaling(self, tokenizer, text_lengths: List[int]) -> Dict:
        """
        Analyze how tokenization memory scales with text length.
        
        This function is PROVIDED to demonstrate scaling analysis.
        """
        print(f"\n🔍 MEMORY SCALING ANALYSIS")
        print("=" * 40)
        
        scaling_results = []
        
        for length in text_lengths:
            # Create text of specified length
            test_text = "Hello world! " * (length // 13 + 1)
            test_text = test_text[:length]
            
            # Measure tokenization
            start_time = time.time()
            tokens = tokenizer.encode(test_text, add_special_tokens=False)
            end_time = time.time()
            
            # Calculate metrics
            time_taken = end_time - start_time
            memory_chars = len(test_text) * 4  # Approximate char memory (bytes)
            memory_tokens = len(tokens) * 4  # Approximate token memory (bytes)
            
            result = {
                'text_length': length,
                'num_tokens': len(tokens),
                'time_ms': time_taken * 1000,
                'memory_chars_bytes': memory_chars,
                'memory_tokens_bytes': memory_tokens,
                'total_memory_bytes': memory_chars + memory_tokens
            }
            
            scaling_results.append(result)
            print(f"   {length:>6} chars → {len(tokens):>4} tokens ({time_taken*1000:.2f}ms)")
        
        # Analyze scaling pattern
        if len(scaling_results) >= 2:
            small = scaling_results[0]
            large = scaling_results[-1]
            
            length_ratio = large['text_length'] / small['text_length']
            time_ratio = large['time_ms'] / small['time_ms']
            memory_ratio = large['total_memory_bytes'] / small['total_memory_bytes']
            
            print(f"\n📈 Scaling Analysis:")
            print(f"   Text length increased {length_ratio:.1f}x")
            print(f"   Time increased {time_ratio:.1f}x")
            print(f"   Memory increased {memory_ratio:.1f}x")
            print(f"   Scaling pattern: {'Linear' if abs(time_ratio - length_ratio) < 1 else 'Non-linear'}")
        
        return scaling_results

def analyze_tokenization_impact():
    """
    Comprehensive analysis of how tokenization affects downstream ML systems.
    
    This function is PROVIDED to show systems-level thinking.
    """
    print("🎯 TOKENIZATION IMPACT ON ML SYSTEMS")
    print("=" * 60)
    
    # Sample texts for analysis
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process tokenized text efficiently.",
        "Byte pair encoding balances vocabulary size and sequence length.",
        "Transformer models use attention mechanisms for sequence processing.",
        "Production systems require fast tokenization for real-time inference."
    ]
    
    # Create tokenizers
    char_tokenizer = CharTokenizer()
    bpe_tokenizer = BPETokenizer(vocab_size=100)
    bpe_tokenizer.train(sample_texts * 3)  # Train with more data
    
    print("\n📊 TOKENIZATION COMPARISON:")
    print(f"{'Strategy':<12} {'Vocab Size':<10} {'Avg Tokens':<10} {'Memory Impact':<15}")
    print("-" * 60)
    
    for tokenizer, name in [(char_tokenizer, "Character"), (bpe_tokenizer, "BPE")]:
        # Analyze average sequence length
        total_tokens = 0
        for text in sample_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
        
        avg_tokens = total_tokens / len(sample_texts)
        
        # Calculate memory impact
        # Embedding table: vocab_size * embedding_dim * 4 bytes (float32)
        embedding_dim = 256  # Typical small model
        embedding_memory_mb = (tokenizer.vocab_size * embedding_dim * 4) / (1024 * 1024)
        
        # Sequence memory: batch_size * seq_length * hidden_dim * 4 bytes
        batch_size = 32
        hidden_dim = 256
        sequence_memory_mb = (batch_size * avg_tokens * hidden_dim * 4) / (1024 * 1024)
        
        total_memory = embedding_memory_mb + sequence_memory_mb
        
        print(f"{name:<12} {tokenizer.vocab_size:<10} {avg_tokens:<10.1f} {total_memory:<15.1f}MB")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🔤 Character tokenizer: Small vocabulary, long sequences")
    print(f"   🧩 BPE tokenizer: Medium vocabulary, shorter sequences")
    print(f"   📈 Memory scaling: O(vocab_size * embed_dim + seq_len * batch_size)")
    print(f"   ⚡ Attention complexity: O(seq_len²) - shorter sequences = faster attention")
    print(f"   🏭 Production trade-off: Vocabulary size vs sequence length vs compute")

# %% [markdown]
"""
### 🧪 Test: Tokenization Performance Analysis

Let's test our tokenization profiler with realistic performance scenarios.
"""

# %% nbgrader={"grade": false, "grade_id": "test-tokenization-profiler", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_tokenization_profiler():
    """Test tokenization profiler with various scenarios."""
    print("🔬 Unit Test: Tokenization Performance Profiler...")
    
    profiler = TokenizationProfiler()
    
    # Create test data
    test_texts = [
        "Hello world!",
        "This is a test sentence.",
        "Tokenization speed matters for ML systems."
    ]
    
    # Test with character tokenizer
    char_tokenizer = CharTokenizer()
    metrics = profiler.measure_tokenization_speed(char_tokenizer, test_texts, "Character")
    
    # Verify metrics structure
    expected_keys = ['tokenizer_name', 'total_time_sec', 'total_texts', 'total_characters', 
                    'total_tokens', 'texts_per_second', 'chars_per_second', 'tokens_per_second',
                    'avg_tokens_per_text', 'avg_sequence_length', 'compression_ratio']
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float, str)), f"Invalid metric type for {key}"
    
    # Verify reasonable values
    assert metrics['total_texts'] == len(test_texts), "Should count texts correctly"
    assert metrics['total_characters'] > 0, "Should count characters"
    assert metrics['total_tokens'] > 0, "Should count tokens"
    assert metrics['texts_per_second'] > 0, "Should measure throughput"
    
    print("✅ Basic profiling functionality test passed")
    
    # Test comparison
    comparison_results = profiler.compare_tokenizers(test_texts)
    assert isinstance(comparison_results, dict), "Should return comparison results"
    assert len(comparison_results) >= 1, "Should test at least one tokenizer"
    
    print("✅ Tokenizer comparison test passed")
    
    # Test scaling analysis
    scaling_results = profiler.analyze_memory_scaling(char_tokenizer, [50, 100])
    assert isinstance(scaling_results, list), "Should return scaling results"
    assert len(scaling_results) == 2, "Should test both sizes"
    
    for result in scaling_results:
        assert 'text_length' in result, "Should include text length"
        assert 'num_tokens' in result, "Should include token count"
        assert result['num_tokens'] > 0, "Should produce tokens"
    
    print("✅ Scaling analysis test passed")
    print("🎯 Tokenization Profiler: All tests passed!")

# Test function defined (called in main block)

# %% [markdown]
"""
## 📊 Systems Analysis: Tokenization Impact on Model Architecture

Let's analyze how different tokenization strategies affect real ML system design choices.
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-systems-analysis", "locked": false, "schema_version": 3, "solution": false, "task": false}
def analyze_tokenization_systems_impact():
    """
    Analyze how tokenization affects ML system design and performance.
    
    This analysis helps students understand the connection between
    tokenization choices and downstream system architecture decisions.
    """
    print("🏗️ TOKENIZATION SYSTEMS IMPACT ANALYSIS")
    print("=" * 60)
    
    # Example model configurations
    model_configs = {
        'Small Model': {'embed_dim': 128, 'hidden_dim': 256, 'batch_size': 16},
        'Medium Model': {'embed_dim': 256, 'hidden_dim': 512, 'batch_size': 32},
        'Large Model': {'embed_dim': 512, 'hidden_dim': 1024, 'batch_size': 64}
    }
    
    # Sample text for analysis
    sample_text = "The transformer architecture revolutionized natural language processing through self-attention mechanisms."
    
    # Create tokenizers
    char_tokenizer = CharTokenizer()
    bpe_tokenizer = BPETokenizer(vocab_size=500)
    bpe_tokenizer.train([sample_text] * 10)
    
    tokenizers = [
        (char_tokenizer, "Character"),
        (bpe_tokenizer, "BPE-500")
    ]
    
    print(f"\n📋 ANALYSIS FOR TEXT: '{sample_text[:50]}...'")
    print(f"   Original length: {len(sample_text)} characters")
    
    for tokenizer, tok_name in tokenizers:
        tokens = tokenizer.encode(sample_text, add_special_tokens=False)
        
        print(f"\n🔤 {tok_name} Tokenization:")
        print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"   Sequence length: {len(tokens)} tokens")
        print(f"   Compression ratio: {len(sample_text)/len(tokens):.2f} chars/token")
        
        print(f"\n💾 Memory Analysis:")
        for model_name, config in model_configs.items():
            # Embedding table memory
            embed_memory = tokenizer.vocab_size * config['embed_dim'] * 4 / (1024**2)  # MB
            
            # Sequence processing memory (attention)
            seq_memory = config['batch_size'] * len(tokens) * config['hidden_dim'] * 4 / (1024**2)  # MB
            
            # Attention memory (O(N²))
            attention_memory = config['batch_size'] * len(tokens)**2 * 4 / (1024**2)  # MB
            
            total_memory = embed_memory + seq_memory + attention_memory
            
            print(f"   {model_name}: {total_memory:.1f}MB total")
            print(f"     Embedding: {embed_memory:.1f}MB, Sequence: {seq_memory:.1f}MB, Attention: {attention_memory:.1f}MB")
    
    print(f"\n🎯 KEY SYSTEM DESIGN INSIGHTS:")
    print(f"   1. Vocabulary Size Trade-offs:")
    print(f"      - Larger vocab = more parameters = more memory")
    print(f"      - Smaller vocab = longer sequences = more compute")
    print(f"   2. Sequence Length Impact:")
    print(f"      - Attention complexity: O(sequence_length²)")
    print(f"      - Memory scales quadratically with sequence length")
    print(f"   3. Production Considerations:")
    print(f"      - Character tokenization: Simple but inefficient")
    print(f"      - BPE tokenization: Balanced approach used in GPT/BERT")
    print(f"      - Vocabulary size affects model download size")
    print(f"   4. Hardware Implications:")
    print(f"      - GPU memory limits sequence length")
    print(f"      - Batch size limited by attention memory")

# Analysis function defined (called in main block)

# %% [markdown]
"""
## 🚀 Advanced: Tokenization Efficiency Techniques

Production tokenization systems use several optimization techniques. Let's implement a few key ones:
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-optimizations", "locked": false, "schema_version": 3, "solution": false, "task": false}
#| export
class OptimizedTokenizer:
    """
    Production-optimized tokenizer with caching and batch processing.
    
    Demonstrates optimization techniques used in real ML systems:
    - Caching for repeated texts
    - Batch processing for efficiency
    - Memory-efficient encoding
    """
    
    def __init__(self, base_tokenizer):
        """Initialize with a base tokenizer and optimization features."""
        self.base_tokenizer = base_tokenizer
        self.encode_cache = {}
        self.decode_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def encode_with_cache(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text with caching for repeated inputs.
        
        This optimization is critical for production systems where
        the same texts are processed repeatedly.
        """
        cache_key = (text, add_special_tokens)
        
        if cache_key in self.encode_cache:
            self.cache_hits += 1
            return self.encode_cache[cache_key]
        
        # Cache miss - compute and cache result
        self.cache_misses += 1
        tokens = self.base_tokenizer.encode(text, add_special_tokens)
        self.encode_cache[cache_key] = tokens
        
        return tokens
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True, 
                    pad_to_max: bool = True) -> List[List[int]]:
        """
        Efficiently encode multiple texts as a batch.
        
        This function is PROVIDED to show batch processing optimization.
        """
        # Encode all texts
        token_sequences = []
        for text in texts:
            tokens = self.encode_with_cache(text, add_special_tokens)
            token_sequences.append(tokens)
        
        # Pad to uniform length if requested
        if pad_to_max and hasattr(self.base_tokenizer, 'pad_sequences'):
            token_sequences = self.base_tokenizer.pad_sequences(token_sequences)
        
        return token_sequences
    
    def get_cache_stats(self) -> Dict:
        """Get caching performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.encode_cache)
        }

def demonstrate_production_optimizations():
    """
    Demonstrate production-level tokenization optimizations.
    
    This function is PROVIDED to show real-world optimization techniques.
    """
    print("🚀 PRODUCTION TOKENIZATION OPTIMIZATIONS")
    print("=" * 60)
    
    # Create optimized tokenizer
    base_tokenizer = CharTokenizer()
    optimized_tokenizer = OptimizedTokenizer(base_tokenizer)
    
    # Test data with repeated texts (common in production)
    test_texts = [
        "Hello world!",
        "Machine learning is amazing.",
        "Hello world!",  # Repeated
        "Tokenization performance matters.",
        "Hello world!",  # Repeated again
        "Machine learning is amazing.",  # Repeated
    ]
    
    print(f"📊 Testing with {len(test_texts)} texts ({len(set(test_texts))} unique)")
    
    # Measure performance without caching
    start_time = time.time()
    tokens_no_cache = []
    for text in test_texts:
        tokens = base_tokenizer.encode(text, add_special_tokens=False)
        tokens_no_cache.append(tokens)
    no_cache_time = time.time() - start_time
    
    # Measure performance with caching
    start_time = time.time()
    tokens_with_cache = []
    for text in test_texts:
        tokens = optimized_tokenizer.encode_with_cache(text, add_special_tokens=False)
        tokens_with_cache.append(tokens)
    cache_time = time.time() - start_time
    
    # Test batch encoding
    start_time = time.time()
    batch_tokens = optimized_tokenizer.batch_encode(test_texts, add_special_tokens=False, pad_to_max=True)
    batch_time = time.time() - start_time
    
    # Report results
    cache_stats = optimized_tokenizer.get_cache_stats()
    
    print(f"\n⚡ PERFORMANCE COMPARISON:")
    print(f"   No caching: {no_cache_time*1000:.2f}ms")
    print(f"   With caching: {cache_time*1000:.2f}ms ({(no_cache_time/cache_time):.1f}x speedup)")
    print(f"   Batch processing: {batch_time*1000:.2f}ms")
    
    print(f"\n📈 CACHE PERFORMANCE:")
    print(f"   Hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"   Cache hits: {cache_stats['cache_hits']}")
    print(f"   Cache misses: {cache_stats['cache_misses']}")
    print(f"   Cache size: {cache_stats['cache_size']} entries")
    
    print(f"\n🎯 PRODUCTION INSIGHTS:")
    print(f"   - Caching provides significant speedup for repeated texts")
    print(f"   - Batch processing enables vectorized operations")
    print(f"   - Memory-efficient encoding reduces allocation overhead")
    print(f"   - Cache hit rates >80% common in production systems")

# Function defined (called in main block)

# %% [markdown]
"""
## Comprehensive Testing & Integration

Let's run comprehensive tests to ensure all tokenization functionality works correctly:
"""

# %% nbgrader={"grade": false, "grade_id": "test-tokenization-comprehensive", "locked": false, "schema_version": 3, "solution": false, "task": false}
def test_tokenization_comprehensive():
    """Comprehensive test suite for all tokenization functionality."""
    print("🧪 Comprehensive Tokenization Tests...")
    
    # Test 1: Character tokenizer edge cases
    print("  Testing character tokenizer edge cases...")
    char_tokenizer = CharTokenizer()
    
    # Empty string
    empty_tokens = char_tokenizer.encode("", add_special_tokens=True)
    assert len(empty_tokens) == 2, "Empty string should have BOS and EOS tokens"
    
    # Single character
    single_tokens = char_tokenizer.encode("A", add_special_tokens=False)
    assert len(single_tokens) == 1, "Single character should produce one token"
    
    # Special characters
    special_text = "!@#$%"
    special_tokens = char_tokenizer.encode(special_text, add_special_tokens=False)
    assert len(special_tokens) == len(special_text), "Should handle special characters"
    
    # Round-trip encoding/decoding
    original = "Hello, World! 123"
    tokens = char_tokenizer.encode(original, add_special_tokens=False)
    decoded = char_tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == original, "Round-trip should preserve text"
    
    print("    ✅ Character tokenizer edge cases passed")
    
    # Test 2: BPE tokenizer robustness
    print("  Testing BPE tokenizer robustness...")
    bpe_tokenizer = BPETokenizer(vocab_size=100)
    
    # Train with diverse data
    training_data = [
        "hello world",
        "the quick brown fox",
        "machine learning systems",
        "neural network training",
        "hello hello world world"  # Repeated patterns for merging
    ]
    
    bpe_tokenizer.train(training_data)
    assert bpe_tokenizer.trained, "BPE should be trained"
    
    # Test encoding various texts
    test_cases = [
        "hello world",
        "new unseen text",
        "machine learning",
        ""  # Empty string
    ]
    
    for test_text in test_cases:
        if test_text:  # Skip empty string for basic tests
            tokens = bpe_tokenizer.encode(test_text, add_special_tokens=False)
            decoded = bpe_tokenizer.decode(tokens, skip_special_tokens=True)
            # BPE decoding might have slightly different spacing due to word boundaries
            assert test_text.replace(" ", "") in decoded.replace(" ", ""), f"BPE round-trip failed for '{test_text}'"
    
    print("    ✅ BPE tokenizer robustness passed")
    
    # Test 3: Memory efficiency with large texts
    print("  Testing memory efficiency...")
    large_text = "This is a test sentence. " * 1000  # ~25k characters
    
    start_time = time.time()
    char_tokens = char_tokenizer.encode(large_text, add_special_tokens=False)
    char_time = time.time() - start_time
    
    assert len(char_tokens) > 20000, "Should handle large texts"
    assert char_time < 1.0, "Should tokenize large text quickly"
    
    print("    ✅ Memory efficiency tests passed")
    
    # Test 4: Integration with optimization features
    print("  Testing optimization features...")
    optimized = OptimizedTokenizer(char_tokenizer)
    
    # Test caching
    test_text = "Repeated text for caching test"
    tokens1 = optimized.encode_with_cache(test_text)
    tokens2 = optimized.encode_with_cache(test_text)  # Should hit cache
    
    assert tokens1 == tokens2, "Cached results should be identical"
    
    cache_stats = optimized.get_cache_stats()
    assert cache_stats['cache_hits'] > 0, "Should have cache hits"
    assert cache_stats['hit_rate'] > 0, "Should have positive hit rate"
    
    # Test batch processing
    batch_texts = ["text one", "text two", "text three"]
    batch_results = optimized.batch_encode(batch_texts, pad_to_max=True)
    
    assert len(batch_results) == len(batch_texts), "Batch size should match input"
    assert all(len(seq) == len(batch_results[0]) for seq in batch_results), "All sequences should be padded to same length"
    
    print("    ✅ Optimization features tests passed")
    
    print("✅ All comprehensive tokenization tests passed!")

# Test function defined (called in main block)

# %% [markdown]
"""
## Main Execution Block

All tokenization tests and demonstrations are run from here when the module is executed directly:
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-main", "locked": false, "schema_version": 3, "solution": false, "task": false}
if __name__ == "__main__":
    # Run all unit tests
    test_unit_char_tokenizer()
    test_unit_bpe_tokenizer()
    test_tokenization_profiler()
    
    # Run comprehensive integration tests
    test_tokenization_comprehensive()
    
    # Performance analysis
    print("\n" + "="*60)
    print("🔍 TOKENIZATION PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Create test data
    sample_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Machine learning models require efficient tokenization for text processing.",
        "Character-level tokenization produces long sequences but small vocabularies.",
        "Byte pair encoding balances vocabulary size with sequence length efficiency.",
        "Production systems need fast tokenization to maintain training throughput."
    ]
    
    print(f"\nTesting with {len(sample_texts)} sample texts...")
    
    # Performance comparison
    profiler = TokenizationProfiler()
    comparison_results = profiler.compare_tokenizers(sample_texts)
    
    # Systems impact analysis
    analyze_tokenization_systems_impact()
    
    # Production optimizations demonstration
    demonstrate_production_optimizations()
    
    print("\n" + "="*60)
    print("🎯 TOKENIZATION MODULE COMPLETE!")
    print("="*60)
    print("All tokenization tests passed!")
    print("Ready for embedding layer integration!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Questions

Now that you've built the text processing foundation for language models, let's connect this work to broader ML systems challenges. These questions help you think critically about how tokenization scales to production language processing systems.

Take time to reflect thoughtfully on each question - your insights will help you understand how tokenization connects to real-world ML systems engineering.
"""

# %% [markdown]
"""
### Question 1: Tokenization Strategy and Model Performance Trade-offs

**Context**: Your tokenization implementations demonstrate the fundamental trade-off between vocabulary size and sequence length. In production language models, this choice affects model parameters, memory usage, training speed, and language understanding capabilities across different domains and languages.

**Reflection Question**: Design a tokenization strategy for a multilingual production language model that needs to handle 50+ languages efficiently while maintaining competitive performance. How would you balance vocabulary size constraints (limited to 100k tokens) with cross-lingual transfer learning, handle languages with different scripts and morphological complexity, and optimize for both training efficiency and inference speed? Consider the challenges of maintaining consistent tokenization quality across languages with vastly different character sets and linguistic structures.

Think about: cross-lingual vocabulary sharing, morphological complexity handling, script normalization, and inference speed optimization.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-1-tokenization-strategy", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON TOKENIZATION STRATEGY AND PERFORMANCE TRADE-OFFS:

TODO: Replace this text with your thoughtful response about multilingual tokenization strategy design.

Consider addressing:
- How would you design a tokenization strategy for 50+ languages within a 100k token limit?
- What approaches would you use to handle different scripts and morphological complexity?
- How would you optimize for both cross-lingual transfer and computational efficiency?
- What trade-offs would you make between vocabulary sharing and language-specific optimization?
- How would you ensure consistent quality across languages with different characteristics?

Write a strategic analysis connecting your tokenization implementations to real multilingual system challenges.

GRADING RUBRIC (Instructor Use):
- Demonstrates understanding of multilingual tokenization challenges (3 points)
- Designs practical approaches to vocabulary size and language coverage (3 points)
- Addresses cross-lingual transfer and efficiency considerations (2 points)
- Shows systems thinking about production language model constraints (2 points)
- Clear strategic reasoning with multilingual optimization insights (bonus points for comprehensive understanding)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring strategic analysis of multilingual tokenization
# Students should demonstrate understanding of cross-lingual efficiency and performance trade-offs
### END SOLUTION

# %% [markdown]
"""
### Question 2: Tokenization Pipeline Integration and Training Efficiency

**Context**: Your tokenization systems will integrate with large-scale training pipelines that process billions of tokens daily. The efficiency of tokenization directly impacts training throughput, data loading bottlenecks, and overall system scalability in production ML training infrastructure.

**Reflection Question**: Architect a tokenization pipeline for large-scale language model training that processes 1TB of text data daily while maintaining training pipeline efficiency. How would you design parallel tokenization processing, implement efficient caching strategies for repeated text patterns, and handle dynamic vocabulary updates during continual learning? Consider the challenges of maintaining tokenization consistency across distributed training nodes while optimizing for storage efficiency and minimizing I/O bottlenecks.

Think about: parallel processing architecture, caching strategies, storage optimization, and distributed training consistency.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-2-pipeline-integration", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON TOKENIZATION PIPELINE INTEGRATION:

TODO: Replace this text with your thoughtful response about large-scale tokenization pipeline design.

Consider addressing:
- How would you architect parallel tokenization for processing 1TB of text daily?
- What caching strategies would you implement for repeated text patterns?
- How would you handle storage optimization and I/O bottleneck minimization?
- What approaches would you use to maintain consistency across distributed training?
- How would you design the system to handle dynamic vocabulary updates?

Write an architectural analysis connecting your tokenization implementations to large-scale training infrastructure.

GRADING RUBRIC (Instructor Use):
- Shows understanding of large-scale tokenization pipeline challenges (3 points)
- Designs practical approaches to parallel processing and caching (3 points)
- Addresses distributed training and consistency requirements (2 points)
- Demonstrates systems thinking about training infrastructure optimization (2 points)
- Clear architectural reasoning with scalability insights (bonus points for comprehensive system design)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of large-scale pipeline integration
# Students should demonstrate knowledge of distributed training and infrastructure optimization
### END SOLUTION

# %% [markdown]
"""
### Question 3: Dynamic Tokenization and Adaptive Systems

**Context**: Your static tokenization implementations work well for fixed domains, but production language models increasingly need to adapt to new domains, evolving language patterns, and emerging terminology. Dynamic tokenization systems must balance stability for existing knowledge with adaptability for new linguistic patterns.

**Reflection Question**: Design an adaptive tokenization system for a production language model that needs to incorporate new domain terminology (like emerging scientific fields or evolving social media language) without degrading performance on existing tasks. How would you implement vocabulary expansion strategies that preserve existing token embeddings, handle tokenization consistency during model updates, and optimize for minimal retraining overhead? Consider the challenges of maintaining backward compatibility while enabling continuous adaptation to language evolution.

Think about: vocabulary expansion techniques, embedding preservation, consistency management, and continuous adaptation strategies.

*Target length: 150-300 words*
"""

# %% nbgrader={"grade": true, "grade_id": "question-3-dynamic-tokenization", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
"""
YOUR REFLECTION ON DYNAMIC TOKENIZATION AND ADAPTIVE SYSTEMS:

TODO: Replace this text with your thoughtful response about adaptive tokenization system design.

Consider addressing:
- How would you design vocabulary expansion for incorporating new domain terminology?
- What strategies would you use to preserve existing token embeddings during updates?
- How would you maintain tokenization consistency during model evolution?
- What approaches would minimize retraining overhead for vocabulary changes?
- How would you balance stability and adaptability in production systems?

Write a design analysis connecting your tokenization work to adaptive language model systems.

GRADING RUBRIC (Instructor Use):
- Understands dynamic tokenization challenges and adaptation requirements (3 points)
- Designs practical approaches to vocabulary evolution and embedding preservation (3 points)
- Addresses consistency and backward compatibility considerations (2 points)
- Shows systems thinking about continuous adaptation in production (2 points)
- Clear design reasoning with adaptive system insights (bonus points for innovative approaches)
"""

### BEGIN SOLUTION
# Student response area - instructor will replace this section during grading setup
# This is a manually graded question requiring understanding of adaptive tokenization systems
# Students should demonstrate knowledge of vocabulary evolution and continuous learning challenges
### END SOLUTION

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Tokenization

Congratulations! You have successfully implemented comprehensive tokenization systems for language processing:

### ✅ What You Have Built
- **Character Tokenizer**: Simple character-level tokenization with special token handling
- **BPE Tokenizer**: Subword tokenization using Byte Pair Encoding algorithm
- **Vocabulary Management**: Efficient mapping between text and numerical representations
- **Padding & Truncation**: Batch processing utilities for uniform sequence lengths
- **Performance Optimization**: Caching and batch processing for production efficiency
- **🆕 Memory Efficiency**: Optimized string processing and token caching systems
- **🆕 Systems Analysis**: Comprehensive performance profiling and scaling analysis

### ✅ Key Learning Outcomes
- **Understanding**: How text becomes numbers that neural networks can process
- **Implementation**: Built character and subword tokenizers from scratch
- **Systems Insight**: How tokenization affects model memory, performance, and capabilities
- **Performance Engineering**: Measured and optimized tokenization throughput
- **Production Context**: Understanding real-world tokenization challenges and solutions

### ✅ Technical Mastery
- **Character Tokenization**: Simple but interpretable text processing
- **BPE Algorithm**: Iterative pair merging for subword discovery
- **Vocabulary Trade-offs**: Balancing vocabulary size vs sequence length
- **Memory Optimization**: Efficient caching and batch processing techniques
- **🆕 Performance Analysis**: Measuring tokenization impact on downstream systems

### ✅ Professional Skills Developed
- **Algorithm Implementation**: Building complex text processing systems
- **Performance Engineering**: Optimizing for speed and memory efficiency
- **Systems Thinking**: Understanding tokenization's role in ML pipelines
- **Production Optimization**: Caching, batching, and scalability techniques

### ✅ Ready for Next Steps
Your tokenization systems are now ready to power:
- **Embedding Layers**: Converting tokens to dense vector representations
- **Language Models**: Processing text for transformer architectures
- **Production Systems**: Efficient text processing pipelines
- **🧠 Text Understanding**: Foundation for natural language processing

### 🔗 Connection to Real ML Systems
Your implementations mirror production systems:
- **GPT Tokenizers**: Modern language models use sophisticated BPE variants
- **SentencePiece**: Unigram language model tokenization used in many systems
- **Hugging Face Tokenizers**: Production-optimized tokenization libraries
- **Industry Applications**: Every language model relies on efficient tokenization

### 🎯 The Power of Text Processing
You have unlocked the bridge between human language and machine understanding:
- **Before**: Text was just strings of characters
- **After**: Text becomes structured numerical sequences for neural networks

**Next Module**: Embeddings - Converting your tokens into rich vector representations that capture semantic meaning!

Your tokenization systems are the first step in language understanding. Now let's build the embeddings that give tokens meaning!
"""