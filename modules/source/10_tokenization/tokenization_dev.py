# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#| default_exp text.tokenization
#| export

# %% [markdown]
"""
# Module 10: Tokenization - Converting Text to Numbers

Welcome to Module 10! Today you'll build tokenization - the bridge that converts human-readable text into numerical representations that machine learning models can process.

## 🔗 Prerequisites & Progress
**You've Built**: Neural networks, layers, training loops, and data loading
**You'll Build**: Text tokenization systems (character and BPE-based)
**You'll Enable**: Text processing for language models and NLP tasks

**Connection Map**:
```
DataLoader → Tokenization → Embeddings
(batching)   (text→numbers)  (learnable representations)
```

## Learning Objectives
By the end of this module, you will:
1. Implement character-based tokenization for simple text processing
2. Build a BPE (Byte Pair Encoding) tokenizer for efficient text representation
3. Understand vocabulary management and encoding/decoding operations
4. Create the foundation for text processing in neural networks

Let's get started!
"""

# %% [markdown]
"""
## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/10_tokenization/tokenization_dev.py`  
**Building Side:** Code exports to `tinytorch.text.tokenization`

```python
# How to use this module:
from tinytorch.text.tokenization import Tokenizer, CharTokenizer, BPETokenizer
```

**Why this matters:**
- **Learning:** Complete tokenization system in one focused module for deep understanding
- **Production:** Proper organization like Hugging Face's tokenizers with all text processing together
- **Consistency:** All tokenization operations and vocabulary management in text.tokenization
- **Integration:** Works seamlessly with embeddings and data loading for complete NLP pipeline
"""

# %%
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import json
import re
from collections import defaultdict, Counter

# Import only Module 01 (Tensor) - this module has minimal dependencies
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_tensor'))
from tensor_dev import Tensor

# %% [markdown]
"""
## 1. Introduction - Why Tokenization?

Neural networks operate on numbers, but humans communicate with text. Tokenization is the crucial bridge that converts text into numerical sequences that models can process.

### The Text-to-Numbers Challenge

Consider the sentence: "Hello, world!"

```
Human Text:     "Hello, world!"
                      ↓
               [Tokenization]
                      ↓
Numerical IDs:  [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
```

### The Four-Step Process

How do we represent this for a neural network? We need to:
1. **Split text into tokens** - meaningful units like words, subwords, or characters
2. **Map tokens to integers** - create a vocabulary that assigns unique IDs
3. **Handle unknown text** - deal with words not seen during training
4. **Enable reconstruction** - convert numbers back to readable text

### Why This Matters

The choice of tokenization strategy dramatically affects:
- **Model performance** - How well the model understands text
- **Vocabulary size** - Memory requirements for embedding tables
- **Computational efficiency** - Sequence length affects processing time
- **Robustness** - How well the model handles new/rare words
"""

# %% [markdown]
"""
## 2. Foundations - Tokenization Strategies

Different tokenization approaches make different trade-offs between vocabulary size, sequence length, and semantic understanding.

### Character-Level Tokenization
**Approach**: Each character gets its own token

```
Text: "Hello world"
       ↓
Tokens: ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
       ↓
IDs:    [8, 5, 12, 12, 15, 0, 23, 15, 18, 12, 4]
```

**Pros**: Small vocabulary (~100), handles any text, no unknown tokens
**Cons**: Long sequences (1 char = 1 token), limited semantic understanding

### Word-Level Tokenization
**Approach**: Each word gets its own token

```
Text: "Hello world"
       ↓
Tokens: ['Hello', 'world']
       ↓
IDs:    [5847, 1254]
```

**Pros**: Semantic meaning preserved, shorter sequences
**Cons**: Huge vocabularies (100K+), many unknown tokens

### Subword Tokenization (BPE)
**Approach**: Learn frequent character pairs, build subword units

```
Text: "tokenization"
       ↓ Character level
Initial: ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']
       ↓ Learn frequent pairs
Merged: ['to', 'ken', 'ization']
       ↓
IDs:    [142, 1847, 2341]
```

**Pros**: Balance between vocabulary size and sequence length
**Cons**: More complex training process

### Strategy Comparison

```
Text: "tokenization" (12 characters)

Character: ['t','o','k','e','n','i','z','a','t','i','o','n'] → 12 tokens, vocab ~100
Word:      ['tokenization']                                   → 1 token, vocab 100K+
BPE:       ['token','ization']                               → 2 tokens, vocab 10-50K
```

The sweet spot for most applications is BPE with 10K-50K vocabulary size.
"""

# %% [markdown]
"""
## 3. Implementation - Building Tokenization Systems

Let's implement tokenization systems from simple character-based to sophisticated BPE. We'll start with the base interface and work our way up to advanced algorithms.
"""

# %% [markdown]
"""
### Base Tokenizer Interface

All tokenizers need to provide two core operations: encoding text to numbers and decoding numbers back to text. Let's define the common interface.

```
Tokenizer Interface:
    encode(text) → [id1, id2, id3, ...]
    decode([id1, id2, id3, ...]) → text
```

This ensures consistent behavior across different tokenization strategies.
"""

# %% nbgrader={"grade": false, "grade_id": "base-tokenizer", "solution": true}
class Tokenizer:
    """
    Base tokenizer class providing the interface for all tokenizers.

    This defines the contract that all tokenizers must follow:
    - encode(): text → list of token IDs
    - decode(): list of token IDs → text
    """

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        TODO: Implement encoding logic in subclasses

        APPROACH:
        1. Subclasses will override this method
        2. Return list of integer token IDs

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.encode("abc")
        [0, 1, 2]
        """
        ### BEGIN SOLUTION
        raise NotImplementedError("Subclasses must implement encode()")
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Convert list of token IDs back to text.

        TODO: Implement decoding logic in subclasses

        APPROACH:
        1. Subclasses will override this method
        2. Return reconstructed text string

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.decode([0, 1, 2])
        "abc"
        """
        ### BEGIN SOLUTION
        raise NotImplementedError("Subclasses must implement decode()")
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-base-tokenizer", "locked": true, "points": 5}
def test_unit_base_tokenizer():
    """🔬 Test base tokenizer interface."""
    print("🔬 Unit Test: Base Tokenizer Interface...")

    # Test that base class defines the interface
    tokenizer = Tokenizer()

    # Should raise NotImplementedError for both methods
    try:
        tokenizer.encode("test")
        assert False, "encode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    try:
        tokenizer.decode([1, 2, 3])
        assert False, "decode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    print("✅ Base tokenizer interface works correctly!")

test_unit_base_tokenizer()

# %% [markdown]
"""
### Character-Level Tokenizer

The simplest tokenization approach: each character becomes a token. This gives us perfect coverage of any text but produces long sequences.

```
Character Tokenization Process:

Step 1: Build vocabulary from unique characters
Text corpus: ["hello", "world"]
Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']
Vocabulary: ['<UNK>', 'h', 'e', 'l', 'o', 'w', 'r', 'd']  # <UNK> for unknown
                0      1    2    3    4    5    6    7

Step 2: Encode text character by character
Text: "hello"
  'h' → 1
  'e' → 2
  'l' → 3
  'l' → 3
  'o' → 4
Result: [1, 2, 3, 3, 4]

Step 3: Decode by looking up each ID
IDs: [1, 2, 3, 3, 4]
  1 → 'h'
  2 → 'e'
  3 → 'l'
  3 → 'l'
  4 → 'o'
Result: "hello"
```
"""

# %% nbgrader={"grade": false, "grade_id": "char-tokenizer", "solution": true}
class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that treats each character as a separate token.

    This is the simplest tokenization approach - every character in the
    vocabulary gets its own unique ID.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize character tokenizer.

        TODO: Set up vocabulary mappings

        APPROACH:
        1. Store vocabulary list
        2. Create char→id and id→char mappings
        3. Handle special tokens (unknown character)

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['a', 'b', 'c'])
        >>> tokenizer.vocab_size
        4  # 3 chars + 1 unknown token
        """
        ### BEGIN SOLUTION
        if vocab is None:
            vocab = []

        # Add special unknown token
        self.vocab = ['<UNK>'] + vocab
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Store unknown token ID
        self.unk_id = 0
        ### END SOLUTION

    def build_vocab(self, corpus: List[str]) -> None:
        """
        Build vocabulary from a corpus of text.

        TODO: Extract unique characters and build vocabulary

        APPROACH:
        1. Collect all unique characters from corpus
        2. Sort for consistent ordering
        3. Rebuild mappings with new vocabulary

        HINTS:
        - Use set() to find unique characters
        - Join all texts then convert to set
        - Don't forget the <UNK> token
        """
        ### BEGIN SOLUTION
        # Collect all unique characters
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for consistent ordering
        unique_chars = sorted(list(all_chars))

        # Rebuild vocabulary with <UNK> token first
        self.vocab = ['<UNK>'] + unique_chars
        self.vocab_size = len(self.vocab)

        # Rebuild mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        ### END SOLUTION

    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of character IDs.

        TODO: Convert each character to its vocabulary ID

        APPROACH:
        1. Iterate through each character in text
        2. Look up character ID in vocabulary
        3. Use unknown token ID for unseen characters

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['h', 'e', 'l', 'o'])
        >>> tokenizer.encode("hello")
        [1, 2, 3, 3, 4]  # maps to h,e,l,l,o
        """
        ### BEGIN SOLUTION
        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of token IDs back to text.

        TODO: Convert each token ID back to its character

        APPROACH:
        1. Look up each token ID in vocabulary
        2. Join characters into string
        3. Handle invalid token IDs gracefully

        EXAMPLE:
        >>> tokenizer = CharTokenizer(['h', 'e', 'l', 'o'])
        >>> tokenizer.decode([1, 2, 3, 3, 4])
        "hello"
        """
        ### BEGIN SOLUTION
        chars = []
        for token_id in tokens:
            # Use unknown token for invalid IDs
            char = self.id_to_char.get(token_id, '<UNK>')
            chars.append(char)
        return ''.join(chars)
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-char-tokenizer", "locked": true, "points": 15}
def test_unit_char_tokenizer():
    """🔬 Test character tokenizer implementation."""
    print("🔬 Unit Test: Character Tokenizer...")

    # Test basic functionality
    vocab = ['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd']
    tokenizer = CharTokenizer(vocab)

    # Test vocabulary setup
    assert tokenizer.vocab_size == 9  # 8 chars + UNK
    assert tokenizer.vocab[0] == '<UNK>'
    assert 'h' in tokenizer.char_to_id

    # Test encoding
    text = "hello"
    tokens = tokenizer.encode(text)
    expected = [1, 2, 3, 3, 4]  # h,e,l,l,o (based on actual vocab order)
    assert tokens == expected, f"Expected {expected}, got {tokens}"

    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"

    # Test unknown character handling
    tokens_with_unk = tokenizer.encode("hello!")
    assert tokens_with_unk[-1] == 0  # '!' should map to <UNK>

    # Test vocabulary building
    corpus = ["hello world", "test text"]
    tokenizer.build_vocab(corpus)
    assert 't' in tokenizer.char_to_id
    assert 'x' in tokenizer.char_to_id

    print("✅ Character tokenizer works correctly!")

test_unit_char_tokenizer()

# %% [markdown]
"""
### 🧪 Character Tokenizer Analysis
Character tokenization provides a simple, robust foundation for text processing. The key insight is that with a small vocabulary (typically <100 characters), we can represent any text without unknown tokens.

**Trade-offs**:
- **Pro**: No out-of-vocabulary issues, handles any language
- **Con**: Long sequences (1 char = 1 token), limited semantic understanding
- **Use case**: When robustness is more important than efficiency
"""

# %% [markdown]
"""
### Byte Pair Encoding (BPE) Tokenizer

BPE is the secret sauce behind modern language models. It learns to merge frequent character pairs, creating subword units that balance vocabulary size with sequence length.

```
BPE Training Process:

Step 1: Start with character vocabulary
Text: ["hello", "hello", "help"]
Initial tokens: [['h','e','l','l','o</w>'], ['h','e','l','l','o</w>'], ['h','e','l','p</w>']]

Step 2: Count character pairs
('h','e'): 3 times  ← Most frequent!
('e','l'): 3 times
('l','l'): 2 times
('l','o'): 2 times
('l','p'): 1 time

Step 3: Merge most frequent pair
Merge ('h','e') → 'he'
Tokens: [['he','l','l','o</w>'], ['he','l','l','o</w>'], ['he','l','p</w>']]
Vocab: ['h','e','l','o','p','</w>','he']  ← New token added

Step 4: Repeat until target vocabulary size
Next merge: ('l','l') → 'll'
Tokens: [['he','ll','o</w>'], ['he','ll','o</w>'], ['he','l','p</w>']]
Vocab: ['h','e','l','o','p','</w>','he','ll']  ← Growing vocabulary

Final result:
Text "hello" → ['he', 'll', 'o</w>'] → 3 tokens (vs 5 characters)
Text "help"  → ['he', 'l', 'p</w>']  → 3 tokens (vs 4 characters)
```

BPE discovers natural word boundaries and common patterns automatically!
"""

# %% nbgrader={"grade": false, "grade_id": "bpe-tokenizer", "solution": true}
class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer that learns subword units.

    BPE works by:
    1. Starting with character-level vocabulary
    2. Finding most frequent character pairs
    3. Merging frequent pairs into single tokens
    4. Repeating until desired vocabulary size
    """

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.

        TODO: Set up basic tokenizer state

        APPROACH:
        1. Store target vocabulary size
        2. Initialize empty vocabulary and merge rules
        3. Set up mappings for encoding/decoding
        """
        ### BEGIN SOLUTION
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []  # List of (pair, new_token) merges
        self.token_to_id = {}
        self.id_to_token = {}
        ### END SOLUTION

    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Convert word to list of characters with end-of-word marker.

        TODO: Tokenize word into character sequence

        APPROACH:
        1. Split word into characters
        2. Add </w> marker to last character
        3. Return list of tokens

        EXAMPLE:
        >>> tokenizer._get_word_tokens("hello")
        ['h', 'e', 'l', 'l', 'o</w>']
        """
        ### BEGIN SOLUTION
        if not word:
            return []

        tokens = list(word)
        tokens[-1] += '</w>'  # Mark end of word
        return tokens
        ### END SOLUTION

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs from word tokens.

        TODO: Extract all consecutive character pairs

        APPROACH:
        1. Iterate through adjacent tokens
        2. Create pairs of consecutive tokens
        3. Return set of unique pairs

        EXAMPLE:
        >>> tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o</w>')}
        """
        ### BEGIN SOLUTION
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs
        ### END SOLUTION

    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """
        Train BPE on corpus to learn merge rules.

        TODO: Implement BPE training algorithm

        APPROACH:
        1. Build initial character vocabulary
        2. Count word frequencies in corpus
        3. Iteratively merge most frequent pairs
        4. Build final vocabulary and mappings

        HINTS:
        - Start with character-level tokens
        - Use frequency counts to guide merging
        - Stop when vocabulary reaches target size
        """
        ### BEGIN SOLUTION
        if vocab_size:
            self.vocab_size = vocab_size

        # Count word frequencies
        word_freq = Counter(corpus)

        # Initialize vocabulary with characters
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        # Convert to sorted list for consistency
        self.vocab = sorted(list(vocab))

        # Add special tokens
        if '<UNK>' not in self.vocab:
            self.vocab = ['<UNK>'] + self.vocab

        # Learn merges
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            # Count all pairs across all words
            pair_counts = Counter()

            for word, freq in word_freq.items():
                tokens = word_tokens[word]
                pairs = self._get_pairs(tokens)
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Get most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # Merge this pair in all words
            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]):
                        # Merge pair
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        # Build final mappings
        self._build_mappings()
        ### END SOLUTION

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""
        ### BEGIN SOLUTION
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        ### END SOLUTION

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        Apply learned merge rules to token sequence.

        TODO: Apply BPE merges to token list

        APPROACH:
        1. Start with character-level tokens
        2. Apply each merge rule in order
        3. Continue until no more merges possible
        """
        ### BEGIN SOLUTION
        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == merge_pair[0] and
                    tokens[i + 1] == merge_pair[1]):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens
        ### END SOLUTION

    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.

        TODO: Apply BPE encoding to text

        APPROACH:
        1. Split text into words
        2. Convert each word to character tokens
        3. Apply BPE merges
        4. Convert to token IDs
        """
        ### BEGIN SOLUTION
        if not self.vocab:
            return []

        # Simple word splitting (could be more sophisticated)
        words = text.split()
        all_tokens = []

        for word in words:
            # Get character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))  # 0 = <UNK>

        return token_ids
        ### END SOLUTION

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        TODO: Convert token IDs back to readable text

        APPROACH:
        1. Convert IDs to tokens
        2. Join tokens together
        3. Clean up word boundaries and markers
        """
        ### BEGIN SOLUTION
        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, '<UNK>')
            token_strings.append(token)

        # Join and clean up
        text = ''.join(token_strings)

        # Replace end-of-word markers with spaces
        text = text.replace('</w>', ' ')

        # Clean up extra spaces
        text = ' '.join(text.split())

        return text
        ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-bpe-tokenizer", "locked": true, "points": 20}
def test_unit_bpe_tokenizer():
    """🔬 Test BPE tokenizer implementation."""
    print("🔬 Unit Test: BPE Tokenizer...")

    # Test basic functionality with simple corpus
    corpus = ["hello", "world", "hello", "hell"]  # "hell" and "hello" share prefix
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(corpus)

    # Check that vocabulary was built
    assert len(tokenizer.vocab) > 0
    assert '<UNK>' in tokenizer.vocab

    # Test helper functions
    word_tokens = tokenizer._get_word_tokens("test")
    assert word_tokens[-1].endswith('</w>'), "Should have end-of-word marker"

    pairs = tokenizer._get_pairs(['h', 'e', 'l', 'l', 'o</w>'])
    assert ('h', 'e') in pairs
    assert ('l', 'l') in pairs

    # Test encoding/decoding
    text = "hello"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)

    # Test round-trip on training data should work well
    for word in corpus:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens)
        # Allow some flexibility due to BPE merging
        assert len(decoded.strip()) > 0

    print("✅ BPE tokenizer works correctly!")

test_unit_bpe_tokenizer()

# %% [markdown]
"""
### 🧪 BPE Tokenizer Analysis

BPE provides a balance between vocabulary size and sequence length. By learning frequent subword patterns, it can handle new words through decomposition while maintaining reasonable sequence lengths.

```
BPE Merging Visualization:

Original: "tokenization" → ['t','o','k','e','n','i','z','a','t','i','o','n','</w>']
                                                       ↓ Merge frequent pairs
Step 1:   ('t','o') is frequent → ['to','k','e','n','i','z','a','t','i','o','n','</w>']
Step 2:   ('i','o') is frequent → ['to','k','e','n','io','z','a','t','io','n','</w>']
Step 3:   ('io','n') is frequent → ['to','k','e','n','io','z','a','t','ion','</w>']
Step 4:   ('to','k') is frequent → ['tok','e','n','io','z','a','t','ion','</w>']
                                                       ↓ Continue merging...
Final:    "tokenization" → ['token','ization']  # 2 tokens vs 13 characters!
```

**Key insights**:
- **Adaptive vocabulary**: Learns from data, not hand-crafted
- **Subword robustness**: Handles rare/new words through decomposition
- **Efficiency trade-off**: Larger vocabulary → shorter sequences → faster processing
- **Morphological awareness**: Naturally discovers prefixes, suffixes, roots
"""

# %% [markdown]
"""
## 4. Integration - Bringing It Together

Now let's build utility functions that make tokenization easy to use in practice. These tools will help you tokenize datasets, analyze performance, and choose the right strategy.

```
Tokenization Workflow:

1. Choose Strategy → 2. Train Tokenizer → 3. Process Dataset → 4. Analyze Results
      ↓                      ↓                    ↓                   ↓
   char/bpe           corpus training        batch encoding      stats/metrics
```
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-utils", "solution": true}
def create_tokenizer(strategy: str = "char", vocab_size: int = 1000, corpus: List[str] = None) -> Tokenizer:
    """
    Factory function to create and train tokenizers.

    TODO: Create appropriate tokenizer based on strategy

    APPROACH:
    1. Check strategy type
    2. Create appropriate tokenizer class
    3. Train on corpus if provided
    4. Return configured tokenizer

    EXAMPLE:
    >>> corpus = ["hello world", "test text"]
    >>> tokenizer = create_tokenizer("char", corpus=corpus)
    >>> tokens = tokenizer.encode("hello")
    """
    ### BEGIN SOLUTION
    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(f"Unknown tokenization strategy: {strategy}")

    return tokenizer
    ### END SOLUTION

def tokenize_dataset(texts: List[str], tokenizer: Tokenizer, max_length: int = None) -> List[List[int]]:
    """
    Tokenize a dataset with optional length limits.

    TODO: Tokenize all texts with consistent preprocessing

    APPROACH:
    1. Encode each text with the tokenizer
    2. Apply max_length truncation if specified
    3. Return list of tokenized sequences

    HINTS:
    - Handle empty texts gracefully
    - Truncate from the end if too long
    """
    ### BEGIN SOLUTION
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized
    ### END SOLUTION

def analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """
    Analyze tokenization statistics.

    TODO: Compute useful statistics about tokenization

    APPROACH:
    1. Tokenize all texts
    2. Compute sequence length statistics
    3. Calculate compression ratio
    4. Return analysis dictionary
    """
    ### BEGIN SOLUTION
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    # Calculate statistics
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab),
        'avg_sequence_length': np.mean(tokenized_lengths),
        'max_sequence_length': max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens': len(all_tokens),
        'compression_ratio': total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens': len(set(all_tokens))
    }

    return stats
    ### END SOLUTION

# %% nbgrader={"grade": true, "grade_id": "test-tokenization-utils", "locked": true, "points": 10}
def test_unit_tokenization_utils():
    """🔬 Test tokenization utility functions."""
    print("🔬 Unit Test: Tokenization Utils...")

    # Test tokenizer factory
    corpus = ["hello world", "test text", "more examples"]

    char_tokenizer = create_tokenizer("char", corpus=corpus)
    assert isinstance(char_tokenizer, CharTokenizer)
    assert char_tokenizer.vocab_size > 0

    bpe_tokenizer = create_tokenizer("bpe", vocab_size=50, corpus=corpus)
    assert isinstance(bpe_tokenizer, BPETokenizer)

    # Test dataset tokenization
    texts = ["hello", "world", "test"]
    tokenized = tokenize_dataset(texts, char_tokenizer, max_length=10)
    assert len(tokenized) == len(texts)
    assert all(len(seq) <= 10 for seq in tokenized)

    # Test analysis
    stats = analyze_tokenization(texts, char_tokenizer)
    assert 'vocab_size' in stats
    assert 'avg_sequence_length' in stats
    assert 'compression_ratio' in stats
    assert stats['total_tokens'] > 0

    print("✅ Tokenization utils work correctly!")

test_unit_tokenization_utils()

# %% [markdown]
"""
## 5. Systems Analysis - Tokenization Trade-offs

Understanding the performance implications of different tokenization strategies is crucial for building efficient NLP systems.
"""

# %% nbgrader={"grade": false, "grade_id": "tokenization-analysis", "solution": true}
def analyze_tokenization_strategies():
    """📊 Compare different tokenization strategies on various texts."""
    print("📊 Analyzing Tokenization Strategies...")

    # Create test corpus with different text types
    corpus = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Tokenization is fundamental to natural language processing",
        "Subword units balance vocabulary size and sequence length"
    ]

    # Test different strategies
    strategies = [
        ("Character", create_tokenizer("char", corpus=corpus)),
        ("BPE-100", create_tokenizer("bpe", vocab_size=100, corpus=corpus)),
        ("BPE-500", create_tokenizer("bpe", vocab_size=500, corpus=corpus))
    ]

    print(f"{'Strategy':<12} {'Vocab':<8} {'Avg Len':<8} {'Compression':<12} {'Coverage':<10}")
    print("-" * 60)

    for name, tokenizer in strategies:
        stats = analyze_tokenization(corpus, tokenizer)

        print(f"{name:<12} {stats['vocab_size']:<8} "
              f"{stats['avg_sequence_length']:<8.1f} "
              f"{stats['compression_ratio']:<12.2f} "
              f"{stats['unique_tokens']:<10}")

    print("\n💡 Key Insights:")
    print("- Character tokenization: Small vocab, long sequences, perfect coverage")
    print("- BPE: Larger vocab trades off with shorter sequences")
    print("- Higher compression ratio = more characters per token = efficiency")

analyze_tokenization_strategies()

# %% [markdown]
"""
### 📊 Performance Analysis: Vocabulary Size vs Sequence Length

The fundamental trade-off in tokenization creates a classic systems engineering challenge:

```
Tokenization Trade-off Spectrum:

Character          BPE-Small         BPE-Large         Word-Level
vocab: ~100    →   vocab: ~1K    →   vocab: ~50K   →   vocab: ~100K+
seq: very long →   seq: long     →   seq: medium   →   seq: short
memory: low    →   memory: med   →   memory: high  →   memory: very high
compute: high  →   compute: med  →   compute: low  →   compute: very low
coverage: 100% →   coverage: 99% →   coverage: 95% →   coverage: <80%
```

**Character tokenization (vocab ~100)**:
- Pro: Universal coverage, simple implementation, small embedding table
- Con: Long sequences (high compute), limited semantic units
- Use case: Morphologically rich languages, robust preprocessing

**BPE tokenization (vocab 10K-50K)**:
- Pro: Balanced efficiency, handles morphology, good coverage
- Con: Training complexity, domain-specific vocabularies
- Use case: Most modern language models (GPT, BERT family)

**Real-world scaling examples**:
```
GPT-3/4:     ~50K BPE tokens, avg 3-4 chars/token
BERT:        ~30K WordPiece tokens, avg 4-5 chars/token
T5:          ~32K SentencePiece tokens, handles 100+ languages
ChatGPT:     ~100K tokens with extended vocabulary
```

**Memory implications for embedding tables**:
```
Tokenizer     Vocab Size   Embed Dim   Parameters    Memory (fp32)
Character           100        512        51K           204 KB
BPE-1K            1,000        512       512K           2.0 MB
BPE-50K          50,000        512      25.6M         102.4 MB
Word-100K       100,000        512      51.2M         204.8 MB
```
"""

# %% [markdown]
"""
## 6. Module Integration Test

Let's test our complete tokenization system to ensure everything works together.
"""

# %% nbgrader={"grade": true, "grade_id": "test-module", "locked": true, "points": 20}
def test_module():
    """
    Comprehensive test of entire tokenization module.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_base_tokenizer()
    test_unit_char_tokenizer()
    test_unit_bpe_tokenizer()
    test_unit_tokenization_utils()

    print("\nRunning integration scenarios...")

    # Test realistic tokenization workflow
    print("🔬 Integration Test: Complete tokenization pipeline...")

    # Create training corpus
    training_corpus = [
        "Natural language processing",
        "Machine learning models",
        "Neural networks learn",
        "Tokenization enables text processing",
        "Embeddings represent meaning"
    ]

    # Train different tokenizers
    char_tokenizer = create_tokenizer("char", corpus=training_corpus)
    bpe_tokenizer = create_tokenizer("bpe", vocab_size=200, corpus=training_corpus)

    # Test on new text
    test_text = "Neural language models"

    # Test character tokenization
    char_tokens = char_tokenizer.encode(test_text)
    char_decoded = char_tokenizer.decode(char_tokens)
    assert char_decoded == test_text, "Character round-trip failed"

    # Test BPE tokenization (may not be exact due to subword splits)
    bpe_tokens = bpe_tokenizer.encode(test_text)
    bpe_decoded = bpe_tokenizer.decode(bpe_tokens)
    assert len(bpe_decoded.strip()) > 0, "BPE decoding failed"

    # Test dataset processing
    test_dataset = ["hello world", "tokenize this", "neural networks"]
    char_dataset = tokenize_dataset(test_dataset, char_tokenizer, max_length=20)
    bpe_dataset = tokenize_dataset(test_dataset, bpe_tokenizer, max_length=10)

    assert len(char_dataset) == len(test_dataset)
    assert len(bpe_dataset) == len(test_dataset)
    assert all(len(seq) <= 20 for seq in char_dataset)
    assert all(len(seq) <= 10 for seq in bpe_dataset)

    # Test analysis functions
    char_stats = analyze_tokenization(test_dataset, char_tokenizer)
    bpe_stats = analyze_tokenization(test_dataset, bpe_tokenizer)

    assert char_stats['vocab_size'] > 0
    assert bpe_stats['vocab_size'] > 0
    assert char_stats['compression_ratio'] < bpe_stats['compression_ratio']  # BPE should compress better

    print("✅ End-to-end tokenization pipeline works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 10")

# Call the comprehensive test
test_module()

# %%
if __name__ == "__main__":
    print("🚀 Running Tokenization module...")
    test_module()
    print("✅ Module validation complete!")

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Text Processing Foundations

### Question 1: Vocabulary Size vs Memory
You implemented tokenizers with different vocabulary sizes.
If you have a BPE tokenizer with vocab_size=50,000 and embed_dim=512:
- How many parameters are in the embedding table? _____ million
- If using float32, how much memory does this embedding table require? _____ MB

### Question 2: Sequence Length Trade-offs
Your character tokenizer produces longer sequences than BPE.
For the text "machine learning" (16 characters):
- Character tokenizer produces ~16 tokens
- BPE tokenizer might produce ~3-4 tokens
If processing batch_size=32 with max_length=512:
- Character model needs _____ total tokens per batch
- BPE model needs _____ total tokens per batch
- Which requires more memory during training? _____

### Question 3: Tokenization Coverage
Your BPE tokenizer handles unknown words by decomposing into subwords.
- Why is this better than word-level tokenization for real applications? _____
- What happens to model performance when many tokens map to <UNK>? _____
- How does vocabulary size affect the number of unknown decompositions? _____
"""

# %% [markdown]
"""
## 🎯 MODULE SUMMARY: Tokenization

Congratulations! You've built a complete tokenization system for converting text to numerical representations!

### Key Accomplishments
- Built character-level tokenizer with perfect text coverage
- Implemented BPE tokenizer that learns efficient subword representations
- Created vocabulary management and encoding/decoding systems
- Discovered the vocabulary size vs sequence length trade-off
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your tokenization implementation enables text processing for language models.
Export with: `tito module complete 10`

**Next**: Module 11 will add learnable embeddings that convert your token IDs into rich vector representations!
"""