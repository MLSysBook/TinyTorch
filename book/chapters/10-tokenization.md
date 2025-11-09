---
title: "Tokenization - Text to Numerical Sequences"
description: "Build tokenizers to convert raw text into sequences for language models"
difficulty: 2
time_estimate: "4-5 hours"
prerequisites: ["Tensor"]
next_steps: ["Embeddings"]
learning_objectives:
  - "Implement character-level and subword tokenization strategies"
  - "Design efficient vocabulary management systems for language models"
  - "Understand trade-offs between vocabulary size and sequence length"
  - "Build BPE tokenizer for optimal subword unit representation"
  - "Apply text processing optimization for production NLP pipelines"
---

# 10. Tokenization

**🏛️ ARCHITECTURE TIER** | Difficulty: ⭐⭐ (2/4) | Time: 4-5 hours

## Overview

Build tokenization systems that convert raw text into numerical sequences for language models. This module implements character-level and subword tokenizers (BPE) that balance vocabulary size, sequence length, and computational efficiency.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement character-level and subword tokenization** strategies for converting text to token sequences
2. **Design efficient vocabulary management** systems with special tokens and encoding/decoding
3. **Understand trade-offs** between vocabulary size (model parameters) and sequence length (computation)
4. **Build BPE (Byte Pair Encoding)** tokenizer for optimal subword unit representation
5. **Apply text processing optimization** techniques for production NLP pipelines at scale

## Why This Matters

### Production Context

Every language model depends on tokenization:

- **GPT-4** uses a 100K-token vocabulary trained on trillions of tokens of text
- **Google Translate** processes billions of sentences daily through tokenization pipelines
- **BERT** pioneered WordPiece tokenization that handles 100+ languages efficiently
- **Code models** like Copilot use specialized tokenizers for programming languages

### Historical Context

Tokenization evolved with language modeling:

- **Word-Level (pre-2016)**: Simple but massive vocabularies (100K+ words); struggles with rare words and typos
- **Character-Level (2015)**: Small vocabulary but extremely long sequences; computationally expensive
- **BPE (2016)**: Subword tokenization balances both; enabled GPT and modern transformers
- **SentencePiece (2018)**: Unified text and multilingual tokenization; powers modern multilingual models
- **Modern (2020+)**: Specialized tokenizers for code, math, and multimodal content

The tokenizers you're building are the foundation of all modern NLP.

## Pedagogical Pattern: Build → Use → Analyze

### 1. Build

Implement from first principles:
- Character-level tokenizer with vocab management
- Special tokens (<PAD>, <UNK>, <BOS>, <EOS>)
- BPE algorithm for learning subword merges
- Encode/decode functions for text ↔ tokens
- Vocabulary serialization for model deployment

### 2. Use

Apply to real problems:
- Tokenize Shakespeare and modern text datasets
- Build vocabularies of different sizes (1K, 10K, 50K tokens)
- Compare character vs BPE on sequence length
- Handle out-of-vocabulary words gracefully
- Measure tokenization throughput (tokens/second)

### 3. Analyze

Deep-dive into design trade-offs:
- How does vocabulary size affect model parameters?
- Why do longer sequences increase computation quadratically (in transformers)?
- What's the sweet spot between vocab size and sequence length?
- How does tokenization affect rare words and morphology?
- Why do multilingual models need larger vocabularies?

## Implementation Guide

### Core Components

**Character-Level Tokenizer**
```python
class CharacterTokenizer:
    """Simple character-level tokenization.
    
    Treats each character as a token. Simple but results in long sequences.
    Vocab size: typically 100-500 (all ASCII or Unicode characters)
    """
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
    
    def build_vocab(self, texts):
        """Build vocabulary from text corpus."""
        # Add special tokens first
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for token in special_tokens:
            self.char_to_idx[token] = len(self.char_to_idx)
        
        # Add all unique characters
        unique_chars = set(''.join(texts))
        for char in sorted(unique_chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = len(self.char_to_idx)
        
        # Create reverse mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]) 
                for char in text]
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return ''.join([self.idx_to_char[idx] for idx in token_ids])
```

**BPE (Byte Pair Encoding) Tokenizer**
```python
class BPETokenizer:
    """Byte Pair Encoding for subword tokenization.
    
    Iteratively merges most frequent character pairs to create subword units.
    Balances vocabulary size and sequence length optimally.
    
    Example:
        "unhappiness" → ["un", "happi", "ness"] (3 tokens)
        vs character-level: ["u","n","h","a","p","p","i","n","e","s","s"] (11 tokens)
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.merges = {}  # Learned merge rules
        self.vocab = {}   # Token to ID mapping
    
    def train(self, texts):
        """Learn BPE merges from corpus.
        
        Algorithm:
        1. Start with character-level vocabulary
        2. Count all adjacent character pairs
        3. Merge most frequent pair
        4. Repeat until vocabulary reaches target size
        """
        # Initialize with character-level vocab
        vocab = self._get_char_vocab(texts)
        
        # Learn merges iteratively
        while len(vocab) < self.vocab_size:
            # Count pairs
            pairs = self._count_pairs(texts, vocab)
            if not pairs:
                break
            
            # Merge most frequent pair
            best_pair = max(pairs, key=pairs.get)
            texts = self._merge_pair(texts, best_pair)
            vocab.add(''.join(best_pair))
            self.merges[best_pair] = ''.join(best_pair)
        
        # Build final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
    
    def encode(self, text):
        """Encode text using learned BPE merges."""
        # Start with characters
        tokens = list(text)
        
        # Apply merges in learned order
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            if not pairs:
                break
            
            # Find first mergeable pair
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break
            
            # Apply merge
            pair = mergeable[0]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(self.merges[pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert tokens to IDs
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
```

**Vocabulary Management**
```python
class Vocabulary:
    """Manage token-to-ID mappings with special tokens.
    
    Provides clean interface for encoding/decoding and vocab serialization.
    """
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Reserve special token IDs
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.BOS_ID = 2
        self.EOS_ID = 3
        
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        special = [('<PAD>', self.PAD_ID), ('<UNK>', self.UNK_ID),
                   ('<BOS>', self.BOS_ID), ('<EOS>', self.EOS_ID)]
        for token, idx in special:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def save(self, path):
        """Save vocabulary for deployment."""
        import json
        with open(path, 'w') as f:
            json.dump(self.token_to_id, f)
    
    def load(self, path):
        """Load vocabulary for inference."""
        import json
        with open(path, 'r') as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
```

### Step-by-Step Implementation

1. **Build Character Tokenizer**
   - Create vocabulary from unique characters
   - Add special tokens (PAD, UNK, BOS, EOS)
   - Implement encode (text → IDs) and decode (IDs → text)
   - Handle unknown characters gracefully

2. **Implement BPE Algorithm**
   - Start with character vocabulary
   - Count adjacent pair frequencies
   - Merge most frequent pairs iteratively
   - Build merge rules and final vocabulary

3. **Add Vocabulary Management**
   - Create token ↔ ID bidirectional mappings
   - Implement serialization for saving/loading
   - Handle special tokens consistently
   - Support vocabulary extension

4. **Optimize for Production**
   - Cache encode/decode results
   - Use efficient data structures (tries, hash maps)
   - Batch process multiple texts
   - Measure throughput (tokens/second)

5. **Compare Tokenization Strategies**
   - Measure sequence lengths for same text
   - Analyze vocabulary size requirements
   - Test on rare words and typos
   - Evaluate multilingual performance

## Testing

### Inline Tests (During Development)

Run inline tests while building:
```bash
cd modules/source/10_tokenization
python tokenization_dev.py
```

Expected output:
```
Unit Test: Character tokenizer...
✅ Vocabulary built with 89 unique characters
✅ Encode/decode round-trip successful
✅ Special tokens handled correctly
Progress: Character Tokenizer ✓

Unit Test: BPE tokenizer...
✅ Learned 5000 merge rules from corpus
✅ Sequence length reduced 3.2x vs character-level
✅ Handles rare words and typos gracefully
Progress: BPE Tokenizer ✓

Unit Test: Vocabulary management...
✅ Token-to-ID mappings bidirectional
✅ Vocabulary saved and loaded correctly
✅ Special token IDs reserved
Progress: Vocabulary ✓
```

### Export and Validate

After completing the module:
```bash
# Export to tinytorch package
tito export 10_tokenization

# Run integration tests
tito test 10_tokenization
```

## Where This Code Lives

```
tinytorch/
├── text/
│   └── tokenization.py         # Your implementation goes here
└── __init__.py                 # Exposes CharTokenizer, BPETokenizer, etc.

Usage in other modules:
>>> from tinytorch.text import BPETokenizer
>>> tokenizer = BPETokenizer(vocab_size=10000)
>>> tokenizer.train(texts)
>>> ids = tokenizer.encode("Hello world!")
```

## Systems Thinking Questions

1. **Vocabulary Size vs Model Size**: GPT-2 has 50K vocabulary with 768-dim embeddings = 38M parameters just for embeddings. How does this scale to GPT-3's 100K vocabulary?

2. **Sequence Length vs Computation**: Transformers have O(n²) attention complexity. If BPE reduces sequence length from 1000 to 300 tokens, how much does this reduce computation?

3. **Rare Word Handling**: A word-level tokenizer marks rare words as <UNK>, losing all information. How does BPE handle rare words like "unhappiness" even if never seen during training?

4. **Multilingual Challenges**: English needs ~30K tokens for good coverage. Chinese needs 50K+. Why? How does this affect multilingual model design?

5. **Tokenization as Compression**: BPE learns common patterns like "ing", "ed", "tion". Why is this similar to data compression? What's the connection to information theory?

## Real-World Connections

### Industry Applications

**OpenAI GPT Series**
- GPT-2: 50K BPE vocabulary, trained on 8M web pages
- GPT-3: 100K vocabulary, handles code and multilingual text
- GPT-4: Advanced tiktoken library with 100K+ tokens
- Tokenization optimization critical for $700/1M token economics

**Google Multilingual Models**
- SentencePiece used in BERT, T5, PaLM for 100+ languages
- Unified tokenization across languages without preprocessing
- Optimized for fast serving at Google-scale traffic

**Code Models (GitHub Copilot, AlphaCode)**
- Specialized tokenizers for programming languages
- Handle indentation, operators, and variable names efficiently
- Balance natural language and code syntax

### Research Impact

This module implements patterns from:
- BPE (Sennrich et al., 2016): Subword tokenization for NMT
- WordPiece (Google, 2016): BERT tokenization strategy
- SentencePiece (Kudo, 2018): Language-agnostic tokenization
- tiktoken (OpenAI, 2023): Fast BPE for GPT-3/4

## What's Next?

In **Module 11: Embeddings**, you'll convert these token IDs into dense vector representations:

- Map discrete token IDs to continuous embeddings
- Learn position encodings for sequence order
- Implement lookup tables for fast embedding retrieval
- Understand how embeddings capture semantic similarity

The tokens you create here become the input to every transformer and language model!

---

**Ready to build tokenizers from scratch?** Open `modules/source/10_tokenization/tokenization_dev.py` and start implementing.
