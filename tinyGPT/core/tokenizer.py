"""
Character-level tokenizer for TinyGPT language models.

Implements character-level tokenization for use with TinyGPT transformer models.
This tokenizer converts text to sequences of character tokens and back.
"""

import numpy as np
from typing import List, Optional, Dict, Union


class CharTokenizer:
    """Character-level tokenizer for language models.
    
    This tokenizer treats each character as a separate token, making it simple
    but effective for learning character-level patterns in text. It's ideal for
    educational purposes and small-scale language modeling experiments.
    
    The tokenizer builds a vocabulary from the training text and provides
    methods for encoding text to token indices and decoding back to text.
    
    Educational Benefits:
    - Simple and transparent tokenization strategy
    - No complex subword algorithms to understand
    - Direct character-to-token mapping
    - Easy to debug and visualize
    """
    
    def __init__(self, vocab_size: Optional[int] = None, 
                 special_tokens: Optional[List[str]] = None):
        """Initialize character tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (None = unlimited)
            special_tokens: List of special tokens to include (e.g., ['<UNK>', '<PAD>'])
        
        Educational Note:
            vocab_size limiting is important for computational efficiency.
            Special tokens handle edge cases like unknown characters.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<UNK>', '<PAD>']
        
        # Core vocabulary mappings
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
        # Special token indices
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.unk_idx = 0  # Will be set in fit()
        self.pad_idx = 1  # Will be set in fit()
        
        # State tracking
        self.is_fitted = False
        self.character_counts: Dict[str, int] = {}
        
        print(f"🔤 CharTokenizer initialized:")
        print(f"   Max vocab size: {vocab_size or 'unlimited'}")
        print(f"   Special tokens: {self.special_tokens}")
    
    def fit(self, text: str) -> None:
        """Build vocabulary from training text.
        
        Args:
            text: Training text to build vocabulary from
            
        Educational Process:
        1. Count character frequencies in the text
        2. Add special tokens first (ensures consistent indices)
        3. Add most frequent characters up to vocab_size limit
        4. Create bidirectional mappings for fast lookup
        """
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text")
        
        print(f"🔍 Analyzing text for vocabulary...")
        print(f"   Text length: {len(text):,} characters")
        
        # Count character frequencies
        self.character_counts = {}
        for char in text:
            self.character_counts[char] = self.character_counts.get(char, 0) + 1
        
        unique_chars = len(self.character_counts)
        print(f"   Unique characters found: {unique_chars}")
        
        # Start building vocabulary with special tokens
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Add special tokens first (ensures consistent indices)
        for i, token in enumerate(self.special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token
        
        # Set special token indices
        self.unk_idx = self.char_to_idx[self.unk_token]
        self.pad_idx = self.char_to_idx[self.pad_token]
        
        # Sort characters by frequency (most frequent first)
        sorted_chars = sorted(self.character_counts.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Add characters to vocabulary up to limit
        current_idx = len(self.special_tokens)
        chars_added = 0
        
        for char, count in sorted_chars:
            # Skip if already in vocabulary (shouldn't happen with char-level)
            if char in self.char_to_idx:
                continue
                
            # Check vocab size limit
            if self.vocab_size and current_idx >= self.vocab_size:
                break
                
            self.char_to_idx[char] = current_idx
            self.idx_to_char[current_idx] = char
            current_idx += 1
            chars_added += 1
        
        self.is_fitted = True
        
        print(f"✅ Vocabulary built successfully:")
        print(f"   Final vocab size: {len(self.char_to_idx)}")
        print(f"   Characters included: {chars_added}")
        if self.vocab_size and chars_added < unique_chars:
            excluded = unique_chars - chars_added
            print(f"   Characters excluded: {excluded} (will map to <UNK>)")
        
        # Show most frequent characters
        print(f"   Most frequent: {sorted_chars[:10]}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to sequence of token indices.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token indices
            
        Educational Note:
            Characters not in vocabulary are mapped to <UNK> token.
            This handles rare characters and maintains fixed vocabulary size.
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding")
        
        if not text:
            return []
        
        indices = []
        unk_count = 0
        
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.unk_idx)
                unk_count += 1
        
        if unk_count > 0:
            unk_rate = unk_count / len(text) * 100
            print(f"⚠️ Encoding: {unk_count} unknown chars ({unk_rate:.1f}%)")
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert sequence of token indices back to text.
        
        Args:
            indices: List of token indices to decode
            
        Returns:
            Decoded text string
            
        Educational Note:
            Invalid indices are skipped to handle generation errors gracefully.
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before decoding")
        
        if not indices:
            return ""
        
        chars = []
        invalid_count = 0
        
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                # Skip special tokens in decoded output (except space-like chars)
                if char not in [self.pad_token]:  # Keep <UNK> for debugging
                    chars.append(char)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"⚠️ Decoding: {invalid_count} invalid indices skipped")
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.char_to_idx)
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                    padding: bool = True, truncation: bool = True) -> np.ndarray:
        """Encode batch of texts with optional padding and truncation.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length (None = longest in batch)
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length
            
        Returns:
            2D numpy array of shape (batch_size, max_length)
            
        Educational Benefits:
        - Demonstrates batch processing for efficiency
        - Shows padding/truncation strategies for variable length sequences
        - Prepares data in format expected by neural networks
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding")
        
        if not texts:
            return np.array([])
        
        # Encode all texts
        encoded_texts = [self.encode(text) for text in texts]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(encoded) for encoded in encoded_texts)
        
        # Prepare batch array
        batch_size = len(texts)
        batch_array = np.full((batch_size, max_length), self.pad_idx, dtype=np.int32)
        
        # Fill batch array
        for i, encoded in enumerate(encoded_texts):
            if truncation and len(encoded) > max_length:
                # Truncate from the end
                sequence = encoded[:max_length]
            else:
                sequence = encoded
            
            # Copy sequence into batch array
            seq_len = min(len(sequence), max_length)
            batch_array[i, :seq_len] = sequence[:seq_len]
        
        return batch_array
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get the complete vocabulary mapping.
        
        Returns:
            Dictionary mapping characters to indices
        """
        return self.char_to_idx.copy()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings.
        
        Returns:
            Dictionary mapping special tokens to indices
        """
        return {token: self.char_to_idx[token] for token in self.special_tokens}
    
    def analyze_text(self, text: str) -> Dict[str, Union[int, float]]:
        """Analyze text with current vocabulary.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis statistics
            
        Educational Purpose:
            Helps understand vocabulary coverage and tokenization quality.
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before analysis")
        
        if not text:
            return {'length': 0, 'tokens': 0, 'coverage': 0.0, 'unk_rate': 0.0}
        
        indices = self.encode(text)
        unk_count = sum(1 for idx in indices if idx == self.unk_idx)
        
        stats = {
            'length': len(text),
            'tokens': len(indices),
            'unique_chars': len(set(text)),
            'vocab_coverage': len(set(text) & set(self.char_to_idx.keys())),
            'unk_count': unk_count,
            'unk_rate': unk_count / len(indices) * 100 if indices else 0.0,
            'compression_ratio': len(text) / len(indices) if indices else 0.0
        }
        
        return stats
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file for reuse.
        
        Args:
            filepath: Path to save vocabulary file
            
        Educational Note:
            In production, you'd want to save/load vocabularies to ensure
            consistency across training and inference.
        """
        import json
        
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted tokenizer")
        
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'character_counts': self.character_counts
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file.
        
        Args:
            filepath: Path to vocabulary file
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.special_tokens = vocab_data['special_tokens']
        self.vocab_size = vocab_data['vocab_size']
        self.character_counts = vocab_data['character_counts']
        
        # Rebuild reverse mapping
        self.idx_to_char = {int(idx): char for char, idx in self.char_to_idx.items()}
        
        # Set special token indices
        self.unk_idx = self.char_to_idx[self.unk_token]
        self.pad_idx = self.char_to_idx[self.pad_token]
        
        self.is_fitted = True
        
        print(f"📁 Vocabulary loaded from {filepath}")
        print(f"   Vocab size: {len(self.char_to_idx)}")


if __name__ == "__main__":
    # Test the CharTokenizer
    print("🧪 Testing CharTokenizer")
    print("=" * 50)
    
    # Sample text for testing
    sample_text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them."""
    
    print(f"📝 Sample text ({len(sample_text)} chars):")
    print(f"'{sample_text[:100]}...'")
    print()
    
    # Test basic tokenization
    print("🔤 Basic Tokenization Test:")
    tokenizer = CharTokenizer(vocab_size=50)
    tokenizer.fit(sample_text)
    print()
    
    # Test encoding/decoding
    test_phrase = "To be or not to be"
    print(f"🔬 Encoding/Decoding Test:")
    print(f"Original: '{test_phrase}'")
    
    encoded = tokenizer.encode(test_phrase)
    print(f"Encoded:  {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded:  '{decoded}'")
    
    print(f"Round-trip successful: {test_phrase == decoded}")
    print()
    
    # Test batch encoding
    print("📦 Batch Encoding Test:")
    batch_texts = [
        "To be",
        "or not to be",
        "that is the question"
    ]
    
    batch_encoded = tokenizer.encode_batch(batch_texts, max_length=20)
    print(f"Batch shape: {batch_encoded.shape}")
    print(f"Batch sample:\n{batch_encoded}")
    print()
    
    # Test vocabulary analysis
    print("📊 Vocabulary Analysis:")
    vocab = tokenizer.get_vocabulary()
    special_tokens = tokenizer.get_special_tokens()
    
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Special tokens: {special_tokens}")
    print(f"Sample characters: {list(vocab.keys())[:20]}")
    print()
    
    # Test text analysis
    print("🔍 Text Analysis:")
    stats = tokenizer.analyze_text(sample_text)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # Test with limited vocabulary
    print("⚠️ Limited Vocabulary Test:")
    small_tokenizer = CharTokenizer(vocab_size=10)  # Very small vocab
    small_tokenizer.fit("abcdefghijklmnopqrstuvwxyz")
    
    test_text = "Hello, World!"
    encoded_small = small_tokenizer.encode(test_text)
    decoded_small = small_tokenizer.decode(encoded_small)
    
    print(f"Original: '{test_text}'")
    print(f"Decoded:  '{decoded_small}'")
    print(f"Small vocab size: {small_tokenizer.get_vocab_size()}")
    print()
    
    # Performance characteristics
    print("⚡ Performance Characteristics:")
    import time
    
    # Encoding speed test
    long_text = sample_text * 100  # Make it longer
    start_time = time.time()
    encoded_long = tokenizer.encode(long_text)
    encoding_time = time.time() - start_time
    
    # Decoding speed test
    start_time = time.time()
    decoded_long = tokenizer.decode(encoded_long)
    decoding_time = time.time() - start_time
    
    print(f"Text length: {len(long_text):,} chars")
    print(f"Encoding time: {encoding_time:.4f}s ({len(long_text)/encoding_time:.0f} chars/s)")
    print(f"Decoding time: {decoding_time:.4f}s ({len(encoded_long)/decoding_time:.0f} tokens/s)")
    print()
    
    print("✅ CharTokenizer tests completed!")
    print("\n💡 Key insights:")
    print("   • Character-level tokenization is simple and transparent")
    print("   • Vocabulary size affects memory usage and unknown token rate")
    print("   • Batch processing enables efficient neural network training")
    print("   • Special tokens handle edge cases gracefully")
    print("   • Round-trip encoding/decoding preserves text (when vocab is sufficient)")
    print("   • 🎉 Ready for integration with TinyGPT!")