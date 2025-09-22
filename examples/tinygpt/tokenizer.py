"""
Character-Level Tokenizer for TinyGPT
Converts text to sequences of character IDs and back.
"""

from typing import List, Dict, Optional
from collections import Counter


class CharTokenizer:
    """
    Simple character-level tokenizer.
    Each unique character gets a unique ID.
    """
    
    def __init__(self):
        """Initialize empty tokenizer."""
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab: List[str] = []
        self.vocab_size = 0
        
    def fit(self, text: str, min_freq: int = 1):
        """
        Build vocabulary from text.
        
        Args:
            text: Text to build vocabulary from
            min_freq: Minimum frequency for character to be included
        """
        # Count character frequencies
        char_counts = Counter(text)
        
        # Build vocabulary (most common first)
        self.vocab = ['<PAD>', '<UNK>']  # Special tokens
        
        for char, count in char_counts.most_common():
            if count >= min_freq:
                self.vocab.append(char)
        
        # Create mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        print(f"✅ Built vocabulary with {self.vocab_size} characters")
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of character IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of character IDs
        """
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.char_to_id['<UNK>'])  # Unknown character
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of IDs back to text.
        
        Args:
            ids: List of character IDs
            
        Returns:
            Decoded text string
        """
        chars = []
        for id in ids:
            if id in self.id_to_char:
                char = self.id_to_char[id]
                if char not in ['<PAD>', '<UNK>']:  # Skip special tokens
                    chars.append(char)
            # Skip invalid IDs
        return ''.join(chars)
    
    def encode_batch(self, texts: List[str], max_length: int = None, 
                    padding: bool = True) -> List[List[int]]:
        """
        Encode multiple texts with optional padding.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            
        Returns:
            List of encoded sequences
        """
        encoded = []
        for text in texts:
            ids = self.encode(text)
            
            # Truncate if needed
            if max_length and len(ids) > max_length:
                ids = ids[:max_length]
            
            # Pad if needed
            if padding and max_length and len(ids) < max_length:
                pad_id = self.char_to_id['<PAD>']
                ids = ids + [pad_id] * (max_length - len(ids))
            
            encoded.append(ids)
        
        return encoded
    
    def decode_batch(self, id_lists: List[List[int]]) -> List[str]:
        """
        Decode multiple ID sequences.
        
        Args:
            id_lists: List of ID sequences
            
        Returns:
            List of decoded texts
        """
        return [self.decode(ids) for ids in id_lists]


def test_tokenizer():
    """Test the tokenizer with a simple example."""
    print("🧪 Testing CharTokenizer...")
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    
    # Sample text
    text = "Hello, World! This is TinyGPT."
    print(f"Original text: '{text}'")
    
    # Fit tokenizer
    tokenizer.fit(text)
    print(f"Vocabulary: {tokenizer.vocab}")
    
    # Encode
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    # Check round-trip
    assert decoded == text.replace('<', '').replace('>', ''), "Round-trip failed!"
    print("✅ Tokenizer test passed!")


if __name__ == "__main__":
    test_tokenizer()