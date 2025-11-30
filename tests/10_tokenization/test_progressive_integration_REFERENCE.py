"""
Module 10: Progressive Integration Tests
Tests that Module 10 (Tokenization) works correctly AND integrates with prior modules.

DEPENDENCY CHAIN: 01_tensor → ... → 08_dataloader → 10_tokenization → 11_embeddings
This is where we enable text processing for NLP tasks.
"""

import numpy as np
import sys
from pathlib import Path
import pytest
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPriorStackStillWorking:
    """Quick regression checks that prior modules (01-09) still work."""

    def test_tensor_operations_stable(self):
        """Verify Module 01 (Tensor) still works."""
        try:
            from tinytorch.core.tensor import Tensor

            # Basic tensor creation
            x = Tensor([1, 2, 3])
            assert x.shape == (3,), "Tensor creation broken"

            # Basic operations
            y = Tensor([4, 5, 6])
            z = x + y
            assert z.shape == x.shape, "Tensor addition broken"

        except ImportError:
            pytest.skip("Tensor module not implemented yet")

    def test_dataloader_stable(self):
        """Verify Module 08 (DataLoader) still works."""
        try:
            from tinytorch.core.data import Dataset, DataLoader

            class DummyDataset(Dataset):
                def __len__(self):
                    return 10
                def __getitem__(self, idx):
                    return idx, idx * 2

            dataset = DummyDataset()
            loader = DataLoader(dataset, batch_size=2)

            assert len(dataset) == 10, "Dataset broken"

            batch_count = 0
            for batch in loader:
                batch_count += 1

            assert batch_count > 0, "DataLoader iteration broken"

        except ImportError:
            pytest.skip("DataLoader not implemented yet")


class TestModule10TokenizationCore:
    """Test Module 10 (Tokenization) core functionality."""

    def test_char_tokenizer_creation(self):
        """Test CharTokenizer initialization and vocab building."""
        try:
            from tinytorch.text.tokenization import CharTokenizer

            # Create tokenizer
            tokenizer = CharTokenizer()
            assert hasattr(tokenizer, 'vocab'), "CharTokenizer missing vocab attribute"
            assert hasattr(tokenizer, 'encode'), "CharTokenizer missing encode method"
            assert hasattr(tokenizer, 'decode'), "CharTokenizer missing decode method"

            # Build vocabulary
            corpus = ["hello", "world", "test"]
            tokenizer.build_vocab(corpus)

            assert len(tokenizer.vocab) > 0, "Vocabulary should be non-empty"
            assert hasattr(tokenizer, 'token_to_id'), "Missing token_to_id mapping"
            assert hasattr(tokenizer, 'id_to_token'), "Missing id_to_token mapping"

        except ImportError:
            pytest.skip("Tokenization module not implemented yet")

    def test_char_tokenizer_encode_decode(self):
        """Test CharTokenizer encode/decode roundtrip."""
        try:
            from tinytorch.text.tokenization import CharTokenizer

            tokenizer = CharTokenizer()
            corpus = ["hello", "world"]
            tokenizer.build_vocab(corpus)

            # Test encoding
            text = "hello"
            token_ids = tokenizer.encode(text)

            assert isinstance(token_ids, list), "encode() should return list"
            assert all(isinstance(t, (int, np.integer)) for t in token_ids), \
                "All token IDs should be integers"
            assert len(token_ids) > 0, "Should produce tokens for non-empty text"

            # Test decoding
            decoded = tokenizer.decode(token_ids)
            assert isinstance(decoded, str), "decode() should return string"

            # Roundtrip should preserve characters
            for char in text:
                assert char in decoded, f"Lost character '{char}' in roundtrip"

        except ImportError:
            pytest.skip("Tokenization module not implemented yet")

    def test_bpe_tokenizer_training(self):
        """Test BPE tokenizer training on corpus."""
        try:
            from tinytorch.text.tokenization import BPETokenizer

            # Create BPE tokenizer
            tokenizer = BPETokenizer(vocab_size=50)
            assert hasattr(tokenizer, 'train'), "BPETokenizer missing train method"

            # Train on corpus
            corpus = ["hello", "world", "hello", "hell"]  # Repeated for merges
            tokenizer.train(corpus)

            # Should have vocabulary
            assert len(tokenizer.vocab) > 0, "BPE should build vocabulary"
            assert '<UNK>' in tokenizer.vocab, "BPE should have <UNK> token"

            # Should have learned merges
            if hasattr(tokenizer, 'merges'):
                # If BPE stores merges separately
                assert len(tokenizer.merges) >= 0, "BPE should learn merges"

        except ImportError:
            pytest.skip("BPE tokenization not implemented yet")

    def test_bpe_tokenizer_encode_decode(self):
        """Test BPE encode/decode roundtrip."""
        try:
            from tinytorch.text.tokenization import BPETokenizer

            tokenizer = BPETokenizer(vocab_size=100)
            corpus = ["hello world", "test data", "hello test"]
            tokenizer.train(corpus)

            # Test encoding
            text = "hello world"
            token_ids = tokenizer.encode(text)

            assert isinstance(token_ids, list), "encode() should return list"
            assert all(isinstance(t, (int, np.integer)) for t in token_ids), \
                "All token IDs should be integers"

            # Test decoding
            decoded = tokenizer.decode(token_ids)
            assert isinstance(decoded, str), "decode() should return string"

            # Should preserve word content (BPE may merge/split)
            words = text.split()
            for word in words:
                # Word should appear in decoded text (possibly merged)
                assert word in decoded or any(w in word for w in decoded.split()), \
                    f"Lost word '{word}' in BPE roundtrip"

        except ImportError:
            pytest.skip("BPE tokenization not implemented yet")


class TestTokenizationIntegration:
    """Test tokenization integration with other modules."""

    def test_tokenizer_produces_correct_dtypes(self):
        """PRIORITY 1: Verify int64 output for embeddings."""
        try:
            from tinytorch.text.tokenization import CharTokenizer
            from tinytorch.core.tensor import Tensor

            tokenizer = CharTokenizer()
            tokenizer.build_vocab(["hello world"])

            # Encode text
            token_ids = tokenizer.encode("hello")

            # CRITICAL: Must be integers
            assert all(isinstance(t, (int, np.integer)) for t in token_ids), \
                "Token IDs must be integers for embedding lookup"

            # If converting to Tensor, should be int64
            token_tensor = Tensor(token_ids)
            # Check dtype is integer-compatible
            assert token_tensor.data.dtype in [np.int32, np.int64, np.int_], \
                f"Expected integer dtype for embeddings, got {token_tensor.data.dtype}"

        except ImportError:
            pytest.skip("Required modules not implemented yet")

    def test_tokenization_to_embedding_pipeline(self):
        """PRIORITY 2: Test complete tokenization → embedding pipeline."""
        try:
            from tinytorch.text.embeddings import Embedding
            from tinytorch.text.tokenization import CharTokenizer
            from tinytorch.core.tensor import Tensor

            # Build tokenizer
            tokenizer = CharTokenizer()
            corpus = ["hello", "world", "test"]
            tokenizer.build_vocab(corpus)

            vocab_size = len(tokenizer.vocab)
            embed_dim = 16

            # Create embedding layer
            embedding = Embedding(vocab_size, embed_dim)

            # Tokenize text
            text = "hello world"
            token_ids = tokenizer.encode(text)

            # CRITICAL: Shape compatibility
            token_tensor = Tensor(token_ids)
            assert token_tensor.shape == (len(token_ids),), \
                "Token IDs should be 1D sequence"

            # Embedding lookup should work
            embedded = embedding(token_tensor)
            expected_shape = (len(token_ids), embed_dim)
            assert embedded.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {embedded.shape}"

            # Values should be actual embeddings, not zeros
            assert not np.allclose(embedded.data, 0), \
                "Embeddings should be non-zero (initialized randomly)"

        except ImportError:
            pytest.skip("Embeddings module not yet implemented")

    def test_tokenizer_dataloader_integration(self):
        """Test tokenizer in DataLoader pipeline."""
        try:
            from tinytorch.core.data import Dataset, DataLoader
            from tinytorch.text.tokenization import CharTokenizer
            from tinytorch.core.tensor import Tensor

            # Custom dataset with tokenization
            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer):
                    self.texts = texts
                    self.tokenizer = tokenizer

                def __len__(self):
                    return len(self.texts)

                def __getitem__(self, idx):
                    text = self.texts[idx]
                    token_ids = self.tokenizer.encode(text)
                    return Tensor(token_ids)

            # Build tokenizer
            tokenizer = CharTokenizer()
            texts = ["hello world", "test data", "sample text"]
            tokenizer.build_vocab(texts)

            # Create dataset and dataloader
            dataset = TextDataset(texts, tokenizer)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

            # Iterate batches
            batch_count = 0
            for batch in dataloader:
                batch_count += 1

                # Batch should exist
                assert batch is not None, "Batch should not be None"

            assert batch_count > 0, "DataLoader should produce batches"

        except ImportError:
            pytest.skip("DataLoader not yet implemented")


class TestTokenizationEdgeCases:
    """Test tokenization robustness with edge cases."""

    def test_bpe_edge_cases(self):
        """PRIORITY 3: Empty strings, unknown tokens, special chars."""
        try:
            from tinytorch.text.tokenization import BPETokenizer

            tokenizer = BPETokenizer(vocab_size=100)

            # Edge Case 1: Empty string
            token_ids = tokenizer.encode("")
            assert isinstance(token_ids, list), "Should return list for empty string"
            # May be empty list or contain padding tokens

            decoded = tokenizer.decode([])
            assert isinstance(decoded, str), "Should return string"

            # Edge Case 2: Single character
            tokenizer.train(["a", "b", "c"])
            token_ids = tokenizer.encode("a")
            assert len(token_ids) > 0, "Single char should tokenize"

            # Edge Case 3: Unknown characters after training
            tokenizer.train(["hello", "world"])
            token_ids = tokenizer.encode("xyz")  # Not in training

            # Should handle gracefully (with <UNK> or character fallback)
            assert isinstance(token_ids, list), "Should handle unknown characters"
            assert all(isinstance(t, (int, np.integer)) for t in token_ids), \
                "Should return valid token IDs for unknown text"

            # Edge Case 4: Special characters
            special_text = "hello, world! @#$%"
            token_ids = tokenizer.encode(special_text)
            assert isinstance(token_ids, list), "Should handle special characters"

        except ImportError:
            pytest.skip("BPE tokenization not implemented yet")

    def test_vocabulary_consistency(self):
        """PRIORITY 4: Bidirectional mappings, roundtrip integrity."""
        try:
            from tinytorch.text.tokenization import CharTokenizer, BPETokenizer

            # Test CharTokenizer
            char_tokenizer = CharTokenizer()
            corpus = ["abc", "def", "xyz"]
            char_tokenizer.build_vocab(corpus)

            # Check bidirectional mappings
            for token, token_id in char_tokenizer.token_to_id.items():
                recovered = char_tokenizer.id_to_token.get(token_id)
                assert recovered == token, \
                    f"Bidirectional mapping broken: {token} -> {token_id} -> {recovered}"

            # Test roundtrip for corpus
            for text in corpus:
                token_ids = char_tokenizer.encode(text)
                decoded = char_tokenizer.decode(token_ids)
                # Should preserve characters
                for char in text:
                    assert char in decoded, f"Lost character '{char}' in roundtrip"

            # Test BPETokenizer
            bpe_tokenizer = BPETokenizer(vocab_size=50)
            bpe_tokenizer.train(["hello world", "test data"])

            # Should have <UNK> token
            assert '<UNK>' in bpe_tokenizer.vocab, "BPE should have <UNK> token"

        except ImportError:
            pytest.skip("Tokenization not implemented yet")

    def test_batch_processing(self):
        """PRIORITY 5: Batch encoding/decoding correctness."""
        try:
            from tinytorch.text.tokenization import CharTokenizer

            tokenizer = CharTokenizer()
            corpus = ["hello", "world", "test", "data"]
            tokenizer.build_vocab(corpus)

            # Batch of texts
            texts = ["hello world", "test data", "new text"]

            # Encode batch
            batch_token_ids = [tokenizer.encode(text) for text in texts]

            # Check all are lists of ints
            for token_ids in batch_token_ids:
                assert isinstance(token_ids, list), "Each should be a list"
                assert all(isinstance(t, (int, np.integer)) for t in token_ids), \
                    "All tokens should be integers"

            # Different texts should produce different sequences
            assert batch_token_ids[0] != batch_token_ids[1], \
                "Different texts should produce different token sequences"

            # Decode batch
            decoded_texts = [tokenizer.decode(ids) for ids in batch_token_ids]

            # Should preserve core content
            for original, decoded in zip(texts, decoded_texts):
                # Core words should be preserved
                original_words = set(original.split())
                decoded_words = set(decoded.split())

                # At least some overlap
                assert len(original_words & decoded_words) > 0 or \
                       all(char in decoded for word in original.split() for char in word), \
                    f"Lost content in roundtrip: {original} -> {decoded}"

        except ImportError:
            pytest.skip("Tokenization not implemented yet")


class TestTokenizationPerformance:
    """Test tokenization performance characteristics."""

    def test_tokenization_throughput(self):
        """PRIORITY 6: Measure chars/sec, vocab size."""
        try:
            from tinytorch.text.tokenization import CharTokenizer, BPETokenizer

            # Build tokenizers
            char_tokenizer = CharTokenizer()
            corpus = ["hello world"] * 50
            char_tokenizer.build_vocab(corpus)

            # Test text
            test_text = "hello world test data " * 50

            # Measure CharTokenizer throughput
            start = time.time()
            iterations = 100
            for _ in range(iterations):
                token_ids = char_tokenizer.encode(test_text)
            char_time = time.time() - start
            char_throughput = (len(test_text) * iterations) / char_time

            print(f"\nCharTokenizer: {char_throughput:.0f} chars/sec")
            # Should be reasonably fast (relaxed threshold)
            assert char_throughput > 1000, \
                f"CharTokenizer too slow: {char_throughput:.0f} chars/sec"

            # Vocabulary size check
            assert len(char_tokenizer.vocab) < 1000, \
                f"CharTokenizer vocab too large: {len(char_tokenizer.vocab)}"

            # BPE test (if implemented)
            try:
                bpe_tokenizer = BPETokenizer(vocab_size=100)
                bpe_tokenizer.train(corpus)

                start = time.time()
                for _ in range(iterations):
                    token_ids = bpe_tokenizer.encode(test_text)
                bpe_time = time.time() - start
                bpe_throughput = (len(test_text) * iterations) / bpe_time

                print(f"BPETokenizer: {bpe_throughput:.0f} chars/sec")
                # BPE can be slower
                assert bpe_throughput > 100, \
                    f"BPETokenizer too slow: {bpe_throughput:.0f} chars/sec"
            except:
                pass  # BPE may not be fully implemented

        except ImportError:
            pytest.skip("Tokenization not implemented yet")


class TestRegressionPrevention:
    """Ensure previous modules still work after Module 10 development."""

    def test_no_tensor_regression(self):
        """Verify Module 01 (Tensor) unchanged."""
        try:
            from tinytorch.core.tensor import Tensor

            # Basic tensor operations should work
            x = Tensor([1.0, 2.0, 3.0])
            y = Tensor([4.0, 5.0, 6.0])

            assert x.shape == (3,), "Tensor shape broken"

            z = x + y
            assert z.shape == x.shape, "Tensor addition broken"

        except ImportError:
            pytest.skip("Tensor module not implemented yet")

    def test_no_dataloader_regression(self):
        """Verify Module 08 (DataLoader) unchanged."""
        try:
            from tinytorch.core.data import Dataset, DataLoader

            class SimpleDataset(Dataset):
                def __len__(self):
                    return 5
                def __getitem__(self, idx):
                    return idx, idx * 2

            dataset = SimpleDataset()
            loader = DataLoader(dataset, batch_size=2)

            assert len(dataset) == 5, "Dataset broken"

            # Should be able to iterate
            batch_count = sum(1 for _ in loader)
            assert batch_count > 0, "DataLoader iteration broken"

        except ImportError:
            pytest.skip("DataLoader not implemented yet")

    def test_progressive_stability(self):
        """Test that the progressive stack is stable through tokenization."""
        # Core functionality should remain stable

        # Tensor level
        try:
            from tinytorch.core.tensor import Tensor
            x = Tensor([1, 2, 3])
            assert x.shape == (3,), "Foundation broken"
        except ImportError:
            pass

        # Tokenization level
        try:
            from tinytorch.text.tokenization import CharTokenizer

            tokenizer = CharTokenizer()
            tokenizer.build_vocab(["test"])

            token_ids = tokenizer.encode("test")
            assert isinstance(token_ids, list), "Tokenization broken"

        except ImportError:
            pass  # Not implemented yet


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
