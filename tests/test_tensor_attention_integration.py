"""
Integration Tests - Tensor and Attention

Tests real integration between Tensor and Attention modules.
Uses actual TinyTorch components to verify they work together correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import (
    scaled_dot_product_attention, 
    SelfAttention,
    create_causal_mask,
    create_padding_mask,
    create_bidirectional_mask
)


class TestTensorAttentionIntegration:
    """Test real integration between Tensor and Attention modules."""
    
    def test_scaled_dot_product_attention_with_real_tensors(self):
        """Test scaled dot-product attention with real Tensor objects."""
        # Create Q, K, V as real Tensors
        seq_len, d_model = 4, 8
        np.random.seed(42)
        
        Q = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        K = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        V = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Apply attention - should work with numpy arrays (tensor.data)
        output, weights = scaled_dot_product_attention(Q.data, K.data, V.data)
        
        # Verify output properties
        assert output.shape == (seq_len, d_model), f"Output shape should be {(seq_len, d_model)}, got {output.shape}"
        assert weights.shape == (seq_len, seq_len), f"Weights shape should be {(seq_len, seq_len)}, got {weights.shape}"
        assert np.allclose(np.sum(weights, axis=-1), 1.0), "Attention weights should sum to 1"
        assert np.all(weights >= 0), "All attention weights should be non-negative"
    
    def test_self_attention_with_real_tensors(self):
        """Test SelfAttention wrapper with real Tensor objects."""
        d_model = 16
        seq_len = 6
        
        # Create self-attention
        self_attn = SelfAttention(d_model)
        
        # Create input tensor
        x = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Apply self-attention - should work with numpy arrays
        output, weights = self_attn(x.data)
        
        # Verify integration
        assert output.shape == x.data.shape, f"Output shape should match input {x.data.shape}, got {output.shape}"
        assert weights.shape == (seq_len, seq_len), f"Weights should be square {(seq_len, seq_len)}, got {weights.shape}"
        assert np.allclose(np.sum(weights, axis=-1), 1.0), "Self-attention weights should sum to 1"
    
    def test_attention_with_masking_and_tensors(self):
        """Test attention with masking using real Tensor inputs."""
        seq_len = 5
        d_model = 12
        
        # Create tensors
        Q = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        K = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        V = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Test causal masking
        causal_mask = create_causal_mask(seq_len)
        output_causal, weights_causal = scaled_dot_product_attention(
            Q.data, K.data, V.data, causal_mask
        )
        
        # Verify causal masking worked
        assert np.all(np.triu(weights_causal, k=1) < 1e-6), "Causal mask should zero upper triangle"
        assert output_causal.shape == (seq_len, d_model), "Causal attention should preserve shape"
        
        # Test padding masking
        lengths = [seq_len, seq_len-2]
        padding_mask = create_padding_mask(lengths, seq_len)
        output_padded, weights_padded = scaled_dot_product_attention(
            Q.data, K.data, V.data, padding_mask[0]
        )
        
        # Verify padding masking worked
        assert output_padded.shape == (seq_len, d_model), "Padding attention should preserve shape"
        assert np.allclose(np.sum(weights_padded, axis=-1), 1.0), "Padded weights should still sum to 1"
    
    def test_attention_batched_tensors(self):
        """Test attention with batched Tensor inputs."""
        batch_size, seq_len, d_model = 3, 4, 8
        
        # Create batched tensors
        Q_batch = Tensor(np.random.randn(batch_size, seq_len, d_model) * 0.1)
        K_batch = Tensor(np.random.randn(batch_size, seq_len, d_model) * 0.1)
        V_batch = Tensor(np.random.randn(batch_size, seq_len, d_model) * 0.1)
        
        # Apply attention to batched data
        output, weights = scaled_dot_product_attention(Q_batch.data, K_batch.data, V_batch.data)
        
        # Verify batched processing
        assert output.shape == (batch_size, seq_len, d_model), "Batched output should preserve batch dimension"
        assert weights.shape == (batch_size, seq_len, seq_len), "Batched weights should preserve batch dimension"
        
        # Verify each batch item has proper attention weights
        for b in range(batch_size):
            assert np.allclose(np.sum(weights[b], axis=-1), 1.0), f"Batch {b} weights should sum to 1"
    
    def test_attention_with_different_tensor_dtypes(self):
        """Test attention works with different Tensor data types."""
        seq_len, d_model = 3, 6
        
        # Test with float32
        Q_f32 = Tensor(np.random.randn(seq_len, d_model).astype(np.float32))
        K_f32 = Tensor(np.random.randn(seq_len, d_model).astype(np.float32))
        V_f32 = Tensor(np.random.randn(seq_len, d_model).astype(np.float32))
        
        output_f32, weights_f32 = scaled_dot_product_attention(Q_f32.data, K_f32.data, V_f32.data)
        
        assert output_f32.dtype == np.float32, "Output should preserve float32 dtype"
        assert weights_f32.dtype == np.float32, "Weights should preserve float32 dtype"
        
        # Test with float64
        Q_f64 = Tensor(np.random.randn(seq_len, d_model).astype(np.float64))
        K_f64 = Tensor(np.random.randn(seq_len, d_model).astype(np.float64))
        V_f64 = Tensor(np.random.randn(seq_len, d_model).astype(np.float64))
        
        output_f64, weights_f64 = scaled_dot_product_attention(Q_f64.data, K_f64.data, V_f64.data)
        
        assert output_f64.dtype == np.float64, "Output should preserve float64 dtype"
        assert weights_f64.dtype == np.float64, "Weights should preserve float64 dtype"
    
    def test_attention_numerical_stability(self):
        """Test attention numerical stability with real Tensor inputs."""
        seq_len, d_model = 4, 8
        
        # Create tensors with extreme values
        Q_extreme = Tensor(np.random.randn(seq_len, d_model) * 10)  # Large values
        K_extreme = Tensor(np.random.randn(seq_len, d_model) * 10)
        V_extreme = Tensor(np.random.randn(seq_len, d_model) * 10)
        
        # Apply attention
        output, weights = scaled_dot_product_attention(Q_extreme.data, K_extreme.data, V_extreme.data)
        
        # Verify numerical stability
        assert not np.any(np.isnan(output)), "Attention output should not contain NaN"
        assert not np.any(np.isinf(output)), "Attention output should not contain Inf"
        assert not np.any(np.isnan(weights)), "Attention weights should not contain NaN"
        assert not np.any(np.isinf(weights)), "Attention weights should not contain Inf"
        assert np.allclose(np.sum(weights, axis=-1), 1.0), "Weights should sum to 1 even with extreme inputs"
    
    def test_attention_gradient_flow_compatibility(self):
        """Test attention compatibility with gradient computation requirements."""
        seq_len, d_model = 3, 4
        
        # Create small tensors for gradient testing
        Q = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        K = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        V = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Apply attention
        output, weights = scaled_dot_product_attention(Q.data, K.data, V.data)
        
        # Verify output is suitable for gradient computation
        assert output.dtype in [np.float32, np.float64], "Output should be floating point for gradients"
        assert weights.dtype in [np.float32, np.float64], "Weights should be floating point for gradients"
        
        # Verify no degenerate cases that would break gradients
        assert np.all(np.abs(output) < 1e6), "Output values should be reasonable for gradient computation"
        assert np.all(weights > 1e-10), "No attention weights should be exactly zero (for gradient flow)"


class TestAttentionMaskingIntegration:
    """Test attention masking utilities with Tensor integration."""
    
    def test_causal_mask_integration(self):
        """Test causal mask creation and usage."""
        seq_len = 6
        
        # Create causal mask
        mask = create_causal_mask(seq_len)
        
        # Verify mask properties
        assert mask.shape == (seq_len, seq_len), f"Causal mask should be {(seq_len, seq_len)}"
        assert mask.dtype in [np.float32, np.float64, np.int32, np.int64], "Mask should be numeric"
        assert np.allclose(mask, np.tril(mask)), "Causal mask should be lower triangular"
        
        # Test with actual attention
        Q = Tensor(np.random.randn(seq_len, 4) * 0.1)
        output, weights = scaled_dot_product_attention(Q.data, Q.data, Q.data, mask)
        
        assert np.all(np.triu(weights, k=1) < 1e-6), "Causal mask should zero future positions"
    
    def test_padding_mask_integration(self):
        """Test padding mask creation and usage."""
        lengths = [4, 2, 3]
        max_length = 4
        
        # Create padding mask
        mask = create_padding_mask(lengths, max_length)
        
        # Verify mask properties
        assert mask.shape == (len(lengths), max_length, max_length), "Padding mask should have batch dimension"
        assert mask.dtype in [np.float32, np.float64, np.int32, np.int64], "Mask should be numeric"
        
        # Test with actual attention (first sequence)
        Q = Tensor(np.random.randn(max_length, 4) * 0.1)
        output, weights = scaled_dot_product_attention(Q.data, Q.data, Q.data, mask[0])
        
        # Verify masking worked for first sequence (length=4, should be all ones, no masking effect)
        assert np.all(weights >= 0), "All visible positions should have non-negative attention"
    
    def test_bidirectional_mask_integration(self):
        """Test bidirectional mask creation and usage."""
        seq_len = 5
        
        # Create bidirectional mask
        mask = create_bidirectional_mask(seq_len)
        
        # Verify mask properties
        assert mask.shape == (seq_len, seq_len), f"Bidirectional mask should be {(seq_len, seq_len)}"
        assert np.all(mask == 1), "Bidirectional mask should be all ones"
        
        # Test with actual attention
        Q = Tensor(np.random.randn(seq_len, 4) * 0.1)
        output, weights = scaled_dot_product_attention(Q.data, Q.data, Q.data, mask)
        
        # Verify all positions can attend to all positions
        assert np.all(weights > 0), "Bidirectional attention should allow all position interactions"


if __name__ == "__main__":
    pytest.main([__file__]) 