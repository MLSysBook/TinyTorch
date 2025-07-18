"""
Integration Tests - Attention Pipeline

Tests attention mechanism in complete ML pipelines.
Uses actual TinyTorch components to verify transformer-like architectures work correctly.
"""

import pytest
import numpy as np
from test_utils import setup_integration_test

# Ensure proper setup before importing
setup_integration_test()

# Import ONLY from TinyTorch package
from tinytorch.core.tensor import Tensor
from tinytorch.core.attention import scaled_dot_product_attention, SelfAttention, create_causal_mask
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU, Softmax
from tinytorch.core.dense import Sequential


class TestAttentionPipelineIntegration:
    """Test attention in complete ML pipelines with other modules."""
    
    def test_attention_dense_pipeline(self):
        """Test attention followed by dense layers (transformer-like)."""
        seq_len, d_model = 8, 16
        vocab_size = 10
        
        # Create input sequence (like word embeddings)
        embeddings = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Step 1: Self-attention (transformer block)
        self_attn = SelfAttention(d_model)
        attended_output, attention_weights = self_attn(embeddings.data)
        
        # Step 2: Dense feedforward network
        dense_network = Sequential([
            Dense(input_size=d_model, output_size=d_model * 2),
            ReLU(),
            Dense(input_size=d_model * 2, output_size=d_model)
        ])
        
        # Apply dense network to each position
        ff_outputs = []
        for i in range(seq_len):
            pos_input = Tensor(attended_output[i:i+1])  # Single position
            pos_output = dense_network(pos_input)
            ff_outputs.append(pos_output.data)
        
        final_output = np.concatenate(ff_outputs, axis=0)
        
        # Step 3: Final classification head
        classifier = Dense(input_size=d_model, output_size=vocab_size)
        
        # Classify each position
        predictions = []
        for i in range(seq_len):
            pos_input = Tensor(final_output[i:i+1])
            pred = classifier(pos_input)
            predictions.append(pred.data)
        
        final_predictions = np.concatenate(predictions, axis=0)
        
        # Verify complete pipeline
        assert final_predictions.shape == (seq_len, vocab_size), f"Expected shape {(seq_len, vocab_size)}, got {final_predictions.shape}"
        assert not np.any(np.isnan(final_predictions)), "Pipeline should not produce NaN"
        assert attention_weights.shape == (seq_len, seq_len), "Attention weights should be preserved"
        
        # Verify attention weights are sensible
        assert np.allclose(np.sum(attention_weights, axis=-1), 1.0), "Attention weights should sum to 1"
    
    def test_multi_layer_attention_pipeline(self):
        """Test multiple attention layers in sequence (like transformer encoder)."""
        seq_len, d_model = 6, 12
        num_layers = 3
        
        # Initial embeddings
        x = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        current_input = x.data
        
        # Multi-layer attention (transformer encoder stack)
        attention_layers = []
        for layer in range(num_layers):
            # Create attention layer
            attn = SelfAttention(d_model)
            attention_layers.append(attn)
            
            # Apply attention
            attn_output, attn_weights = attn(current_input)
            
            # Simple residual connection (add input to output)
            if current_input.shape == attn_output.shape:
                current_input = current_input + attn_output
            else:
                current_input = attn_output
            
            # Verify intermediate outputs
            assert attn_output.shape == (seq_len, d_model), f"Layer {layer} output shape wrong"
            assert attn_weights.shape == (seq_len, seq_len), f"Layer {layer} attention weights wrong"
            assert np.allclose(np.sum(attn_weights, axis=-1), 1.0), f"Layer {layer} weights should sum to 1"
        
        # Final output verification
        assert current_input.shape == (seq_len, d_model), "Multi-layer output should preserve shape"
        assert not np.any(np.isnan(current_input)), "Multi-layer pipeline should not produce NaN"
    
    def test_attention_with_causal_masking_pipeline(self):
        """Test attention with causal masking in language modeling pipeline."""
        seq_len, d_model = 10, 16
        vocab_size = 20
        
        # Create sequence (like tokens in a sentence)
        token_embeddings = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Create causal mask for autoregressive generation
        causal_mask = create_causal_mask(seq_len)
        
        # Apply causal self-attention
        attn_output, attn_weights = scaled_dot_product_attention(
            token_embeddings.data, token_embeddings.data, token_embeddings.data, causal_mask
        )
        
        # Language modeling head (predict next token)
        lm_head = Dense(input_size=d_model, output_size=vocab_size)
        
        # Generate predictions for each position
        predictions = []
        for pos in range(seq_len):
            pos_input = Tensor(attn_output[pos:pos+1])
            pred = lm_head(pos_input)
            predictions.append(pred.data)
        
        all_predictions = np.concatenate(predictions, axis=0)
        
        # Verify causal masking worked
        assert np.all(np.triu(attn_weights, k=1) < 1e-6), "Causal mask should prevent future attention"
        assert all_predictions.shape == (seq_len, vocab_size), "Should predict vocabulary for each position"
        
        # Verify predictions are reasonable for language modeling
        for pos in range(seq_len):
            pos_pred = all_predictions[pos]
            assert not np.any(np.isnan(pos_pred)), f"Position {pos} predictions should not be NaN"
            assert len(pos_pred) == vocab_size, f"Position {pos} should predict full vocabulary"
    
    def test_attention_encoder_decoder_pipeline(self):
        """Test attention in encoder-decoder architecture."""
        src_len, tgt_len, d_model = 6, 8, 12
        
        # Source sequence (encoder input)
        src_embeddings = Tensor(np.random.randn(src_len, d_model) * 0.1)
        
        # Target sequence (decoder input)
        tgt_embeddings = Tensor(np.random.randn(tgt_len, d_model) * 0.1)
        
        # Encoder: self-attention on source
        encoder = SelfAttention(d_model)
        encoder_output, encoder_weights = encoder(src_embeddings.data)
        
        # Decoder: self-attention on target with causal mask
        decoder = SelfAttention(d_model)
        causal_mask = create_causal_mask(tgt_len)
        decoder_self_attn, decoder_weights = scaled_dot_product_attention(
            tgt_embeddings.data, tgt_embeddings.data, tgt_embeddings.data, causal_mask
        )
        
        # Cross-attention: target queries attend to source keys/values
        cross_attn_output, cross_attn_weights = scaled_dot_product_attention(
            decoder_self_attn,  # Queries from decoder
            encoder_output,      # Keys from encoder
            encoder_output       # Values from encoder
        )
        
        # Final output layer
        output_layer = Dense(input_size=d_model, output_size=d_model)
        final_outputs = []
        for pos in range(tgt_len):
            pos_input = Tensor(cross_attn_output[pos:pos+1])
            pos_output = output_layer(pos_input)
            final_outputs.append(pos_output.data)
        
        final_sequence = np.concatenate(final_outputs, axis=0)
        
        # Verify encoder-decoder pipeline
        assert encoder_output.shape == (src_len, d_model), "Encoder output should preserve source shape"
        assert decoder_self_attn.shape == (tgt_len, d_model), "Decoder self-attention should preserve target shape"
        assert cross_attn_output.shape == (tgt_len, d_model), "Cross-attention should output target length"
        assert final_sequence.shape == (tgt_len, d_model), "Final output should match target length"
        
        # Verify attention patterns
        assert encoder_weights.shape == (src_len, src_len), "Encoder attention should be source x source"
        assert decoder_weights.shape == (tgt_len, tgt_len), "Decoder attention should be target x target"
        assert cross_attn_weights.shape == (tgt_len, src_len), "Cross-attention should be target x source"
        
        # Verify causal masking in decoder
        assert np.all(np.triu(decoder_weights, k=1) < 1e-6), "Decoder should have causal masking"
    
    def test_attention_with_multiple_architectures(self):
        """Test attention integrated with different architecture types."""
        seq_len, d_model = 8, 16
        
        # Input sequence
        x = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Path 1: Attention → Dense layers
        attn1 = SelfAttention(d_model)
        attn_out1, _ = attn1(x.data)
        
        dense_path = Sequential([
            Dense(input_size=d_model, output_size=d_model // 2),
            ReLU(),
            Dense(input_size=d_model // 2, output_size=d_model)
        ])
        
        # Apply dense to sequence (position by position)
        dense_outputs = []
        for i in range(seq_len):
            pos_input = Tensor(attn_out1[i:i+1])
            pos_output = dense_path(pos_input)
            dense_outputs.append(pos_output.data)
        
        dense_result = np.concatenate(dense_outputs, axis=0)
        
        # Path 2: Multiple attention layers
        attn2 = SelfAttention(d_model)
        attn3 = SelfAttention(d_model)
        
        attn_out2, _ = attn2(x.data)
        attn_out3, _ = attn3(attn_out2)
        
        # Verify both paths work
        assert dense_result.shape == (seq_len, d_model), "Dense path should preserve sequence shape"
        assert attn_out3.shape == (seq_len, d_model), "Multi-attention path should preserve shape"
        assert not np.any(np.isnan(dense_result)), "Dense path should not produce NaN"
        assert not np.any(np.isnan(attn_out3)), "Multi-attention path should not produce NaN"
        
        # Verify they can be combined
        combined = dense_result + attn_out3
        assert combined.shape == (seq_len, d_model), "Combined paths should work"
        assert not np.any(np.isnan(combined)), "Combined result should not be NaN"
    
    def test_attention_scalability_pipeline(self):
        """Test attention pipeline with different sequence lengths and dimensions."""
        test_configs = [
            (4, 8),    # Small sequence, small dimension
            (16, 32),  # Medium sequence, medium dimension
            (32, 16),  # Long sequence, smaller dimension
            (8, 64),   # Short sequence, large dimension
        ]
        
        for seq_len, d_model in test_configs:
            # Create test data
            x = Tensor(np.random.randn(seq_len, d_model) * 0.1)
            
            # Attention pipeline
            attn = SelfAttention(d_model)
            attn_output, attn_weights = attn(x.data)
            
            # Dense post-processing
            post_process = Dense(input_size=d_model, output_size=min(d_model, 10))
            
            # Process each position
            processed_outputs = []
            for i in range(seq_len):
                pos_input = Tensor(attn_output[i:i+1])
                pos_output = post_process(pos_input)
                processed_outputs.append(pos_output.data)
            
            final_output = np.concatenate(processed_outputs, axis=0)
            
            # Verify scalability
            assert attn_output.shape == (seq_len, d_model), f"Config {test_configs} attention output wrong"
            assert attn_weights.shape == (seq_len, seq_len), f"Config {test_configs} attention weights wrong"
            assert final_output.shape == (seq_len, min(d_model, 10)), f"Config {test_configs} final output wrong"
            assert not np.any(np.isnan(final_output)), f"Config {test_configs} should not produce NaN"


class TestAttentionRealWorldPipelines:
    """Test attention in realistic ML scenarios."""
    
    def test_sequence_classification_pipeline(self):
        """Test attention for sequence classification (like sentiment analysis)."""
        seq_len, d_model = 12, 24
        num_classes = 3
        
        # Input sequence (like sentence embeddings)
        sentence = Tensor(np.random.randn(seq_len, d_model) * 0.1)
        
        # Self-attention to capture dependencies
        attn = SelfAttention(d_model)
        attended_sequence, attn_weights = attn(sentence.data)
        
        # Global pooling (mean over sequence dimension)
        pooled_representation = np.mean(attended_sequence, axis=0, keepdims=True)
        
        # Classification head
        classifier = Sequential([
            Dense(input_size=d_model, output_size=d_model // 2),
            ReLU(),
            Dense(input_size=d_model // 2, output_size=num_classes)
        ])
        
        # Get classification scores
        pooled_tensor = Tensor(pooled_representation)
        class_scores = classifier(pooled_tensor)
        
        # Verify classification pipeline
        assert attended_sequence.shape == (seq_len, d_model), "Attention should preserve sequence shape"
        assert pooled_representation.shape == (1, d_model), "Pooling should create single representation"
        assert class_scores.shape == (1, num_classes), "Should output class scores"
        assert not np.any(np.isnan(class_scores.data)), "Classification should not produce NaN"
    
    def test_sequence_to_sequence_pipeline(self):
        """Test attention for sequence-to-sequence tasks (like translation)."""
        src_len, tgt_len, d_model = 10, 8, 20
        vocab_size = 30
        
        # Source sequence (input language)
        src_seq = Tensor(np.random.randn(src_len, d_model) * 0.1)
        
        # Target sequence (output language, teacher forcing)
        tgt_seq = Tensor(np.random.randn(tgt_len, d_model) * 0.1)
        
        # Encoder: process source sequence
        encoder = SelfAttention(d_model)
        encoded_src, src_attn = encoder(src_seq.data)
        
        # Decoder: process target with causal masking
        decoder = SelfAttention(d_model)
        causal_mask = create_causal_mask(tgt_len)
        decoded_tgt, tgt_attn = scaled_dot_product_attention(
            tgt_seq.data, tgt_seq.data, tgt_seq.data, causal_mask
        )
        
        # Cross-attention: target attends to source
        cross_attended, cross_attn = scaled_dot_product_attention(
            decoded_tgt,  # Target queries
            encoded_src,  # Source keys
            encoded_src   # Source values
        )
        
        # Output projection to vocabulary
        vocab_proj = Dense(input_size=d_model, output_size=vocab_size)
        
        # Generate output tokens
        output_tokens = []
        for pos in range(tgt_len):
            pos_input = Tensor(cross_attended[pos:pos+1])
            token_logits = vocab_proj(pos_input)
            output_tokens.append(token_logits.data)
        
        output_sequence = np.concatenate(output_tokens, axis=0)
        
        # Verify seq2seq pipeline
        assert encoded_src.shape == (src_len, d_model), "Encoder should preserve source shape"
        assert cross_attended.shape == (tgt_len, d_model), "Cross-attention should output target length"
        assert output_sequence.shape == (tgt_len, vocab_size), "Should generate vocab logits for each target position"
        
        # Verify attention patterns make sense
        assert src_attn.shape == (src_len, src_len), "Source attention should be src x src"
        assert tgt_attn.shape == (tgt_len, tgt_len), "Target attention should be tgt x tgt" 
        assert cross_attn.shape == (tgt_len, src_len), "Cross-attention should be tgt x src"
        
        # Verify causal masking
        assert np.all(np.triu(tgt_attn, k=1) < 1e-6), "Target should have causal masking"


if __name__ == "__main__":
    pytest.main([__file__]) 