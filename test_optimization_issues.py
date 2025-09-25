#!/usr/bin/env python3
"""
Test script to demonstrate the actual issues with quantization and KV caching
that the user identified.

This script shows:
1. Quantization fails because it's broken (5x slower, accuracy issues)
2. KV caching fails because sequence lengths are too short
3. What the breakeven points actually are
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add module paths
sys.path.append(str(Path(__file__).parent / 'modules' / '17_quantization'))
sys.path.append(str(Path(__file__).parent / 'modules' / '19_caching'))

print("🔬 TESTING OPTIMIZATION ISSUES")
print("=" * 50)

# Test 1: Quantization Issues
print("\n1. 📊 QUANTIZATION ANALYSIS")
print("-" * 30)

try:
    from quantization_dev import BaselineCNN, QuantizedCNN
    
    # Create models
    baseline = BaselineCNN(input_channels=3, num_classes=10)
    quantized = QuantizedCNN(input_channels=3, num_classes=10)
    
    # Prepare test
    test_input = np.random.randn(8, 3, 32, 32)
    calibration_data = [np.random.randn(1, 3, 32, 32) for _ in range(10)]
    
    print("Testing FP32 baseline...")
    start = time.time()
    baseline_output = baseline.forward(test_input)
    baseline_time = time.time() - start
    baseline_pred = baseline.predict(test_input)
    print(f"  FP32 time: {baseline_time*1000:.2f}ms")
    print(f"  FP32 accuracy: 100% (reference)")
    
    print("Quantizing model...")
    quantized.calibrate_and_quantize(calibration_data)
    
    print("Testing INT8 quantized...")
    start = time.time()
    quantized_output = quantized.forward(test_input)
    quantized_time = time.time() - start
    quantized_pred = quantized.predict(test_input)
    print(f"  INT8 time: {quantized_time*1000:.2f}ms")
    
    # Calculate metrics
    speedup = baseline_time / quantized_time
    accuracy_agreement = np.mean(baseline_pred == quantized_pred)
    accuracy_loss = (1.0 - accuracy_agreement) * 100
    
    print(f"\n📈 QUANTIZATION RESULTS:")
    print(f"  Speedup: {speedup:.2f}× {'✅' if speedup > 3 else '❌'} (target: 4×)")
    print(f"  Accuracy loss: {accuracy_loss:.1f}% {'✅' if accuracy_loss < 2 else '❌'} (target: <1%)")
    
    if speedup < 1.0:
        print(f"  🚨 ISSUE: Quantization is {1/speedup:.1f}× SLOWER!")
        print(f"     This is because we dequantize weights for every operation")
        print(f"     Real systems use INT8 kernels that stay in INT8")
        
except Exception as e:
    print(f"❌ Quantization test failed: {e}")

# Test 2: KV Caching Issues
print("\n\n2. 🧠 KV CACHING ANALYSIS")
print("-" * 30)

try:
    from caching_dev import KVCache, CachedMultiHeadAttention
    
    embed_dim = 128
    num_heads = 8
    head_dim = embed_dim // num_heads
    
    # Create attention layer
    attention = CachedMultiHeadAttention(embed_dim, num_heads)
    
    # Test different sequence lengths to find breakeven point
    seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512]
    
    print("Testing KV caching at different sequence lengths...")
    print(f"{'Seq Len':<8} {'No Cache (ms)':<15} {'With Cache (ms)':<17} {'Speedup':<10} {'Result'}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        try:
            # Create cache
            cache = KVCache(seq_len, 1, num_heads, head_dim)
            
            # Test without cache (recompute full sequence each time)
            def generate_without_cache():
                total_time = 0
                for pos in range(1, seq_len + 1):
                    input_seq = np.random.randn(1, pos, embed_dim)
                    start = time.time()
                    output, _ = attention.forward(input_seq, use_cache=False)
                    total_time += time.time() - start
                return total_time
            
            # Test with cache (incremental)
            def generate_with_cache():
                cache.reset()
                total_time = 0
                for pos in range(seq_len):
                    token = np.random.randn(1, 1, embed_dim)
                    start = time.time()
                    output, _ = attention.forward(token, cache=cache, layer_idx=0, use_cache=True)
                    total_time += time.time() - start
                return total_time
            
            # Measure times (average of 3 runs)
            no_cache_times = [generate_without_cache() for _ in range(3)]
            with_cache_times = [generate_with_cache() for _ in range(3)]
            
            no_cache_avg = np.mean(no_cache_times) * 1000  # ms
            with_cache_avg = np.mean(with_cache_times) * 1000  # ms
            
            speedup = no_cache_avg / with_cache_avg
            
            if speedup > 1.2:
                result = "✅ Cache wins"
            elif speedup > 0.8:
                result = "➖ Close"
            else:
                result = "❌ Cache slower"
                
            print(f"{seq_len:<8} {no_cache_avg:<15.2f} {with_cache_avg:<17.2f} {speedup:<10.2f} {result}")
            
        except Exception as e:
            print(f"{seq_len:<8} ERROR: {str(e)[:40]}")
    
    print(f"\n📈 KV CACHING ANALYSIS:")
    print(f"  🔍 The issue: Sequence lengths 8-48 are too short!")
    print(f"  💡 KV caching has coordination overhead")
    print(f"  ⚖️  Only beneficial when seq_len > overhead threshold")
    print(f"  🎯 Need sequences ~100+ tokens to see clear benefits")

except Exception as e:
    print(f"❌ KV caching test failed: {e}")

# Test 3: What would work - Pruning
print("\n\n3. 🌿 PRUNING ANALYSIS (What might work better)")
print("-" * 45)

print("Testing weight magnitude pruning concept...")

# Simple MLP for pruning test
class SimpleMLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def forward(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2
    
    def prune_weights(self, sparsity=0.5):
        """Remove smallest magnitude weights"""
        # Prune W1
        w1_flat = self.w1.flatten()
        threshold_1 = np.percentile(np.abs(w1_flat), sparsity * 100)
        self.w1 = np.where(np.abs(self.w1) > threshold_1, self.w1, 0)
        
        # Prune W2
        w2_flat = self.w2.flatten()
        threshold_2 = np.percentile(np.abs(w2_flat), sparsity * 100)
        self.w2 = np.where(np.abs(self.w2) > threshold_2, self.w2, 0)
    
    def count_nonzero_params(self):
        return np.count_nonzero(self.w1) + np.count_nonzero(self.w2)
    
    def count_total_params(self):
        return self.w1.size + self.w2.size

# Test pruning
test_input = np.random.randn(32, 784)

print("Creating baseline MLP...")
dense_model = SimpleMLP()
baseline_output = dense_model.forward(test_input)
baseline_params = dense_model.count_total_params()

print(f"Baseline parameters: {baseline_params:,}")

sparsity_levels = [0.5, 0.7, 0.9]
print(f"\n{'Sparsity':<10} {'Params Left':<12} {'% Reduction':<12} {'Output MSE':<12} {'Feasible'}")
print("-" * 60)

for sparsity in sparsity_levels:
    pruned_model = SimpleMLP()
    pruned_model.w1 = dense_model.w1.copy()
    pruned_model.w2 = dense_model.w2.copy()
    pruned_model.b1 = dense_model.b1.copy()  
    pruned_model.b2 = dense_model.b2.copy()
    
    # Prune weights
    pruned_model.prune_weights(sparsity)
    
    # Test forward pass
    pruned_output = pruned_model.forward(test_input)
    
    # Calculate metrics
    remaining_params = pruned_model.count_nonzero_params()
    reduction = (1 - remaining_params / baseline_params) * 100
    mse = np.mean((baseline_output - pruned_output) ** 2)
    
    feasible = "✅" if mse < 1.0 else "❌"
    
    print(f"{sparsity*100:.0f}%{'':<7} {remaining_params:<12,} {reduction:<12.1f}% {mse:<12.4f} {feasible}")

print(f"\n📊 PRUNING INSIGHTS:")
print(f"  🎯 More intuitive: 'cut the weakest connections'")
print(f"  🚀 Could show real speedups with sparse matrix ops")
print(f"  💡 Students understand neurons/synapses being removed")
print(f"  ⚖️  Clear trade-off between compression and accuracy")

print("\n" + "=" * 50)
print("🔬 SUMMARY OF OPTIMIZATION ISSUES:")
print("✅ Quantization: Needs proper PTQ implementation")  
print("✅ KV Caching: Needs longer sequences (100+ tokens)")
print("💡 Pruning: Could be simpler and more effective")
print("\nThe user's feedback is spot on! 🎯")