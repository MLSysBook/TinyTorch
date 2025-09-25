#!/usr/bin/env python3
"""
Quantization and Compression Demo

Demonstrates how to reduce model size using TinyTorch modules:
- Module 17: Quantization (INT8 precision reduction)
- Module 18: Compression (magnitude-based pruning)

Shows the memory vs accuracy tradeoffs in model optimization.
"""

import numpy as np
from tinytorch.core.quantization import INT8Quantizer
from tinytorch.core.compression import calculate_sparsity, CompressionMetrics


class DemoModel:
    """Simple model for compression demonstration."""
    
    def __init__(self, layer_sizes=[784, 256, 128, 10]):
        """Initialize model with specified layer sizes."""
        self.layer_sizes = layer_sizes
        self.weights = {}
        self.biases = {}
        
        # Create random weights
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            self.weights[f'W{i+1}'] = np.random.randn(in_size, out_size).astype(np.float32) * 0.01
            self.biases[f'b{i+1}'] = np.random.randn(out_size).astype(np.float32) * 0.01
    
    def get_model_stats(self):
        """Get model statistics."""
        total_params = sum(w.size for w in self.weights.values()) + sum(b.size for b in self.biases.values())
        total_size_mb = total_params * 4 / (1024 * 1024)  # 32-bit floats
        
        return {
            'total_parameters': total_params,
            'size_mb': total_size_mb,
            'layers': len(self.weights)
        }
    
    def forward(self, x):
        """Forward pass through the model."""
        h = x
        for i in range(len(self.weights)):
            W = self.weights[f'W{i+1}']
            b = self.biases[f'b{i+1}']
            
            # Linear transformation
            h = np.dot(h, W) + b
            
            # ReLU activation (except last layer)
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)
        
        return h


def demonstrate_quantization():
    """Demonstrate INT8 quantization effects."""
    print("🔢 QUANTIZATION DEMONSTRATION")
    print("=" * 50)
    print("Using Module 17: Quantization for precision reduction")
    
    # Create model
    model = DemoModel()
    baseline_stats = model.get_model_stats()
    
    print(f"\\n📊 Baseline Model (FP32):")
    print(f"   Parameters: {baseline_stats['total_parameters']:,}")
    print(f"   Model Size: {baseline_stats['size_mb']:.2f} MB")
    print(f"   Precision: 32-bit floating point")
    
    # Simulate quantization analysis
    quantizer = INT8Quantizer()
    
    print(f"\\n🔄 Applying INT8 Quantization...")
    
    # Calculate quantized model statistics
    quantized_params = baseline_stats['total_parameters']
    quantized_size_mb = quantized_params * 1 / (1024 * 1024)  # INT8 = 1 byte per param
    compression_ratio = baseline_stats['size_mb'] / quantized_size_mb
    
    print(f"\\n📉 Quantized Model (INT8):")
    print(f"   Parameters: {quantized_params:,} (unchanged)")
    print(f"   Model Size: {quantized_size_mb:.2f} MB")
    print(f"   Precision: 8-bit integer")
    print(f"   🗜️  Compression: {compression_ratio:.2f}x smaller")
    
    # Analyze quantization effects
    print(f"\\n🎯 Quantization Analysis:")
    print(f"   • Memory Reduction: {compression_ratio:.2f}x")
    print(f"   • Typical Accuracy Loss: ~1-3%")
    print(f"   • Inference Speed: ~2x faster on modern hardware")
    print(f"   • Energy Efficiency: Significantly improved")
    
    # Show weight distribution effects
    sample_weight = model.weights['W1'][:50, :50]  # Sample for visualization
    
    # Simulate quantization effects on weight distribution
    weight_range = np.max(sample_weight) - np.min(sample_weight)
    quantization_step = weight_range / 256  # 8-bit = 256 levels
    
    print(f"\\n📈 Weight Quantization Effects:")
    print(f"   Original Range: [{np.min(sample_weight):.6f}, {np.max(sample_weight):.6f}]")
    print(f"   Quantization Step: {quantization_step:.8f}")
    print(f"   Quantization Levels: 256 discrete values")
    
    return {
        'baseline_size_mb': baseline_stats['size_mb'],
        'quantized_size_mb': quantized_size_mb,
        'quantization_compression': compression_ratio
    }


def demonstrate_pruning():
    """Demonstrate magnitude-based pruning."""
    print("\\n\\n✂️  PRUNING DEMONSTRATION")
    print("=" * 50)
    print("Using Module 18: Compression for sparsity-based reduction")
    
    # Create model
    model = DemoModel()
    baseline_stats = model.get_model_stats()
    
    print(f"\\n📊 Baseline Model:")
    print(f"   Total Parameters: {baseline_stats['total_parameters']:,}")
    print(f"   Model Size: {baseline_stats['size_mb']:.2f} MB")
    print(f"   Sparsity: 0% (all weights non-zero)")
    
    # Apply different pruning levels
    sparsity_levels = [0.25, 0.50, 0.75, 0.90]
    
    print(f"\\n🎯 Testing Different Pruning Levels:")
    
    results = {}
    
    for target_sparsity in sparsity_levels:
        print(f"\\n   🔍 Applying {target_sparsity:.0%} sparsity...")
        
        # Apply pruning to each weight matrix
        total_params = 0
        total_pruned = 0
        
        pruned_model = {
            'weights': {},
            'biases': model.biases.copy()  # Don't prune biases
        }
        
        for name, weight in model.weights.items():
            # Calculate magnitude-based threshold
            flat_weights = weight.flatten()
            threshold = np.percentile(np.abs(flat_weights), target_sparsity * 100)
            
            # Create pruned weight matrix
            pruned_weight = weight.copy()
            pruned_weight[np.abs(pruned_weight) < threshold] = 0
            
            # Calculate actual sparsity achieved
            actual_sparsity = calculate_sparsity(pruned_weight)
            
            pruned_model['weights'][name] = pruned_weight
            
            layer_params = weight.size
            layer_pruned = np.sum(pruned_weight == 0)
            
            total_params += layer_params
            total_pruned += layer_pruned
            
            print(f"      {name}: {layer_pruned:,}/{layer_params:,} pruned ({actual_sparsity:.1%})")
        
        # Calculate overall metrics
        overall_sparsity = total_pruned / total_params
        effective_params = total_params - total_pruned
        
        # Calculate compressed size (sparse representation)
        # In practice, sparse matrices need overhead for indices
        sparse_overhead = 1.2  # 20% overhead for storing indices
        compressed_size_mb = (effective_params * 4 * sparse_overhead) / (1024 * 1024)
        compression_ratio = baseline_stats['size_mb'] / compressed_size_mb
        
        results[target_sparsity] = {
            'achieved_sparsity': overall_sparsity,
            'effective_params': effective_params,
            'compressed_size_mb': compressed_size_mb,
            'compression_ratio': compression_ratio
        }
        
        print(f"      Overall Sparsity: {overall_sparsity:.1%}")
        print(f"      Compressed Size: {compressed_size_mb:.2f} MB")
        print(f"      🗜️  Compression: {compression_ratio:.2f}x")
    
    # Analyze pruning effectiveness
    print(f"\\n📈 Pruning Analysis:")
    print(f"   Sparsity Level  | Compression | Est. Accuracy Loss")
    print(f"   --------------- | ----------- | ------------------")
    
    accuracy_loss_estimates = {0.25: 0.5, 0.50: 2.0, 0.75: 5.0, 0.90: 15.0}
    
    for sparsity in sparsity_levels:
        result = results[sparsity]
        acc_loss = accuracy_loss_estimates[sparsity]
        print(f"   {sparsity:.0%}            | {result['compression_ratio']:.2f}x       | ~{acc_loss:.1f}%")
    
    return results


def demonstrate_combined_compression():
    """Demonstrate combined quantization + pruning."""
    print("\\n\\n🚀 COMBINED COMPRESSION DEMONSTRATION")
    print("=" * 60)
    print("Applying both quantization AND pruning for maximum compression")
    
    # Get individual results
    quantization_results = demonstrate_quantization()
    pruning_results = demonstrate_pruning()
    
    # Calculate combined compression
    best_pruning = pruning_results[0.50]  # 50% sparsity as reasonable trade-off
    
    print(f"\\n🎯 Combined Optimization Results:")
    print(f"=" * 40)
    
    baseline_size = quantization_results['baseline_size_mb']
    quantized_size = quantization_results['quantized_size_mb']
    pruned_size = best_pruning['compressed_size_mb']
    
    # Combined: quantized AND pruned
    combined_size = pruned_size / quantization_results['quantization_compression']
    total_compression = baseline_size / combined_size
    
    print(f"📊 Compression Pipeline:")
    print(f"   Original Model:           {baseline_size:.2f} MB")
    print(f"   After Quantization (INT8): {quantized_size:.2f} MB ({quantization_results['quantization_compression']:.1f}x)")
    print(f"   After Pruning (50%):      {pruned_size:.2f} MB ({best_pruning['compression_ratio']:.1f}x)")
    print(f"   After BOTH:               {combined_size:.2f} MB")
    print(f"   🏆 TOTAL COMPRESSION:     {total_compression:.2f}x")
    
    print(f"\\n💡 Key Insights:")
    print(f"   • Quantization: Universal 4x compression with minimal accuracy loss")
    print(f"   • Pruning: Additional compression but with accuracy trade-offs")
    print(f"   • Combined: Multiplicative benefits = {total_compression:.1f}x total compression")
    print(f"   • Best for: Deployment on resource-constrained devices")
    
    print(f"\\n🎯 Production Recommendations:")
    print(f"   • Start with quantization (safe 4x compression)")
    print(f"   • Add pruning gradually while monitoring accuracy")
    print(f"   • 50% sparsity usually provides good compression/accuracy balance")
    print(f"   • Always benchmark on your specific use case!")


def main():
    """Run quantization and compression demonstration."""
    print("🚀 QUANTIZATION & COMPRESSION DEMONSTRATION")
    print("=" * 80)
    print("Learning how to reduce model size using TinyTorch optimization modules")
    print("• Module 17 (Quantization): Precision reduction (FP32 → INT8)")  
    print("• Module 18 (Compression): Sparsity through magnitude-based pruning")
    print("=" * 80)
    
    try:
        # Run comprehensive demonstration
        demonstrate_combined_compression()
        
        print("\\n\\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 50)
        print("You've learned model compression techniques:")
        print("✓ INT8 quantization for 4x memory reduction")
        print("✓ Magnitude-based pruning for sparsity")
        print("✓ Combined techniques for maximum compression")
        print("✓ Understanding accuracy vs compression trade-offs")
        
        print("\\n📚 Next Steps:")
        print("• Apply these techniques to your TinyTorch models")
        print("• Experiment with different sparsity levels")
        print("• Use TinyMLPerf to benchmark compressed models")
        print("• Consider deployment constraints when choosing compression levels")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)