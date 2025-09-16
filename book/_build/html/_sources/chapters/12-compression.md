---
title: "Compression & Optimization"
description: "Making AI models efficient for real-world deployment"
difficulty: "⭐⭐⭐⭐"
time_estimate: "8-10 hours"
prerequisites: []
next_steps: ['Module 11: Kernels - Hardware-aware optimization', 'Module 12: Benchmarking - Performance measurement', 'Module 13: MLOps - Production deployment']
learning_objectives: []
---

# Module: Compression

```{div} badges
⭐⭐⭐⭐ | ⏱️ 8-10 hours
```


## 📊 Module Info
- **Difficulty**: ⭐⭐⭐⭐ Expert
- **Time Estimate**: 8-10 hours
- **Prerequisites**: Networks, Training modules
- **Next Steps**: Kernels, MLOps modules

Build model compression systems that make neural networks smaller, faster, and more efficient for real-world deployment. This module teaches the optimization techniques that bridge the gap between research-quality models and production-ready AI systems.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Understand deployment constraints**: Analyze model size, memory usage, and computational requirements for real-world systems
- **Implement pruning techniques**: Build magnitude-based and structured pruning to remove unimportant weights and neurons
- **Master quantization methods**: Reduce memory usage by 75% through FP32 → INT8 precision reduction
- **Apply knowledge distillation**: Train compact models using larger teacher models for better performance
- **Design compression strategies**: Combine techniques optimally for different deployment scenarios and constraints

## 🧠 Build → Use → Optimize

This module follows TinyTorch's **Build → Use → Optimize** framework:

1. **Build**: Implement pruning, quantization, knowledge distillation, and structured optimization from engineering principles
2. **Use**: Apply compression techniques to real neural networks with accuracy vs efficiency analysis
3. **Optimize**: Combine compression methods strategically for production deployment scenarios with specific constraints

## 📚 What You'll Build

### Model Compression Analysis System
```python
# Comprehensive model analysis for compression planning
metrics = CompressionMetrics()

# Analyze original model
original_size = metrics.calculate_model_size(model)
param_count = metrics.count_parameters(model)
weight_dist = metrics.analyze_weight_distribution(model)

print(f"Original model: {original_size:.2f} MB, {param_count:,} parameters")
print(f"Weight distribution: mean={weight_dist['mean']:.4f}, std={weight_dist['std']:.4f}")
```

### Pruning Systems for Model Sparsity
```python
# Magnitude-based pruning: remove smallest weights
pruned_model = prune_model_by_magnitude(model, sparsity=0.5)  # Remove 50% of weights
sparsity = calculate_sparsity(pruned_model)
print(f"Achieved sparsity: {sparsity:.2%}")

# Structured pruning: remove entire neurons/channels
optimized_model = prune_layer_neurons(model, layer_idx=0, neurons_to_remove=32)
print(f"Removed 32 neurons from layer 0")

# Sparsity analysis and performance impact
original_acc = evaluate_model(model, test_loader)
pruned_acc = evaluate_model(pruned_model, test_loader)
print(f"Accuracy: {original_acc:.4f} → {pruned_acc:.4f} ({pruned_acc-original_acc:+.4f})")
```

### Quantization for Memory Efficiency
```python
# Quantize model weights from FP32 to INT8
quantized_model = quantize_model_weights(model)
compressed_size = metrics.calculate_model_size(quantized_model)

print(f"Size reduction: {original_size:.2f} MB → {compressed_size:.2f} MB")
print(f"Compression ratio: {original_size/compressed_size:.1f}x smaller")

# Test quantization impact on accuracy
quantized_acc = evaluate_model(quantized_model, test_loader)
print(f"Quantization accuracy impact: {quantized_acc-original_acc:+.4f}")
```

### Knowledge Distillation for Compact Models
```python
# Train small model using large teacher model
teacher_model = load_pretrained_large_model()
student_model = create_compact_model(compression_ratio=0.25)  # 4x smaller

# Distillation training with temperature scaling
distillation_loss = DistillationLoss(temperature=4.0, alpha=0.7)

# Training loop with teacher guidance
for batch_inputs, batch_labels in train_loader:
    teacher_outputs = teacher_model(batch_inputs)
    student_outputs = student_model(batch_inputs)
    
    # Combined loss: distillation + task loss
    loss = distillation_loss(student_outputs, teacher_outputs, batch_labels)
    loss.backward()
    optimizer.step()

print(f"Student model size: {metrics.calculate_model_size(student_model):.2f} MB")
print(f"Student accuracy: {evaluate_model(student_model, test_loader):.4f}")
```

### Comprehensive Compression Pipeline
```python
# End-to-end compression with multiple techniques
def compress_for_mobile_deployment(model, target_size_mb=5.0):
    """Compress model for mobile deployment under 5MB constraint"""
    
    # Step 1: Structured pruning for architecture optimization
    model = prune_redundant_neurons(model, importance_threshold=0.1)
    
    # Step 2: Magnitude-based pruning for sparsity
    model = prune_model_by_magnitude(model, sparsity=0.6)
    
    # Step 3: Quantization for memory reduction
    model = quantize_model_weights(model)
    
    # Step 4: Verify size constraint
    final_size = CompressionMetrics().calculate_model_size(model)
    print(f"Final compressed model: {final_size:.2f} MB")
    
    return model

mobile_model = compress_for_mobile_deployment(trained_model)
```

## 🚀 Getting Started

### Prerequisites
Ensure you have mastered the training foundation:

```bash
# Activate TinyTorch environment
source bin/activate-tinytorch.sh

# Verify prerequisite modules
tito test --module networks
tito test --module training
```

### Development Workflow
1. **Open the development file**: `modules/source/11_compression/compression_dev.py`
2. **Implement compression metrics**: Build model analysis tools for size and parameter counting
3. **Create pruning algorithms**: Implement magnitude-based and structured pruning techniques
4. **Build quantization system**: Add FP32 → INT8 weight quantization with scale/offset mapping
5. **Add knowledge distillation**: Implement teacher-student training for compact models
6. **Export and verify**: `tito export --module compression && tito test --module compression`

## 🧪 Testing Your Implementation

### Comprehensive Test Suite
Run the full test suite to verify compression system functionality:

```bash
# TinyTorch CLI (recommended)
tito test --module compression

# Direct pytest execution
python -m pytest tests/ -k compression -v
```

### Test Coverage Areas
- ✅ **Compression Metrics**: Verify accurate model size and parameter analysis
- ✅ **Pruning Algorithms**: Test magnitude-based and structured pruning correctness
- ✅ **Quantization System**: Ensure proper FP32 ↔ INT8 conversion and accuracy preservation
- ✅ **Knowledge Distillation**: Verify teacher-student training and loss computation
- ✅ **Integrated Compression**: Test combined techniques on real neural networks

### Inline Testing & Compression Analysis
The module includes comprehensive compression validation and performance analysis:
```python
# Example inline test output
🔬 Unit Test: Model compression metrics...
✅ Parameter counting accurate
✅ Model size calculation correct
✅ Weight distribution analysis working
📈 Progress: Compression Analysis ✓

# Pruning validation
🔬 Unit Test: Magnitude-based pruning...
✅ Smallest weights identified correctly
✅ Sparsity calculation accurate
✅ Model functionality preserved
📈 Progress: Pruning Systems ✓

# Quantization testing
🔬 Unit Test: Weight quantization...
✅ FP32 → INT8 conversion correct
✅ Dequantization recovers values
✅ 75% memory reduction achieved
📈 Progress: Quantization ✓
```

### Manual Testing Examples
```python
from compression_dev import CompressionMetrics, prune_model_by_magnitude, quantize_model_weights
from networks_dev import Sequential
from layers_dev import Dense
from activations_dev import ReLU

# Create test model
model = Sequential([
    Dense(784, 128), ReLU(),
    Dense(128, 64), ReLU(),
    Dense(64, 10)
])

# Analyze original model
metrics = CompressionMetrics()
original_size = metrics.calculate_model_size(model)
original_params = metrics.count_parameters(model)
print(f"Original: {original_size:.2f} MB, {original_params:,} parameters")

# Test pruning
pruned_model = prune_model_by_magnitude(model, sparsity=0.5)
pruned_size = metrics.calculate_model_size(pruned_model)
print(f"After 50% pruning: {pruned_size:.2f} MB ({original_size/pruned_size:.1f}x smaller)")

# Test quantization
quantized_model = quantize_model_weights(model)
quantized_size = metrics.calculate_model_size(quantized_model)
print(f"After quantization: {quantized_size:.2f} MB ({original_size/quantized_size:.1f}x smaller)")
```

## 🎯 Key Concepts

### Real-World Applications
- **Mobile AI**: Smartphone apps require models under 10MB for fast download and inference
- **Edge Computing**: IoT devices have severe memory constraints requiring aggressive compression
- **Cloud Cost Optimization**: Reducing model size decreases inference costs at scale
- **Autonomous Systems**: Real-time requirements demand efficient models for safety-critical applications

### Compression Techniques
- **Magnitude-based Pruning**: Remove weights with smallest absolute values to create sparse networks
- **Structured Pruning**: Remove entire neurons/channels for actual hardware speedup benefits
- **Quantization**: Reduce precision from FP32 to INT8 for 75% memory reduction
- **Knowledge Distillation**: Transfer knowledge from large teacher to small student models

### Production Deployment Considerations
- **Hardware Constraints**: Different devices have different memory, compute, and energy limitations
- **Accuracy vs Efficiency Trade-offs**: Balancing model performance with deployment requirements
- **Inference Speed**: Compression techniques that actually improve runtime performance
- **Model Serving**: Considerations for batch processing, latency, and throughput

### Systems Engineering Patterns
- **Compression Pipeline Design**: Sequential application of techniques for maximum benefit
- **Performance Profiling**: Measuring actual improvements in memory, speed, and energy usage
- **Quality Assurance**: Maintaining model accuracy while achieving compression targets
- **Deployment Validation**: Testing compressed models in realistic production scenarios

## 🎉 Ready to Build?

You're about to master the optimization techniques that make AI practical for real-world deployment! From the smartphone in your pocket to autonomous vehicles, they all depend on compressed models that balance intelligence with efficiency.

This module teaches you the systems engineering that separates research prototypes from production AI. You'll learn to think like a deployment engineer, balancing accuracy against constraints and building systems that work in the real world. Take your time, understand the trade-offs, and enjoy building AI that actually ships!

 


Choose your preferred way to engage with this module:

````{grid} 1 2 3 3

```{grid-item-card} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/12_compression/compression_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{grid-item-card} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/12_compression/compression_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{grid-item-card} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/12_compression/compression_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{admonition} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

<div class="prev-next-area">
<a class="left-prev" href="../chapters/11_training.html" title="previous page">← Previous Module</a>
<a class="right-next" href="../chapters/13_kernels.html" title="next page">Next Module →</a>
</div>
