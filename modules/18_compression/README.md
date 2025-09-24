# Module 18: Compression - Model Size Optimization

## Overview
Reduce model size by 90% while maintaining accuracy through pruning and distillation. Learn how production systems deploy efficient models at scale.

## What You'll Build
- **Magnitude Pruner**: Remove unimportant weights
- **Structured Pruning**: Remove entire channels/layers
- **Knowledge Distillation**: Transfer knowledge to smaller models
- **Sparse Inference**: Efficient computation with pruned models

## Learning Objectives
1. **Sparsity Patterns**: Structured vs unstructured pruning
2. **Pruning Strategies**: Magnitude, gradient, lottery ticket
3. **Distillation**: Teacher-student knowledge transfer
4. **Deployment**: Optimize sparse models for production

## Prerequisites
- Module 10: Training (models to compress)
- Module 17: Precision (understanding of optimization tradeoffs)

## Key Concepts

### Magnitude-Based Pruning
```python
# Remove 90% of smallest weights
def prune_magnitude(model, sparsity=0.9):
    for layer in model.layers:
        threshold = torch.quantile(abs(layer.weight), sparsity)
        mask = abs(layer.weight) > threshold
        layer.weight *= mask  # Zero out small weights
```

### Structured Pruning
```python
# Remove entire filters/channels
def prune_structured(conv_layer, num_filters_to_remove):
    # Compute filter importance (L2 norm)
    importance = conv_layer.weight.norm(dim=(1,2,3))
    
    # Keep only important filters
    keep_indices = importance.topk(n_keep).indices
    conv_layer.weight = conv_layer.weight[keep_indices]
```

### Knowledge Distillation
```python
# Small student learns from large teacher
teacher = LargeModel()  # 100M parameters
student = SmallModel()  # 10M parameters

# Student learns both from labels and teacher
loss = alpha * cross_entropy(student(x), y) + \
       beta * kl_divergence(student(x), teacher(x))
```

## Performance Impact
- **Model Size**: 10x reduction with pruning
- **Inference Speed**: 3-5x faster with structured pruning  
- **Accuracy**: Maintain 95%+ of original performance
- **Memory**: Deploy large models on edge devices

## Real-World Applications
- **MobileNet**: Designed for mobile deployment
- **DistilBERT**: 60% faster, 97% performance
- **Lottery Ticket Hypothesis**: Finding efficient subnetworks
- **Neural Architecture Search**: Automated compression

## Module Structure
1. **Sparsity Theory**: Why neural networks are compressible
2. **Magnitude Pruning**: Simple but effective compression
3. **Structured Pruning**: Hardware-friendly sparsity
4. **Knowledge Distillation**: Learning from larger models
5. **Deployment**: Optimizing sparse models

## Hands-On Projects
```python
# Project 1: Prune your CNN
cnn = load_model("cifar10_cnn.pt")
pruned = progressive_prune(cnn, target_sparsity=0.9)
print(f"Parameters: {count_params(cnn)} → {count_params(pruned)}")
print(f"Accuracy: {evaluate(cnn)}% → {evaluate(pruned)}%")

# Project 2: Distill transformer to CNN
teacher = TinyTransformer()  
student = SimpleCNN()
distilled = distill(teacher, student, data_loader)
```

## Success Criteria
- ✅ Achieve 90% sparsity with <5% accuracy loss
- ✅ 3x inference speedup with structured pruning
- ✅ Successfully distill large models to small ones
- ✅ Deploy compressed models efficiently