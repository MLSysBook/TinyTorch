# Module 17: Precision - Numerical Optimization through Quantization

## Overview
Reduce model size by 75% and accelerate inference by 2-4x through INT8 quantization. Learn how production systems deploy billion-parameter models on edge devices.

## What You'll Build
- **INT8 Quantizer**: Convert FP32 models to INT8
- **Calibration System**: Find optimal scaling factors
- **Quantized Operations**: Fast integer arithmetic
- **Accuracy Validator**: Measure precision/performance tradeoffs

## Learning Objectives
1. **Numerical Representation**: FP32 vs FP16 vs INT8 tradeoffs
2. **Post-Training Quantization**: Convert trained models efficiently
3. **Calibration Techniques**: Minimize accuracy loss
4. **Hardware Acceleration**: Why INT8 is 4x faster on modern hardware

## Prerequisites
- Module 15: Acceleration (backend dispatch)
- Module 10: Training (trained models to quantize)

## Key Concepts

### The Problem: Model Size and Speed
```python
# FP32 Model - High precision, slow, large
model = TinyGPT()  # 400MB, 100ms/token

# After quantization - Lower precision, fast, small  
quantized = quantize_int8(model)  # 100MB, 25ms/token
```

### Quantization Process
```python
# 1. Calibration - Find scale factors
scales = calibrate(model, calibration_data)

# 2. Quantization - Convert weights
quantized_weights = (weights / scales).round().clip(-128, 127)

# 3. Inference - Use integer ops
output = quantized_forward(input, quantized_weights, scales)
```

## Performance Impact
- **Model Size**: 4x reduction (FP32 → INT8)
- **Inference Speed**: 2-4x faster on CPU/GPU
- **Accuracy**: Typically <1% loss with good calibration
- **Memory Bandwidth**: 4x reduction

## Real-World Applications
- **Mobile Deployment**: Run LLMs on phones
- **Edge AI**: Raspberry Pi inference
- **Datacenter Efficiency**: 4x more models per GPU
- **TensorFlow Lite**: Production quantization

## Module Structure
1. **Numerical Basics**: Understanding precision and range
2. **Quantization Math**: Scale factors and rounding
3. **Calibration**: Finding optimal quantization parameters
4. **Implementation**: Building quantized operations
5. **Evaluation**: Accuracy vs performance analysis

## Hands-On Examples
```python
# Quantize your trained CNN
cnn = load_trained_model("cifar10_cnn.pt")
quantized = quantize_model(cnn, calibration_loader)

# Compare accuracy
original_acc = evaluate(cnn, test_loader)      # 75.2%
quantized_acc = evaluate(quantized, test_loader)  # 74.8%

# Measure speedup
original_time = benchmark(cnn)      # 45ms/batch
quantized_time = benchmark(quantized)  # 12ms/batch (3.75x faster!)
```

## Success Criteria
- ✅ Quantize models to INT8 with <1% accuracy loss
- ✅ Achieve 2-4x inference speedup
- ✅ Reduce model size by 75%
- ✅ Understand hardware acceleration principles