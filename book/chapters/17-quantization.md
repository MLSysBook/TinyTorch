---
title: "Quantization - Reduced Precision for Efficiency"
description: "INT8 quantization, calibration, and mixed-precision strategies"
difficulty: 3
time_estimate: "5-6 hours"
prerequisites: ["Acceleration"]
next_steps: ["Compression"]
learning_objectives:
  - "Implement INT8 quantization for weights and activations"
  - "Design calibration strategies to minimize accuracy loss"
  - "Apply mixed-precision training and inference patterns"
  - "Understand quantization-aware training vs post-training quantization"
  - "Measure memory and speed improvements from reduced precision"
---

# 17. Quantization

**⚡ OPTIMIZATION TIER** | Difficulty: ⭐⭐⭐ (3/4) | Time: 5-6 hours

## Overview

Reduce model precision from FP32 to INT8 for 4× memory reduction and 2-4× inference speedup. This module implements quantization, calibration, and mixed-precision strategies used in production deployment.

## Learning Objectives

By completing this module, you will be able to:

1. **Implement INT8 quantization** for model weights and activations with scale/zero-point parameters
2. **Design calibration strategies** using representative data to minimize accuracy degradation
3. **Apply mixed-precision training** (FP16/FP32) for faster training with maintained accuracy
4. **Understand quantization-aware training** vs post-training quantization trade-offs
5. **Measure memory and speed improvements** while tracking accuracy impact

## Why This Matters

### Production Context

Quantization is mandatory for edge deployment:

- **TensorFlow Lite** uses INT8 quantization for mobile deployment; 4× smaller models
- **ONNX Runtime** supports INT8 inference; 2-4× faster on CPUs
- **Apple Core ML** quantizes models for iPhone Neural Engine; enables on-device ML
- **Google Edge TPU** requires INT8; optimized hardware for quantized operations

### Historical Context

- **Pre-2017**: FP32 standard; quantization for special cases only
- **2017-2019**: INT8 post-training quantization; TensorFlow Lite adoption
- **2019-2021**: Quantization-aware training; maintains accuracy better
- **2021+**: INT4, mixed-precision, dynamic quantization; aggressive compression

Quantization enables deployment where FP32 models wouldn't fit or run fast enough.

## Implementation Guide

### Core Components

**Symmetric INT8 Quantization**
```
Quantization: x_int8 = round(x_fp32 / scale)
Dequantization: x_fp32 = x_int8 * scale

where scale = max(|x|) / 127
```

**Asymmetric Quantization (with zero-point)**
```
Quantization: x_int8 = round(x_fp32 / scale) + zero_point
Dequantization: x_fp32 = (x_int8 - zero_point) * scale
```

**Calibration**: Use representative data to find optimal scale/zero-point parameters

## Testing

```bash
tito export 17_quantization
tito test 17_quantization
```

## Where This Code Lives

```
tinytorch/
├── quantization/
│   └── quantize.py
└── __init__.py
```

## Systems Thinking Questions

1. **Accuracy vs Efficiency**: INT8 loses precision. When is <1% accuracy drop acceptable? When must you use QAT?

2. **Per-Tensor vs Per-Channel**: Per-channel quantization preserves accuracy better but increases complexity. When is it worth it?

3. **Quantized Operations**: INT8 matmul is faster, but quantize/dequantize adds overhead. When does quantization win overall?

## Real-World Connections

**Mobile Deployment**: TensorFlow Lite, Core ML use INT8 for on-device inference
**Cloud Serving**: ONNX Runtime, TensorRT use INT8 for cost-effective serving
**Edge AI**: INT8 required for Coral Edge TPU, Jetson Nano deployment

## What's Next?

In **Module 18: Compression**, you'll combine quantization with pruning:
- Remove unimportant weights (pruning)
- Quantize remaining weights (INT8)
- Achieve 10-50× compression with minimal accuracy loss

---

**Ready to quantize models?** Open `modules/source/17_quantization/quantization_dev.py` and start implementing.
