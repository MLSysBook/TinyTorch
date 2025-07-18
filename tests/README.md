# 🧪 TinyTorch Integration Tests

## ⚠️ **CRITICAL DIRECTORY - DO NOT DELETE**

This directory contains **17 integration test files** that verify cross-module functionality across the entire TinyTorch system. These tests represent significant development effort and are essential for:

- **Module integration validation**
- **Cross-component compatibility**  
- **Real-world ML pipeline testing**
- **System-level regression detection**

## 📁 **Test Structure**
- `test_*_integration.py` - Cross-module integration tests
- `test_utils.py` - Shared testing utilities
- `test_integration_report.md` - Test documentation

## 🧪 **Integration Test Coverage**

### Foundation Integration
- `test_tensor_activations_integration.py` - Tensor + Activations
- `test_layers_networks_integration.py` - Layers + Dense Networks
- `test_tensor_autograd_integration.py` - Tensor + Autograd

### Architecture Integration  
- `test_tensor_attention_integration.py` - **NEW**: Tensor + Attention mechanisms
- `test_attention_pipeline_integration.py` - **NEW**: Complete transformer-like pipelines
- `test_tensor_cnn_integration.py` - Tensor + Spatial/CNN
- `test_cnn_networks_integration.py` - Spatial + Dense Networks
- `test_cnn_pipeline_integration.py` - Complete CNN pipelines

### Training & Data Integration
- `test_dataloader_tensor_integration.py` - DataLoader + Tensor
- `test_training_integration.py` - Complete training workflows
- `test_ml_pipeline_integration.py` - End-to-end ML pipelines

### Inference Serving Integration
- `test_compression_integration.py` - Model compression
- `test_kernels_integration.py` - Custom operations
- `test_benchmarking_integration.py` - Performance measurement
- `test_mlops_integration.py` - Deployment and serving

## 🔧 **Usage**
```bash
# Run all integration tests
pytest tests/ -v

# Run specific module integration
pytest tests/test_tensor_attention_integration.py -v
pytest tests/test_attention_pipeline_integration.py -v

# Run attention-related tests
pytest tests/ -k "attention" -v
```

## 🚨 **Recovery Instructions**
If accidentally deleted:
```bash
git checkout HEAD -- tests/
git status  # Verify recovery
```

## 📊 **Test Coverage**
These integration tests complement the inline tests in each module's `*_dev.py` files, providing comprehensive system validation with focus on:
- **Real component integration** (not mocks)
- **Cross-module compatibility**
- **Realistic ML workflows** (classification, seq2seq, transformers)
- **Performance and scalability** 