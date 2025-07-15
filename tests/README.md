# 🧪 TinyTorch Integration Tests

## ⚠️ **CRITICAL DIRECTORY - DO NOT DELETE**

This directory contains **15 integration test files** that verify cross-module functionality across the entire TinyTorch system. These tests represent significant development effort and are essential for:

- **Module integration validation**
- **Cross-component compatibility**  
- **Real-world ML pipeline testing**
- **System-level regression detection**

## 📁 **Test Structure**
- `test_*_integration.py` - Cross-module integration tests
- `test_utils.py` - Shared testing utilities
- `test_integration_report.md` - Test documentation

## 🔧 **Usage**
```bash
# Run all integration tests
pytest tests/ -v

# Run specific module integration
pytest tests/test_tensor_activations_integration.py -v
```

## 🚨 **Recovery Instructions**
If accidentally deleted:
```bash
git checkout HEAD -- tests/
git status  # Verify recovery
```

## 📊 **Test Coverage**
These integration tests complement the inline tests in each module's `*_dev.py` files, providing comprehensive system validation. 