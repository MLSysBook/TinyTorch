# TinyTorch Integration Testing Rules

**Context**: Cross-module testing patterns for TinyTorch's educational ML systems course  
**Reference**: `tests/` directory with 17 integration test files  
**Philosophy**: Interface compatibility over functionality re-testing

## 🎯 **Core Principle: Interface Testing, Not Function Re-testing**

Integration tests in TinyTorch focus on **cross-module interfaces and compatibility**, NOT re-testing individual module functionality. Individual functions should already work from inline tests within each module's `*_dev.py` files.

### ✅ **DO: Interface Compatibility Testing**
```python
def test_tensor_attention_interface():
    """Test Tensor operations work correctly with Attention mechanisms."""
    # Test interface compatibility
    tensor = Tensor([[1, 2, 3, 4]])
    attention = SelfAttention(embed_size=4, num_heads=1)
    
    # Focus on: Can these components work together?
    result = attention.forward(tensor)
    assert isinstance(result, Tensor)
    assert result.shape == tensor.shape
```

### ❌ **DON'T: Functionality Re-testing**
```python
def test_attention_math_correctness():
    """DON'T: Re-test attention computation correctness."""
    # This should already be verified in attention_dev.py inline tests
    Q, K, V = create_qkv_matrices()
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    # ... detailed math verification belongs in inline tests
```

## 🏗️ **Integration Test Categories**

### 1. **Foundation Integration** (Tensor + Core Operations)
- **Tensor + Activations**: Do activation functions work with Tensor operations?
- **Tensor + Autograd**: Does gradient computation integrate with Tensor?
- **Layers + Dense Networks**: Do individual layers compose into networks?

### 2. **Architecture Integration** (Component Composition)
- **Tensor + Attention**: Do attention mechanisms work with Tensor operations?
- **Spatial + Dense**: Do CNN layers integrate with fully connected layers?
- **Complete Pipelines**: Do end-to-end architectures work together?

### 3. **Training & Data Integration** (Learning Workflows)
- **DataLoader + Tensor**: Does data loading work with tensor operations?
- **Training Integration**: Do complete training workflows function?
- **ML Pipeline**: Do end-to-end machine learning pipelines work?

### 4. **Inference Serving Integration** (Production Systems)
- **Compression + Models**: Do compressed models maintain functionality?
- **Kernels + Operations**: Do custom kernels integrate with standard ops?
- **Benchmarking + Systems**: Does performance measurement work across components?

## 📋 **Integration Test Structure**

### **File Naming Convention**
```
test_{module1}_{module2}_integration.py    # Two-module interface testing
test_{workflow}_pipeline_integration.py    # Multi-module workflow testing
```

### **Test Function Structure**
```python
def test_interface_compatibility():
    """Test that ModuleA outputs work as ModuleB inputs."""
    # Setup components
    module_a = ModuleA()
    module_b = ModuleB()
    
    # Test interface compatibility
    output_a = module_a.process(input_data)
    result = module_b.process(output_a)  # Key: Does this work?
    
    # Assert compatibility, not correctness
    assert isinstance(result, expected_type)
    assert result.shape == expected_shape
    
def test_workflow_integration():
    """Test complete workflow across multiple modules."""
    # Test realistic usage patterns
    pipeline = setup_realistic_pipeline()
    result = pipeline.run(real_data)
    
    # Assert workflow completion
    assert pipeline.successful_completion()
```

## 🎓 **Educational Testing Principles**

### **Real Components, Not Mocks**
```python
# ✅ Good: Use actual TinyTorch components
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Dense

# ❌ Bad: Mock components that don't reflect real behavior
class MockTensor:
    def __init__(self, data): pass
```

### **Realistic Scenarios**
```python
def test_student_workflow():
    """Test scenarios students will actually encounter."""
    # Use realistic data sizes and patterns
    data = np.random.randn(32, 10)  # Reasonable batch size
    
    # Test common student workflows
    network = create_simple_network()
    predictions = network.forward(data)
    loss = compute_loss(predictions, targets)
```

### **Clear Success Criteria**
```python
def test_cross_module_compatibility():
    """Integration test with clear educational objectives."""
    print("🔬 Integration Test: Tensor + Activation compatibility...")
    
    # Test specific interface
    result = test_compatibility()
    
    # Clear feedback
    if result.success:
        print("✅ Tensor and Activation modules integrate correctly")
        print("📈 Progress: Cross-module interfaces ✓")
    else:
        print(f"❌ Integration issue: {result.error}")
```

## 🔧 **Test Implementation Guidelines**

### **Focus Areas for Integration Tests**

1. **Data Flow Compatibility**
   - Output of ModuleA → Input of ModuleB
   - Shape and type consistency across modules
   - Error propagation across component boundaries

2. **Interface Contracts**
   - Method signatures work across modules
   - Expected behaviors are maintained in composition
   - Error handling integrates properly

3. **Workflow Validation**
   - Complete educational scenarios work end-to-end
   - Student-facing APIs function together
   - Real-world usage patterns succeed

### **Testing Anti-Patterns to Avoid**

❌ **Don't Re-test Module Internals**
```python
# Bad: This should be in activation_dev.py inline tests
def test_relu_computation():
    assert relu(np.array([-1, 1])) == np.array([0, 1])
```

❌ **Don't Test Contrived Scenarios**
```python
# Bad: Students will never use 10000x10000 tensors
def test_massive_tensor_integration():
    huge_tensor = Tensor(np.random.randn(10000, 10000))
```

❌ **Don't Duplicate Inline Test Coverage**
```python
# Bad: If inline tests pass, mathematical correctness is verified
def test_attention_math_again():
    # Detailed mathematical verification already done inline
```

### ✅ **Good Integration Test Examples**

```python
def test_attention_tensor_interface():
    """Verify Attention mechanisms work with Tensor operations."""
    # Setup realistic scenario
    sequence_length, embed_size = 10, 64
    tensor_input = Tensor(np.random.randn(1, sequence_length, embed_size))
    
    # Test interface compatibility
    attention = SelfAttention(embed_size=embed_size)
    result = attention.forward(tensor_input)
    
    # Assert interface contract
    assert isinstance(result, Tensor)
    assert result.shape == tensor_input.shape
    print("✅ Attention-Tensor interface compatibility verified")

def test_complete_transformer_pipeline():
    """Test realistic transformer-like pipeline."""
    # Realistic student project scenario
    vocab_size, seq_len, embed_size = 1000, 20, 128
    
    # Complete pipeline components
    embedding = Embedding(vocab_size, embed_size)
    attention = SelfAttention(embed_size)
    dense = Dense(embed_size, vocab_size)
    
    # Test end-to-end workflow
    tokens = np.random.randint(0, vocab_size, (1, seq_len))
    embedded = embedding(tokens)
    attended = attention.forward(embedded)
    output = dense.forward(attended)
    
    # Verify complete workflow
    assert output.shape == (1, seq_len, vocab_size)
    print("✅ Complete transformer pipeline integration successful")
```

## 🚀 **Integration Testing Workflow**

### **Development Process**
1. **Module completion**: Ensure inline tests pass first
2. **Interface design**: Define cross-module contracts
3. **Integration tests**: Write interface compatibility tests
4. **Pipeline tests**: Add workflow validation tests
5. **Student validation**: Test realistic usage scenarios

### **Running Integration Tests**
```bash
# All integration tests
pytest tests/ -v

# Specific integration area
pytest tests/test_tensor_attention_integration.py -v
pytest tests/test_attention_pipeline_integration.py -v

# Integration test pattern
pytest tests/ -k "integration" -v
```

### **Quality Standards**
- ✅ **Interface Focus**: Tests verify component compatibility
- ✅ **Real Components**: Uses actual TinyTorch modules, not mocks
- ✅ **Student Scenarios**: Reflects realistic educational workflows
- ✅ **Clear Feedback**: Provides educational progress indicators
- ✅ **Complementary**: Adds value beyond inline testing

## 📚 **Reference Implementation**

**Best Examples**: 
- `test_tensor_activations_integration.py` - Interface compatibility testing
- `test_attention_pipeline_integration.py` - Complete workflow testing
- `test_layers_networks_integration.py` - Component composition testing

**Avoid These Patterns**:
- Tests that duplicate inline test functionality
- Mock-based testing that doesn't reflect real component behavior
- Contrived scenarios that students won't encounter

Integration tests should answer: *"Do these modules work together?"* not *"Do these modules work correctly?"* - correctness is verified by inline tests. 