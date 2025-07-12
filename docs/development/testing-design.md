# TinyTorch Testing Design Document

## Overview

This document defines the four-tier testing architecture for TinyTorch, ensuring comprehensive validation while maintaining educational clarity and avoiding dependency cascades.

## Four-Tier Testing Architecture

### 1. Unit Tests (In Notebooks)
**Goal**: Immediate feedback on individual functions during development

**Location**: Embedded in `*_dev.py` files as NBGrader cells  
**Dependencies**: None (or minimal, well-controlled)  
**Scope**: Individual functions and methods  
**Purpose**: Catch basic implementation errors immediately  

**Example**:
```python
# %% nbgrader={"grade": true, "grade_id": "test-relu-basic", "locked": true, "points": 5}
# Quick validation of ReLU function
def test_relu_basic():
    # Test with simple inputs
    result = relu([-1, 0, 1, 2])
    expected = [0, 0, 1, 2]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ ReLU function works!")

test_relu_basic()
```

**Characteristics**:
- **Fast**: Execute in seconds
- **Simple**: Easy to understand and debug
- **Focused**: Test one function at a time
- **Visual**: Clear pass/fail feedback with emojis
- **Educational**: Explain what's being tested

### 2. Module Tests (Separate Files with Mocks)
**Goal**: Comprehensive validation of module functionality using simple, visible mocks

**Location**: `tests/test_{module}.py` files  
**Dependencies**: Simple, visible mock objects (no cross-module dependencies)  
**Scope**: Complete module functionality  
**Purpose**: Verify module works correctly with well-defined interfaces  

**Example**:
```python
# tests/test_layers.py
"""
Comprehensive Layers Module Tests

Tests Dense layer functionality using simple mock objects.
No dependencies on other TinyTorch modules.
"""

class SimpleTensor:
    """
    Simple mock of what Dense layer expects from Tensor.
    
    Your Dense layer should work with any object that has:
    - .data (numpy array): The actual numerical data
    - .shape (tuple): Dimensions of the data
    
    This mock shows exactly what interface your layer needs.
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"SimpleTensor(shape={self.shape})"

class TestDenseLayer:
    """Comprehensive tests for Dense layer implementation."""
    
    def test_initialization(self):
        """Test Dense layer creation and weight initialization."""
        layer = Dense(input_size=3, output_size=2)
        
        # Check weights and bias are created
        assert hasattr(layer, 'weights'), "Dense layer should have weights"
        assert hasattr(layer, 'bias'), "Dense layer should have bias"
        assert layer.weights.shape == (3, 2), f"Expected weights shape (3, 2), got {layer.weights.shape}"
        assert layer.bias.shape == (2,), f"Expected bias shape (2,), got {layer.bias.shape}"
    
    def test_forward_pass(self):
        """Test Dense layer forward pass with mock tensor."""
        layer = Dense(input_size=3, output_size=2)
        
        # Create mock input
        input_tensor = SimpleTensor([[1.0, 2.0, 3.0]])  # Batch size 1, 3 features
        
        # Forward pass
        output = layer(input_tensor)
        
        # Verify output
        assert hasattr(output, 'data'), "Layer should return tensor-like object with .data"
        assert hasattr(output, 'shape'), "Layer should return tensor-like object with .shape"
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        
        # Verify computation (y = Wx + b)
        expected = np.dot(input_tensor.data, layer.weights) + layer.bias
        np.testing.assert_array_almost_equal(output.data, expected)
    
    def test_batch_processing(self):
        """Test Dense layer with batch of inputs."""
        layer = Dense(input_size=2, output_size=3)
        
        # Batch of 4 samples, 2 features each
        batch_input = SimpleTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        
        output = layer(batch_input)
        
        assert output.shape == (4, 3), f"Expected batch output shape (4, 3), got {output.shape}"
    
    def test_edge_cases(self):
        """Test Dense layer with edge cases."""
        layer = Dense(input_size=1, output_size=1)
        
        # Single feature, single output
        single_input = SimpleTensor([[5.0]])
        output = layer(single_input)
        assert output.shape == (1, 1)
        
        # Large batch
        large_batch = SimpleTensor([[1.0]] * 100)  # 100 samples
        output = layer(large_batch)
        assert output.shape == (100, 1)
```

**Characteristics**:
- **Self-contained**: No dependencies on other TinyTorch modules
- **Comprehensive**: Test all functionality, edge cases, error conditions
- **Clear interfaces**: Mocks show exactly what the module expects
- **Debuggable**: Students can easily understand and modify mocks
- **Professional**: Use pytest structure and best practices

### 3. Integration Tests (With Vetted Solutions)
**Goal**: Verify new module composes correctly with other vetted modules

**Location**: `tests/integration/` directory  
**Dependencies**: Instructor-provided working implementations of prerequisite modules  
**Scope**: Cross-module workflows and realistic ML scenarios  
**Purpose**: Ensure modules work together in real ML pipelines  

**Example**:
```python
# tests/integration/test_layers_integration.py
"""
Integration Tests for Layers Module

Tests how student's layer implementation works with vetted Tensor and Activation modules.
Uses instructor-provided working implementations to avoid dependency cascades.
"""

from tinytorch.solutions.tensor import Tensor  # Instructor-provided working version
from tinytorch.solutions.activations import ReLU  # Instructor-provided working version
from student_layers import Dense  # Student's implementation

class TestLayersIntegration:
    """Test student's layers with working tensor and activation implementations."""
    
    def test_neural_network_forward_pass(self):
        """Test complete neural network forward pass using student's Dense layer."""
        # Create network components
        layer1 = Dense(input_size=4, output_size=3)  # Student's implementation
        activation = ReLU()  # Working implementation
        layer2 = Dense(input_size=3, output_size=2)  # Student's implementation
        
        # Create input data
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])  # Working tensor
        
        # Forward pass through network
        h1 = layer1(x)  # Student's layer with working tensor
        h1_activated = activation(h1)  # Working activation
        output = layer2(h1_activated)  # Student's layer
        
        # Verify complete pipeline works
        assert output.shape == (1, 2), "Network should produce correct output shape"
        assert isinstance(output, Tensor), "Network should produce Tensor output"
        
        print("✅ Student's Dense layers work in complete neural network!")
    
    def test_image_classification_pipeline(self):
        """Test realistic image classification scenario."""
        # Simulate flattened MNIST image (28x28 = 784 pixels)
        image_data = Tensor([np.random.randn(1, 784)])
        
        # Create classification network
        hidden_layer = Dense(784, 128)  # Student's implementation
        relu = ReLU()  # Working activation
        output_layer = Dense(128, 10)  # Student's implementation (10 classes)
        
        # Forward pass
        hidden = hidden_layer(image_data)
        activated = relu(hidden)
        predictions = output_layer(activated)
        
        # Verify realistic ML workflow
        assert predictions.shape == (1, 10), "Should output 10 class predictions"
        
        print("✅ Student's layers work for image classification!")
```

**Characteristics**:
- **Realistic workflows**: Test actual ML scenarios students will encounter
- **Vetted dependencies**: Use working implementations to isolate testing
- **No cascade failures**: Student's module tested independently
- **Production-like**: Mirror real-world ML development patterns

### 4. System Tests (Production Scenarios)
**Goal**: Validate performance, scalability, and robustness in production-like scenarios

**Location**: `tests/system/` directory  
**Dependencies**: Complete working system  
**Scope**: Performance, scalability, robustness, production workflows  
**Purpose**: Ensure system works at scale and handles real-world conditions  

**Example**:
```python
# tests/system/test_performance.py
"""
System Performance Tests

Tests TinyTorch performance with realistic datasets and workloads.
Ensures system can handle production-scale scenarios.
"""

import time
import psutil
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Dense
from tinytorch.core.networks import Sequential

class TestSystemPerformance:
    """Test system performance with realistic workloads."""
    
    def test_large_batch_processing(self):
        """Test system with large batch sizes."""
        # Create large network
        network = Sequential([
            Dense(1000, 500),
            Dense(500, 250),
            Dense(250, 10)
        ])
        
        # Large batch (1000 samples)
        large_batch = Tensor(np.random.randn(1000, 1000))
        
        # Time the forward pass
        start_time = time.time()
        output = network(large_batch)
        duration = time.time() - start_time
        
        # Verify performance
        assert duration < 5.0, f"Large batch processing took {duration:.2f}s, expected < 5s"
        assert output.shape == (1000, 10), "Should handle large batches correctly"
        
        print(f"✅ Processed 1000 samples in {duration:.2f}s")
    
    def test_memory_usage(self):
        """Test memory usage with realistic workloads."""
        # Monitor memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use multiple large tensors
        tensors = []
        for i in range(10):
            tensor = Tensor(np.random.randn(1000, 1000))
            tensors.append(tensor)
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Verify reasonable memory usage
        assert memory_used < 500, f"Memory usage {memory_used:.1f}MB seems excessive"
        
        print(f"✅ Memory usage: {memory_used:.1f}MB for large tensor operations")
    
    def test_cifar10_training_simulation(self):
        """Test system with CIFAR-10 scale workload."""
        # Simulate CIFAR-10 training batch
        batch_size = 32
        image_size = 32 * 32 * 3  # 3072 pixels
        num_classes = 10
        
        # Create realistic CNN-like network
        network = Sequential([
            Dense(image_size, 512),
            Dense(512, 256),
            Dense(256, 128),
            Dense(128, num_classes)
        ])
        
        # Simulate training batches
        total_time = 0
        num_batches = 100
        
        for batch in range(num_batches):
            # Create batch
            images = Tensor(np.random.randn(batch_size, image_size))
            
            # Forward pass
            start = time.time()
            predictions = network(images)
            batch_time = time.time() - start
            total_time += batch_time
            
            # Verify batch processing
            assert predictions.shape == (batch_size, num_classes)
        
        avg_batch_time = total_time / num_batches
        
        # Performance requirements
        assert avg_batch_time < 0.1, f"Average batch time {avg_batch_time:.3f}s too slow"
        
        print(f"✅ Processed {num_batches} CIFAR-10 batches, avg time: {avg_batch_time:.3f}s")
```

**Characteristics**:
- **Production scale**: Test with realistic dataset sizes and batch sizes
- **Performance monitoring**: Measure time, memory, throughput
- **Robustness testing**: Handle edge cases and stress conditions
- **Real-world scenarios**: Mirror actual ML training and inference workloads

## Testing Workflow

### For Students
1. **Develop with unit tests**: Get immediate feedback in notebooks
2. **Validate with module tests**: Run comprehensive tests before moving on
3. **Verify integration**: See how module works with broader system
4. **Optional system tests**: Understand production requirements

### For Instructors
1. **Grade module tests**: Assess individual module functionality
2. **Verify integration**: Ensure modules compose correctly
3. **Monitor system performance**: Track overall system health
4. **Provide solutions**: Maintain working implementations for integration tests

## Key Principles

### 1. **Dependency Isolation**
- Unit tests: No dependencies
- Module tests: Simple, visible mocks only
- Integration tests: Vetted solutions for dependencies
- System tests: Complete working system

### 2. **Clear Interfaces**
- Mocks document expected interfaces explicitly
- Students can see exactly what their module needs to provide
- Interface evolution is visible and documented

### 3. **Educational Value**
- Each test level serves a specific learning purpose
- Tests explain what they're checking and why
- Failures provide actionable feedback

### 4. **Professional Standards**
- Use pytest structure and best practices
- Include comprehensive edge case testing
- Mirror real-world ML development patterns

### 5. **Scalable Architecture**
- No cascade failures from broken dependencies
- Independent module development and grading
- Realistic integration without penalty for past bugs

## Implementation Guidelines

### Mock Design Principles
1. **Minimal**: Only implement what the module actually needs
2. **Visible**: Put mocks at the top of test files with clear documentation
3. **Simple**: Easy to understand and modify
4. **Evolving**: Update mocks as interfaces grow

### Test Organization
```
tests/
├── test_{module}.py          # Module tests with mocks
├── integration/              # Cross-module integration tests
│   ├── test_basic_ml.py      # Tensor → Layers → Networks
│   ├── test_vision.py        # CNN pipelines
│   └── test_data.py          # DataLoader → Networks
└── system/                   # Production-scale tests
    ├── test_performance.py   # Speed and memory
    ├── test_scalability.py   # Large datasets
    └── test_robustness.py    # Error handling
```

### CLI Integration
```bash
# Run unit tests (embedded in notebooks)
tito test --unit --module tensor

# Run module tests (with mocks)
tito test --module tensor

# Run integration tests
tito test --integration

# Run system tests
tito test --system

# Run all tests
tito test --all
```

## Benefits

### For Students
- **Clear progression**: Unit → Module → Integration → System
- **Immediate feedback**: Catch issues early
- **No cascade failures**: Broken dependencies don't block progress
- **Realistic experience**: See how modules work in complete systems

### For Instructors
- **Independent grading**: Assess modules separately
- **Clear diagnostics**: Know exactly where issues are
- **Flexible pacing**: Students can progress at different rates
- **Quality assurance**: Comprehensive validation at every level

### For the System
- **Maintainable**: Clear separation of concerns
- **Scalable**: Add new modules without breaking existing tests
- **Professional**: Industry-standard testing practices
- **Educational**: Every test serves a learning purpose

This four-tier architecture ensures comprehensive testing while maintaining educational clarity and avoiding the dependency cascade problem that plagued our earlier approaches. 