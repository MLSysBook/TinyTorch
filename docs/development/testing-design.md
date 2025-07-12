# TinyTorch Testing Design Document

## Overview

This document defines the **inline-first testing architecture** for TinyTorch, prioritizing student learning effectiveness and flow state over context switching overhead.

## Core Philosophy: Learning-First Testing

**Primary Goal**: Student learning and confidence building  
**Secondary Goal**: Comprehensive validation for grading  
**Tertiary Goal**: Professional testing practices  

### Key Insight
Students learning ML concepts are already at cognitive capacity. Every context switch is expensive and disrupts the learning flow. We prioritize immediate feedback and confidence building over professional tool complexity.

## Three-Tier Testing Architecture

### 1. Inline Tests (Primary - In Notebooks)
**Goal**: Immediate feedback and confidence building during development

**Location**: Embedded in `*_dev.py` files as NBGrader cells  
**Dependencies**: None (or minimal, well-controlled)  
**Scope**: Comprehensive testing with educational context  
**Purpose**: Build confidence step-by-step while ensuring correctness  

**Example**:
```python
# %% [markdown]
"""
### 🧪 Test Your ReLU Implementation

Let's verify your ReLU function works correctly with various inputs.
This tests the core functionality you just implemented.
"""

# %% nbgrader={"grade": true, "grade_id": "test-relu-comprehensive", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def test_relu_comprehensive():
    """Comprehensive test of ReLU function."""
    print("🔬 Testing ReLU function...")
    
    # Test 1: Basic positive numbers
    try:
        result = relu([1, 2, 3])
        expected = [1, 2, 3]
        assert result == expected, f"Positive numbers: expected {expected}, got {result}"
        print("✅ Positive numbers work correctly")
    except Exception as e:
        print(f"❌ Positive numbers failed: {e}")
        return False
    
    # Test 2: Negative numbers (should become 0)
    try:
        result = relu([-1, -2, -3])
        expected = [0, 0, 0]
        assert result == expected, f"Negative numbers: expected {expected}, got {result}"
        print("✅ Negative numbers correctly clipped to 0")
    except Exception as e:
        print(f"❌ Negative numbers failed: {e}")
        return False
    
    # Test 3: Mixed positive and negative
    try:
        result = relu([-2, -1, 0, 1, 2])
        expected = [0, 0, 0, 1, 2]
        assert result == expected, f"Mixed numbers: expected {expected}, got {result}"
        print("✅ Mixed positive/negative numbers work correctly")
    except Exception as e:
        print(f"❌ Mixed numbers failed: {e}")
        return False
    
    # Test 4: Edge case - zero
    try:
        result = relu([0])
        expected = [0]
        assert result == expected, f"Zero: expected {expected}, got {result}"
        print("✅ Zero handled correctly")
    except Exception as e:
        print(f"❌ Zero failed: {e}")
        return False
    
    print("🎉 All ReLU tests passed! Your implementation works correctly.")
    print("📈 Progress: ReLU function ✓")
    return True

# Run the test
success = test_relu_comprehensive()
if not success:
    print("\n💡 Hints for fixing ReLU:")
    print("- ReLU should return max(0, x) for each element")
    print("- Negative numbers should become 0")
    print("- Positive numbers should stay unchanged")
    print("- Zero should remain zero")
```

**Characteristics**:
- **Comprehensive**: Test all functionality, edge cases, error conditions
- **Educational**: Explain what's being tested and why
- **Visual**: Clear pass/fail feedback with emojis and progress tracking
- **Immediate**: No context switching required
- **Encouraging**: Build confidence with positive reinforcement
- **Helpful**: Provide hints and guidance when tests fail

### 2. Module Tests (For Grading - Separate Files with Mocks)
**Goal**: Comprehensive validation for instructor grading using simple, visible mocks

**Location**: `tests/test_{module}.py` files  
**Dependencies**: Simple, visible mock objects (no cross-module dependencies)  
**Scope**: Complete module functionality with professional structure  
**Purpose**: Verify module works correctly for grading and assessment  

**Example**:
```python
# tests/test_layers.py
"""
Comprehensive Layers Module Tests

Tests Dense layer functionality using simple mock objects.
Used for instructor grading and comprehensive validation.
"""

class SimpleTensor:
    """
    Simple mock tensor for testing layers.
    
    Shows exactly what interface the Dense layer expects:
    - .data (numpy array): The actual numerical data
    - .shape (tuple): Dimensions of the data
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"SimpleTensor(shape={self.shape})"

class TestDenseLayer:
    """Comprehensive tests for Dense layer - used for grading."""
    
    def test_initialization(self):
        """Test Dense layer creation and weight initialization."""
        layer = Dense(input_size=3, output_size=2)
        
        assert hasattr(layer, 'weights'), "Dense layer should have weights"
        assert hasattr(layer, 'bias'), "Dense layer should have bias"
        assert layer.weights.shape == (3, 2), f"Expected weights shape (3, 2), got {layer.weights.shape}"
        assert layer.bias.shape == (2,), f"Expected bias shape (2,), got {layer.bias.shape}"
    
    def test_forward_pass_comprehensive(self):
        """Comprehensive test of Dense layer forward pass."""
        layer = Dense(input_size=3, output_size=2)
        
        # Test single sample
        input_tensor = SimpleTensor([[1.0, 2.0, 3.0]])
        output = layer(input_tensor)
        
        assert hasattr(output, 'data'), "Layer should return tensor-like object with .data"
        assert hasattr(output, 'shape'), "Layer should return tensor-like object with .shape"
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        
        # Verify computation (y = Wx + b)
        expected = np.dot(input_tensor.data, layer.weights) + layer.bias
        np.testing.assert_array_almost_equal(output.data, expected)
        
        # Test batch processing
        batch_input = SimpleTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        batch_output = layer(batch_input)
        assert batch_output.shape == (2, 2), f"Expected batch output shape (2, 2), got {batch_output.shape}"
        
        # Test edge cases
        edge_input = SimpleTensor([[0.0, 0.0, 0.0]])
        edge_output = layer(edge_input)
        assert edge_output.shape == (1, 2), "Should handle zero input"
```

**Characteristics**:
- **Self-contained**: No dependencies on other TinyTorch modules
- **Mock-based**: Simple, visible mocks that document interfaces
- **Comprehensive**: Test all functionality, edge cases, error conditions
- **Professional**: Use pytest structure and best practices
- **Grading-focused**: Designed for instructor assessment

### 3. Integration Tests (Cross-Module Validation)
**Goal**: Verify modules work together using vetted solutions

**Location**: `tests/integration/` directory  
**Dependencies**: Instructor-provided working implementations  
**Scope**: Cross-module workflows and realistic ML scenarios  
**Purpose**: Ensure modules compose correctly without cascade failures  

**Example**:
```python
# tests/integration/test_basic_ml_pipeline.py
"""
Integration Tests - Basic ML Pipeline

Tests how student modules work together with vetted solutions.
No cascade failures - student code tested with known-working dependencies.
"""

from tinytorch.solutions.tensor import Tensor  # Working implementation
from tinytorch.solutions.activations import ReLU  # Working implementation
from student_layers import Dense  # Student's implementation

class TestBasicMLPipeline:
    """Test student modules in realistic ML workflows."""
    
    def test_neural_network_forward_pass(self):
        """Test complete neural network using student's Dense layer."""
        # Create network with student's Dense layers
        layer1 = Dense(input_size=4, output_size=3)  # Student implementation
        activation = ReLU()  # Working implementation
        layer2 = Dense(input_size=3, output_size=2)  # Student implementation
        
        # Create input with working Tensor
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])  # Working tensor
        
        # Forward pass through network
        h1 = layer1(x)  # Student layer with working tensor
        h1_activated = activation(h1)  # Working activation
        output = layer2(h1_activated)  # Student layer
        
        # Verify pipeline works end-to-end
        assert output.shape == (1, 2), "Network should produce correct output shape"
        assert isinstance(output, Tensor), "Network should produce Tensor output"
        
        print("✅ Student's Dense layers work in complete neural network!")
```

## Testing Workflow

### For Students (Primary Path)
1. **Implement in notebook**: Write code with educational guidance
2. **Test inline**: Get immediate feedback without leaving notebook
3. **Build confidence**: See progress with visual indicators
4. **Complete module**: All inline tests pass before moving on

### For Instructors (Grading Path)
1. **Run module tests**: Comprehensive validation with mocks
2. **Run integration tests**: Verify cross-module functionality
3. **Grade systematically**: Clear separation of concerns

## Key Principles

### 1. **Learning-First Design**
- Prioritize student understanding over tool complexity
- Minimize context switching and cognitive overhead
- Provide immediate, encouraging feedback
- Build confidence step by step

### 2. **Flow State Preservation**
- Keep students in their notebooks
- Provide instant validation
- Use visual, encouraging feedback
- Avoid workflow disruption

### 3. **Comprehensive Coverage**
- Inline tests are thorough, not just quick checks
- Test functionality, edge cases, and error conditions
- Provide educational context for each test
- Explain what's being tested and why

### 4. **Professional Structure (Where Appropriate)**
- Module tests use professional pytest structure
- Integration tests mirror real-world workflows
- Maintain quality standards for grading
- Prepare students for industry practices

## Implementation Guidelines

### Inline Test Design
1. **Comprehensive**: Test all functionality thoroughly
2. **Educational**: Explain what's being tested
3. **Visual**: Use emojis and clear progress indicators
4. **Helpful**: Provide hints when tests fail
5. **Encouraging**: Build confidence with positive feedback

### Mock Design for Module Tests
1. **Simple**: Only implement what's needed
2. **Visible**: Put mocks at top of files with clear documentation
3. **Educational**: Show exactly what interfaces are expected
4. **Minimal**: Don't over-engineer the mocks

### Test Organization
```
modules/source/{module}/{module}_dev.py    # Implementation + comprehensive inline tests
tests/test_{module}.py                     # Module tests with mocks (for grading)
tests/integration/                         # Cross-module tests with vetted solutions
```

### CLI Integration
```bash
# Primary workflow - students work in notebooks
# Tests run automatically as cells execute

# Instructor grading workflow
tito test --module tensor                  # Run module tests with mocks
tito test --integration                    # Run integration tests
tito test --all                           # Run all tests
```

## Benefits

### For Students
- **No context switching**: Stay in flow state
- **Immediate feedback**: Know instantly if code works
- **Confidence building**: Step-by-step validation
- **Clear guidance**: Helpful error messages and hints
- **Educational value**: Learn what good testing looks like

### For Instructors
- **Comprehensive validation**: Thorough testing for grading
- **Clear diagnostics**: Know exactly what's working/broken
- **Independent assessment**: Module tests don't depend on student's other modules
- **Professional standards**: Maintain quality without overwhelming students

### For the System
- **Maintainable**: Clear separation of learning vs grading concerns
- **Scalable**: Easy to add new modules
- **Educational**: Every test serves a learning purpose
- **Practical**: Balances learning effectiveness with quality assurance

## Conclusion

The inline-first testing approach prioritizes student learning effectiveness over tool complexity. Students get comprehensive testing within their learning context, building confidence and understanding step by step. Instructors maintain professional testing standards for grading while avoiding the cognitive overhead that disrupts the learning process.

**Key insight**: Context switching is expensive for learners. Keep them in flow state while ensuring comprehensive validation. 