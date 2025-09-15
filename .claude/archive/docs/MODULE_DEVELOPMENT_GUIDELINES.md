# TinyTorch Module Development Guidelines

This guide provides comprehensive instructions for creating TinyTorch modules that deliver consistent, high-quality educational experiences.

## Core Philosophy

TinyTorch modules follow the **"Implement → Test → Implement → Test"** pattern where students get immediate feedback after implementing each concept. This creates a tight feedback loop that builds confidence and validates understanding.

**CRITICAL**: Every piece of code must be preceded by markdown that explains WHY we're implementing it, WHAT it does, and HOW it connects to the bigger picture. See `MARKDOWN_BEST_PRACTICES.md` for detailed guidelines.

## Educational Framework: Build → Use → Understand

Every module follows this three-phase learning approach:

1. **Build**: Students implement the core functionality from scratch
2. **Use**: Students immediately use their implementation with real data  
3. **Understand**: Students gain deeper insights into how the component fits into larger systems

## Implementation-Test Cycle Pattern

### The Golden Rule: Immediate Testing
**Every concept gets tested immediately after implementation.** This is non-negotiable.

```python
# 1. Introduce concept with theory
# %% [markdown] - Conceptual introduction

# 2. Implementation with scaffolding  
# %% nbgrader={"solution": true} - Student implements

# 3. IMMEDIATE unit test
# %% nbgrader={"grade": true} - Validate implementation

# 4. Move to next concept
```

### Testing Hierarchy

1. **Unit Tests** (`test_unit_[function_name]`): Test individual functions immediately after implementation
2. **Comprehensive Tests** (`test_unit_[module]_comprehensive`): Test multiple functions working together
3. **Integration Tests** (`test_module_[module]_[component]_integration`): Test cross-module compatibility

## Module Structure Requirements

### 1. File Organization
- **One dev file**: `[module]_dev.py` containing all implementations and tests
- **Module metadata**: `module.yaml` with dependencies and export information
- **Documentation**: README.md with educational content (only if explicitly requested)

### 2. Import Pattern (MANDATORY)
```python
# Always use this exact pattern for maximum compatibility
try:
    from tinytorch.core.[dependency] import [Class]
except ImportError:
    # For development, import from local modules
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '[dependency_module]'))
    from [dependency]_dev import [Class]
```

### 3. NBGrader Configuration
- **Implementation cells**: `{"grade": false, "solution": true, "task": false}`
- **Test cells**: `{"grade": true, "locked": true, "points": N, "solution": false, "task": false}`
- **Documentation cells**: `{"grade": false, "locked": false, "solution": false, "task": false}`

**IMPORTANT**: See `NBGRADER_INTEGRATION_GUIDE.md` for complete explanation of why we use NBGrader and what each metadata field means.

## Implementation Guidelines

### 1. Scaffolding Pattern (REQUIRED)
Every implementation must include:

```python
def method_name(self, parameters):
    """
    [Clear description]
    
    TODO: Implement [specific functionality].
    
    STEP-BY-STEP IMPLEMENTATION:
    1. [Specific action with clear guidance]
    2. [Next specific action]
    3. [Continue building up solution]
    4. [Final step]
    
    EXAMPLE USAGE:
    ```python
    [Working example showing exactly how to use]
    ```
    
    IMPLEMENTATION HINTS:
    - [Specific function/method to use]
    - [Specific approach or algorithm]
    - [Common pitfall to avoid]
    - [Expected return format]
    
    LEARNING CONNECTIONS:
    - This is like [PyTorch/TensorFlow equivalent]
    - Used in [real applications]
    - [Broader conceptual connection]
    """
    ### BEGIN SOLUTION
    [Implementation]
    ### END SOLUTION
```

### 2. Mathematical Foundation
Every concept must include:
- **Clear mathematical definition**
- **Visual/intuitive explanation**
- **Real-world examples**
- **Connection to ML systems**

### 3. Error Handling
- **Educational error messages**: Help students debug issues
- **Input validation**: Check for common mistakes
- **Graceful degradation**: Handle edge cases appropriately

## Testing Standards

### 1. Unit Test Requirements
```python
def test_unit_[function_name]():
    """[Clear description of what's being tested]."""
    print("🔬 Unit Test: [Test Description]...")
    
    # Test 1: Basic functionality
    [test implementation]
    assert [condition], f"[Descriptive error with expected vs actual]"
    
    # Test 2: Edge case
    [test implementation]
    assert [condition], f"[Descriptive error]"
    
    # Test 3: Integration scenario
    [test implementation]
    assert [condition], f"[Descriptive error]"
    
    print("✅ [Function] tests passed!")
    print(f"✅ [Specific behavior validated]")
    print(f"✅ [Another behavior validated]")

# CRITICAL: Run test immediately
test_unit_[function_name]()
```

### 2. Test Placement Rules
- **Immediate**: Unit tests run right after implementation
- **Descriptive**: Tests explain what behavior is being validated
- **Progressive**: Each test builds on previous understanding
- **Comprehensive**: Cover normal cases, edge cases, and integration

### 3. Integration Test Standards
```python
def test_module_[module]_[component]_integration():
    """
    Integration test for [module] with [other components].
    
    Tests that [module] properly integrates with [other systems]
    and maintains compatibility for [use cases].
    """
    print("🔬 Running Integration Test: [Module]-[Component] Integration...")
    
    # Always test real-world scenarios
    # Always verify type consistency
    # Always check cross-module compatibility
    
    print("✅ Integration Test Passed: [Description].")
```

## Content Standards

### 1. Markdown Before Every Code Block (MANDATORY)
Every code implementation must be preceded by markdown that includes:
- **Context**: Why are we doing this?
- **Concept**: What is this technically?
- **Connection**: How does this relate to what they know?
- **Concrete**: Specific example with real values
- **Confidence**: What success looks like

See `MARKDOWN_BEST_PRACTICES.md` for the complete stencil.

### 2. Conceptual Introductions
- **Start with "why"**: Why does this concept matter?
- **Mathematical foundation**: Precise definitions
- **Visual intuition**: Simple examples students can follow
- **Real applications**: Where is this used in practice?

### 3. Implementation Guidance
- **Step-by-step breakdown**: Never leave students guessing
- **Specific hints**: Name exact functions/approaches to use
- **Example walkthrough**: Show expected inputs/outputs
- **Connection to theory**: Link implementation back to concepts

### 4. Professional Context
- **Industry relevance**: How does this relate to PyTorch/TensorFlow?
- **Production patterns**: What professional practices are being taught?
- **System integration**: How does this fit into larger architectures?

## Quality Assurance Checklist

### Before Module Release
- [ ] Every concept has immediate unit test
- [ ] All tests pass and provide clear feedback
- [ ] Integration tests verify cross-module compatibility
- [ ] Mathematical foundations are clearly explained
- [ ] Implementation scaffolding follows required pattern
- [ ] Error messages are educational and helpful
- [ ] Code exports properly to tinytorch package
- [ ] Dependencies are correctly specified in module.yaml
- [ ] Professional connections are clearly articulated

### Educational Effectiveness
- [ ] Students can complete without prior knowledge of implementation details
- [ ] Each step builds logically on previous steps
- [ ] Immediate feedback validates understanding
- [ ] Real-world connections are clear and compelling
- [ ] Module prepares students for subsequent modules

## Common Anti-Patterns to Avoid

### ❌ Don't Do This
```python
# Vague instructions
def forward(self, x):
    """Implement forward pass."""
    # TODO: Fill this in
    pass

# No immediate testing
# Students implement but don't know if it works

# Unclear error messages
assert x.shape[0] > 0  # What does this mean?

# Missing context
# Why are we doing this? How does it connect?
```

### ✅ Do This Instead
```python
# Clear, specific instructions
def forward(self, x):
    """
    Apply ReLU activation: f(x) = max(0, x)
    
    TODO: Implement ReLU activation function.
    
    STEP-BY-STEP:
    1. Use np.maximum(0, x.data) for element-wise max
    2. Return new Tensor with same type as input
    
    EXAMPLE:
    relu = ReLU()
    result = relu(Tensor([-1, 0, 1]))  # Should be [0, 0, 1]
    """
    ### BEGIN SOLUTION
    result = np.maximum(0, x.data)
    return type(x)(result)
    ### END SOLUTION

# Immediate testing with clear validation
def test_unit_relu():
    print("🔬 Testing ReLU activation...")
    # [comprehensive test with clear assertions]
    print("✅ ReLU tests passed!")

test_unit_relu()  # Run immediately!
```

## Module Integration Protocol

### 1. Dependency Management
- **Verify prerequisites**: Ensure required modules are complete
- **Test integration**: Verify compatibility with existing modules
- **Update module.yaml**: Specify dependencies and export paths

### 2. Package Export Protocol
- **Use `#| export`**: Mark code for package inclusion
- **Test exports**: Verify `tito export` and `tito test` work
- **Integration testing**: Ensure exported code works with other modules

### 3. Documentation Standards
- **Self-documenting code**: Implementation should be clear without external docs
- **Educational comments**: Explain the "why" not just the "what"
- **Professional examples**: Show real-world usage patterns

## Success Metrics

A successful TinyTorch module should:

1. **Enable immediate progress**: Students can implement and test each concept
2. **Build confidence**: Clear scaffolding and immediate feedback reduce frustration
3. **Create understanding**: Students learn both implementation and conceptual foundations
4. **Prepare for advancement**: Module prepares students for subsequent modules
5. **Mirror industry**: Professional patterns and practices are clearly taught

## Module Templates

Use the `MODULE_STRUCTURE_TEMPLATE.md` as your starting point for every new module. This ensures consistency and reduces development time while maintaining educational quality.

Remember: **Great modules are built through the implement-test-implement-test cycle.** Students should never implement large chunks without validation. Every small step should provide immediate feedback and build toward the larger learning objectives.