# TinyTorch NBGrader Style Guide

## Purpose
This guide establishes the standard format for all NBGrader solution blocks across TinyTorch modules to ensure consistency and maximize educational value.

## Standard Solution Block Format

```python
def function_name(self, parameters):
    """
    Brief function description (1-2 sentences).
    
    Args:
        param1: Parameter description
        param2: Parameter description
    
    Returns:
        Return type and description
    
    TODO: Implement [specific task] with [key requirements].
    
    STEP-BY-STEP IMPLEMENTATION:
    1. [Action verb] [specific task] - [brief explanation]
    2. [Action verb] [specific task] - [brief explanation]  
    3. [Action verb] [specific task] - [brief explanation]
    4. [Action verb] [specific task] - [brief explanation]
    
    EXAMPLE USAGE:
    ```python
    # Realistic example with clear input/output
    input_data = ClassName(example_data)
    result = function_name(input_data, parameters)
    print(result)  # Expected: [specific output]
    ```
    
    IMPLEMENTATION HINTS:
    - Use [specific function/method] for [specific purpose]
    - Handle [edge case] by [specific approach]
    - Remember to [critical requirement]
    - Common error: [specific mistake to avoid]
    
    LEARNING CONNECTIONS:
    - This is equivalent to [PyTorch/TensorFlow function]
    - Used in [real-world application/system]
    - Foundation for [advanced concept]
    - Enables [specific capability]
    """
    ### BEGIN SOLUTION
    # Implementation code (typically 3-10 lines)
    # Focus on clarity and correctness
    # Follow the steps outlined above
    ### END SOLUTION
```

## Required Sections

### 1. TODO
- **Purpose**: Clear task description
- **Format**: `TODO: Implement [specific task] with [key requirements].`
- **Example**: `TODO: Implement forward pass for ReLU activation with proper handling of negative values.`

### 2. STEP-BY-STEP IMPLEMENTATION
- **Purpose**: Guide implementation approach
- **Format**: Numbered list with action verbs
- **Guidelines**:
  - Start each step with an action verb (Create, Calculate, Apply, Return)
  - Include brief explanation after dash
  - Keep to 3-5 steps for later modules, 5-7 for early modules
- **Example**:
  ```
  1. Check input dimensions - ensure tensor is valid
  2. Apply element-wise maximum - compare with zero
  3. Return activated tensor - maintain original shape
  ```

### 3. EXAMPLE USAGE
- **Purpose**: Demonstrate correct usage
- **Format**: Python code block with comments
- **Must Include**:
  - Realistic input data
  - Function call with proper parameters
  - Expected output with comment
- **Example**:
  ```python
  # Create sample input
  x = Tensor([[-1, 0, 2], [3, -4, 5]])
  relu = ReLU()
  output = relu(x)
  print(output)  # Expected: [[0, 0, 2], [3, 0, 5]]
  ```

### 4. IMPLEMENTATION HINTS
- **Purpose**: Technical guidance and common pitfalls
- **Format**: Bulleted list
- **Should Include**:
  - Specific functions/methods to use
  - Edge cases to handle
  - Common errors to avoid
  - Performance considerations (for later modules)
- **Example**:
  ```
  - Use np.maximum() for element-wise comparison
  - Handle None inputs gracefully
  - Remember to preserve input shape
  - Common error: forgetting to handle batch dimensions
  ```

### 5. LEARNING CONNECTIONS
- **Purpose**: Connect to real-world ML systems
- **Format**: Bulleted list
- **Should Include**:
  - Framework equivalents (PyTorch/TensorFlow)
  - Real-world applications
  - Connection to other modules
  - Why this implementation matters
- **Example**:
  ```
  - This is equivalent to torch.nn.ReLU() in PyTorch
  - Used in every modern neural network architecture
  - Foundation for understanding gradient flow
  - Enables training deep networks without vanishing gradients
  ```

## Optional Enhancement Sections

### VISUAL STEP-BY-STEP (Early modules)
- **When to Use**: Complex mathematical operations or data flow
- **Format**: ASCII diagrams or visual explanations
- **Example**:
  ```
  Input: [1, -2, 3, -4, 5]
           ↓ ReLU
  Output: [1, 0, 3, 0, 5]
  ```

### DEBUGGING HINTS (When helpful)
- **When to Use**: Functions with common implementation errors
- **Format**: Specific debugging strategies
- **Example**:
  ```
  - Print shapes at each step to verify dimensions
  - Check for NaN values after operations
  - Verify gradient flow in backward pass
  ```

### MATHEMATICAL FOUNDATION (Math-heavy modules)
- **When to Use**: Complex mathematical operations
- **Format**: LaTeX-style equations with explanations
- **Example**:
  ```
  Softmax formula: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
  ```

## Module-Specific Guidelines

### Early Modules (01-07): Foundation & Architecture
- More detailed STEP-BY-STEP (5-7 steps)
- Include VISUAL STEP-BY-STEP where helpful
- Focus on educational clarity
- Simpler EXAMPLE USAGE

### Middle Modules (08-11): Training
- Balance detail with conciseness (4-5 steps)
- Include gradient flow considerations
- Real dataset examples
- Performance hints become important

### Later Modules (12-16): Production
- Concise STEP-BY-STEP (3-5 steps)
- Production-focused IMPLEMENTATION HINTS
- Complex, real-world EXAMPLE USAGE
- Strong emphasis on LEARNING CONNECTIONS to industry

## Quality Checklist

Before finalizing any solution block, verify:

- [ ] TODO clearly states the task
- [ ] STEP-BY-STEP has numbered action steps
- [ ] EXAMPLE USAGE has realistic code with expected output
- [ ] IMPLEMENTATION HINTS cover key technical points
- [ ] LEARNING CONNECTIONS link to real ML systems
- [ ] Solution code follows the outlined steps
- [ ] All code is tested and working
- [ ] Docstring has proper Args/Returns sections

## Common Mistakes to Avoid

1. **Inconsistent section names**: Always use exact section headers
2. **Missing expected output**: Every example needs `# Expected:` comment
3. **Too vague TODOs**: Be specific about requirements
4. **Untested examples**: All example code must actually work
5. **Missing Learning Connections**: Always connect to real-world ML

## Example: Well-Formatted Solution Block

```python
def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Apply softmax activation function along specified axis.
    
    Args:
        x: Input array of any shape
        axis: Axis along which to apply softmax (default: -1)
    
    Returns:
        Array with same shape as input with softmax applied
    
    TODO: Implement numerically stable softmax with overflow protection.
    
    STEP-BY-STEP IMPLEMENTATION:
    1. Subtract maximum value - prevent overflow in exponential
    2. Compute exponentials - apply exp() to shifted values
    3. Sum exponentials - calculate normalization factor
    4. Divide by sum - normalize to get probabilities
    
    EXAMPLE USAGE:
    ```python
    logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
    probs = softmax(logits)
    print(probs.sum(axis=1))  # Expected: [1.0, 1.0]
    print(probs[0])  # Expected: [0.659, 0.242, 0.099] (approx)
    ```
    
    IMPLEMENTATION HINTS:
    - Use x.max(axis=axis, keepdims=True) for stable computation
    - Apply np.exp() after shifting by maximum
    - Use keepdims=True to maintain broadcasting shape
    - Common error: forgetting to handle arbitrary axis parameter
    
    LEARNING CONNECTIONS:
    - This is equivalent to torch.nn.functional.softmax() in PyTorch
    - Critical for multi-class classification in final layers
    - Used in attention mechanisms for weight normalization
    - Foundation for cross-entropy loss computation
    """
    ### BEGIN SOLUTION
    x_max = x.max(axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    sum_exp = exp_x.sum(axis=axis, keepdims=True)
    return exp_x / sum_exp
    ### END SOLUTION
```

## Enforcement

1. All new modules MUST follow this style guide
2. Existing modules should be updated when modified
3. Use this guide for code reviews
4. Include compliance in module testing

---

*Last Updated: [Current Date]*
*Version: 1.0*