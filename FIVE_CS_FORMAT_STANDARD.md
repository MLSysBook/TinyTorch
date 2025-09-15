# The 5 C's Format Standard for TinyTorch

## Standard Structure

Use this exact format before every major implementation:

```markdown
### Before We Code: The 5 C's

```python
# CONCEPT: What is [Component]?
# Brief, clear definition with analogy to familiar concepts

# CODE STRUCTURE: What We're Building  
class ComponentName:
    def method1():     # Key method 1
    def method2():     # Key method 2
    # Properties: .prop1, .prop2

# CONNECTIONS: Real-World Equivalents
# PyTorch equivalent - same concept, production optimized
# TensorFlow equivalent - industry alternative
# NumPy/other relationship - how it relates to known tools

# CONSTRAINTS: Key Implementation Requirements
# - Technical requirement 1 with why it matters
# - Technical requirement 2 with why it matters
# - Technical requirement 3 with why it matters

# CONTEXT: Why This Matters in ML Systems
# Specific applications in ML:
# - Use case 1: How it's used in neural networks
# - Use case 2: How it's used in training
# - Use case 3: How it's used in production
```

**Compelling closing statement about impact.**
```

## Example: Tensor Implementation

```markdown
### Before We Code: The 5 C's

```python
# CONCEPT: What is a Tensor?
# Tensors are N-dimensional arrays that carry data through neural networks.
# Think NumPy arrays with ML superpowers - same math, more capabilities.

# CODE STRUCTURE: What We're Building
class Tensor:
    def __init__(self, data):     # Create from any data type
    def __add__(self, other):     # Enable tensor + tensor
    def __mul__(self, other):     # Enable tensor * tensor
    # Properties: .shape, .size, .dtype, .data

# CONNECTIONS: Real-World Equivalents  
# torch.Tensor (PyTorch) - same concept, production optimized
# tf.Tensor (TensorFlow) - distributed computing focus
# np.ndarray (NumPy) - we wrap this with ML operations

# CONSTRAINTS: Key Implementation Requirements
# - Handle broadcasting (auto-shape matching for operations)
# - Support multiple data types (float32, int32, etc.)
# - Efficient memory usage (copy only when necessary)
# - Natural math notation (tensor + tensor should just work)

# CONTEXT: Why This Matters in ML Systems
# Every ML operation flows through tensors:
# - Neural networks: All computations operate on tensors
# - Training: Gradients flow through tensor operations  
# - Hardware: GPUs optimized for tensor math
# - Production: Millions of tensor ops per second in real systems
```

**You're building the universal language of machine learning.**
```

## Key Design Principles

### 1. Code-Comment Integration
- Present concepts within code structure
- Show exactly where each principle applies
- Feel like practical guidance, not academic theory

### 2. Scannable Format
- Each C is clearly labeled
- Bullet points for easy scanning
- Concise but complete information

### 3. Implementation Focus
- CODE STRUCTURE shows actual methods being built
- CONSTRAINTS are technical requirements, not abstract concepts
- CONTEXT explains specific ML applications

### 4. Professional Connection
- CONNECTIONS always include PyTorch/TensorFlow equivalents
- Show how student code relates to production systems
- Emphasize real-world relevance

### 5. Motivational Closing
- End with compelling statement about impact
- Connect to bigger picture of ML systems
- Build student excitement for implementation

## When to Use

- **Always before major class implementations**
- Before complex algorithms or mathematical concepts
- When introducing new ML paradigms
- Before components that integrate with other modules

## When NOT to Use

- Before simple utility functions
- For minor method implementations within a class
- When students are already familiar with the concept
- For debugging or testing functions

## Implementation Checklist

- [ ] CONCEPT: Clear definition with analogy
- [ ] CODE STRUCTURE: Shows actual methods being built
- [ ] CONNECTIONS: Includes PyTorch/TensorFlow equivalents
- [ ] CONSTRAINTS: Lists 3-4 technical requirements
- [ ] CONTEXT: Explains specific ML applications
- [ ] Compelling closing statement
- [ ] Fits in code comment format
- [ ] Scannable and concise