# WORKING EXAMPLE: Activations Module with Interactive NBGrader Text Responses
# This demonstrates the complete implementation pattern for ML Systems Thinking

# %% [markdown]
"""
## 🤔 ML Systems Thinking: Interactive Reflection

Now that you've implemented ReLU, Sigmoid, Tanh, and Softmax activation functions, let's explore how these 
simple mathematical operations power real-world ML systems through focused questions requiring thoughtful analysis.

**Instructions:** 
- Provide thoughtful 150-300 word responses to each question
- Draw connections between your implementation and production ML systems
- Use specific examples from your code and real-world scenarios
- These responses will be manually graded for insight and understanding
"""

# %% [markdown] nbgrader={"grade": false, "grade_id": "systems-thinking-task-1", "locked": true, "schema_version": 3, "solution": false, "task": true}
"""
### Question 1: Computational Efficiency in Production

**Context:** Your ReLU implementation uses simple NumPy operations: `np.maximum(0, x)`, while your Softmax requires 
exponentials and normalization with overflow protection.

**Question:** In production neural networks with billions of activations computed per forward pass, every operation 
matters. How might the computational complexity differences between ReLU and Softmax impact training speed and memory 
usage in large-scale deployments? What specific optimizations do you think GPU kernels implement for these activation 
functions, and why has ReLU become the dominant choice in deep learning?

**Expected Response:** 150-300 words analyzing computational efficiency, GPU optimizations, and production performance implications.
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-response-1", "locked": false, "schema_version": 3, "solution": true, "task": false, "points": 10}
"""
=== BEGIN MARK SCHEME ===
GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Deep understanding of computational complexity differences between activation functions
- Insightful analysis of ReLU's efficiency advantages (no exponentials, sparse outputs, simple gradient)
- Shows awareness of GPU kernel optimizations (vectorization, memory coalescing, etc.)
- Makes specific connections to real-world performance implications in large models
- Discusses memory benefits of ReLU sparsity and vanishing gradient mitigation
- Clear, technical writing with concrete examples

GOOD (7-8 points):
- Good understanding of activation function efficiency differences
- Some awareness of GPU optimizations and production considerations
- Generally accurate technical content about ReLU advantages
- Makes some connections to real systems

SATISFACTORY (5-6 points):
- Basic understanding of computational differences between activations
- Limited insight into production optimizations
- General discussion without specific technical depth

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of efficiency implications
- Unclear or inaccurate technical content
- No connection to real-world systems

NO CREDIT (0 points):
- No response or completely off-topic
- Factually incorrect fundamental concepts
=== END MARK SCHEME ===

**Your Response:**
[Student writes their analysis here - this cell will be editable by students]
"""

# %% [markdown] nbgrader={"grade": false, "grade_id": "systems-thinking-task-2", "locked": true, "schema_version": 3, "solution": false, "task": true}
"""
### Question 2: Numerical Stability in Large Systems

**Context:** Your Softmax implementation includes overflow protection by clipping large values, preventing 
`exp(x)` from causing numerical overflow.

**Question:** In production systems training massive language models with hundreds of layers, numerical instability 
can cascade and destroy training. How do frameworks like PyTorch handle numerical stability for activation functions 
at scale? What happens when a single unstable activation propagates through a deep network, and how do production 
systems prevent this? Consider both forward pass stability and gradient computation implications.

**Expected Response:** 150-300 words discussing numerical stability challenges, cascading effects, and production solutions.
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-response-2", "locked": false, "schema_version": 3, "solution": true, "task": false, "points": 10}
"""
=== BEGIN MARK SCHEME ===
GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Demonstrates deep understanding of numerical stability challenges in deep networks
- Explains cascading effects of instability through multiple layers
- Shows awareness of production solutions (gradient clipping, mixed precision, normalization)
- Discusses both forward and backward pass stability considerations
- Makes connections to real framework implementations and training practices
- Technical accuracy and clear communication

GOOD (7-8 points):
- Good understanding of numerical stability issues
- Some awareness of cascading effects and production solutions
- Generally accurate technical content
- Makes some connections to real systems

SATISFACTORY (5-6 points):
- Basic understanding of stability problems
- Limited insight into production solutions
- General discussion without deep technical analysis

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of stability implications
- Unclear or inaccurate analysis

NO CREDIT (0 points):
- No response or off-topic content
=== END MARK SCHEME ===

**Your Response:**
[Student writes their analysis here - this cell will be editable by students]
"""

# %% [markdown] nbgrader={"grade": false, "grade_id": "systems-thinking-task-3", "locked": true, "schema_version": 3, "solution": false, "task": true}
"""
### Question 3: Hardware Abstraction and API Design

**Context:** Your activation functions use callable classes (`relu(x)`) that provide a consistent interface 
regardless of the underlying mathematical complexity.

**Question:** Modern ML frameworks must run the same activation code on CPUs, GPUs, TPUs, and other specialized 
hardware. How does your simple, consistent API design enable this hardware flexibility? What challenges do 
framework designers face when ensuring that `relu(x)` produces identical results whether running on a laptop 
CPU or a datacenter GPU cluster? Consider precision, parallelization, and hardware-specific optimizations.

**Expected Response:** 150-300 words analyzing hardware abstraction, cross-platform consistency, and framework design challenges.
"""

# %% [markdown] nbgrader={"grade": true, "grade_id": "systems-thinking-response-3", "locked": false, "schema_version": 3, "solution": true, "task": false, "points": 10}
"""
=== BEGIN MARK SCHEME ===
GRADING CRITERIA (10 points total):

EXCELLENT (9-10 points):
- Insightful analysis of hardware abstraction principles in ML frameworks
- Understands challenges of cross-platform consistency (precision differences, threading, etc.)
- Shows awareness of hardware-specific optimizations while maintaining API consistency
- Discusses specific examples of framework design decisions
- Makes connections to real-world deployment scenarios
- Clear technical communication

GOOD (7-8 points):
- Good understanding of hardware abstraction concepts
- Some awareness of cross-platform challenges
- Generally accurate technical content about framework design

SATISFACTORY (5-6 points):
- Basic understanding of hardware differences
- Limited insight into framework design challenges
- General discussion without specific examples

NEEDS IMPROVEMENT (1-4 points):
- Minimal understanding of hardware abstraction
- Unclear analysis of framework challenges

NO CREDIT (0 points):
- No response or inaccurate fundamental concepts
=== END MARK SCHEME ===

**Your Response:**
[Student writes their analysis here - this cell will be editable by students]
"""

# %% [markdown]
"""
**💡 Systems Insight**: The activation functions you've implemented are computational primitives that must work reliably across every major computing platform powering modern AI. Your simple mathematical operations translate into highly optimized kernel code that processes trillions of activations daily in production ML systems.

From your `np.maximum(0, x)` ReLU to the complex exponentials in Softmax, each operation represents careful trade-offs between mathematical expressiveness, computational efficiency, and numerical stability that framework designers have refined over decades of ML system evolution.
"""

print("✅ Example implementation complete!")
print("This demonstrates the complete pattern for NBGrader text response cells")
print("with proper metadata, grading rubrics, and mark schemes.")