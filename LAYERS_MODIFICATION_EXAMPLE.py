#!/usr/bin/env python3
"""
Example: How to Modify Existing Layers to Use Backend System

This shows the minimal changes needed to existing tinytorch.core.layers
to support the backend dispatch system for competition optimization.
"""

# This is how you would modify the existing matmul function in layers_dev.py:

# BEFORE (Original Implementation):
def matmul_original(a, b):
    """Original matrix multiplication implementation"""
    return a.data @ b.data  # Simple NumPy operation

# AFTER (Backend-Aware Implementation):  
def matmul_backend_aware(a, b):
    """Matrix multiplication with backend dispatch"""
    from kernels_dev import get_backend  # Import the backend system
    
    backend = get_backend()
    result_data = backend.matmul(a.data, b.data)
    
    from tensor_dev import Tensor
    return Tensor(result_data)

# The Dense layer automatically inherits the optimization!
# NO CHANGES needed to Dense.forward() method

print("""
🔧 MODIFICATION STRATEGY:

1. MINIMAL CHANGES: Only modify the low-level operation functions
   - matmul() gets backend dispatch
   - conv2d() gets backend dispatch  
   - Other layers inherit optimizations automatically

2. PRESERVE EXISTING APIs: No changes to:
   - Dense layer implementation
   - Module base class
   - Training loops
   - Student-facing code

3. ADDITIVE OPTIMIZATIONS: 
   - Add backend system alongside existing code
   - Default to naive backend (safe for learning)
   - Students opt-in to optimized backend for competition

4. EXPORT COMPATIBILITY:
   - `tito module complete` still works
   - NBGrader integration preserved
   - Learning progression unchanged

RESULT: Students can run EXACTLY THE SAME CODE with 10-100x speedup
just by calling set_backend('optimized') before their training loop!
""")

# Example usage in student code:
example_student_code = '''
# Student writes this code normally (learning mode):
import tinytorch
model = MyNetwork()
optimizer = Adam(model.parameters())

# Train normally with naive backend (default)
for epoch in range(10):
    loss = train_epoch(model, data, optimizer)
    print(f"Epoch {epoch}: {loss:.4f}")

# NOW COMPETITION MODE - same code, much faster!
tinytorch.set_backend("optimized")  # Only line that changes!

# Re-run the EXACT SAME training code - 10x faster!
for epoch in range(10):  
    loss = train_epoch(model, data, optimizer)  # Same function!
    print(f"Fast Epoch {epoch}: {loss:.4f}")
'''

print("💡 STUDENT EXPERIENCE:")
print(example_student_code)