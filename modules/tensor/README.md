# 🔥 Module 02: Tensor Implementation

Welcome to the Tensor module! This is where you'll build the foundation of TinyTorch by implementing the core Tensor class.

## 🎯 Learning Objectives

By the end of this module, you will:
- ✅ Understand what tensors are and why they're fundamental to ML
- ✅ Implement a basic Tensor class with core operations
- ✅ Handle tensor shapes, data types, and basic arithmetic
- ✅ Test your implementation with automated tests
- ✅ Be ready to build neural networks in the next module

## 📋 Module Structure

```
modules/tensor/
├── README.md                    # 📖 This file - Module overview
├── notebook/                    # 📓 Interactive development
│   └── tensor_dev.ipynb        # Main development notebook
├── tutorials/                   # 🎓 Step-by-step learning guides
│   ├── 01_tensor_basics.ipynb  # Introduction to tensors
│   └── 02_tensor_ops.ipynb     # Tensor operations tutorial
├── test_tensor.py              # 🧪 Automated tests
├── check_tensor.py             # ✅ Manual verification
└── solutions/                  # 🔑 Reference solutions (instructors)
    └── solution_tensor.py
```

## 🚀 Getting Started

### Step 1: Environment Setup
```bash
# Make sure your environment is activated
source .venv/bin/activate

# Navigate to the tensor module
cd modules/tensor/
```

### Step 2: Read the Tutorials (Recommended)
```bash
# Start Jupyter
tito jupyter --lab

# Then open these notebooks in order:
# 1. tutorials/01_tensor_basics.ipynb - Learn about tensors
# 2. tutorials/02_tensor_ops.ipynb - Learn about operations
```

### Step 3: Implement Your Tensor Class
```bash
# Open the main development notebook
# Navigate to: notebook/tensor_dev.ipynb

# Work through the implementation step by step:
# 1. Basic Tensor class with constructor
# 2. Properties (shape, size, dtype)
# 3. Arithmetic operations (add, subtract, multiply, divide)
# 4. Utility methods (reshape, transpose, sum, mean, etc.)
```

### Step 4: Test Your Implementation
```bash
# Run automated tests
tito test --module tensor

# Run manual verification
python check_tensor.py

# Test integration
python -c "from tinytorch.core.tensor import Tensor; print('Success!')"
```

### Step 5: Submit Your Work
```bash
# Submit when ready
tito submit --module tensor

# Move to next module
cd ../mlp/
cat README.md
```

## 📚 What You'll Implement

### Core Tensor Class
Your Tensor class should support:

1. **Constructor**: Initialize with data and optional dtype
2. **Properties**: shape, size, dtype
3. **Arithmetic**: +, -, *, / with tensors and scalars
4. **Utilities**: reshape, transpose, sum, mean, max, min, flatten
5. **Error Handling**: Proper exceptions for invalid operations

### Example Usage
```python
# Create tensors
a = Tensor([1, 2, 3])
b = Tensor([[1, 2], [3, 4]])

# Basic operations
c = a + 5          # Scalar addition
d = a * b          # Element-wise multiplication
e = b.reshape(1, 4)  # Reshape
f = b.sum(axis=0)    # Sum along axis
```

## 🧪 Testing Your Implementation

### Automated Tests
```bash
tito test --module tensor
```
This runs comprehensive tests for:
- ✅ Tensor creation (scalar, list, matrix)
- ✅ Arithmetic operations (add, subtract, multiply, divide)
- ✅ Utility methods (reshape, transpose, sum, mean, etc.)
- ✅ Error handling (invalid inputs, operations)

### Manual Verification
```bash
python check_tensor.py
```
This provides human-readable feedback on:
- 📊 Progress tracking
- 🔍 Specific test results
- 💡 Helpful error messages
- 📋 Next steps guidance

## 🎯 Success Criteria

Your implementation is complete when:

1. **All automated tests pass**: `tito test --module tensor` shows ✅
2. **Manual verification passes**: `python check_tensor.py` shows all tests passed
3. **Integration works**: Can import and use Tensor from tinytorch.core.tensor
4. **Error handling works**: Invalid operations raise appropriate exceptions

## 📖 Learning Resources

### Tutorial Notebooks
- **01_tensor_basics.ipynb**: Introduction to tensor concepts
- **02_tensor_ops.ipynb**: Deep dive into tensor operations

### Key Concepts to Understand
- **Tensors as multi-dimensional arrays**
- **Shape and size properties**
- **Element-wise operations**
- **Broadcasting rules**
- **Reduction operations (sum, mean, etc.)**

## 🔧 Implementation Tips

### Start Simple
1. **Basic constructor** that handles lists and numpy arrays
2. **Properties** (shape, size, dtype) that return correct values
3. **Simple arithmetic** (add, multiply) with tensors and scalars
4. **Basic utilities** (reshape, sum) that work correctly

### Test Frequently
- Run tests after each major feature
- Use the manual verification for detailed feedback
- Test edge cases (empty tensors, different dtypes)

### Common Pitfalls
- **Shape handling**: Make sure 0D tensors are handled correctly
- **Data types**: Ensure dtype parameter works properly
- **Error messages**: Provide clear, helpful error messages
- **Memory efficiency**: Avoid unnecessary copies

## 🚀 Next Steps

Once you complete this module:

1. **Move to MLP module**: `cd ../mlp/`
2. **Build neural networks**: Use your tensors to create layers
3. **Add autograd**: Implement automatic differentiation
4. **Train models**: Put it all together with training loops

## 💡 Need Help?

### Common Issues
- **Import errors**: Make sure your Tensor class is in `tinytorch/core/tensor.py`
- **Test failures**: Check the specific error messages for guidance
- **Shape mismatches**: Verify your shape calculations are correct

### Getting Support
- **Check tutorials**: The tutorial notebooks have detailed explanations
- **Run verification**: `python check_tensor.py` provides specific feedback
- **Review tests**: The test files show exactly what's expected

## 🎉 You're Building Something Amazing!

This Tensor implementation will be the foundation for everything else in TinyTorch:
- **Neural networks** will use your tensors for weights and activations
- **Optimizers** will update your tensors during training
- **Data loading** will convert datasets into your tensors
- **And much more!**

Good luck, and happy coding! 🔥 