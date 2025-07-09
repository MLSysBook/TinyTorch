# 🔥 Module 01: Setup & Environment

Welcome to your first TinyTorch module! This setup project gets your development environment ready and introduces you to the workflow you'll use throughout the course.

## 🎯 Learning Objectives

By the end of this module, you will:
- ✅ Have a fully working development environment
- ✅ Understand the `tito` CLI workflow  
- ✅ Know how to check system status and run tests
- ✅ Have implemented your first TinyTorch component
- ✅ Be ready to build the entire ML system

## 📋 Module Structure

```
modules/setup/
├── README.md                    # 📖 This file - Module overview
├── notebook/                    # 📓 Interactive development
│   └── setup_dev.ipynb         # Main development notebook
├── tutorials/                   # 🎓 Step-by-step learning guides
│   └── 01_setup_basics.ipynb   # Setup fundamentals tutorial
├── test_setup.py               # 🧪 Automated tests
├── check_setup.py              # ✅ Manual verification
├── create_env.py               # 🔧 Environment creation
├── QUICKSTART.md               # ⚡ Quick start guide
└── solutions/                  # 🔑 Reference solutions (instructors)
    └── solution_setup.py
```

## 🚀 Getting Started

### Step 1: Environment Setup
```bash
# The only command you need!
source bin/activate-tinytorch.sh
```

This smart script handles everything:
- 🆕 **First time**: Creates environment, installs dependencies, activates it
- 💤 **Already exists**: Just activates the existing environment  
- ✅ **Already active**: Already good to go!

**Important**: Use `source` (not `./`) to activate in your current shell!

### Step 2: System Verification
```bash
# Check that everything is working
tito --version
tito info
tito info --show-architecture
```

### Step 3: Read the Tutorial (Recommended)
```bash
# Start Jupyter
tito jupyter --lab

# Then open this tutorial:
# tutorials/01_setup_basics.ipynb - Learn the workflow
```

### Step 4: Implement Your First Function
```bash
# Open the main development notebook
# Navigate to: notebook/setup_dev.ipynb

# Work through the implementation step by step:
# 1. Environment check
# 2. Import verification
# 3. Function implementation
# 4. Testing and validation
```

### Step 5: Test Your Implementation
```bash
# Run automated tests
tito test --module setup

# Run manual verification
python check_setup.py

# Test integration
python -c "from tinytorch.core.utils import hello_tinytorch; print(hello_tinytorch())"
```

### Step 6: Submit Your Work
```bash
# Submit when ready
tito submit --module setup

# Move to next module
cd ../tensor/
cat README.md
```

## 📚 What You'll Implement

### Hello World Function
Your task is to implement a `hello_tinytorch()` function in `tinytorch/core/utils.py`:

```python
def hello_tinytorch() -> str:
    """
    Return a greeting message for new TinyTorch users.
    
    Returns:
        A welcoming message string
    """
    return "🔥 Welcome to TinyTorch! Ready to build ML systems from scratch! 🔥"
```

### Requirements
1. **Function signature**: Must be named `hello_tinytorch()` with return type `-> str`
2. **Return value**: Must return a non-empty string
3. **Content**: Must contain welcoming content (welcome, hello, tinytorch, ready)
4. **Branding**: Must include the 🔥 emoji (TinyTorch branding)
5. **Documentation**: Must have proper docstring explaining the function

## 🧪 Testing Your Implementation

### Automated Tests
```bash
tito test --module setup
```
This runs comprehensive tests for:
- ✅ Function exists and can be imported
- ✅ Returns correct type (string)
- ✅ Returns non-empty content
- ✅ Contains welcoming content
- ✅ Contains 🔥 emoji
- ✅ Has proper documentation

### Manual Verification
```bash
python check_setup.py
```
This provides human-readable feedback on:
- 📊 Environment status
- 🔍 Function implementation
- 💡 Specific error messages
- 📋 Next steps guidance

### Direct Testing
```python
from tinytorch.core.utils import hello_tinytorch
result = hello_tinytorch()
print(result)
```

## 🎯 Success Criteria

Your implementation is complete when:

1. **All automated tests pass**: `tito test --module setup` shows ✅
2. **Manual verification passes**: `python check_setup.py` shows all tests passed
3. **Function works correctly**: Returns proper greeting with 🔥 emoji
4. **Environment is ready**: All CLI commands work properly

## 📖 Learning Resources

### Tutorial Notebooks
- **01_setup_basics.ipynb**: Introduction to TinyTorch workflow and development process

### Key Concepts to Understand
- **Project structure**: How modules are organized
- **Development workflow**: Using notebooks, tests, and CLI
- **Testing process**: Automated vs manual verification
- **Environment management**: Virtual environments and dependencies

## 🔧 Implementation Tips

### Start Simple
1. **Read the tutorial**: Understand the workflow first
2. **Check the environment**: Make sure everything is working
3. **Implement the function**: Add it to the correct file
4. **Test frequently**: Run tests after each change

### Common Patterns
- **File location**: Always add functions to the correct module file
- **Import testing**: Test imports before implementing
- **Error messages**: Read error messages carefully for guidance
- **Documentation**: Always add proper docstrings

### Common Pitfalls
- **Wrong file**: Make sure you're editing `tinytorch/core/utils.py`
- **Syntax errors**: Check your Python syntax carefully
- **Missing requirements**: Ensure your function meets all requirements
- **Import issues**: Make sure your function can be imported

## 🚀 Next Steps

Once you complete this module:

1. **Move to Tensor module**: `cd ../tensor/`
2. **Build core tensors**: Implement the fundamental data structure
3. **Learn operations**: Add arithmetic and utility methods
4. **Test thoroughly**: Use the comprehensive testing framework

## 💡 Need Help?

### Common Issues
- **Environment not working**: Run `bin/activate-tinytorch.sh`
- **Import errors**: Check file paths and syntax
- **Test failures**: Read error messages for specific guidance
- **CLI not working**: Make sure environment is activated

### Getting Support
- **Check tutorials**: The tutorial notebook has detailed explanations
- **Run verification**: `python check_setup.py` provides specific feedback
- **Review tests**: The test files show exactly what's expected

## 🎉 You're Starting Something Amazing!

This setup module introduces you to the TinyTorch development workflow:
- **Environment management**: Virtual environments and dependencies
- **Testing framework**: Automated and manual verification
- **CLI workflow**: Using `tito` commands for development
- **Project structure**: Understanding how modules are organized

This foundation will support everything you build in the coming modules!

Good luck, and happy coding! 🔥 