# 🚀 TinyTorch Quick Start Guide

Get up and running with TinyTorch in 10 minutes! This guide will walk you through setting up your environment and implementing your first ML component.

## 📋 Prerequisites

- **Python 3.8+** (check with `python --version`)
- **Git** for cloning the repository
- **Basic Python knowledge** (functions, classes, imports)
- **Jupyter** familiarity (we'll install it for you)

## ⚡ 5-Minute Setup

### Step 1: Clone and Navigate
```bash
git clone https://github.com/tinytorch/TinyTorch.git
cd TinyTorch
```

### Step 2: Create Virtual Environment
```bash
# Create isolated environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# OR: .venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# This installs: numpy, matplotlib, jupyter, nbdev, pytest, and more
```

### Step 4: Verify Installation
```bash
# Check TinyTorch CLI
python bin/tito.py --version

# Check environment
python bin/tito.py info
```

**✅ If you see system information, you're ready to go!**

## 🎯 Your First Module: Setup

Let's implement your first TinyTorch component to understand the workflow.

### Step 1: Navigate to Setup Module
```bash
cd modules/setup/
```

### Step 2: Read the Module Overview
```bash
# This explains what you'll build
cat README.md
```

### Step 3: Open the Development Notebook
```bash
# Start Jupyter Lab
jupyter lab setup_dev.ipynb

# The notebook will open in your browser at http://localhost:8888
```

### Step 4: Follow the Notebook
The notebook guides you through:
1. **Environment check** - Verify everything works
2. **Hello function** - Implement `hello_tinytorch()`
3. **Export process** - Learn the `#| export` directive
4. **Testing** - Run interactive tests

### Step 5: Export Your Code
```bash
# Back in terminal (new tab/window):
cd /path/to/TinyTorch  # Navigate back to root

# Export notebook code to Python package
python bin/tito.py sync
```

This creates `tinytorch/core/utils.py` with your function!

### Step 6: Test Your Implementation
```bash
# Run automated tests
python bin/tito.py test --module setup

# Should show: ✅ All tests passed!
```

### Step 7: Verify Integration
```bash
# Test that your function is importable
python -c "from tinytorch.core.utils import hello_tinytorch; print(hello_tinytorch())"
```

**🎉 Congratulations! You've completed your first module!**

## 🔄 The TinyTorch Development Workflow

Now you understand the core workflow that you'll use for every module:

```mermaid
graph LR
    A[Read README] --> B[Open Notebook]
    B --> C[Implement Code]
    C --> D[Mark #|export]
    D --> E[Test in Notebook]
    E --> F[Export with tito sync]
    F --> G[Run Tests]
    G --> H{Tests Pass?}
    H -->|No| C
    H -->|Yes| I[Next Module]
```

### Key Commands Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `cat README.md` | Read module overview | Start of each module |
| `jupyter lab [module]_dev.ipynb` | Open development environment | Implement code |
| `python bin/tito.py sync` | Export notebooks to package | After coding in notebook |
| `python bin/tito.py test --module [name]` | Test your implementation | Verify correctness |
| `python bin/tito.py info` | Check system status | Anytime |

## 🧗 Next Steps: Building Your First Tensor

Ready for the real challenge? Let's build the foundation of TinyTorch!

### Step 1: Move to Tensor Module
```bash
cd modules/tensor/
```

### Step 2: Read the Overview
```bash
cat README.md
```

### Step 3: Start Building
```bash
jupyter lab tensor_dev.ipynb
```

In this module, you'll implement:
- **Tensor class** - Multi-dimensional arrays
- **Arithmetic operations** - Addition, multiplication
- **Utility methods** - Reshape, transpose, sum
- **Error handling** - Robust edge cases

### Expected Timeline
- **Setup module**: 15-30 minutes
- **Tensor module**: 2-4 hours
- **Complete course**: 40-80 hours (over several weeks)

## 🛠️ Development Environment Tips

### Jupyter Lab Shortcuts
- **Shift + Enter**: Run cell and move to next
- **Ctrl + Enter**: Run cell and stay
- **A**: Insert cell above
- **B**: Insert cell below
- **DD**: Delete cell

### TinyTorch Best Practices
1. **Read module READMEs first** - Understand objectives
2. **Test frequently** - Don't implement everything at once
3. **Use `#| export` correctly** - Only mark code for the package
4. **Write docstrings** - Document your functions
5. **Check tests** - They show exactly what's expected

### Common Issues and Solutions

#### "ModuleNotFoundError"
```bash
# Make sure you've activated your environment
source .venv/bin/activate

# And exported your notebooks
python bin/tito.py sync
```

#### "Command 'tito' not found"
```bash
# Use the full path
python bin/tito.py [command]

# Make sure you're in the TinyTorch root directory
pwd  # Should end with /TinyTorch
```

#### "Tests failing"
```bash
# Run with verbose output to see details
python bin/tito.py test --module setup -v

# Check the specific error messages
# Fix in the notebook, then re-export and test
```

#### "Jupyter not starting"
```bash
# Make sure it's installed
pip install jupyter jupyterlab

# Try classic notebook instead
jupyter notebook modules/setup/setup_dev.ipynb
```

## 📚 Learning Path

### Beginner Track (Start Here)
1. **Setup** (30 min) - Learn the workflow
2. **Tensor** (4 hours) - Core data structures
3. **MLP** (6 hours) - Basic neural networks

### Intermediate Track
4. **Autograd** (8 hours) - Automatic differentiation
5. **CNN** (6 hours) - Convolutional networks
6. **Training** (4 hours) - Training loops and optimizers

### Advanced Track
7. **Data** (3 hours) - Efficient data loading
8. **Kernels** (10 hours) - Custom GPU operations
9. **Compression** (6 hours) - Model optimization

### Expert Track
10. **Profiling** (4 hours) - Performance analysis
11. **Benchmarking** (4 hours) - Systematic evaluation
12. **Config** (2 hours) - Configuration management
13. **MLOps** (8 hours) - Production systems

## 🎯 Success Metrics

You'll know you're succeeding when:

### After Each Module
- ✅ All tests pass: `python bin/tito.py test --module [name]`
- ✅ Code is importable: `from tinytorch.core.X import Y`
- ✅ You understand what you built
- ✅ Ready for the next module

### After Major Milestones
- **Tensor complete**: Can create and manipulate multi-dimensional arrays
- **MLP complete**: Can build and train simple neural networks
- **CNN complete**: Can implement convolutional architectures
- **Course complete**: You've built a complete ML framework!

## 💡 Getting Help

### Self-Help Resources
1. **Module READMEs** - Detailed explanations and tips
2. **Test files** - Show exactly what's expected
3. **Notebook examples** - See reference implementations
4. **Error messages** - Often contain helpful guidance

### Debugging Workflow
1. **Read the error** - Understand what failed
2. **Check the test** - See what was expected
3. **Review the notebook** - Look for implementation issues
4. **Test incrementally** - Don't implement everything at once
5. **Use print statements** - Debug your logic

### Community Support
- **GitHub Issues** - Report bugs or ask questions
- **Discussions** - Share tips and solutions
- **Study Groups** - Find fellow learners

## 🚀 Ready to Build?

You now have everything you need to start building ML systems from scratch!

```bash
# Quick reminder of the workflow:
cd modules/setup/          # Navigate to module
cat README.md             # Read overview
jupyter lab setup_dev.ipynb   # Start implementing
# [work in notebook]
python bin/tito.py sync   # Export to package
python bin/tito.py test --module setup  # Test implementation
```

**Welcome to TinyTorch! Let's build something amazing! 🔥**

---

## 📖 What's Next?

After completing the QUICKSTART:
- **Dive deeper**: Read `COURSE_GUIDE.md` for the complete curriculum
- **Understand the structure**: Review `STRUCTURE_PROPOSAL.md` for architecture details
- **Stay updated**: Check `README.md` for the latest information

Happy coding! 🎉 