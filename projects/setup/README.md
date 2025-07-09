# Project Setup: Environment & Onboarding

Welcome to Tiny🔥Torch! This setup project gets your development environment ready and introduces you to the workflow you'll use throughout the course.

## 🎯 Learning Objectives

By the end of this project, you will:
- ✅ Have a fully working development environment
- ✅ Understand the `tito` CLI workflow  
- ✅ Know how to check system status and run tests
- ✅ Have implemented your first TinyTorch component
- ✅ Be ready to build the entire ML system

## 📋 Setup Checklist

### Step 1: One-Command Environment Setup

**The Only Command You Need** 🚀
```bash
bin/activate-tinytorch.sh
```

This smart script handles everything:
- 🆕 **First time**: Creates environment, installs dependencies, activates it
- 💤 **Already exists**: Just activates the existing environment  
- ✅ **Already active**: Already good to go!

**Super Simple Workflow!** ⚡
```bash
# First time or any time - just run:
bin/activate-tinytorch.sh

# Then you're ready:
tito info      # Check system status
tito test      # Run tests
tito doctor    # Diagnose any issues

# When done:
deactivate     # Exit the environment
```

> **💡 Dead simple!** One script does everything - no setup commands, no complexity.

### Step 2: System Verification

**Test the CLI**
```bash
tito --version    # (after running bin/activate-tinytorch.sh)
tito info
```
You should see the Tiny🔥Torch banner and system status.

**Check Project Structure**
```bash
tito info --show-architecture
```
This shows the system architecture you'll be building.

### Step 3: Development Workflow

**Learn Status Commands**
```bash
# Check implementation status
tito info

# Test specific project (you'll use this constantly)
tito test --project setup

# Get help on any command
tito --help
```

## 🚀 Hello World Implementation

Now let's implement your first TinyTorch component! You'll add a simple greeting function to the system.

**Your Task**: Implement a `hello_tinytorch()` function in `tinytorch/core/utils.py`

**Step 1: Open the utils file**
```bash
# Look at the current file
cat tinytorch/core/utils.py
```

**Step 2: Implement the function**

Add this function to `tinytorch/core/utils.py`:

```python
def hello_tinytorch() -> str:
    """
    Return a greeting message for new TinyTorch users.
    
    Returns:
        A welcoming message string
    """
    return "🔥 Welcome to TinyTorch! Ready to build ML systems from scratch! 🔥"
```

**Step 3: Test your implementation**
```bash
# Run the pytest test suite
python3 -m pytest projects/setup/test_setup.py -v

# Or run the CLI test command
tito test --project setup

# Run comprehensive setup check
python3 projects/setup/check_setup.py
```

## 🧪 Verification & Next Steps

**Run the setup checker**
```bash
python3 projects/setup/check_setup.py
```

This will verify:
- ✅ Environment is correctly configured
- ✅ All dependencies are installed
- ✅ CLI commands work properly
- ✅ Your hello world function is implemented
- ✅ You're ready for the next project

**Expected Output:**
```
🔥 TinyTorch Setup Verification 🔥
==================================

✅ Python version: 3.x.x (compatible)
✅ Dependencies: All installed correctly
✅ CLI commands: Working properly
✅ hello_tinytorch(): Implemented correctly
✅ Test suite: All tests passing

🎉 Setup complete! You're ready to build an ML system from scratch.

Next steps:
  cd ../tensor/
  cat README.md

You can now submit this project:
  tito submit --project setup
```

## 📚 What You've Learned

- **Environment setup**: Python, dependencies, development tools
- **CLI workflow**: Using `tito` commands for testing and status checks
- **Project structure**: How code is organized in TinyTorch
- **Implementation pattern**: Where to write code and how to test it
- **Verification process**: Using automated checkers to validate your work

## 🎯 Next Project

Once setup passes, move to your first real implementation:

```bash
cd ../tensor/
cat README.md  # Start building the core tensor system
```

---

**Need help?** Check the main README.md or ask in office hours! 