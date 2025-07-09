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

### Step 1: Environment Setup

**Check Python Version**
```bash
python3 --version  # Should be 3.8 or higher
```

**Create Virtual Environment** (REQUIRED)
```bash
# Create isolated environment for TinyTorch
python3 -m venv tinytorch-env

# Activate it (you'll do this every time)
source tinytorch-env/bin/activate  # macOS/Linux
# OR on Windows: tinytorch-env\Scripts\activate

# Verify you're in the virtual environment
which python  # Should show path with tinytorch-env
```

**Install Dependencies**
```bash
# Make sure you're in the activated virtual environment!
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify Installation**
```bash
python3 -c "import numpy, matplotlib, yaml; print('✅ Core dependencies installed')"
```

> **⚠️ Important**: Always activate your virtual environment before working:
> ```bash
> source tinytorch-env/bin/activate  # Run this every time you start working
> ```

### Step 2: System Verification

**Test the CLI**
```bash
python3 bin/tito.py --version
python3 bin/tito.py info
```
You should see the Tiny🔥Torch banner and system status.

**Check Project Structure**
```bash
python3 bin/tito.py info --show-architecture
```
This shows the system architecture you'll be building.

### Step 3: Development Workflow

**Learn Status Commands**
```bash
# Check implementation status
python3 bin/tito.py info

# Test specific project (you'll use this constantly)
python3 bin/tito.py test --project setup

# Get help on any command
python3 bin/tito.py train --help
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
python3 bin/tito.py test --project setup

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
  python3 bin/tito.py submit --project setup
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