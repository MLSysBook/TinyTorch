# Setup Module

Welcome to TinyTorch! This is your first module in the Machine Learning Systems course.

## Overview

The setup module is a simple introduction to TinyTorch that displays beautiful ASCII art to get you started on your ML systems journey.

## Files

- `setup_dev.py` - Main development file with the hello_tinytorch() function
- `setup_dev.ipynb` - Jupyter notebook version (auto-generated)
- `tinytorch_flame.txt` - ASCII art file containing the TinyTorch flame design
- `tests/test_setup.py` - pytest test suite for the module
- `README.md` - This file

## Usage

### Python Script
```python
from setup_dev import hello_tinytorch

hello_tinytorch()
```

### Jupyter Notebook
Open `setup_dev.ipynb` and run the cells to see the ASCII art displayed.

## Testing

Run the tests using pytest:

```bash
# Using the TinyTorch CLI (recommended)
python bin/tito.py test --module setup

# Or directly with pytest
python -m pytest modules/setup/tests/test_setup.py -v
```

### Test Coverage

The test suite includes:
- ✅ Function execution without errors
- ✅ Correct output content (ASCII art and branding)
- ✅ ASCII art file existence and content validation
- ✅ Graceful handling of missing files
- ✅ Error recovery and fallback behavior

## ASCII Art Customization

The ASCII art is loaded from `tinytorch_flame.txt`. You can customize it by:

1. **Edit the file directly**: Modify `tinytorch_flame.txt` with your own ASCII art
2. **Create your own design**: Replace the flame with your initials, logo, or any design you like

## What You'll Learn

This simple module introduces:
- Basic Python file structure
- File I/O operations
- Error handling (fallback when file not found)
- pytest testing framework and best practices
- The TinyTorch development workflow

## Next Steps

Once you've explored this module, you're ready to move on to the tensor module where you'll build the core data structures for TinyTorch! 