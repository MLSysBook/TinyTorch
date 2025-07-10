# Setup Module

Welcome to TinyTorch! This is your first module in the Machine Learning Systems course.

## Overview

The setup module is a simple introduction to TinyTorch that displays beautiful ASCII art to get you started on your ML systems journey.

## Files

- `setup_dev.py` - Main development file with the hello_tinytorch() function
- `setup_dev.ipynb` - Jupyter notebook version (auto-generated)
- `tinytorch_flame.txt` - ASCII art file containing the TinyTorch flame design
- `tests/test_setup.py` - Simple tests for the module
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

Run the tests to verify everything works:

```bash
python3 tests/test_setup.py
```

## ASCII Art Customization

The ASCII art is loaded from `tinytorch_flame.txt`. You can customize it by:

1. **Edit the file directly**: Modify `tinytorch_flame.txt` with your own ASCII art
2. **Create your own design**: Replace the flame with your initials, logo, or any design you like

## What You'll Learn

This simple module introduces:
- Basic Python file structure
- File I/O operations
- Error handling (fallback when file not found)
- Testing with simple assertions
- The TinyTorch development workflow

## Next Steps

Once you've explored this module, you're ready to move on to the tensor module where you'll build the core data structures for TinyTorch! 