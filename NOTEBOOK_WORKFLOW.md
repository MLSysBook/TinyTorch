# TinyTorch Notebook Conversion Workflow

This document provides comprehensive instructions for converting TinyTorch Python modules to Jupyter notebooks using the TITO CLI.

## Overview

The TinyTorch project uses a sophisticated workflow where educational content is developed in Python files with special cell markers (`%%`) and then converted to interactive Jupyter notebooks using Jupytext. This approach provides:

- **Version control friendly**: Python files are easier to track in git
- **Cell-based development**: Use `%%` markers to define notebook cells
- **Automatic conversion**: TITO CLI handles the conversion seamlessly
- **Student-friendly**: Students get interactive notebooks for learning

## Quick Start

### 1. Environment Setup

**Automated Setup (Recommended):**
```bash
# Clone and navigate to repository
git clone https://github.com/your-org/TinyTorch.git
cd TinyTorch

# Run automated setup script
chmod +x setup-dev.sh
./setup-dev.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install essential dependencies
pip install rich>=13.0.0 jupytext>=1.14.0 pytest>=7.0.0 numpy>=1.21.0

# Install full requirements (optional)
pip install -r requirements.txt

# Install TinyTorch in development mode
pip install -e .
```

### 2. Environment Verification

```bash
# Activate virtual environment
source .venv/bin/activate

# Run environment diagnosis
python -m tito.main system doctor
```

This should show:
- ✅ Python 3.8+ detected
- ✅ Virtual environment active
- ✅ Essential dependencies installed

### 3. Convert Modules to Notebooks

**Convert Single Module:**
```bash
python -m tito.main module notebooks --module 03_activations
```

**Convert All Modules:**
```bash
python -m tito.main module notebooks
```

**Dry Run (Preview):**
```bash
python -m tito.main module notebooks --dry-run
```

## Detailed Workflow

### Python Module Structure

Python modules are located in `modules/source/` with the following structure:

```
modules/source/
├── 01_setup/
│   ├── setup_dev.py        # Python module with %% cell markers
│   ├── module.yaml         # Module metadata
│   └── README.md          # Module documentation
├── 02_tensor/
│   ├── tensor_dev.py       # Enhanced with educational content
│   └── ...
├── 03_activations/
│   ├── activations_dev.py  # Enhanced with educational content
│   └── ...
```

### Cell Markers in Python Files

Use `%%` to define notebook cells in Python files:

```python
# %% [markdown]
"""
# Activations - Nonlinearity in Neural Networks

Welcome to the Activations module!
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
"""
## Understanding ReLU Activation
"""

# %%
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)
```

### TITO CLI Commands

The TITO CLI provides comprehensive module management:

```bash
# Get help
python -m tito.main --help
python -m tito.main module --help
python -m tito.main module notebooks --help

# System commands
python -m tito.main system info        # System information
python -m tito.main system doctor      # Environment diagnosis

# Module management
python -m tito.main module status      # Module status
python -m tito.main module notebooks   # Convert to notebooks
python -m tito.main module test        # Run tests
python -m tito.main module export      # Export to package

# Notebook conversion
python -m tito.main module notebooks --module 02_tensor    # Single module
python -m tito.main module notebooks --dry-run             # Preview only
python -m tito.main module notebooks --force               # Force rebuild
```

## Working with Generated Notebooks

### Opening Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Navigate to modules/source/03_activations/
# Open activations_dev.ipynb
```

### Student Workflow

1. **Get the repository:**
   ```bash
   git clone https://github.com/your-org/TinyTorch.git
   cd TinyTorch
   ```

2. **Setup environment:**
   ```bash
   ./setup-dev.sh
   source .venv/bin/activate
   ```

3. **Convert modules to notebooks:**
   ```bash
   python -m tito.main module notebooks
   ```

4. **Work with notebooks:**
   ```bash
   jupyter lab
   ```

### Developer Workflow

1. **Edit Python modules** in `modules/source/*/` directories
2. **Test conversion:** `python -m tito.main module notebooks --dry-run`
3. **Convert to notebooks:** `python -m tito.main module notebooks`
4. **Export to package:** `python -m tito.main module export`
5. **Run tests:** `python -m tito.main module test`

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'rich'`**
```bash
# Solution: Install essential dependencies
pip install rich>=13.0.0 jupytext>=1.14.0 pytest>=7.0.0
```

**Issue: `jupytext not found`**
```bash
# Solution: Install jupytext
pip install jupytext>=1.14.0
```

**Issue: `Virtual environment not activated`**
```bash
# Solution: Activate virtual environment
source .venv/bin/activate
```

**Issue: `No *_dev.py files found`**
```bash
# Check you're in the right directory
pwd  # Should be /path/to/TinyTorch

# Check modules exist
ls modules/source/
```

**Issue: Architecture mismatch with NumPy**
```bash
# Create fresh virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy>=1.21.0,<2.0.0
```

### Environment Diagnosis

Always start troubleshooting with the doctor command:

```bash
python -m tito.main system doctor
```

This will show:
- Python version and virtual environment status
- Installed dependencies and versions
- Module structure validation
- Actionable recommendations

### Getting Help

```bash
# CLI help
python -m tito.main --help

# Specific command help
python -m tito.main module notebooks --help

# Environment diagnosis
python -m tito.main system doctor

# Module status
python -m tito.main module status
```

## CI/CD Integration

The repository includes automated testing of the notebook conversion workflow:

- **On every push**: Tests notebook conversion across Python 3.8-3.12
- **Validates**: Generated notebooks have correct structure
- **Archives**: Generated notebooks as artifacts
- **Student workflow**: Tests the complete student setup process

See `.github/workflows/test-notebooks.yml` for implementation details.

## Advanced Usage

### Custom Module Conversion

```bash
# Convert specific modules only
python -m tito.main module notebooks --module 02_tensor --module 03_activations

# Force rebuild existing notebooks
python -m tito.main module notebooks --force

# Preview what would be converted
python -m tito.main module notebooks --dry-run
```

### Batch Operations

```bash
# Convert all modules and export to package
python -m tito.main module notebooks && python -m tito.main module export --all

# Test after conversion
python -m tito.main module notebooks && python -m tito.main module test --all
```

## File Locations

After successful conversion, you'll find:

```
modules/source/03_activations/
├── activations_dev.py      # Original Python module
├── activations_dev.ipynb   # Generated notebook ← NEW!
├── module.yaml             # Module metadata
└── README.md              # Module documentation
```

The generated `.ipynb` files are fully functional Jupyter notebooks that can be opened in Jupyter Lab, VS Code, or any notebook environment.

## Summary

The TinyTorch notebook conversion workflow provides a robust, automated way to transform educational Python modules into interactive notebooks. The TITO CLI handles all the complexity, providing students with a simple, reliable workflow to get started with interactive learning.

**Key Benefits:**
- **One-command setup**: `./setup-dev.sh`
- **Reliable conversion**: TITO CLI with comprehensive error handling
- **Student-friendly**: Clear error messages and helpful guidance
- **Developer-friendly**: Automated CI/CD testing across Python versions
- **Production-ready**: Used in real educational environments

For questions or issues, run `python -m tito.main system doctor` for comprehensive environment diagnosis.