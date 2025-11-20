# Binder & Colab Verification Guide

This guide helps you verify that Binder and Colab links are working correctly.

## Quick Verification Checklist

- [ ] Binder build completes successfully
- [ ] TinyTorch package imports correctly in Binder
- [ ] Colab can clone repository and install dependencies
- [ ] Launch buttons appear on notebook pages in documentation
- [ ] All three deployment environments work (JupyterHub, Colab, Local)

## Step-by-Step Verification

### 1. Test Binder Build

**Direct URL Test:**
```
https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main
```

**What to check:**
- Build completes without errors (may take 2-5 minutes first time)
- JupyterLab launches successfully
- No import errors in terminal or notebook

**Test in Binder Notebook:**
```python
# Test 1: Import TinyTorch
import tinytorch
print(f"TinyTorch version: {tinytorch.__version__}")

# Test 2: Verify modules are accessible
import os
assert os.path.exists("modules"), "Modules directory not found"
assert os.path.exists("assignments"), "Assignments directory not found"

# Test 3: Test basic functionality
from tinytorch.core import Tensor
x = Tensor([1, 2, 3])
print(f"Tensor created: {x}")
```

### 2. Test Colab Integration

**For a specific notebook:**
```
https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/assignments/source/02_tensor/02_tensor.ipynb
```

**What to check:**
- Notebook opens in Colab
- Can run cells without errors
- Dependencies install correctly

**Colab Setup Cell (add to notebooks if needed):**
```python
# Install TinyTorch
!pip install -e /content/TinyTorch

# Verify installation
import tinytorch
print("TinyTorch installed successfully!")
```

### 3. Verify Launch Buttons in Documentation

**Check that launch buttons appear:**
1. Build the site: `cd site && jupyter-book build .`
2. Open `_build/html/index.html` in browser
3. Navigate to any page with notebooks
4. Look for "Launch" buttons in the top-right corner

**Expected buttons:**
- 🚀 Launch Binder
- 🔵 Open in Colab
- 📥 Download notebook

### 4. Test All Three Deployment Environments

As documented in `paper/paper.tex`, TinyTorch supports:

#### A. JupyterHub (Institutional)
- Requires: 8-core/32GB server
- Supports: ~50 concurrent students
- Setup: Install via `pip install tinytorch` or mount repository

#### B. Google Colab (Zero Installation)
- Best for: MOOCs and self-paced learning
- Setup: Automatic via launch buttons
- Verify: Test with sample notebooks

#### C. Local Installation
- Best for: Self-paced learning and development
- Setup: `pip install tinytorch`
- Verify: Run `python -c "import tinytorch; print(tinytorch.__version__)"`

## Common Issues & Solutions

### Issue: Binder build times out

**Solution:**
- Check `binder/requirements.txt` for unnecessary heavy dependencies
- Ensure `postBuild` script is fast (< 2 minutes)
- Consider using `environment.yml` instead if you need conda packages

### Issue: "Module not found" errors in Binder

**Solution:**
- Verify `postBuild` script runs `pip install -e .`
- Check that `pyproject.toml` is in repository root
- Ensure all dependencies are in `binder/requirements.txt`

### Issue: Colab can't access repository

**Solution:**
- Ensure repository is public (Colab can't access private repos)
- Check that notebook path is correct in URL
- Verify GitHub repository URL in `site/_config.yml`

### Issue: Launch buttons don't appear

**Solution:**
- Verify `launch_buttons` configuration in `site/_config.yml`
- Ensure repository URL and branch are correct
- Rebuild the site: `jupyter-book build . --all`

## Automated Testing

You can add a GitHub Actions workflow to test Binder builds:

```yaml
# .github/workflows/test-binder.yml
name: Test Binder Build

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  test-binder:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test Binder Build
        uses: jupyterhub/repo2docker-action@master
        with:
          image-name: tinytorch-binder-test
```

## Monitoring

**Binder Status:**
- Check build status: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main
- View build logs: Add `?urlpath=lab/tree/logs%2Fbuild.log` to URL

**Colab Status:**
- Test with sample notebooks from `assignments/` directory
- Monitor for import errors or dependency issues

## References

- [Binder Documentation](https://mybinder.readthedocs.io/)
- [Jupyter Book Launch Buttons](https://jupyterbook.org/en/stable/interactive/launchbuttons.html)
- [Google Colab GitHub Integration](https://colab.research.google.com/github/)

