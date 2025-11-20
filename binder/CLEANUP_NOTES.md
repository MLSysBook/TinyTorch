# Cleanup Notes: Old 01_setup Module

## Issue
The `assignments/source/01_setup/` directory contains an outdated notebook from when Module 01 was "Setup". Module 01 is now "Tensor" (`modules/01_tensor/`).

## Current State
- ✅ **Current Module 01**: `modules/01_tensor/` (Tensor)
- ⚠️ **Old Assignment**: `assignments/source/01_setup/` (outdated)
- ✅ **Current Assignment**: `assignments/source/02_tensor/` (Tensor)

## Impact on Binder/Colab
**No impact** - Binder setup doesn't depend on specific assignment notebooks. The `binder/` configuration:
- Installs TinyTorch package (`pip install -e .`)
- Provides JupyterLab environment
- Students can access any notebooks in the repository

## References Updated
- ✅ `binder/VERIFY.md` - Updated Colab example to use `02_tensor`
- ✅ `site/usage-paths/classroom-use.md` - Updated nbgrader commands
- ✅ `docs/STUDENT_QUICKSTART.md` - Updated module references

## Recommendation
The old `assignments/source/01_setup/` directory can be:
1. **Removed** if no longer needed (cleanest option)
2. **Kept** if you want to preserve old assignments for reference
3. **Moved** to an archive directory if you want to keep history

**For Binder/Colab**: No action needed - they work regardless of this old directory.

