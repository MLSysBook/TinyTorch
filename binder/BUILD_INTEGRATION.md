# Automatic Notebook Preparation in Site Build

## Overview

Notebook preparation is now **automatically integrated** into the site build process. When you build the site, notebooks are automatically prepared for launch buttons to work.

## How It Works

### Automatic Integration

The build process now includes notebook preparation:

```bash
cd site
make html        # Automatically prepares notebooks, then builds site
jupyter-book build .  # Also prepares notebooks automatically
```

### Build Flow

```
1. User runs: make html
   ↓
2. prepare_notebooks.sh runs automatically
   ↓
3. Script looks for existing assignment notebooks
   ↓
4. Copies them to site/chapters/modules/
   ↓
5. Jupyter Book builds site
   ↓
6. Launch buttons appear on notebook pages!
```

## What Gets Prepared

### Source: Assignment Notebooks
The script uses notebooks from `assignments/source/` (generated via `tito nbgrader generate`):

```
assignments/source/01_tensor/01_tensor.ipynb
    ↓ (copied during build)
site/chapters/modules/01_tensor.ipynb
```

### Why Assignment Notebooks?
- Already processed with nbgrader markers
- Student-ready format
- Generated from Python source files
- Consistent with assignment workflow

## Build Commands

All build commands now include notebook preparation:

### HTML Build
```bash
cd site
make html
# Or directly:
jupyter-book build .
```

### PDF Builds
```bash
make pdf-simple   # HTML-to-PDF (includes notebook prep)
make pdf          # LaTeX PDF (includes notebook prep)
```

## Manual Preparation (Optional)

If you want to prepare notebooks manually:

```bash
cd site
./prepare_notebooks.sh
```

This is useful for:
- Testing notebook preparation
- Debugging launch button issues
- Preparing notebooks before CI/CD builds

## Workflow Summary

### Complete Development → Site Flow

```
1. Development
   Edit: modules/01_tensor/tensor_dev.py
   
2. Generate Assignments
   Run: tito nbgrader generate 01_tensor
   Creates: assignments/source/01_tensor/01_tensor.ipynb
   
3. Build Site (automatic)
   Run: cd site && make html
   Auto-prepares: Copies notebooks to site/chapters/modules/
   Builds: Jupyter Book with launch buttons
   
4. Launch Buttons Work!
   Users click → Binder/Colab opens with notebook
```

## Benefits

✅ **Automatic**: No manual steps needed
✅ **Consistent**: Always uses latest notebooks
✅ **Fast**: Uses existing assignment notebooks when available
✅ **Robust**: Falls back gracefully if notebooks don't exist
✅ **Integrated**: Works with all build commands

## Troubleshooting

### Launch Buttons Don't Appear

1. **Check notebooks exist**:
   ```bash
   ls site/chapters/modules/*.ipynb
   ```

2. **Regenerate assignments**:
   ```bash
   tito nbgrader generate --all
   ```

3. **Rebuild site**:
   ```bash
   cd site && make html
   ```

### Notebooks Not Found

If you see "No notebooks prepared":
- Run `tito nbgrader generate --all` first
- Ensure modules have Python source files
- Check that `tito` command is available

### Build Fails

The prepare script is designed to fail gracefully:
- If `tito` is not available, it skips preparation
- If notebooks don't exist, it warns but continues
- Build continues even if preparation fails

## CI/CD Integration

For automated builds (GitHub Actions, etc.):

```yaml
# Example GitHub Actions step
- name: Build site
  run: |
    cd site
    make html
```

The prepare script automatically handles:
- Missing `tito` command (skips gracefully)
- Missing notebooks (warns but continues)
- Non-git environments (works in CI/CD)

## Next Steps

1. ✅ Notebook preparation integrated into build
2. ✅ Launch buttons will work automatically
3. ⏳ Test Binder/Colab links after build
4. ⏳ Verify launch buttons appear on site

