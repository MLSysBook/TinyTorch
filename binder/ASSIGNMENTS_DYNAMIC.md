# Assignments Are Dynamically Generated

## Important: Assignments Directory Structure

**All assignments are dynamically generated** from the `modules/` directory using `tito nbgrader` commands. The `assignments/` directory should **not** be manually maintained.

## How It Works

### Source of Truth: `modules/` Directory

The actual module content lives in:
```
modules/
├── 01_tensor/
│   └── tensor_dev.py      ← Source of truth
├── 02_activations/
│   └── activations_dev.py  ← Source of truth
└── ...
```

### Dynamic Generation: `assignments/` Directory

Assignments are **generated** from modules using:

```bash
# Generate assignment for a single module
tito nbgrader generate 01_tensor

# Generate assignments for all modules
tito nbgrader generate --all

# Generate assignments for a range
tito nbgrader generate --range 01-05
```

This creates:
```
assignments/
├── source/
│   ├── 01_tensor/
│   │   └── 01_tensor.ipynb    ← Generated from modules/01_tensor/tensor_dev.py
│   └── 02_activations/
│       └── 02_activations.ipynb ← Generated from modules/02_activations/activations_dev.py
└── release/
    └── ... (student versions, generated via 'tito nbgrader release')
```

## Process Flow

```
modules/01_tensor/tensor_dev.py
    ↓ (tito nbgrader generate)
    ↓ (jupytext converts .py → .ipynb)
    ↓ (NotebookGenerator processes with nbgrader markers)
assignments/source/01_tensor/01_tensor.ipynb
    ↓ (tito nbgrader release)
assignments/release/01_tensor/01_tensor.ipynb (student version)
```

## What This Means

1. **Don't manually edit** `assignments/source/` files - they're generated
2. **Edit modules** in `modules/` directory instead
3. **Regenerate assignments** when modules change: `tito nbgrader generate`
4. **Old assignments** (like `01_setup`) are outdated - regenerate from current modules

## Outdated Assignment: `01_setup`

The `assignments/source/01_setup/` directory is **outdated** because:
- Module 01 is now "Tensor" (`modules/01_tensor/`)
- It was created when Module 01 was "Setup" (old structure)
- Should be regenerated: `tito nbgrader generate 01_tensor`

## For Binder/Colab

**No impact** - Binder setup doesn't depend on assignment notebooks. However:
- If you want to include assignments in Binder, regenerate them first:
  ```bash
  tito nbgrader generate --all
  ```
- Students can access modules directly from `modules/` directory
- Assignments are optional - modules are the source of truth

## Best Practices

1. **Always regenerate** assignments after modifying modules
2. **Don't commit** manually edited assignment files
3. **Use `tito nbgrader generate`** to create assignments
4. **Keep modules/** as the single source of truth

## Commands Reference

```bash
# Generate assignments
tito nbgrader generate 01_tensor          # Single module
tito nbgrader generate --all             # All modules
tito nbgrader generate --range 01-05     # Range

# Release to students (removes solutions)
tito nbgrader release 01_tensor

# Generate feedback
tito nbgrader feedback 01_tensor
```

