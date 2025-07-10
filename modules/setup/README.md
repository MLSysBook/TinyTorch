# Module 0: Setup

## Learning Objectives

This module teaches you the **TinyTorch development workflow**. By the end, you'll be comfortable with:

- Writing code in Jupyter notebooks using nbdev conventions
- Exporting notebook code to Python modules
- Running tests and using the TinyTorch CLI
- Understanding the development rhythm you'll use for all modules

## What You'll Build

A simple "Hello World" system that demonstrates the complete development cycle:

- Basic utility functions
- A simple `SystemInfo` class
- Tests to verify everything works
- Experience with the full notebook → export → test workflow

## Module Structure

```
modules/setup/
├── setup_dev.ipynb        # 📓 Main development notebook
├── README.md              # 📖 This guide
└── __init__.py            # 📦 Module marker
```

## Development Workflow

### 1. Work in the Notebook
```bash
cd modules/setup
jupyter lab setup_dev.ipynb
```

### 2. Export Your Code
```bash
python bin/tito.py sync
```

### 3. Test Your Implementation
```bash
python bin/tito.py test --module setup
```

### 4. Check Your Progress
```bash
python bin/tito.py info
```

## Key Concepts

- **nbdev workflow**: Write in notebooks, export to Python
- **Export directive**: Use `#| export` to mark code for export
- **Module → Package mapping**: This module exports to `tinytorch/core/utils.py`
- **Teaching vs. Building**: Learn by modules, build by function (see VISION.md)
- **Test integration**: Tests run automatically via CLI
- **Module development**: Each module is self-contained

## Success Criteria

✅ All tests pass  
✅ Code exports cleanly to `tinytorch/core/utils.py`  
✅ You understand the development rhythm  
✅ Ready to tackle the Tensor module  

---

**Next Module**: [Tensor](../tensor/) - Core data structures and operations 