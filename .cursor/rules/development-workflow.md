# Development Workflow Rules

## Branch-First Development
- **Always create a branch** for any work - never work directly on main
- **Branch naming**: `feature/description`, `fix/issue`, `refactor/component`
- **Remind user** to create branches if they forget

## 🚨 CRITICAL: TinyTorch Development Workflow

### The Golden Rule: Source → Export → Use

```
modules/     →     tito export     →     tinytorch/     →     milestones/
  (EDIT HERE!)          (BUILD STEP)         (NEVER EDIT!)       (USE IT!)
```

### Three Sacred Principles

1. **ONLY edit files in `modules/`** - This is your source of truth
2. **ALWAYS use `tito export`** to build the `tinytorch/` package
3. **NEVER modify anything in `tinytorch/` directly** - It's generated code!

### Why This Matters

- **`modules/`**: Educational module sources (Python `.py` files)
- **`tinytorch/`**: Generated package (like `node_modules/` or `dist/`)
- **`milestones/`**: Student projects that import from `tinytorch`

**If you edit `tinytorch/` directly, your changes will be LOST on next export!**

### Complete Development Workflow

```bash
# 1. Edit the module source (ONLY place to make changes)
vim modules/12_attention/attention.py

# 2. Export to tinytorch package (Build step)
tito export

# 3. Test the exported module
pytest tests/12_attention/

# 4. Use in milestones
cd milestones/05_2017_transformer/
python tinytalks_dashboard.py  # Uses tinytorch.core.attention
```

## 🚨 CRITICAL: Notebook Development Workflow
**NEVER EDIT .ipynb FILES DIRECTLY**

TinyTorch uses a literate programming approach with nbdev:

1. **Edit ONLY `.py` files** in `modules/*/`
2. **Export to tinytorch** using `tito export`
3. **Run tests** with `pytest` to verify changes
4. **Never manually edit .ipynb files** - they are generated artifacts
5. **Never manually edit tinytorch/** - it's generated from modules/

### Why This Matters
- `.ipynb` files are JSON and hard to merge/review
- `.py` files are the **source of truth**
- `tinytorch/` is **generated code** (like compiled binaries)
- nbdev ensures proper sync between code, tests, and documentation
- Manual .ipynb edits will be overwritten on next export
- Manual tinytorch/ edits will be overwritten on next export

### Correct Workflow Example
```bash
# 1. Edit the Python source
vim modules/12_attention/attention.py

# 2. Export to tinytorch package
tito export

# 3. Run tests
pytest tests/12_attention/

# 4. If tests pass, commit source changes
git add modules/12_attention/attention.py
git commit -m "fix(attention): Handle 3D attention masks"
```

## Work Process
1. **Plan**: Define what changes are needed and why
2. **Reason**: Think through the approach and potential issues  
3. **Test**: Write tests to verify success before implementing
4. **Execute**: Implement changes in a new Git branch
5. **Verify**: Run all tests and ensure everything works
6. **Merge**: Only merge when fully tested and verified

## Testing Standards
- **Always use pytest** for all tests
- **Test before implementing** - write tests that define success
- **Test after implementing** - verify everything works
- **Test edge cases** and error conditions

## Documentation
- **Prefer Quarto** for documentation generation
- **Keep rules short** and actionable
- **Update rules** as patterns emerge

This ensures quality, traceability, and prevents breaking main branch. 