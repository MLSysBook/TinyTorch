# TinyTorch Command Hierarchy

## Implemented Command Structure

### Top Level (Essential Only)
```bash
tito setup          # First-time setup (prompts for user info)
tito status         # Overall progress/status
tito logo           # Philosophy
```

### Student Commands (Work in notebooks/)
```bash
tito module start 01           # Start module (marks progress)
tito module test 01            # Test implementation
tito module complete 01        # Test + export + mark complete
tito module reset 01           # Reset to clean state
tito module status             # Show progress
```

### Developer Commands (Work in src/)
```bash
tito source export 01_tensor   # Export: src/*.py → modules/*.ipynb → tinytorch/*.py
tito source export --all       # Export all modules
tito source test 01            # Test module (planned)
```

### Instructor Commands
```bash
tito nbgrader generate 01      # Create student versions
tito nbgrader release 01       # Distribute to students
tito nbgrader collect 01       # Collect submissions
tito nbgrader autograde 01     # Auto-grade
```

### Milestone Commands
```bash
tito milestone list            # List all milestones
tito milestone run 03          # Run milestone test
tito milestone status          # Show achievement progress
```

---

## Directory Mapping

| Directory | Command Hierarchy | Purpose |
|-----------|------------------|---------|
| `src/` | `tito source` | Python source files (developers) |
| `modules/` | `tito module` | Notebooks (students/learners) |
| `tinytorch/` | N/A | Generated package (import only) |

---

## Workflow Examples

### Student Workflow
```bash
# Clone and setup
git clone tinytorch
tito source export --all        # Generate all notebooks

# Work on modules
tito module start 01
# Edit modules/01_tensor/01_tensor.ipynb
tito module complete 01         # Tests + exports

# Check progress
tito module status
tito milestone list
```

### Developer Workflow
```bash
# Edit source
vim src/01_tensor/01_tensor.py

# Export changes
tito source export 01_tensor    # Generates notebook + package

# Test
tito source test 01_tensor      # (to be implemented)

# Run milestones
tito milestone run 03
```

---

## Command Status

### ✅ Working
- `tito source export <num>` - Exports src/ to modules/ to tinytorch/
- `tito source export --all` - Exports all modules
- `tito module status` - Shows progress
- `tito milestone list` - Lists milestones

### ⚠️ Partially Working
- `tito module start` - Works but tries to launch Jupyter (not installed)
- `tito module complete` - Has export bug (passes wrong args)

### ❌ Broken
- `tito milestone run` - Missing progress_tracker module
- `tito setup` - Prompts for user input (hangs in automated flow)

### 🚧 Not Yet Implemented
- `tito source test` - Defined but not implemented
- `tito nbgrader` subcommands - Exist but may need updates for new structure

---

## Next Steps

1. Fix `tito module complete` to use source command
2. Fix `tito milestone run` progress_tracker import
3. Implement `tito source test`
4. Add `--non-interactive` flag to setup
5. Update nbgrader to work with new structure

