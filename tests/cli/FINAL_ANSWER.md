# Final Answer: CLI Command Cleanup

## What the Tests Found ✅

**Good news**: No broken or dangling commands! Everything is accounted for.

**However**: Found some cleanup opportunities:

### 1. Dead Imports in main.py

These 2 commands are imported but **never used**:
```python
# tito/main.py lines 28 and 37
from .commands.notebooks import NotebooksCommand  # ❌ DELETE
from .commands.status import StatusCommand        # ❌ DELETE
```

They're only in `__init__.py` exports, not actually used anywhere.

### 2. Orphaned Command Files (8 files)

These files exist but are **not imported anywhere**:
```bash
tito/commands/check.py
tito/commands/clean.py
tito/commands/clean_workspace.py
tito/commands/help.py
tito/commands/protect.py
tito/commands/report.py
tito/commands/version.py
tito/commands/view.py
```

### 3. Internal Helper Commands (6 files) ✅ KEEP

These are used by other commands:
- `reset.py` → used by `package.py`
- `nbdev.py` → used by `package.py`
- `info.py` → used by `system.py`
- `health.py` → used by `system.py`
- `jupyter.py` → used by `system.py`
- `module_reset.py` → used by `module_workflow.py`

## Recommended Actions

### Option A: Full Cleanup (Recommended)

```bash
# 1. Delete truly orphaned files
rm tito/commands/check.py
rm tito/commands/clean.py
rm tito/commands/clean_workspace.py
rm tito/commands/help.py
rm tito/commands/protect.py
rm tito/commands/report.py
rm tito/commands/version.py
rm tito/commands/view.py

# 2. Delete unused imported files
rm tito/commands/notebooks.py
rm tito/commands/status.py

# 3. Remove dead imports from main.py
# Edit tito/main.py and remove lines 28 and 37
```

### Option B: Conservative (Move to Archive)

```bash
# Move to archive instead of deleting
mkdir -p tito/commands/_archived
mv tito/commands/{check,clean,clean_workspace,help,protect,report,version,view}.py tito/commands/_archived/
mv tito/commands/{notebooks,status}.py tito/commands/_archived/
```

### Option C: Do Nothing

Current state is **fine** - tests prove nothing is broken. The extra files just create clutter but don't hurt.

## After Cleanup

Update `tests/cli/test_cli_registry.py`:

```python
# Remove these from known_internal since they'll be deleted:
known_internal = {
    'health.py',       # Used by system
    'info.py',         # Used by system
    'jupyter.py',      # Used by system
    'nbdev.py',        # Used by package
    'reset.py',        # Used by package
    'module_reset.py'  # Used by module_workflow
}
```

## Summary

Your CLI is **healthy**! The tests caught:
- ✅ 18 working registered commands
- ✅ 6 internal helper commands (properly used)
- ❌ 2 dead imports (should remove)
- ❌ 8 orphaned files (safe to delete)
- ❌ 2 unused command files (safe to delete)

**Total cleanup**: 12 files/imports that can be safely removed without breaking anything.

Want me to do the cleanup for you?
