# CLI Command Files - Usage Report

## Summary

**Status**: ✅ All files are accounted for. Some are imported but not exposed as top-level commands.

## File Categories

### 1. ✅ Registered Top-Level Commands (18)
These are in `TinyTorchCLI.commands` and accessible via `tito <command>`:

```
benchmark, book, checkpoint, community, demo, export,
grade, leaderboard, logo, milestones, module, nbgrader,
olympics, package, setup, src, system, test
```

### 2. 🔧 Internal Subcommands (7)
**Imported and used by other commands, but not top-level:**

| File | Used By | Purpose |
|------|---------|---------|
| `reset.py` | `package.py` | Reset functionality for package command |
| `module_reset.py` | `module_workflow.py` | Module reset subcommand |
| `status.py` | - | Imported in main.py but not clearly used |
| `nbdev.py` | `package.py` | NBDev integration for package command |
| `info.py` | `system.py`, `health.py` | System info subcommand |
| `health.py` | `system.py` | System health check subcommand |
| `jupyter.py` | `system.py` | Jupyter integration subcommand |

**Action**: ✅ **KEEP THESE** - They're used by other commands

### 3. ❓ Imported but Unclear Usage (1)

| File | Issue | Recommendation |
|------|-------|----------------|
| `notebooks.py` | Imported in main.py, but no usage found | Check if used, otherwise remove import |
| `status.py` | Imported in main.py, but no clear usage | Check if used, otherwise remove import |

**Action**: Need to verify these

### 4. 🗑️ Likely Unused/Deprecated (9)

| File | Status |
|------|--------|
| `check.py` | Not imported anywhere |
| `clean.py` | Not imported anywhere |
| `clean_workspace.py` | Not imported anywhere |
| `help.py` | Not imported anywhere |
| `protect.py` | Not imported anywhere |
| `report.py` | Not imported anywhere |
| `version.py` | Not imported anywhere |
| `view.py` | Not imported anywhere |

**Action**: ⚠️ Safe to delete (not imported anywhere)

## Cleanup Actions

### Step 1: Remove Dead Imports from main.py

These are imported but not registered or used:

```python
# Remove from tito/main.py lines 28-37:
from .commands.notebooks import NotebooksCommand  # ❌ Not used
from .commands.status import StatusCommand        # ❌ Not used (verify first)
```

### Step 2: Delete Truly Unused Files

```bash
# These are safe to delete (not imported anywhere)
rm tito/commands/check.py
rm tito/commands/clean.py
rm tito/commands/clean_workspace.py
rm tito/commands/help.py
rm tito/commands/protect.py
rm tito/commands/report.py
rm tito/commands/version.py
rm tito/commands/view.py
```

### Step 3: Verify and Update Tests

Update `test_cli_registry.py` to remove deleted files from `known_internal`:

```python
known_internal = {
    'health.py',      # Used by system command
    'info.py',        # Used by system command
    'jupyter.py',     # Used by system command
    'nbdev.py',       # Used by package command
    'notebooks.py',   # Verify if needed, otherwise remove
    'reset.py',       # Used by package command
    'status.py',      # Verify if needed, otherwise remove
    'module_reset.py' # Used by module_workflow command
}
```

## Verification Commands

Check if status.py is actually used:
```bash
grep -r "StatusCommand" tito/ --include="*.py" | grep -v "^tito/main.py:from" | grep -v "class StatusCommand"
```

Check if notebooks.py is actually used:
```bash
grep -r "NotebooksCommand" tito/ --include="*.py" | grep -v "^tito/main.py:from" | grep -v "class NotebooksCommand"
```

## Final Architecture

After cleanup, you'll have:
- **18 top-level commands** (user-facing via `tito <cmd>`)
- **7-8 internal commands** (used as helpers by other commands)
- **0 orphaned files** (everything has a purpose)

Clean CLI with clear separation between public API and internal helpers!
