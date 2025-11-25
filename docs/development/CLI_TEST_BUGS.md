# TinyTorch CLI Testing - Bugs Found

> Tracking bugs discovered during comprehensive CLI testing

---

## Bug #1: Reset Command Uses Wrong Directory Path

**Test**: Test 4.3 - Reset Module
**Command**: `tito module reset 04 --force`
**Expected**: Module resets successfully with backup, unexport, and restore
**Actual**:
```
Dev file not found: /Users/VJ/GitHub/TinyTorch/modules/04_losses/losses.py
File not tracked in git: /Users/VJ/GitHub/TinyTorch/modules/04_losses/losses.py
Restore failed. Module may be in inconsistent state.
```

**Error Output**:
```
Step 2: Removing package exports...
Dev file not found: /Users/VJ/GitHub/TinyTorch/modules/04_losses/losses.py

Step 3: Restoring pristine source...
Restoring from git: modules/04_losses/losses.py
File not tracked in git: /Users/VJ/GitHub/TinyTorch/modules/04_losses/losses.py
Restore failed. Module may be in inconsistent state.
```

**Root Cause**:
1. Config (`tito/core/config.py` line 50-54) points to `modules/` directory
2. Actual module files are in `src/` directory (new structure from `restructure/src-modules-separation` branch)
3. File naming: Reset command looks for `losses.py` but actual file is `04_losses.py` (full module name)

**Files Affected**:
- `/Users/VJ/GitHub/TinyTorch/tito/core/config.py` - line 50: `modules_path = project_root / 'modules'`
- `/Users/VJ/GitHub/TinyTorch/tito/commands/module_reset.py` - lines 189, 246, 305, 362: uses `self.config.modules_dir`
- `/Users/VJ/GitHub/TinyTorch/tito/commands/module_reset.py` - line 248: `dev_file = module_dir / f"{short_name}.py"` (should be `{module_name}.py`)

**Fix Required**:
1. Update `config.py` to point to `src/` instead of `modules/`
2. Update `module_reset.py` line 248 to use full module name instead of short name:
   - Current: `dev_file = module_dir / f"{short_name}.py"`
   - Should be: `dev_file = module_dir / f"{module_name}.py"`
3. Same fix needed in lines 307, 362 of `module_reset.py`

**Status**: ✅ Fixed

**Severity**: HIGH - Reset command completely broken, can't reset any modules

**Fix Applied**:
- Updated `/Users/VJ/GitHub/TinyTorch/tito/core/config.py` line 50: changed `modules/` to `src/`
- Updated `/Users/VJ/GitHub/TinyTorch/tito/commands/module_reset.py` line 248: changed `f"{short_name}.py"` to `f"{module_name}.py"`
- Updated `/Users/VJ/GitHub/TinyTorch/tito/commands/module_reset.py` line 307: changed `f"{short_name}.py"` to `f"{module_name}.py"`

**Verification**: Successfully tested `tito module reset 04 --force`. Module was backed up, unexported, restored from git, and removed from completed modules. Progress correctly updated from 4/21 to 3/21.

---

## Bug #2: Reset Command Doesn't Update __init__.py Imports

**Test**: Test 4.3 - Reset Module (side effect discovered)
**Command**: `tito module reset 04 --force`
**Expected**: Module reset should remove imports from `tinytorch/__init__.py`
**Actual**: Imports remain in `__init__.py`, causing ModuleNotFoundError when trying to use other modules

**Error Output**:
```
ModuleNotFoundError: No module named 'tinytorch.core.losses'
```

**Root Cause**:
The reset command's `unexport_module()` method (line 241-298) removes the exported `.py` file from `tinytorch/core/` but doesn't update the top-level `tinytorch/__init__.py` which still has:
```python
from .core.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
```

This causes circular import errors when other modules try to import tinytorch.

**Fix Required**:
1. Add a new method `update_init_imports()` in `module_reset.py`
2. Parse `tinytorch/__init__.py` to find and comment out/remove imports for the reset module
3. Call this method from `unexport_module()` after removing the exported file
4. Also update the `__all__` export list

**Status**: 🔴 Open - Needs Fix

**Severity**: HIGH - Resetting any module breaks imports for all other modules

**Workaround**: Manually comment out the import in `__init__.py`

---

## Bug #3: Complete Command Doesn't Check Prerequisites

**Test**: Test 5.2 - Complete Without Starting
**Command**: `tito module complete 06` (when module 05 is not complete)
**Expected**: Should fail with prerequisite check, similar to `start` command
**Actual**: Command runs tests and export, but silently doesn't mark as complete

**Root Cause**:
The `complete_module()` method in `module_workflow.py` (line 245-370) does not check prerequisites before running tests and export. It only checks if the module exists in the mapping, then proceeds to:
1. Run tests
2. Export to package
3. Update progress

The `update_progress()` method likely checks prerequisites internally, preventing the module from being marked complete, but this wastes time running tests and export.

**Impact**:
- Wastes computational resources running tests for locked modules
- Confusing user experience - command appears to succeed but module not marked complete
- Inconsistent with `start` command which has clear prerequisite checking

**Fix Required**:
Add prerequisite checking at the beginning of `complete_module()` method, similar to the check in `start_module()`:

```python
def complete_module(self, module_number: Optional[str] = None, ...):
    # ... existing code ...

    module_num = int(normalized)

    # NEW: Check prerequisites before running tests
    if module_num > 1:
        progress = self.get_progress_data()
        completed = progress.get('completed_modules', [])
        missing_prereqs = []
        for i in range(1, module_num):
            prereq_num = f"{i:02d}"
            if prereq_num not in completed:
                missing_prereqs.append((prereq_num, module_mapping.get(prereq_num, "Unknown")))

        if missing_prereqs:
            # Show locked module panel and return early
            self.console.print(Panel(...))
            return 1

    # ... continue with tests and export ...
```

**Status**: 🔴 Open - Needs Fix

**Severity**: MEDIUM - Wastes resources but doesn't break core functionality

---

## Bug #4: [Template for Next Bug]

**Test**: Test X.X - [Name]
**Command**: `...`
**Expected**: ...
**Actual**: ...
**Error Output**: ...
**Root Cause**: ...
**Fix**: ...
**Status**: [ ] Open / [ ] Fixed / [ ] Won't Fix
