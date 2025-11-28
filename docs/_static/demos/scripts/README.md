# TinyTorch Demo Workflow Scripts

## Overview

This directory contains scripts for testing and generating demo GIFs for the TinyTorch website carousel.

## Scripts

### `test_demo_workflow.sh`

Comprehensive testing script that validates the TinyTorch demo workflow before generating VHS tapes.

**Purpose:**
- Ensures demo commands actually work before recording
- Prevents "directory already exists" errors during GIF generation
- Validates the complete student onboarding flow
- Logs all output for debugging and verification

**What it does:**
1. Cleans `/tmp/TinyTorch` (prevents git clone conflicts)
2. Clones TinyTorch repository from GitHub
3. Runs `setup-environment.sh` (creates venv, installs dependencies)
4. Verifies TITO CLI is properly installed
5. Tests all demo commands:
   - `tito system health` - Environment health check
   - `tito module status` - Module progress display
   - `tito milestones list` - Historical milestone browser

**Usage:**
```bash
# Run from TinyTorch project root
./docs/_static/demos/scripts/test_demo_workflow.sh

# Check detailed log
cat /tmp/tinytorch_demo_log.txt
```

**Exit behavior:**
- Exits with code 0 if all tests pass
- Exits with non-zero code if any command fails (set -e)
- Creates log file at `/tmp/tinytorch_demo_log.txt`

### `generate.sh`

Generates all demo GIFs using VHS (Video Helper Script).

**Requirements:**
- VHS installed: `brew install vhs` (macOS) or [other platforms](https://github.com/charmbracelet/vhs)
- VHS tape files in `../tapes/` directory

**Usage:**
```bash
# Run from TinyTorch project root
bash docs/_static/demos/scripts/generate.sh

# GIFs will be created in docs/_static/demos/
```

## Workflow Best Practices

### Before Generating GIFs

1. **Test the workflow first:**
   ```bash
   ./docs/_static/demos/scripts/test_demo_workflow.sh
   ```

2. **Review the log:**
   ```bash
   cat /tmp/tinytorch_demo_log.txt
   ```

3. **Update VHS tapes** if commands changed

4. **Generate GIFs:**
   ```bash
   bash docs/_static/demos/scripts/generate.sh
   ```

### Updating Demo Commands

If you modify TITO commands (rename, add flags, change output):

1. Update test script first (`test_demo_workflow.sh`)
2. Run test to verify new commands work
3. Update VHS tape files (`.tape` files in `tapes/`)
4. Regenerate GIFs

This ensures the demos always show working, current commands.

## VHS Tape Files

Located in `../tapes/`:

- `01-zero-to-ready.tape` - Setup workflow (clone → setup → activate)
- `02-build-test-ship.tape` - Module workflow (start → complete → import)
- `03-milestone-unlocked.tape` - Milestone system demo
- `04-share-journey.tape` - Community features demo

Each tape file includes:
- Cleanup command: `cd /tmp && rm -rf TinyTorch` (prevents conflicts)
- Actual working commands (validated by test script)
- Proper timing and sleep delays for readability

## Troubleshooting

### "fatal: destination path 'TinyTorch' already exists"

**Cause:** Previous demo run left `/tmp/TinyTorch` directory

**Fix:** Test script now automatically cleans this up. VHS tapes also include cleanup.

```bash
# Manual cleanup if needed
rm -rf /tmp/TinyTorch
```

### "tito: command not found"

**Cause:** Virtual environment not activated or TITO not installed

**Fix:** Test script uses `.venv/bin/tito` directly (no activation needed)

### GIFs showing wrong commands

**Cause:** VHS tapes not updated after command changes

**Fix:**
1. Run test script to see current command output
2. Update `.tape` files to match
3. Regenerate GIFs

## File Locations

```
docs/_static/demos/
├── scripts/
│   ├── README.md                    # This file
│   ├── test_demo_workflow.sh        # Workflow testing
│   └── generate.sh                  # GIF generation
├── tapes/
│   ├── 01-zero-to-ready.tape        # VHS tape files
│   ├── 02-build-test-ship.tape
│   ├── 03-milestone-unlocked.tape
│   └── 04-share-journey.tape
└── *.gif                            # Generated GIFs
```

## Implementation Notes

### Why separate test script?

Before this script existed, demo GIFs would:
- Show commands that didn't work
- Fail on re-runs due to directory conflicts
- Display outdated command syntax

The test script ensures:
- ✅ All commands actually work
- ✅ Reproducible on any machine
- ✅ Synchronized with current TITO CLI

### Cleanup strategy

**Test script:** Cleans before testing to validate fresh setup
```bash
if [ -d "/tmp/TinyTorch" ]; then
    rm -rf "$DEMO_DIR"
fi
```

**Generate script:** Cleans before VHS to ensure clean demo environment
```bash
rm -rf /tmp/TinyTorch 2>/dev/null || true
```

**VHS tapes:** Show clean student workflow (no cleanup commands visible)
```
Type "cd /tmp"
Type "git clone https://github.com/mlsysbook/TinyTorch.git"
```

This separation ensures demos show the ideal first-time experience while maintaining reproducibility behind the scenes.

### Command validation

Test script runs commands exactly as shown in demos:
- Same paths (`/tmp/TinyTorch`)
- Same sequence (clone → setup → activate → tito)
- Same flags and arguments

If the test passes, the GIF will show working commands.
