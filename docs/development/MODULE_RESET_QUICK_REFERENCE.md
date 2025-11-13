# `tito module reset` Quick Reference Guide

Quick reference for the new module reset functionality.

---

## Basic Usage

### Full Reset (Default - Recommended)
```bash
tito module reset 01
```
**What it does:**
1. ✅ Backs up your current work
2. ✅ Removes package exports
3. ✅ Restores module from git
4. ✅ Updates progress tracking

**Use when:** You want to completely start over on a module

---

## Common Scenarios

### 1. Made Mistakes - Start Fresh
```bash
# Completely reset module 01
tito module reset 01

# Then start working again
tito module start 01
```

### 2. Keep Package Exports (Soft Reset)
```bash
# Reset source but keep package
tito module reset 01 --soft

# Good for: Testing different approaches without re-exporting
```

### 3. View Available Backups
```bash
# List all backups for module 01
tito module reset 01 --list-backups

# Output shows:
# - Timestamp of backup
# - Git hash at time of backup
# - Number of files backed up
```

### 4. Restore from Specific Backup
```bash
# First, list backups to find timestamp
tito module reset 01 --list-backups

# Then restore from specific backup
tito module reset 01 --restore-backup 20251112_143022
```

### 5. Quick Reset (Skip Confirmation)
```bash
# Useful for automation or when you're sure
tito module reset 01 --force
```

---

## All Command Flags

| Flag | Description | Use When |
|------|-------------|----------|
| `--soft` | Keep package exports | You want to preserve exports |
| `--hard` | Remove everything [DEFAULT] | You want a complete reset |
| `--from-git` | Restore from git HEAD [DEFAULT] | Reset to repository version |
| `--restore-backup <timestamp>` | Restore from backup | You want previous work back |
| `--list-backups` | Show available backups | Checking backup history |
| `--no-backup` | Skip backup creation | **DANGEROUS** - only for testing |
| `--force` | Skip confirmations | You're absolutely sure |

---

## Safety Features

### Automatic Backup
**Every reset creates a backup** (unless `--no-backup`)

**Backup location:**
```
.tito/backups/01_tensor_20251112_143022/
├── tensor.py                 # Your work
└── backup_metadata.json      # Info about backup
```

### Confirmation Prompts
**Always asks before destructive actions** (unless `--force`)

Example:
```
This will reset the module to a clean state.
Your current work will be backed up.

Continue with reset? (y/N):
```

### Git Status Check
**Warns if you have uncommitted changes:**
```
⚠️  You have uncommitted changes in your repository!
Consider committing your work before resetting.
```

---

## Workflow Integration

### Natural Learning Flow
```bash
# 1. Start module
tito module start 01

# 2. Work in Jupyter (save changes)

# 3. Try to complete
tito module complete 01
# Tests fail? Code doesn't work?

# 4. Reset and try again
tito module reset 01

# 5. Start fresh with new approach
tito module resume 01
```

### Progress Tracking
**Reset updates your progress:**
- Removes module from "completed"
- Clears completion date
- Preserves backup history

**View progress:**
```bash
tito module status
```

---

## Common Questions

### Q: Will I lose my work?
**A:** No! Unless you use `--no-backup`, your work is automatically backed up to `.tito/backups/`

### Q: Can I undo a reset?
**A:** Yes! Use `--restore-backup` with the timestamp of the backup you want:
```bash
tito module reset 01 --restore-backup 20251112_143022
```

### Q: What's the difference between soft and hard reset?
**A:**
- **Hard (default):** Removes everything (source + package exports)
- **Soft:** Only resets source, keeps package exports intact

### Q: Where are my backups stored?
**A:** In `.tito/backups/<module_name>_<timestamp>/`

Each backup includes:
- Your Python files
- Metadata (timestamp, git hash, file list)

### Q: How do I see all my backups?
**A:**
```bash
tito module reset 01 --list-backups
```

### Q: Can I reset multiple modules?
**A:** Not in one command, but you can run reset for each:
```bash
tito module reset 01
tito module reset 02
tito module reset 03
```

---

## Examples by Use Case

### Beginner: Made a mistake, start over
```bash
tito module reset 01
# Easy! Just reset and start fresh
```

### Intermediate: Want to try different approach
```bash
# Keep exports but reset source
tito module reset 01 --soft

# Work on new approach
tito module resume 01
```

### Advanced: Restore previous working version
```bash
# Check what backups exist
tito module reset 01 --list-backups

# Restore the one that was working
tito module reset 01 --restore-backup 20251112_140000
```

### Developer: Quick iteration during testing
```bash
# No confirmations, fast workflow
tito module reset 01 --force
```

---

## Troubleshooting

### "Backup failed. Reset aborted."
**Cause:** Can't create backup directory or files

**Solution:**
1. Check `.tito/` directory permissions
2. Ensure you have write access
3. Try: `mkdir -p .tito/backups`

### "Git checkout failed"
**Cause:** File not tracked in git or git issues

**Solutions:**
1. Check if file is in git: `git ls-files modules/01_tensor/tensor.py`
2. Commit the file first: `git add modules/01_tensor/tensor.py && git commit`
3. Or use backup restore: `--restore-backup` instead

### "Export file not found"
**Cause:** Module wasn't exported yet

**Solution:** This is fine! If there's no export, nothing to remove. Reset will still restore source.

### "Module directory not found"
**Cause:** Invalid module number or name

**Solution:**
```bash
# Use correct module number (01-20)
tito module reset 01  # ✅ Correct
tito module reset 1   # ✅ Also works (normalized to 01)
tito module reset 99  # ❌ Invalid - no module 99
```

---

## Best Practices

### 1. **Commit Before Major Changes**
```bash
# Before trying something risky
git add .
git commit -m "Working version before experiment"

# Now safe to reset if needed
tito module reset 01
```

### 2. **Use Soft Reset for Iteration**
```bash
# When experimenting with implementations
tito module reset 01 --soft
# Keeps package exports, faster iteration
```

### 3. **Check Backups Periodically**
```bash
# See your backup history
tito module reset 01 --list-backups

# Clean old backups manually if needed
rm -rf .tito/backups/01_tensor_20251110_*
```

### 4. **Force Only When Sure**
```bash
# Interactive (asks confirmation) - SAFE
tito module reset 01

# Force (no confirmation) - FAST but RISKY
tito module reset 01 --force
```

---

## Integration with Other Commands

### Complete Workflow
```bash
# Start
tito module start 01

# Work...

# Try to complete
tito module complete 01

# If tests fail, reset and try again
tito module reset 01

# Resume work
tito module resume 01

# Complete successfully
tito module complete 01

# Check progress
tito module status
```

### With Package Management
```bash
# Reset removes exports
tito module reset 01

# Manually re-export if needed
tito module export 01_tensor

# Or just use complete (exports automatically)
tito module complete 01
```

---

## Quick Command Reference

| What You Want | Command |
|---------------|---------|
| Full reset | `tito module reset 01` |
| Keep exports | `tito module reset 01 --soft` |
| List backups | `tito module reset 01 --list-backups` |
| Restore backup | `tito module reset 01 --restore-backup <time>` |
| Quick reset | `tito module reset 01 --force` |
| No backup | `tito module reset 01 --no-backup --force` |

---

## Help Text

```bash
# Show reset help
tito module reset --help

# Show module workflow help
tito module --help

# Show all TITO commands
tito --help
```

---

**Remember:** Reset is your safety net! Don't be afraid to use it when you need a fresh start. Your work is backed up automatically.

**Pro Tip:** Check your backups occasionally with `--list-backups` to see your learning progress history!
