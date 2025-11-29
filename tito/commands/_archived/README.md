# Archived Commands

These command files are no longer top-level commands but are kept for reference.

## Archived Files

- `clean.py` - Deprecated cleanup command
- `help.py` - Old help command (now handled by argparse)
- `notebooks.py` - Deprecated notebooks command
- `status.py` - Old status command (functionality moved to module workflow)
- `checkpoint.py` - Old checkpoint tracking (superseded by milestones command)

## Note

During the CLI reorganization on 2025-11-28, commands with subcommands were moved into logical subfolders:
- `module/` - Module workflow and reset
- `system/` - System commands (info, health, jupyter, check, version, clean_workspace, report, protect)
- `package/` - Package management (nbdev, reset)

These archived files are truly deprecated and not used anywhere in the codebase.
