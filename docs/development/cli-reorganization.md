# CLI Reorganization: Hierarchical Command Structure

TinyTorch CLI has been reorganized into a clear hierarchical structure with three main command groups: `system`, `module`, and `package`. This provides better organization and makes it clear which subsystem each command operates on.

## New Command Structure

### System Commands (`tito system`)
**Environment, configuration, and system tools**

- `tito system info` - Show system information and course navigation
- `tito system doctor` - Run environment diagnosis  
- `tito system jupyter` - Start Jupyter notebook server

### Module Commands (`tito module`)
**Development workflow and module management**

- `tito module status` - Check status of all modules
- `tito module test` - Run module tests
- `tito module notebooks` - Build notebooks from Python files

### Package Commands (`tito package`)
**nbdev integration and package management**

- `tito package sync` - Export notebook code to Python package
- `tito package reset` - Reset tinytorch package to clean state
- `tito package nbdev` - nbdev notebook development commands

## Benefits of Hierarchical Structure

### 1. **Clear Subsystem Separation**
Each command group operates on a specific subsystem:
- **System**: Environment and configuration
- **Module**: Individual module development
- **Package**: Overall package management

### 2. **Intuitive Command Discovery**
Users can explore commands by subsystem:
```bash
tito system    # Shows all system commands
tito module    # Shows all module commands  
tito package   # Shows all package commands
```

### 3. **Extensibility**
New commands can be easily categorized:
- Adding deployment tools? → `tito system deploy`
- Adding module generators? → `tito module create`
- Adding package publishing? → `tito package publish`

### 4. **Reduced Cognitive Load**
Instead of remembering 10+ flat commands, users think in terms of:
- What subsystem am I working with?
- What do I want to do in that subsystem?

## Usage Examples

### System Operations
```bash
# Check environment setup
tito system doctor

# Get system information
tito system info

# Start development environment
tito system jupyter
```

### Module Development
```bash
# Check module status with metadata
tito module status --metadata

# Test all modules
tito module test --all

# Test specific module
tito module test --module tensor

# Generate notebooks
tito module notebooks --module tensor
```

### Package Management
```bash
# Export modules to package
tito package sync

# Export specific module
tito package sync --module tensor

# Reset package to clean state
tito package reset

# Run nbdev commands
tito package nbdev --export
```

## Backward Compatibility

The old flat command structure is still supported for backward compatibility:

```bash
# These still work (legacy commands)
tito status
tito test --all
tito sync
tito info
```

However, the new hierarchical structure is recommended for new usage.

## Migration Guide

### For Users
**Old → New Command Mapping:**

```bash
# System commands
tito info           → tito system info
tito doctor         → tito system doctor
tito jupyter        → tito system jupyter

# Module commands  
tito status         → tito module status
tito test           → tito module test
tito notebooks      → tito module notebooks

# Package commands
tito sync           → tito package sync
tito reset          → tito package reset
tito nbdev          → tito package nbdev
```

### For Scripts
Update automation scripts to use the new structure:

```bash
# Old
./scripts/test-all.sh:
tito test --all

# New (recommended)
./scripts/test-all.sh:
tito module test --all
```

## Implementation Details

### Command Group Classes
Each command group is implemented as a separate class:

- `SystemCommand` - Handles system subcommands
- `ModuleCommand` - Handles module subcommands  
- `PackageCommand` - Handles package subcommands

### Argument Parsing
Uses argparse subparsers for clean hierarchical structure:

```python
# Main parser
parser = argparse.ArgumentParser(prog="tito")
subparsers = parser.add_subparsers(dest='command')

# System group
system_parser = subparsers.add_parser('system')
system_subparsers = system_parser.add_subparsers(dest='system_command')

# System subcommands
system_subparsers.add_parser('info')
system_subparsers.add_parser('doctor')
system_subparsers.add_parser('jupyter')
```

### Help System
Each level provides contextual help:

```bash
tito              # Shows command groups
tito system       # Shows system subcommands
tito module       # Shows module subcommands
tito package      # Shows package subcommands
```

## Future Enhancements

### Additional Command Groups
The structure supports adding new command groups:

```bash
# Deployment commands
tito deploy status
tito deploy docker
tito deploy cloud

# Research commands
tito research benchmark
tito research profile
tito research compare
```

### Command Aliases
Short aliases for common commands:

```bash
# System
tito sys info      # alias for tito system info

# Module  
tito mod status    # alias for tito module status
tito mod test      # alias for tito module test

# Package
tito pkg sync      # alias for tito package sync
```

### Interactive Mode
Enhanced interactive command discovery:

```bash
tito interactive
# → Shows menu of command groups
# → Allows drilling down into subcommands
# → Provides guided command building
```

## Best Practices

### 1. **Use Hierarchical Commands**
Prefer the new structure for clarity:
```bash
# Good
tito module status --metadata

# Avoid (legacy)
tito status --metadata
```

### 2. **Think in Subsystems**
When adding new functionality, consider which subsystem it belongs to:
- Environment/config → `system`
- Individual modules → `module`
- Overall package → `package`

### 3. **Consistent Naming**
Use consistent naming patterns within each group:
- `status` for checking state
- `test` for running tests
- `sync` for synchronization
- `reset` for cleanup

### 4. **Help Documentation**
Always provide clear help text for new commands:
```python
parser.add_parser('new-command', help='Clear description of what this does')
```

## Conclusion

The hierarchical CLI structure provides:
- **Better organization** with clear subsystem separation
- **Improved discoverability** through logical grouping
- **Enhanced extensibility** for future commands
- **Maintained compatibility** with existing workflows

This structure scales well as TinyTorch grows and provides a professional CLI experience that matches industry standards. 