# Appendix C: TITO CLI Reference

TITO (TinyTorch Interactive Teaching Orchestrator) is your command-line companion for navigating the TinyTorch learning journey. This appendix provides a quick reference for essential commands.

## System Commands

### Installation & Setup

```bash
# Install TinyTorch package
pip install tinytorch

# Verify installation
tito system verify

# Check your environment
tito system info

# View system status
tito system status
```

### Package Management

```bash
# Export modules from src/ to modules/ directory
tito system export {module}
tito system export 01  # Export Module 01

# Export all modules
tito system export-all

# Build package from student-implemented code
tito system build
```

## Module Commands

### Module Navigation

```bash
# Check module status
tito module status

# List all modules
tito module list

# Start a specific module (opens Jupyter Lab)
tito module start 01
tito module start 05

# Complete a module
tito module complete 01
```

### Module Testing

```bash
# Run tests for current module
tito test

# Run tests for specific module
tito test 01
tito test 05

# Run all tests
tito test all

# Run specific test file
tito test tests/01_tensor/test_tensor.py
```

## Milestone Commands

Historical milestones validate your implementation against ML history:

```bash
# List all milestones
tito milestone list

# Run specific milestone
tito milestone run perceptron    # 1958: Rosenblatt's Perceptron
tito milestone run alexnet       # 2012: AlexNet
tito milestone run transformer   # 2017: Attention Is All You Need

# Run all milestones (comprehensive validation)
tito milestone run-all
```

## Development Commands

### Code Quality

```bash
# Format code with black
tito format

# Run linters
tito lint

# Type checking
tito typecheck
```

### Conversion Utilities

```bash
# Convert between Jupytext and notebook formats
tito convert notebook-to-py modules/01_tensor/01_tensor.ipynb
tito convert py-to-notebook src/01_tensor/01_tensor.py
```

## Community & Benchmarking

### Capstone Competition (Module 20)

```bash
# Generate benchmark submission
tito community benchmark

# Submit to leaderboard
tito community submit submission.json

# View leaderboard
tito community leaderboard

# Compare with other submissions
tito community compare
```

## Quick Workflows

### Starting a New Module

```bash
# 1. Check status and prerequisites
tito module status

# 2. Start the module (opens Jupyter Lab)
tito module start 03

# 3. Work on implementation...

# 4. Run tests frequently
tito test

# 5. Complete when tests pass
tito module complete 03
```

### Validating Your Implementation

```bash
# Run module tests
tito test 05

# Run related milestone
tito milestone run backprop

# Check system integration
tito system verify
```

### Troubleshooting

```bash
# System diagnostics
tito system info
tito system status

# Environment verification
tito system verify

# Clean build artifacts
tito system clean

# Reset module (if needed)
tito module reset 05
```

## Configuration

TITO configuration is managed via `.tito/config.yml` in your working directory:

```yaml
# Example configuration
modules:
  auto_export: true
  default_editor: "jupyter-lab"

testing:
  verbose: true
  coverage: true

milestones:
  strict_mode: false
```

## Help System

```bash
# General help
tito --help

# Command-specific help
tito module --help
tito test --help
tito milestone --help

# Version information
tito --version
```

## Environment Variables

TITO respects these environment variables:

- `TITO_HOME`: Override default TinyTorch directory
- `TITO_EDITOR`: Default editor for opening modules
- `TITO_NO_COLOR`: Disable colored output
- `TITO_VERBOSE`: Enable verbose logging

Example:
```bash
export TITO_EDITOR="code"  # Use VS Code instead of Jupyter Lab
export TITO_VERBOSE=1      # Enable detailed logging
```

## Exit Codes

TITO commands use standard Unix exit codes:

- `0`: Success
- `1`: General error
- `2`: Command usage error
- `3`: Test failure
- `4`: Module not found
- `5`: Dependency missing

## Tips & Best Practices

1. **Run tests frequently**: Use `tito test` after each implementation step
2. **Check module status**: `tito module status` shows your progress
3. **Use milestones for validation**: Historical milestones confirm your implementation correctness
4. **Leverage auto-export**: Enable `auto_export` to sync src/ and modules/ automatically
5. **Read test failures carefully**: Test output guides you toward correct implementation

## Getting Help

- **Documentation**: https://mlsysbook.ai/tinytorch
- **GitHub Issues**: https://github.com/mlsysbook/TinyTorch/issues
- **Community Forum**: https://mlsysbook.ai/community

---

*For detailed command documentation, use `tito {command} --help` or visit the online documentation.*
