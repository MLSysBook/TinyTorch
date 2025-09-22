# Testing Framework

## Comprehensive Testing with TinyTorch

TinyTorch includes a robust testing framework that ensures all modules work correctly and integrate seamlessly. The testing system provides multiple levels of validation from individual module testing to full system integration.

## Quick Start: Run All Tests

```bash
# Run comprehensive test suite (recommended)
tito test --comprehensive

# Output shows module tests, integration tests, and example validation
# ✅ Overall TinyTorch Health: 100.0%
```

## Testing Levels

### 1. Module-Level Testing
Test individual components in isolation:

```bash
# Test specific modules
tito test --module tensor
tito test --module training
tito test --module autograd

# All modules have dedicated test suites
python tests/run_all_modules.py
```

### 2. Integration Testing
Verify modules work together correctly:

```bash
# Run integration tests
python tests/integration/run_integration_tests.py

# Test module dependencies
python tests/integration/test_module_dependencies.py
```

### 3. Checkpoint Testing
Validate learning progression capabilities:

```bash
# Test specific capabilities
tito checkpoint test 01  # Foundation capabilities
tito checkpoint test 10  # Training capabilities

# Run all checkpoint tests
for i in {00..15}; do tito checkpoint test $i; done
```

### 4. Example Validation
Ensure real-world examples work:

```bash
# Example tests are included in comprehensive testing
# Validates XOR training and CIFAR-10 baseline performance
```

## Test Architecture

### Progressive Testing Pattern
Every module follows the same testing pattern:
- **Core functionality tests**: Basic operations work correctly
- **Integration tests**: Module exports and imports properly
- **Progressive tests**: Dependencies on previous modules work
- **Capability tests**: Real-world usage scenarios pass

### Test Files Organization
```
tests/
├── checkpoints/           # Capability validation tests
├── integration/           # Cross-module integration tests
├── module_XX/            # Individual module test suites
└── run_all_modules.py    # Master test runner
```

## Test Results Interpretation

### Health Status
- **100.0%**: All tests passing, framework fully functional
- **95%+**: Minor issues, core functionality intact
- **90%+**: Some failing tests, investigate specific modules
- **<90%**: Significant issues, check recent changes

### Module Status Indicators
- ✅ **Passing**: Module works correctly
- ⚠️ **Warning**: Minor issues detected
- ❌ **Failing**: Module has errors requiring fixes

## Best Practices

### During Development
1. **Test early and often**: Run `tito test --module X` after each change
2. **Use progressive testing**: Ensure dependencies work before building on them
3. **Validate with examples**: Real usage should work, not just unit tests

### Before Commits
1. **Run comprehensive tests**: `tito test --comprehensive`
2. **Check all modules pass**: Verify 100% health status
3. **Test checkpoint progression**: Ensure learning path works

### Troubleshooting Failed Tests
1. **Check specific module**: `tito test --module failing_module`
2. **Review integration**: Look for dependency issues
3. **Validate exports**: Ensure `tito module complete` works
4. **Check environment**: Run `tito system doctor`

## Testing Philosophy

The TinyTorch testing framework embodies our core principles:

### KISS Principle in Testing
- **Simple test patterns**: Every module follows the same structure
- **Clear feedback**: Tests provide specific, actionable error messages
- **Minimal complexity**: Tests focus on essential functionality

### Systems Thinking
- **Integration focus**: Tests verify components work together
- **Real-world validation**: Examples demonstrate practical usage
- **Performance awareness**: Tests include basic performance checks

### Educational Value
- **Learning verification**: Tests confirm understanding, not just implementation
- **Progressive validation**: Capabilities build on each other systematically
- **Immediate feedback**: Students know instantly if their code works

This testing framework ensures that your TinyTorch implementation is not just educational, but production-ready and reliable.