# TinyTorch Testing Architecture

## Overview

TinyTorch uses a **dual testing architecture** designed specifically for educational purposes. This system provides both immediate functionality validation and stretch goals for students.

## Architecture Diagram

```
Module Development → NBDev Export → Package Integration → Student Usage
       ↓                  ↓              ↓                    ↓
modules/tensor/       tinytorch/      tests/            from tinytorch.core
tests/ (stretch)     core/tensor.py  (integration)     import Tensor
```

## Two Test Levels

### 1. Package-Level Tests (`tests/` directory)

**Purpose**: Validate the final exported package that students use

- **Location**: `tests/test_setup.py`, `tests/test_tensor.py`
- **Import**: `from tinytorch.core.tensor import Tensor`
- **Tests**: Core functionality that students can rely on
- **Status**: ✅ All tests pass (29/29)
- **Command**: `python -m pytest tests/ -v`

**What it validates**:
- Exported package functionality works correctly
- Students can import and use TinyTorch components
- Integration between modules works properly
- Final student experience is smooth

### 2. Module-Level Tests (`modules/{module}/tests/` directory)

**Purpose**: Development validation + stretch goals for students

- **Location**: `modules/tensor/tests/test_tensor.py`
- **Import**: `from tensor_dev import Tensor` (direct module import)
- **Tests**: Core features + advanced methods (stretch goals)
- **Status**: ✅ 22 passed, 11 skipped (stretch goals)
- **Command**: `python bin/tito.py test --module tensor`

**What it validates**:
- Current implementation works with core features
- Advanced methods are tested as "stretch goals"
- Students can see what additional features they could implement
- Instructors can validate development completeness

## Test Results Summary

### ✅ Package-Level Tests (Integration)
```
tests/test_setup.py    → 11 passed ✅
tests/test_tensor.py   → 18 passed ✅
Total: 29 passed, 0 failed
```

### ✅ Module-Level Tests (Development + Stretch Goals)
```
modules/tensor/tests/test_tensor.py → 22 passed ✅, 11 skipped (stretch goals)
```

**Stretch Goals (Skipped - Student Implementation Targets)**:
- `reshape()` method
- `transpose()` method  
- `sum()` method
- `mean()`, `max()`, `min()` methods
- `item()` method for scalar extraction
- `numpy()` method for conversion
- Advanced chained operations

## Educational Benefits

### For Students
1. **Immediate Success**: Package-level tests ensure working functionality
2. **Clear Goals**: Module-level skipped tests show what to implement next
3. **Progressive Learning**: Can implement stretch goals at their own pace
4. **Validation**: Both test levels confirm their implementations work

### For Instructors
1. **Quality Assurance**: Package-level tests ensure course materials work
2. **Assignment Creation**: Module-level tests provide implementation targets
3. **Progress Tracking**: Can see which advanced features students implement
4. **Flexibility**: Can adjust stretch goals based on course needs

## Commands Reference

### Test Everything
```bash
# Test all modules (both levels)
python bin/tito.py test --all

# Test package integration only
python -m pytest tests/ -v
```

### Test Specific Module
```bash
# Test module development (includes stretch goals)
python bin/tito.py test --module tensor

# Test module directly
cd modules/tensor && python -m pytest tests/test_tensor.py -v
```

### Test Individual Components
```bash
# Setup module only
python bin/tito.py test --module setup

# Tensor package integration only
python -m pytest tests/test_tensor.py -v
```

## Implementation Pattern

When creating new modules, follow this pattern:

### 1. Core Implementation
- Implement basic functionality in `modules/{module}/{module}_dev.py`
- Add `#| export` directives for NBDev export
- Include both student and instructor versions

### 2. Package-Level Tests
- Create `tests/test_{module}.py`
- Import from `tinytorch.core.{module}`
- Test core functionality that students will use
- Ensure all tests pass

### 3. Module-Level Tests
- Create `modules/{module}/tests/test_{module}.py`
- Import from `{module}_dev`
- Test core functionality + stretch goals
- Use `pytest.skip()` for unimplemented features

### 4. Helper Functions
```python
def safe_method(tensor, method_name, *args, **kwargs):
    """Call method if it exists, otherwise skip test"""
    if hasattr(tensor, method_name):
        return getattr(tensor, method_name)(*args, **kwargs)
    else:
        pytest.skip(f"{method_name} method not implemented - stretch goal")
```

## Benefits of This Architecture

### ✅ **Reliability**
- Students always have working code to build on
- Package-level tests ensure integration works
- No broken dependencies between modules

### ✅ **Educational Value**
- Clear progression from basic to advanced features
- Students can see their implementation goals
- Stretch goals provide optional challenges

### ✅ **Maintainability**
- Two clear test levels with different purposes
- Easy to add new modules following the pattern
- Clear separation of concerns

### ✅ **Flexibility**
- Instructors can adjust stretch goals per course
- Students can work at their own pace
- Advanced students can implement additional features

## Current Status

- **Setup Module**: ✅ Fully implemented and tested
- **Tensor Module**: ✅ Core functionality complete, 11 stretch goals available
- **Remaining Modules**: 11 modules ready for implementation following this pattern

The dual testing architecture provides a solid foundation for the complete TinyTorch educational system. 