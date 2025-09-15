# NBGrader Integration Guide for TinyTorch

## What is NBGrader?

NBGrader is a Jupyter notebook autograding system that allows instructors to create, distribute, and automatically grade programming assignments. TinyTorch uses NBGrader metadata to enable automated assessment of student implementations.

## Why We Use NBGrader in TinyTorch

1. **Automated Assessment**: Instructors can automatically grade student implementations
2. **Immediate Feedback**: Students get instant validation of their solutions
3. **Scalable Education**: Enables TinyTorch to be used in large courses
4. **Consistent Grading**: Ensures fair and uniform evaluation across all students
5. **Learning Analytics**: Tracks student progress and identifies common issues

## NBGrader Metadata Fields Explained

### The Complete Metadata Structure
```python
# %% nbgrader={"grade": true, "grade_id": "test-relu-basic", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
```

### Field Definitions

#### `grade` (boolean)
- **`true`**: This cell will be graded (test cells)
- **`false`**: This cell won't be graded (implementation cells, documentation)
- **Purpose**: Tells NBGrader which cells contain grading logic

#### `grade_id` (string)
- **Format**: `"[type]-[concept]-[variant]"` (e.g., `"test-relu-basic"`)
- **Requirement**: Must be UNIQUE within the entire notebook
- **Purpose**: Identifies specific cells for grading database
- **Warning**: Duplicate grade_ids cause autograding failures!

#### `locked` (boolean)
- **`true`**: Students cannot modify this cell (test cells)
- **`false`**: Students can edit this cell (implementation cells)
- **Purpose**: Protects test code from tampering

#### `points` (integer)
- **Value**: Number of points awarded if tests pass
- **Only for**: Cells where `grade=true`
- **Purpose**: Defines grading weight for each test

#### `schema_version` (integer)
- **Current**: 3 (as of NBGrader 0.9.x)
- **Purpose**: Ensures compatibility with NBGrader version
- **Note**: Always use 3 for modern Jupyter/NBGrader

#### `solution` (boolean)
- **`true`**: This cell contains instructor solution to be removed
- **`false`**: This cell doesn't contain removable solution
- **Purpose**: Identifies which cells have BEGIN/END SOLUTION blocks

#### `task` (boolean)
- **`true`**: This is a manually graded task (essays, plots)
- **`false`**: This is not a manually graded task
- **Purpose**: Differentiates auto-graded from human-graded work

## TinyTorch Cell Type Patterns

### 1. Implementation Cell (Student Writes Code)
```python
# %% nbgrader={"grade": false, "grade_id": "relu-implementation", "locked": false, "schema_version": 3, "solution": true, "task": false}
def relu(x):
    """
    TODO: Implement ReLU activation
    """
    ### BEGIN SOLUTION
    return np.maximum(0, x)
    ### END SOLUTION
```
**Configuration**: Not graded, unlocked, contains solution

### 2. Test Cell (Validates Implementation)
```python
# %% nbgrader={"grade": true, "grade_id": "test-relu-basic", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": false}
def test_unit_relu():
    """Test ReLU implementation"""
    result = relu(np.array([-1, 0, 1]))
    assert np.array_equal(result, np.array([0, 0, 1]))
    print("✅ ReLU test passed!")

test_unit_relu()
```
**Configuration**: Graded, locked, awards points, no solution

### 3. Documentation Cell (Explanatory Content)
```python
# %% nbgrader={"grade": false, "grade_id": "relu-intro", "locked": false, "schema_version": 3, "solution": false, "task": false}
# %% [markdown]
"""
## Understanding ReLU Activation
The ReLU function is defined as f(x) = max(0, x)
"""
```
**Configuration**: Not graded, not locked, no solution

### 4. Read-Only Cell (Protected Instructions)
```python
# %% nbgrader={"grade": false, "grade_id": "imports", "locked": true, "schema_version": 3, "solution": false, "task": false}
import numpy as np
import matplotlib.pyplot as plt
```
**Configuration**: Not graded, locked, prevents tampering

## The BEGIN/END SOLUTION Pattern

### How It Works
```python
def matrix_multiply(A, B):
    """
    TODO: Implement matrix multiplication
    
    HINTS: Use three nested loops
    """
    ### BEGIN SOLUTION
    # Instructor's reference implementation
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C
    ### END SOLUTION
```

### Student Version (After NBGrader Processing)
```python
def matrix_multiply(A, B):
    """
    TODO: Implement matrix multiplication
    
    HINTS: Use three nested loops
    """
    # YOUR CODE HERE
    raise NotImplementedError()
```

## NBGrader Workflow in TinyTorch

### 1. Instructor Creates Assignment
- Write complete solution in `modules/source/XX_module/module_dev.py`
- Add NBGrader metadata to cells
- Include BEGIN/END SOLUTION blocks
- Create comprehensive tests

### 2. Generate Student Version
```bash
nbgrader generate_assignment module_name
```
- Removes solution code
- Replaces with `raise NotImplementedError()`
- Locks test cells
- Creates distributable notebook

### 3. Student Workflow
- Opens generated notebook
- Implements solutions replacing `NotImplementedError`
- Runs test cells for immediate feedback
- Submits completed notebook

### 4. Automated Grading
```bash
nbgrader autograde module_name
```
- Runs all test cells
- Awards points for passing tests
- Generates grade report
- Identifies common errors

## Best Practices for TinyTorch Modules

### 1. Unique grade_id Convention
```python
# Implementation cells
"grade_id": "[module]-[concept]-implementation"

# Test cells  
"grade_id": "test-[module]-[concept]-[variant]"

# Documentation cells
"grade_id": "[module]-[section]-doc"
```

### 2. Point Distribution
- **Unit tests**: 5-10 points (basic functionality)
- **Comprehensive tests**: 10-15 points (edge cases)
- **Integration tests**: 15-20 points (cross-module)
- **Total per module**: ~100 points

### 3. Test Placement
- ALWAYS place test immediately after implementation
- Name pattern: `test_unit_[function_name]()`
- Run test at cell bottom: `test_unit_[function_name]()`

### 4. Solution Quality
```python
### BEGIN SOLUTION
# Include comments explaining approach
# Use clear variable names
# Follow TinyTorch coding standards
implementation_code_here()
### END SOLUTION
```

### 5. Error Messages
```python
assert condition, "Descriptive error: Expected X but got Y"
# Not just: assert condition
```

## Common NBGrader Issues and Solutions

### Issue 1: Duplicate grade_id
**Error**: "Cell with id 'test_1' exists multiple times!"
**Solution**: Ensure every grade_id is unique across entire notebook

### Issue 2: Student Modifies Locked Cell
**Problem**: Test cells edited by student
**Solution**: NBGrader automatically restores from database during grading

### Issue 3: Missing Metadata
**Problem**: Cells without NBGrader metadata aren't processed correctly
**Solution**: Add metadata to every cell that needs grading/protection

### Issue 4: Solution Not Hidden
**Problem**: BEGIN/END SOLUTION not working
**Solution**: Ensure `"solution": true` in metadata

## Integration with TITO CLI

TinyTorch's TITO CLI works with NBGrader:

```bash
# Convert to notebook (preserves NBGrader metadata)
tito module notebooks tensor

# Validate NBGrader compatibility
tito validate --nbgrader tensor

# Generate student version
tito module generate --student tensor

# Run autograding locally
tito module grade tensor
```

## Why This Matters for Students

1. **Immediate Validation**: Know instantly if implementation is correct
2. **Clear Expectations**: Tests show exactly what's required
3. **Fair Grading**: Automated tests ensure consistency
4. **Learning Focus**: Spend time learning, not worrying about grading
5. **Professional Practice**: Mirrors industry test-driven development

## Why This Matters for Instructors

1. **Scalability**: Grade hundreds of submissions automatically
2. **Consistency**: Every student evaluated identically
3. **Time Savings**: Focus on teaching, not grading
4. **Analytics**: Identify common misconceptions from test failures
5. **Flexibility**: Combine auto-grading with manual review

## Summary

NBGrader integration in TinyTorch enables:
- **Automated assessment** at scale
- **Immediate feedback** for students
- **Consistent evaluation** across all submissions
- **Professional TDD practices** in education
- **Seamless workflow** with TITO CLI

The metadata we include in every cell ensures that TinyTorch modules can be used effectively in both self-study and classroom environments, providing a robust educational framework that scales from individual learners to massive online courses.