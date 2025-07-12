# NBGrader Integration Plan

## Overview
This plan outlines the systematic integration of nbgrader with TinyTorch, starting with the setup module and progressing through all modules. Each module will be worth 100 points, allocated based on difficulty.

## Current Directory Structure
```
TinyTorch/
├── modules/
│   ├── 00_setup/
│   │   ├── setup_dev.py           # Complete implementation
│   │   ├── tests/test_setup.py    # pytest tests
│   │   └── README.md
│   ├── 01_tensor/
│   │   ├── tensor_dev.py          # Complete implementation
│   │   ├── tests/test_tensor.py   # pytest tests
│   │   └── README.md
│   └── [other modules...]
├── bin/
│   ├── tito                       # Current CLI
│   └── generate_student_notebooks.py
└── tinytorch/                     # Package output
```

## Proposed Directory Structure (After Integration)
```
TinyTorch/
├── modules/                       # Source modules (unchanged)
│   ├── 00_setup/
│   │   ├── setup_dev.py          # Enhanced with nbgrader markers
│   │   ├── tests/test_setup.py   # pytest tests
│   │   └── README.md
│   └── [other modules...]
├── assignments/                   # NEW: nbgrader assignments
│   ├── source/                   # Instructor versions
│   │   ├── setup/
│   │   │   └── setup.ipynb       # Generated from setup_dev.py
│   │   ├── tensor/
│   │   │   └── tensor.ipynb      # Generated from tensor_dev.py
│   │   └── [other assignments...]
│   ├── release/                  # Student versions (nbgrader managed)
│   ├── submitted/                # Student submissions (nbgrader managed)
│   ├── autograded/              # Auto-graded submissions (nbgrader managed)
│   └── feedback/                # Generated feedback (nbgrader managed)
├── nbgrader_config.py           # NEW: nbgrader configuration
├── bin/
│   ├── tito                     # Enhanced with nbgrader commands
│   └── generate_student_notebooks.py
└── tinytorch/                   # Package output (unchanged)
```

## Point Allocation System (100 Points Per Module)

### Module 00: Setup (100 Points)
- **Basic Functions** (30 points)
  - `hello_tinytorch()` - 10 points (easy)
  - `add_numbers()` - 10 points (easy)
  - Function execution tests - 10 points (easy)
- **SystemInfo Class** (35 points)
  - Constructor implementation - 15 points (medium)
  - `__str__` method - 10 points (easy)
  - `is_compatible()` method - 10 points (medium)
- **DeveloperProfile Class** (35 points)
  - Constructor with defaults - 15 points (medium)
  - Profile display methods - 10 points (easy)
  - ASCII art handling - 10 points (hard)

### Module 01: Tensor (100 Points)
- **Basic Properties** (30 points)
  - Constructor - 10 points (easy)
  - Properties (shape, size, dtype) - 10 points (easy)
  - String representation - 10 points (medium)
- **Element-wise Operations** (35 points)
  - Addition - 15 points (medium)
  - Multiplication - 15 points (medium)
  - Broadcasting support - 5 points (hard)
- **Matrix Operations** (35 points)
  - Matrix multiplication - 20 points (hard)
  - Shape validation - 10 points (medium)
  - Error handling - 5 points (hard)

### Module 02: Activations (100 Points)
- **ReLU** (25 points)
  - Implementation - 15 points (easy)
  - Edge cases - 10 points (medium)
- **Sigmoid** (25 points)
  - Implementation - 15 points (medium)
  - Numerical stability - 10 points (hard)
- **Tanh** (25 points)
  - Implementation - 15 points (medium)
  - Range validation - 10 points (medium)
- **Softmax** (25 points)
  - Implementation - 15 points (hard)
  - Numerical stability - 10 points (hard)

### Module 03: Layers (100 Points)
- **Dense Layer** (60 points)
  - Weight initialization - 15 points (medium)
  - Forward pass - 20 points (medium)
  - Bias handling - 15 points (medium)
  - Shape validation - 10 points (hard)
- **Layer Composition** (40 points)
  - Sequential implementation - 20 points (hard)
  - Layer chaining - 15 points (medium)
  - Error propagation - 5 points (hard)

## CLI Integration: `tito nbgrader` Commands

### Proposed CLI Structure
```bash
# Setup and configuration
tito nbgrader init                 # Initialize nbgrader environment
tito nbgrader config              # Configure nbgrader settings
tito nbgrader validate            # Validate nbgrader setup

# Assignment management
tito nbgrader generate --module setup     # Generate assignment from module
tito nbgrader generate --all              # Generate all assignments
tito nbgrader release --assignment setup  # Release assignment to students
tito nbgrader collect --assignment setup  # Collect student submissions
tito nbgrader autograde --assignment setup # Auto-grade submissions
tito nbgrader feedback --assignment setup  # Generate feedback

# Batch operations
tito nbgrader batch --release             # Release all pending assignments
tito nbgrader batch --collect             # Collect all submitted assignments
tito nbgrader batch --autograde          # Auto-grade all submissions
tito nbgrader batch --feedback           # Generate all feedback

# Analytics and reporting
tito nbgrader status                      # Show assignment status
tito nbgrader analytics --assignment setup # Show assignment analytics
tito nbgrader report --format csv         # Export grades report
```

## Implementation Strategy

### Phase 1: Setup Module (Week 1)
1. **Enhance setup_dev.py** with nbgrader markers
2. **Create nbgrader configuration** files
3. **Implement `tito nbgrader init`** command
4. **Generate first assignment** from setup module
5. **Test complete workflow** with setup module
6. **Validate point allocation** (100 points total)

### Phase 2: Core CLI Integration (Week 2)
1. **Implement core nbgrader commands** in tito
2. **Create assignment generation** pipeline
3. **Set up auto-grading workflow**
4. **Implement feedback generation**
5. **Add analytics and reporting**

### Phase 3: Module Enhancement (Week 3-4)
1. **Enhance tensor module** with nbgrader markers
2. **Enhance activations module** with nbgrader markers
3. **Enhance layers module** with nbgrader markers
4. **Test each module** individually
5. **Validate point allocations** for each module

### Phase 4: Integration Testing (Week 5)
1. **Test complete course workflow**
2. **Validate grade calculations**
3. **Test error handling and edge cases**
4. **Performance testing** with multiple submissions
5. **Documentation and training materials**

## Technical Implementation Details

### 1. NBGrader Configuration
```python
# nbgrader_config.py
c = get_config()
c.CourseDirectory.course_id = "tinytorch-ml-systems"
c.CourseDirectory.source_directory = "assignments/source"
c.CourseDirectory.release_directory = "assignments/release"
c.CourseDirectory.submitted_directory = "assignments/submitted"
c.CourseDirectory.autograded_directory = "assignments/autograded"
c.CourseDirectory.feedback_directory = "assignments/feedback"

# Point allocation
c.ClearSolutions.code_stub = {
    "python": "# YOUR CODE HERE\nraise NotImplementedError()"
}
```

### 2. Enhanced Module Structure
```python
# modules/00_setup/setup_dev.py (enhanced)
#| export
def hello_tinytorch():
    """Display TinyTorch welcome message"""
    #| exercise_start
    #| hint: Load ASCII art from tinytorch_flame.txt
    #| solution_test: Function should display ASCII art
    #| difficulty: easy
    #| points: 10
    
    ### BEGIN SOLUTION
    # Implementation here
    ### END SOLUTION
    
    #| exercise_end

### BEGIN HIDDEN TESTS
def test_hello_tinytorch():
    """Test hello_tinytorch function (10 points)"""
    # Test implementation
    pass
### END HIDDEN TESTS
```

### 3. CLI Integration
```python
# tito/commands/nbgrader.py (new file)
class NBGraderCommand(BaseCommand):
    """NBGrader integration commands"""
    
    def init(self):
        """Initialize nbgrader environment"""
        pass
    
    def generate(self, module=None, all=False):
        """Generate assignments from modules"""
        pass
    
    def release(self, assignment):
        """Release assignment to students"""
        pass
```

## Testing Strategy

### Unit Tests
- Test nbgrader marker parsing
- Test assignment generation
- Test point allocation calculations
- Test CLI command functionality

### Integration Tests
- Test complete workflow (generate → release → collect → grade)
- Test error handling and edge cases
- Test with multiple student submissions
- Test grade calculations and reporting

### User Acceptance Tests
- Test instructor workflow
- Test student experience
- Test grading accuracy
- Test feedback quality

## Success Metrics

### Technical Metrics
- **Assignment Generation**: < 30 seconds per module
- **Auto-grading**: < 5 minutes per 100 submissions
- **Accuracy**: 100% grade calculation accuracy
- **Reliability**: 99.9% uptime for critical workflows

### Educational Metrics
- **Student Completion**: > 80% assignment completion rate
- **Feedback Quality**: Student satisfaction > 4.0/5.0
- **Learning Outcomes**: Improved performance on subsequent modules
- **Instructor Efficiency**: 80% reduction in grading time

## Risk Mitigation

### Technical Risks
- **NBGrader Compatibility**: Test thoroughly with latest nbgrader version
- **Performance**: Optimize for large class sizes (100+ students)
- **Data Loss**: Implement backup strategies for submissions
- **Integration Complexity**: Maintain backward compatibility with existing workflow

### Educational Risks
- **Learning Quality**: Ensure auto-grading doesn't compromise learning
- **Cheating Prevention**: Implement appropriate security measures
- **Feedback Quality**: Ensure meaningful feedback generation
- **Student Support**: Provide clear documentation and support

## Next Steps

1. **Review and approve** this plan
2. **Start with Phase 1** (setup module enhancement)
3. **Implement `tito nbgrader init`** command
4. **Create first assignment** from setup module
5. **Test complete workflow** with setup module
6. **Iterate and improve** based on feedback

This plan provides a structured approach to integrating nbgrader with TinyTorch while maintaining the educational quality and philosophy of the existing system. 