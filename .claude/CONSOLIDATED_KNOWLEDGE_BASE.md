# TinyTorch Consolidated Knowledge Base

This document consolidates all critical knowledge for TinyTorch development. The information here has been embedded into individual agent knowledge bases in `.claude/agents/`.

## Quick Access

### Agent Knowledge Bases (Primary References)
- `.claude/agents/module-developer.md` - Complete implementation guidelines
- `.claude/agents/education-architect.md` - Learning design principles  
- `.claude/agents/quality-assurance.md` - Validation requirements
- `.claude/agents/devops-engineer.md` - Release management
- `.claude/agents/documentation-publisher.md` - Publishing standards

### System References
- `.claude/AGENT_REFERENCE.md` - Agent team structure and workflow
- `.claude/GIT_WORKFLOW_STANDARDS.md` - Git branching and commits

## Core Knowledge Summary

### NBGrader Integration
**Purpose**: Enable automatic generation of student versions where solutions are removed.

**Key Points**:
- Solutions go between `### BEGIN SOLUTION` and `### END SOLUTION`
- Scaffolding (TODOs, HINTS) must be OUTSIDE solution blocks
- Test cells must be locked: `"locked": true`
- Every cell needs unique `grade_id`
- Schema version is always 3

### The 5 C's Educational Pattern
Before every implementation:
1. **Context** - Why are we implementing this?
2. **Concept** - What is this technically?
3. **Connection** - How does this relate to prior knowledge?
4. **Concrete** - Specific examples with real values
5. **Confidence** - What success looks like

### Test-Immediately Pattern
```
Introduce Concept → Implementation → IMMEDIATE Test → Next Concept
```
- Never implement large chunks without validation
- Tests named: `test_unit_[function_name]()`
- Tests run at cell bottom
- Educational error messages

### Module Structure Template
```python
# 1. Module Introduction
# Learning goals, Build→Use→Understand

# 2. Setup and Imports
# NBGrader metadata, fallback imports

# 3. Development Section
# For each concept:
#   - Markdown with 5 C's
#   - Implementation with scaffolding
#   - Immediate test
#   - Checkpoint

# 4. Integration Tests
# Cross-module validation

# 5. Module Summary
# Achievements, skills, next steps
```

### NBGrader Metadata Reference
```python
# Implementation cells
{"grade": false, "grade_id": "unique-id", "locked": false, 
 "schema_version": 3, "solution": true, "task": false}

# Test cells  
{"grade": true, "grade_id": "test-unique-id", "locked": true,
 "points": 10, "schema_version": 3, "solution": false, "task": false}

# Documentation cells
{"grade": false, "grade_id": "doc-id", "locked": false,
 "schema_version": 3, "solution": false, "task": false}
```

### Implementation Scaffolding Pattern
```python
def method(self, params):
    """
    [Description]
    
    TODO: [Clear task]                    # OUTSIDE solution
    
    APPROACH:                             # OUTSIDE solution
    1. [Step with reason]
    2. [Step with reason]
    
    EXAMPLE:                              # OUTSIDE solution
    ```python
    result = method([1, 2, 3])
    # Expected: [4, 5, 6]
    ```
    
    HINTS:                                # OUTSIDE solution
    - Use np.function()
    - Handle edge case
    """
    ### BEGIN SOLUTION
    # Actual implementation (removed for students)
    actual_code = here()
    ### END SOLUTION
```

## Workflow Commands

### Development
```bash
tito module create [name]
tito module test [name]
tito module validate [name]
```

### Release
```bash
nbgrader generate_assignment [name]  # Create student version
nbgrader validate [name]            # Check structure
nbgrader autograde [name]           # Grade submissions
```

### Documentation
```bash
jupyter-book build docs/
ghp-import -n -p -f docs/_build/html
```

## Quality Checklist

### Must Have
- [ ] Unique grade_ids
- [ ] Solution blocks with complete code
- [ ] Scaffolding outside solutions
- [ ] Locked test cells
- [ ] Points assigned
- [ ] 5 C's before implementations
- [ ] Immediate tests
- [ ] Integration tests

### Should Have  
- [ ] Checkpoints after implementations
- [ ] Real-world connections
- [ ] PyTorch/TensorFlow comparisons
- [ ] Performance considerations
- [ ] Module summary

## Important Notes

1. **Agents have embedded knowledge**: Each agent's `.md` file in `.claude/agents/` contains all necessary information
2. **No external lookups needed**: Agents don't need to read separate documentation
3. **Single source of truth**: Agent knowledge bases are the authoritative reference
4. **Workflow is codified**: NBGrader → Student Release → Autograding

## Deprecation Notice

The following standalone documentation files have been consolidated into agent knowledge bases:
- MODULE_DEVELOPMENT_GUIDELINES.md → Embedded in agents
- NBGRADER_INTEGRATION_GUIDE.md → Embedded in agents  
- MARKDOWN_BEST_PRACTICES.md → Embedded in agents
- MODULE_STRUCTURE_TEMPLATE.md → Embedded in agents
- AGENT_MODULE_CHECKLIST.md → Embedded in agents

These files are retained for historical reference but agents should use their embedded knowledge bases.