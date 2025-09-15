# TinyTorch Agent Module Development Checklist

This checklist ensures all TinyTorch agents follow the complete module development workflow, including proper NBGrader integration for student release functionality.

## 📋 Pre-Development Verification

### Development Environment Setup
- [ ] Read and internalized `MODULE_DEVELOPMENT_GUIDELINES.md`
- [ ] Read and internalized `MARKDOWN_BEST_PRACTICES.md`
- [ ] Read and internalized `MODULE_STRUCTURE_TEMPLATE.md`
- [ ] Read and internalized `NBGRADER_INTEGRATION_GUIDE.md`
- [ ] Understand the complete student release workflow via NBGrader

### Module Planning
- [ ] Learning objectives clearly defined
- [ ] Prerequisite knowledge identified
- [ ] Progressive complexity planned (simple → complex)
- [ ] Real-world connections identified
- [ ] Testing strategy planned

## 🏗️ Module Structure Implementation

### File Structure
- [ ] `[module]_dev.py` follows the percent format structure
- [ ] Correct Jupyter/Jupytext header included
- [ ] `module.yaml` with dependencies and exports
- [ ] README.md (only if explicitly requested)

### Header and Imports
- [ ] **Correct NBGrader metadata for imports cell**:
  ```python
  # %% nbgrader={"grade": false, "grade_id": "[module]-imports", "locked": false, "schema_version": 3, "solution": false, "task": false}
  ```
- [ ] `#| default_exp core.[module_name]` directive
- [ ] `#| export` for package-bound code
- [ ] Dependency imports with try/except fallback pattern
- [ ] Welcome cell with version information

### Module Introduction
- [ ] **Educational markdown follows 5C pattern**:
  - [ ] **Context**: Why this module matters
  - [ ] **Concept**: What we're building technically
  - [ ] **Connection**: How it relates to what students know
  - [ ] **Concrete**: Specific examples with real values
  - [ ] **Confidence**: What success looks like

## 📝 Content Development

### Markdown Documentation (Before EVERY Code Block)
- [ ] **Context**: Why are we implementing this?
- [ ] **Concept**: What is this technically? 
- [ ] **Connection**: How does this relate to what they know?
- [ ] **Concrete**: Specific example with real values
- [ ] **Confidence**: What success looks like
- [ ] Mathematical foundation included where relevant
- [ ] Real-world applications clearly stated
- [ ] Professional connections (PyTorch/TensorFlow equivalents)

### Implementation Cells
- [ ] **Correct NBGrader metadata for implementation cells**:
  ```python
  # %% nbgrader={"grade": false, "grade_id": "[unique-id]", "locked": false, "schema_version": 3, "solution": true, "task": false}
  ```
- [ ] `#| export` directive for package code
- [ ] Clear TODO statements guide implementation
- [ ] **STEP-BY-STEP IMPLEMENTATION** section with concrete actions
- [ ] **EXAMPLE USAGE** with working code examples
- [ ] **IMPLEMENTATION HINTS** with specific functions/approaches
- [ ] **LEARNING CONNECTIONS** to real-world systems
- [ ] **BEGIN SOLUTION / END SOLUTION blocks** properly placed
- [ ] Implementation includes educational comments (why, not what)
- [ ] Error handling with educational messages

## 🧪 Testing Implementation (CRITICAL)

### Immediate Testing Pattern
- [ ] **Every concept tested immediately after implementation**
- [ ] **Test-immediately-after-implementation pattern followed**
- [ ] Test placement: Directly after each implementation cell
- [ ] Tests run automatically (called at cell bottom)

### Test Cell Configuration
- [ ] **Correct NBGrader metadata for test cells**:
  ```python
  # %% nbgrader={"grade": true, "grade_id": "test-[concept]-immediate", "locked": true, "points": [N], "schema_version": 3, "solution": false, "task": false}
  ```
- [ ] **All test cells are locked** (`"locked": true`)
- [ ] **Points assigned appropriately** (5-15 for unit tests, 15-25 for integration)
- [ ] **Unique grade_id for every test cell**
- [ ] Clear test descriptions explaining what's being validated
- [ ] Comprehensive assertions with descriptive error messages
- [ ] Success messages that build confidence

### Test Hierarchy Implementation
- [ ] **Unit Tests** (`test_unit_[function_name]`): Individual functions tested immediately
- [ ] **Comprehensive Tests** (`test_unit_[module]_comprehensive`): Multiple functions together
- [ ] **Integration Tests** (`test_module_[module]_[component]_integration`): Cross-module compatibility

### Test Quality Standards
- [ ] Descriptive test names following convention
- [ ] Clear print statements showing test progress
- [ ] Comprehensive coverage: normal cases, edge cases, integration scenarios
- [ ] Assertions with descriptive error messages
- [ ] Success confirmations with specific behavior validation

## ✅ NBGrader Integration Verification

### Metadata Consistency
- [ ] **Every cell has NBGrader metadata**
- [ ] **All grade_id values are unique** across the entire notebook
- [ ] **schema_version: 3** used throughout
- [ ] **Implementation cells**: `"grade": false, "solution": true`
- [ ] **Test cells**: `"grade": true, "locked": true, "points": N`
- [ ] **Documentation cells**: `"grade": false, "solution": false`

### Solution Block Requirements
- [ ] **BEGIN SOLUTION / END SOLUTION** blocks in all implementation cells
- [ ] **Solution blocks contain complete, working implementations**
- [ ] **Clear comments explaining approach within solution blocks**
- [ ] **Solution blocks follow TinyTorch coding standards**
- [ ] **All TODOs are outside the solution blocks**

### Student Release Compatibility
- [ ] **All implementation cells have `"solution": true`**
- [ ] **All test cells have `"locked": true`**
- [ ] **Clear TODOs provide guidance without giving away solution**
- [ ] **Scaffolding (STEP-BY-STEP, EXAMPLES, HINTS) outside solution blocks**
- [ ] **Tests provide immediate feedback on student implementations**

## 🔗 Professional Integration

### Real-World Connections
- [ ] PyTorch equivalent functionality identified and explained
- [ ] TensorFlow equivalent functionality identified and explained
- [ ] Industry applications clearly articulated
- [ ] Production patterns and practices taught
- [ ] Professional development practices demonstrated

### Package Integration
- [ ] Proper import pattern with try/except fallback
- [ ] `#| export` directives for package-bound code
- [ ] Module.yaml dependencies correctly specified
- [ ] Integration tests verify cross-module compatibility
- [ ] Type consistency maintained across module boundaries

## 📊 Quality Assurance

### Code Quality
- [ ] Clean, readable implementations that teach
- [ ] Consistent naming conventions throughout
- [ ] Appropriate error handling with educational messages
- [ ] Performance considerations explained where relevant
- [ ] Memory efficiency considered

### Educational Effectiveness
- [ ] **Implement → Test → Implement → Test pattern** followed strictly
- [ ] Progressive complexity from simple to advanced
- [ ] Each step builds logically on previous steps
- [ ] Real-world relevance clear and compelling
- [ ] Students can complete without prior implementation knowledge

### Module Completion
- [ ] **Module Summary** section with achievements highlighted
- [ ] **Learning outcomes** clearly articulated
- [ ] **Next steps** and connections to subsequent modules
- [ ] **Professional skills developed** emphasized
- [ ] **Real-world impact** demonstrated

## 🚀 Final Validation

### NBGrader Workflow Testing
- [ ] **Module can be converted to student notebook** (solution blocks removed)
- [ ] **All tests remain locked and functional**
- [ ] **Student version provides clear guidance**
- [ ] **Automatic grading works correctly**
- [ ] **Point distribution is appropriate**

### Export and Integration
- [ ] Code exports properly with `tito export`
- [ ] All tests pass with `tito test`
- [ ] Integration with other modules verified
- [ ] Package imports work correctly
- [ ] Documentation builds without errors

### Student Experience Validation
- [ ] Students can implement without seeing solutions
- [ ] Immediate feedback helps identify and fix issues
- [ ] Learning progression is clear and logical
- [ ] Real-world connections motivate learning
- [ ] Professional skills are developed

## 🎯 Success Metrics

A successful TinyTorch module achieves:
- **>70% students implement correctly on first attempt**
- **Clear understanding of concepts (not just coding)**
- **Zero breaking changes with other modules**
- **Positive student feedback on relevance and clarity**
- **Professional practices clearly demonstrated**

## ⚠️ Common Pitfalls to Avoid

### NBGrader Issues
- [ ] Duplicate grade_id values (causes autograding failures)
- [ ] Missing metadata on cells (prevents proper processing)
- [ ] BEGIN/END SOLUTION blocks in wrong location
- [ ] Test cells not locked (students could modify)
- [ ] Points not assigned to test cells

### Educational Issues
- [ ] Implementation without immediate testing
- [ ] Vague TODOs without specific guidance
- [ ] Missing real-world connections
- [ ] Complex concepts without progressive building
- [ ] Error messages that don't help learning

### Technical Issues
- [ ] Import patterns that break in package environment
- [ ] Type inconsistencies across operations
- [ ] Missing export directives for package code
- [ ] Integration failures with other modules
- [ ] Performance issues without explanation

## 📚 Reference Documentation

This checklist is based on:
- `MODULE_DEVELOPMENT_GUIDELINES.md` - Core development principles
- `MARKDOWN_BEST_PRACTICES.md` - 5C pattern and educational scaffolding
- `MODULE_STRUCTURE_TEMPLATE.md` - Exact structure requirements
- `NBGRADER_INTEGRATION_GUIDE.md` - Student release workflow

## 🤝 Agent Collaboration

### Handoffs
- **From Education Architect**: Learning objectives and progression plan
- **To Quality Assurance**: Complete implementation ready for testing
- **Feedback Loop**: Iterate based on test results and student feedback

Remember: **Every line of code should teach through doing.** Students should understand not just HOW to implement ML systems, but WHY design decisions matter and WHAT impact they have on real-world applications.

The goal is educational masterpieces - code that teaches, guides, and empowers students to build their own ML framework while understanding every decision along the way.