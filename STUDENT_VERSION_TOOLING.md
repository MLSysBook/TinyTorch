# Student Version Generation Tooling

## Status: 🚧 Available but Untested

TinyTorch includes tooling to generate student versions from complete implementations, but these are **not yet validated for classroom use**.

## Current State

### ✅ What Exists
- **NBGrader integration**: `tito nbgrader generate [module]`
- **Student notebook generator**: `bin/generate_student_notebooks.py`
- **Solution markers**: Modules use `### BEGIN SOLUTION` / `### END SOLUTION`
- **Release workflow**: `tito nbgrader release [module]`

### 🚧 What's Not Ready
- Student versions have not been tested with actual students
- NBGrader autograding rubrics need validation
- Generated notebooks may need manual review

## Planned Workflow (Untested)

### For Instructors (Future)

```bash
# 1. Generate student version of a module
tito nbgrader generate 01_tensor

# 2. Release to students (removes solutions)
tito nbgrader release 01_tensor

# 3. Collect student submissions
tito nbgrader collect 01_tensor

# 4. Auto-grade submissions
tito nbgrader autograde 01_tensor

# 5. Generate feedback
tito nbgrader feedback 01_tensor
```

### Current Best Practice

**December 2024 Release**: Focus on complete implementations for pedagogical review

**Spring 2025**: Validate student version generation with pilot users

**Fall 2025**: Full classroom deployment with tested workflows

## Student Version Features (Planned)

When mature, student versions will:
- Remove solutions between `### BEGIN SOLUTION` / `### END SOLUTION` markers
- Replace with `raise NotImplementedError()` stubs
- Preserve tests and validation cells
- Include hints and learning guidance
- Support NBGrader auto-grading

## Contributing

If you're interested in helping validate student version workflows:
1. Try generating student versions for modules 01-07
2. Report issues with missing hints, unclear instructions, or broken tests
3. Suggest improvements to the pedagogical flow

## Architecture

### Markers System
```python
### BEGIN SOLUTION
# Complete implementation here
# This will be removed in student version
### END SOLUTION

### BEGIN HIDDEN TESTS
# Tests hidden from students
# Used for auto-grading
### END HIDDEN TESTS
```

### Generation Process
1. Parse `_dev.py` files in `modules/source/`
2. Convert to notebooks using Jupytext
3. Remove solution blocks
4. Add NBGrader metadata
5. Output to `assignments/release/`

## Why Not Ship Student Versions Now?

1. **Focus on pedagogy first**: Get feedback on complete implementations
2. **Avoid half-baked releases**: Student versions need proper testing
3. **Community validation**: Let others help identify gaps before classroom use
4. **Honest communication**: Better to say "untested" than to ship broken workflows

## Timeline

- **December 2024**: Release complete implementations + this documentation
- **January-March 2025**: Community testing of student version generation
- **April 2025**: Validated student version workflows
- **Fall 2025**: Full classroom deployment

---

**For December reviewers**: Please focus feedback on:
- Pedagogical progression (modules 01-20)
- Implementation quality and correctness
- Documentation clarity
- Learning objectives and systems thinking

Student version feedback is welcome but secondary for now.

