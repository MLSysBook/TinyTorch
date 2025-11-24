# TinyTorch Release Check Workflow

## Overview

The **Release Check** workflow is a comprehensive quality assurance system that validates TinyTorch meets all educational, technical, and documentation standards before any release.

## Workflow Structure

The workflow consists of **6 parallel quality gates** that run sequentially to ensure comprehensive validation:

```
Educational Standards → Implementation Standards → Testing Standards
        ↓                         ↓                       ↓
Package Integration → Documentation → Systems Analysis → Release Report
```

### Quality Gates

#### 1. Educational Validation
- ✅ Module structure and learning objectives
- ✅ Progressive disclosure patterns (no forward references)
- ✅ Cognitive load management
- ✅ NBGrader compatibility

#### 2. Implementation Validation
- ✅ Time estimate consistency (LEARNING_PATH.md ↔ ABOUT.md)
- ✅ Difficulty rating consistency
- ✅ Testing patterns (test_unit_*, test_module())
- ✅ Dependency chain validation
- ✅ NBGrader metadata

#### 3. Test Validation
- ✅ All unit tests passing
- ✅ Integration tests passing
- ✅ Checkpoint validation
- ✅ Test coverage ≥80%

#### 4. Package Validation
- ✅ Export directives correct
- ✅ Import paths consistent
- ✅ Package builds successfully
- ✅ Installation works

#### 5. Documentation Validation
- ✅ ABOUT.md files consistent
- ✅ Checkpoint markers in long modules
- ✅ Jupyter Book builds successfully

#### 6. Systems Analysis Validation
- ✅ Memory profiling present
- ✅ Performance analysis included
- ✅ Production context provided

## Triggering the Workflow

### Manual Trigger (Recommended for Releases)

```bash
# Via GitHub UI:
# 1. Go to Actions → TinyTorch Release Check
# 2. Click "Run workflow"
# 3. Select:
#    - Release Type: patch | minor | major
#    - Check Level: quick | standard | comprehensive
```

### Automatic Trigger (PRs)

The workflow runs automatically on:
- Pull requests to `main` or `dev` branches
- When PRs are opened or synchronized

## Check Levels

### Quick (5-10 minutes)
- Essential validations only
- Time estimates, difficulty ratings, testing patterns
- Good for: Small fixes, documentation updates

### Standard (15-20 minutes) - **Default**
- All quality gates
- Complete validation suite
- Good for: Regular releases, feature additions

### Comprehensive (30-40 minutes)
- Extended testing
- Performance benchmarks
- Full documentation rebuild
- Good for: Major releases, significant changes

## Running Locally

You can run individual validation scripts before pushing:

```bash
# Time estimates
python .github/scripts/validate_time_estimates.py

# Difficulty ratings
python .github/scripts/validate_difficulty_ratings.py

# Testing patterns
python .github/scripts/validate_testing_patterns.py

# Checkpoint markers
python .github/scripts/check_checkpoints.py
```

## Validation Scripts

Located in `.github/scripts/`:

### Core Validators (Fully Implemented)
- `validate_time_estimates.py` - Time consistency across docs
- `validate_difficulty_ratings.py` - Star rating consistency
- `validate_testing_patterns.py` - test_unit_* and test_module() patterns
- `check_checkpoints.py` - Checkpoint markers in long modules (8+ hours)

### Stub Validators (To Be Implemented)
- `validate_educational_standards.py` - Learning objectives, scaffolding
- `check_learning_objectives.py` - Objective alignment
- `check_progressive_disclosure.py` - No forward references
- `validate_dependencies.py` - Module dependency chain
- `validate_nbgrader.py` - NBGrader metadata
- `validate_exports.py` - Export directive validation
- `validate_imports.py` - Import path consistency
- `validate_documentation.py` - ABOUT.md validation
- `validate_systems_analysis.py` - Memory/performance/production analysis

## Release Report

After all gates pass, the workflow generates a comprehensive **Release Readiness Report**:

```markdown
# TinyTorch Release Readiness Report

✅ Educational Standards
✅ Implementation Standards
✅ Testing Standards
✅ Package Integration
✅ Documentation
✅ Systems Analysis

Status: APPROVED FOR RELEASE
```

The report is:
- ✅ Uploaded as workflow artifact
- ✅ Posted as PR comment (if applicable)
- ✅ Includes quality metrics and module inventory

## Integration with Agent Workflow

This GitHub Actions workflow complements the manual agent review process:

### Agent-Driven Reviews (Pre-Release)
```
TPM coordinates:
├── Education Reviewer → Pedagogical validation
├── Module Developer → Implementation review
├── Quality Assurance → Testing validation
└── Package Manager → Integration check
```

### Automated CI/CD (Every Commit/PR)
```
GitHub Actions runs:
├── Educational Validation
├── Implementation Validation
├── Test Validation
├── Package Validation
├── Documentation Validation
└── Systems Analysis Validation
```

## Failure Handling

If any quality gate fails:

1. **Workflow stops** at the failed gate
2. **Error details** are displayed in the job log
3. **PR is blocked** (if configured)
4. **Notifications** sent to team

To fix:
1. Review the failed job log
2. Run the specific validation script locally
3. Fix the identified issues
4. Push changes
5. Workflow re-runs automatically

## Configuration

### Branch Protection

Recommended settings for `main` and `dev` branches:

```yaml
# In GitHub Repository Settings → Branches
- Require status checks to pass before merging
  ✓ TinyTorch Release Check / educational-validation
  ✓ TinyTorch Release Check / implementation-validation
  ✓ TinyTorch Release Check / test-validation
  ✓ TinyTorch Release Check / package-validation
  ✓ TinyTorch Release Check / documentation-validation
```

### Workflow Permissions

The workflow requires:
- ✅ Read access to repository
- ✅ Write access to pull requests (for comments)
- ✅ Artifact upload permissions

## Continuous Improvement

The validation scripts are designed to evolve:

### Adding New Validators

1. Create script in `.github/scripts/`
2. Add to appropriate job in `release-check.yml`
3. Update this README
4. Test locally before committing

### Enhancing Existing Validators

1. Update script logic
2. Add tests for the validator itself
3. Document new checks in README
4. Version the changes

## Success Metrics

### Educational Excellence
- All modules have consistent metadata
- Progressive disclosure maintained
- Cognitive load appropriate

### Technical Quality
- All tests passing
- Package builds and installs correctly
- Integration validated

### Documentation Quality
- All ABOUT.md files complete
- Checkpoint markers in place
- Jupyter Book builds successfully

## Troubleshooting

### Common Issues

**"Time estimate mismatch"**
- Check LEARNING_PATH.md and module ABOUT.md
- Ensure format: "X-Y hours" (with space)

**"Missing test_module()"**
- Add integration test at end of module
- Must be named exactly `test_module()`

**"Checkpoint markers recommended"**
- Informational only for modules 8+ hours
- Add 2+ checkpoint markers in ABOUT.md

**"Build failed"**
- Check for Python syntax errors
- Verify all dependencies in requirements.txt

## Related Documentation

- [Agent Descriptions](../.claude/agents/README.md)
- [Module Development Guide](../../modules/DEFINITIVE_MODULE_PLAN.md)
- [Contributing Guidelines](../../CONTRIBUTING.md)

---

**Maintained by:** TinyTorch Team
**Last Updated:** 2024-11-24
**Version:** 1.0.0
