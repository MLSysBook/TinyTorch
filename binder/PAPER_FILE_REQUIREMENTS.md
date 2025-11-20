# Exact File Requirements from paper.tex

## Files Explicitly Mentioned in Paper

Based on line-by-line analysis of `paper/paper.tex`, here are the exact files the paper says should exist:

### Line 988: Repository Instructor Resources

The paper states:
> "The repository includes instructor resources: \texttt{CONTRIBUTING.md} (guidelines for bug reports and curriculum improvements), \texttt{INSTRUCTOR.md} (30-minute setup guide, grading rubrics, common student errors), and \texttt{MAINTENANCE.md} (support commitment through 2027, succession planning for community governance)."

**Required Files**:
1. `CONTRIBUTING.md` - Guidelines for bug reports and curriculum improvements
2. `INSTRUCTOR.md` - 30-minute setup guide, grading rubrics, common student errors
3. ~~`MAINTENANCE.md`~~ - **User doesn't want this** (removed)

### Line 999: TA Guide

The paper states:
> "The repository provides \texttt{TA\_GUIDE.md} documenting frequent student errors (gradient shape mismatches, disconnected computational graphs, broadcasting failures) and debugging strategies."

**Required File**:
4. `TA_GUIDE.md` - Frequent student errors and debugging strategies

### Line 1003: Sample Solutions

The paper states:
> "Sample solutions and grading rubrics in \texttt{INSTRUCTOR.md} calibrate evaluation standards."

**Required Content** (in INSTRUCTOR.md):
- Sample solutions
- Grading rubrics

## Summary: Required Files

| File | Purpose | Status |
|------|---------|--------|
| `CONTRIBUTING.md` | Bug reports, curriculum improvements | ✅ Exists |
| `INSTRUCTOR.md` | Setup guide, grading rubrics, common errors, sample solutions | ✅ Created |
| `TA_GUIDE.md` | Common errors, debugging strategies | ✅ Created |
| `MAINTENANCE.md` | Support commitment | ❌ Removed (user preference) |

## What Each File Should Contain

### CONTRIBUTING.md
- Guidelines for bug reports
- Guidelines for curriculum improvements
- Contribution process

### INSTRUCTOR.md
- 30-minute setup guide
- Grading rubrics
- Common student errors
- Sample solutions (for grading calibration)

### TA_GUIDE.md
- Frequent student errors:
  - Gradient shape mismatches
  - Disconnected computational graphs
  - Broadcasting failures
- Debugging strategies
- TA preparation guidance

## Files NOT Mentioned in Paper

These are NOT required by the paper (but may be useful):
- `TEAM_ONBOARDING.md` - Not explicitly mentioned (but Model 3 is described)
- `MAINTENANCE.md` - Mentioned but user doesn't want it

## Action Items

1. ✅ Remove MAINTENANCE.md (done)
2. ✅ Verify CONTRIBUTING.md matches paper description
3. ✅ Verify INSTRUCTOR.md contains all required content:
   - 30-minute setup guide ✅
   - Grading rubrics ✅
   - Common student errors ✅
   - Sample solutions ⚠️ Need to verify
4. ✅ Verify TA_GUIDE.md contains:
   - Gradient shape mismatches ✅
   - Disconnected computational graphs ✅
   - Broadcasting failures ✅
   - Debugging strategies ✅

