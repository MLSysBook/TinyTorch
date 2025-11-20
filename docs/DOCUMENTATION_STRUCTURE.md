# Documentation Structure - Single Source of Truth

## Site Documentation (`site/`)
**Purpose**: User-facing website content (built with Jupyter Book)

**Files**:
- `site/community.md` - Community features for website visitors
- `site/quickstart-guide.md` - Quick start guide
- `site/student-workflow.md` - Student workflow guide
- `site/instructor-guide.md` - Instructor guide (copied from docs/)
- `site/usage-paths/classroom-use.md` - Classroom usage guide

**Build**: These files are built into the website via `make html` in `site/`

## Developer Documentation (`docs/`)
**Purpose**: Technical documentation for developers and experts

**Files**:
- `docs/COMMUNITY_BENCHMARK_IMPLEMENTATION.md` - Full implementation details
- `docs/EXPERT_FEEDBACK_ANALYSIS.md` - Expert feedback analysis
- `docs/EXPERT_FEEDBACK_REQUEST.md` - Questions for experts
- `docs/PRIVACY_DATA_RETENTION.md` - Privacy policy
- `docs/CONFIGURATION_SETUP.md` - Configuration guide
- `docs/COMMUNITY_FEATURES_SUMMARY.md` - Quick summary

**Note**: These are NOT included in the website build - they're for developers/experts

## Root Documentation
**Purpose**: Repository-level documentation

**Files**:
- `README.md` - Main repository README
- `CONTRIBUTING.md` - Contribution guidelines
- `INSTRUCTOR.md` - Instructor guide (root copy)
- `TA_GUIDE.md` - TA guide (root copy)

## Single Source Principle

**Site files** (`site/*.md`):
- ✅ Single source: `site/community.md` is the ONLY community page for website
- ✅ No duplicates in `docs/` for website content

**Developer docs** (`docs/*.md`):
- ✅ Technical details for developers
- ✅ NOT built into website (separate purpose)

**Root docs** (`*.md`):
- ✅ Repository-level documentation
- ✅ Referenced by paper.tex

## File Locations Summary

| Content Type | Location | Purpose | Built into Site? |
|-------------|----------|---------|------------------|
| Community features | `site/community.md` | Website page | ✅ Yes |
| Quick start | `site/quickstart-guide.md` | Website page | ✅ Yes |
| Student workflow | `site/student-workflow.md` | Website page | ✅ Yes |
| Implementation details | `docs/COMMUNITY_*.md` | Developer docs | ❌ No |
| Privacy policy | `docs/PRIVACY_*.md` | Developer docs | ❌ No |
| Expert feedback | `docs/EXPERT_*.md` | Developer docs | ❌ No |

**All documentation is in the correct location with no duplicates.**
