# Book Directory Cleanup Summary

Date: November 7, 2025
Branch: website-content-improvements

## Files Deleted (Duplicates)

### 1. user-manual.md (17K)
- **Reason**: Complete duplicate of quickstart-guide.md
- **Status**: quickstart-guide.md is in TOC and actively maintained

### 2. instructor-guide.md (12K)
- **Reason**: Duplicate of usage-paths/classroom-use.md
- **Status**: classroom-use.md is in TOC ("For Instructors")

### 3. leaderboard.md (6.2K)
- **Reason**: Old "TinyTorch Olympics" content
- **Status**: Superseded by community.md and Module 20 (MLPerf® Edu Competition)

**Total Deleted**: ~35KB of duplicate content

## Files Archived (Development/Reference)

Moved to: `docs/archive/book-development/`

### Development Files:
- THEME_DESIGN.md (4.5K) - Design documentation
- convert_modules.py (17K) - Build script
- convert_readmes.py (11K) - Build script  
- verify_build.py (3.2K) - Build script

### Documentation (Not in TOC):
- faq.md (18K) - FAQ content (may add to TOC later)
- kiss-principle.md (6.7K) - Design philosophy
- vision.md (7.3K) - Project vision document

### Unused Usage Paths:
- quick-exploration.md (2.7K) - Alternative usage path
- serious-development.md (6.6K) - Alternative usage path
- **Note**: Only classroom-use.md is in active TOC

**Total Archived**: ~77KB of reference content

## Images Archived

Moved to: `book/_static/archive/`

- Gemini_Generated_Image_1as0881as0881as0.png
- Gemini_Generated_Image_b34tigb34tigb34t.png
- Gemini_Generated_Image_b34tiib34tiib34t.png

**Reason**: AI-generated images not used in current site

## Files Remaining in book/ (All Active)

### Root Level (In TOC):
- intro.md (12K) - Homepage
- quickstart-guide.md (8.4K) - Getting Started
- tito-essentials.md (8.9K) - CLI reference
- learning-progress.md (6.4K) - Progress tracking
- checkpoint-system.md (11K) - Checkpoint system
- testing-framework.md (13K) - Testing guide
- resources.md (5.4K) - Additional resources
- community.md (1.5K) - Community page

### Subdirectories:
- chapters/ (21 files) - All 20 modules + introduction
- chapters/milestones.md - Referenced in intro
- appendices/api-reference.md - API documentation
- usage-paths/classroom-use.md - Instructor guide (in TOC)

### Assets:
- logo-tinytorch-*.png (3 files) - Active logos
- tensortorch.png - Project image
- _static/ - CSS, JS, favicon

## Result

**Before Cleanup**: 39 markdown files in book/
**After Cleanup**: 29 markdown files (26% reduction)

**Benefits**:
- ✅ No duplicate content
- ✅ Clear separation of active vs archived content  
- ✅ Easier maintenance
- ✅ Cleaner repository structure
- ✅ All active files are in TOC or properly referenced

## Files to Consider Adding to TOC

If we want to surface this content:
- appendices/api-reference.md - Could add to Resources section
- chapters/milestones.md - Already referenced, could add to TOC
- docs/archive/book-development/faq.md - Could add if FAQ is needed

## Notes

All archived files are preserved and can be:
1. Restored if needed
2. Referenced in documentation
3. Updated and added to TOC later
4. Used as reference for future content

