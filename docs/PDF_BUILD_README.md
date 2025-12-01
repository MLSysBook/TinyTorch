# TinyTorch PDF Book - Build Instructions

## Quick Start

To build the PDF book from the TinyTorch educational content:

```bash
cd docs
./build-tinytorch-pdf.sh
```

The script will generate a PDF at `dist/tinytorch-book.pdf`.

## What Was Built

**Proof-of-Concept PDF Generated!** ✅

- **File**: `dist/tinytorch-book-draft.pdf`
- **Size**: 598 KB
- **Pages**: 220 pages
- **Format**: Letter (8.5" × 11")
- **Status**: Draft with some LaTeX formatting issues to fix

## Current PDF Structure

```
Cover: "TinyTorch: Don't Import It. Build It."
Preface
Introduction
Course Overview

PART I: Foundation Tier (Modules 01-07)
- 01. Tensor
- 02. Activations
- 03. Layers
- 04. Losses
- 05. Autograd
- 06. Optimizers
- 07. Training

PART II: Architecture Tier (Modules 08-13)
- 08. DataLoader
- 09. Spatial Convolutions
- 10. Tokenization
- 11. Embeddings
- 12. Attention
- 13. Transformers

PART III: Optimization Tier (Modules 14-19)
- 14. Profiling
- 15. Quantization
- 16. Compression
- 17. Memoization
- 18. Acceleration
- 19. Benchmarking

PART IV: Capstone
- 20. MLPerf® Edu Competition

APPENDICES
- A. Historical Milestones
- B. Quick Start Guide
- C. TITO CLI Reference
- D. Additional Resources
```

## Infrastructure Created

### 1. Directory Structure
- `docs/modules/` - Symlinks to `../modules/*/` for all 20 modules
- Module ABOUT.md files copied from `src/` to `modules/`

### 2. Configuration Files
- **`_config_pdf.yml`** - PDF-specific Jupyter Book configuration
- **`_toc_pdf.yml`** - Table of contents with correct module paths
- **`cover.md`** - Professional title page
- **`tito-essentials.md`** - CLI reference appendix

### 3. Build Script
- **`build-tinytorch-pdf.sh`** - Automated PDF builder
  - Sets up directory structure
  - Verifies required files
  - Attempts multiple build methods
  - Restores original configuration
  - Reports build status

## Single Source of Truth Maintained

The workflow preserves src/ as the authoritative source:

```
src/01_tensor/ABOUT.md  ← SINGLE SOURCE OF TRUTH
        ↓
    [manual copy or nbdev export]
        ↓
modules/01_tensor/ABOUT.md
        ↓
    [symlink from docs/modules/]
        ↓
docs/modules/01_tensor/ABOUT.md
        ↓
    [jupyter-book build --pdf]
        ↓
_build/exports/docs.pdf → dist/tinytorch-book.pdf
```

## Known Issues (To Fix for Production)

### LaTeX Formatting Errors
The current PDF has ~50 LaTeX errors related to:
- **Bullet lists in math mode**: Some markdown lists are being interpreted as math
- **Probable cause**: Underscore characters (`_`) in text triggering math mode

### Solutions:
1. **Quick fix**: Escape underscores in module ABOUT.md files (`\_` instead of `_`)
2. **Better fix**: Add LaTeX preamble to handle underscores correctly
3. **Best fix**: Use custom LaTeX template with proper formatting rules

## Improving the PDF

### Phase 1: Fix Formatting (2-3 hours)
1. Fix underscore/math mode issues in ABOUT.md files
2. Adjust LaTeX configuration for better list handling
3. Test rebuild

### Phase 2: Polish Content (4-6 hours)
1. Add professional cover design
2. Improve typography (fonts, spacing, margins)
3. Add page headers/footers with module names
4. Create proper index
5. Fix any remaining formatting issues

### Phase 3: Production Ready (8-10 hours)
1. Professional proofreading
2. Technical accuracy review
3. Add ISBN (if selling)
4. Copyright and licensing pages
5. Final design polish

## Build Methods Attempted

The build script tries three methods in order:

### Method 1: Direct PDF Export (MyST-MD v2.0)
```bash
jupyter-book build --pdf .
```
**Status**: ✅ Works but has LaTeX errors
**Output**: `_build/exports/docs.pdf`

### Method 2: LaTeX Compilation
```bash
jupyter-book build .  # Generate LaTeX
cd _build/latex
pdflatex python.tex
```
**Status**: ❌ Failed (no LaTeX generated in this version)

### Method 3: HTML Fallback
```bash
jupyter-book build .
# Then print HTML to PDF manually
```
**Status**: ❌ Not triggered (Method 1 succeeded)

## File Locations

- **Source content**: `src/*/ABOUT.md` (single source of truth)
- **Module copies**: `modules/*/ABOUT.md`
- **Build config**: `docs/_config_pdf.yml`, `docs/_toc_pdf.yml`
- **Build script**: `docs/build-tinytorch-pdf.sh`
- **PDF output**: `dist/tinytorch-book-draft.pdf`
- **Build logs**: `/tmp/jb-build.log`

## Customization

### Change Title/Branding
Edit `docs/_config_pdf.yml`:
```yaml
title: "TinyTorch: Don't Import It. Build It."
author: "Prof. Vijay Janapa Reddi (Harvard University)"
```

### Change Module Selection
Edit `docs/_toc_pdf.yml` to add/remove chapters

### Change LaTeX Settings
Edit `docs/_config_pdf.yml`:
```yaml
latex:
  latex_engine: pdflatex
  latex_elements:
    papersize: 'letterpaper'
    pointsize: '10pt'
    preamble: |
      \usepackage{fancyhdr}
      # Add custom LaTeX here
```

## Next Steps

1. **Review the draft PDF**: `open dist/tinytorch-book-draft.pdf`
2. **Identify formatting issues**: Check which modules have problems
3. **Fix LaTeX errors**: Start with underscore escaping
4. **Rebuild**: Run `./build-tinytorch-pdf.sh` again
5. **Iterate**: Repeat until satisfactory

## Dependencies

- Python 3.8+
- jupyter-book v2.0+
- LaTeX distribution (MacTeX on macOS, texlive on Linux)
- pdfinfo (from poppler-utils)

Install with:
```bash
pip install jupyter-book
brew install --cask mactex-no-gui  # macOS
# OR
sudo apt-get install texlive-latex-extra  # Ubuntu
```

## FAQ

**Q: Why 220 pages instead of estimated 350-450?**
A: This is a draft with only the ABOUT.md module descriptions. A full book would include:
- More extensive front matter
- Code examples with explanations
- Exercise solutions
- Expanded appendices
- Index

**Q: Can I sell this PDF?**
A: The current draft needs significant polish for commercial sale:
- Fix all LaTeX errors
- Professional proofreading
- Cover design
- ISBN registration
- Copyright page
- Quality assurance

**Q: How do I update content?**
A: Edit the source files in `src/*/ABOUT.md`, copy to `modules/`, then rebuild:
```bash
cp src/01_tensor/ABOUT.md modules/01_tensor/
cd docs
./build-tinytorch-pdf.sh
```

**Q: Can I use a different PDF generator?**
A: Yes! The markdown content can be used with:
- Pandoc → LaTeX → PDF
- Sphinx → LaTeX → PDF
- mdBook → PDF
- Custom LaTeX pipeline

The current approach (Jupyter Book/MyST-MD) was chosen because the infrastructure was already 80% complete.

---

**Generated**: November 30, 2025
**PDF Status**: Proof-of-concept draft (220 pages, 598 KB)
**Next Action**: Review draft and fix LaTeX formatting errors
