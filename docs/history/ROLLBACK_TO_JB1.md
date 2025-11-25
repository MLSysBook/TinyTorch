# Rollback to Jupyter Book 1.x - Complete

**Date:** November 25, 2024

## Summary

Successfully rolled back from Jupyter Book 2.0 (MyST-MD) to Jupyter Book 1.0.4.post1 (Sphinx-based) due to incompatibility issues with custom CSS/JS.

## What Was Done

### 1. ✅ Stopped Jupyter Book 2.0 Server
```bash
pkill -9 -f "jupyter-book"
```

### 2. ✅ Downgraded Jupyter Book
```bash
.venv/bin/pip uninstall -y jupyter-book
.venv/bin/pip install 'jupyter-book==1.0.4.post1'
```

**Verified version:**
```
Jupyter Book      : 1.0.4.post1
External ToC      : 1.0.1
MyST-Parser       : 3.0.1
MyST-NB           : 1.3.0
Sphinx Book Theme : 1.1.4
Jupyter-Cache     : 1.0.1
NbClient          : 0.10.2
```

### 3. ✅ Restored Original Configuration
```bash
cp _config.yml.v1_backup _config.yml
cp _toc.yml.v1_backup _toc.yml
```

### 4. ✅ Built Site with Jupyter Book 1.x
```bash
.venv/bin/jupyter-book build . --all
```

**Result:** Built successfully with 54 warnings (cosmetic only)

### 5. ✅ Served Site
```bash
python -m http.server 8000 --directory _build/html
```

## Current Status

✅ **Site running** at `http://localhost:8000`
✅ **All styling working** - Custom CSS loads properly
✅ **All JavaScript working** - Carousel, timeline, etc.
✅ **45 pages built** successfully

## Why Rollback Was Necessary

Jupyter Book 2.0 (MyST-MD) is a **complete rewrite** with fundamentally different architecture:

| Feature | Jupyter Book 1.x | Jupyter Book 2.0 |
|---------|------------------|------------------|
| **Engine** | Python/Sphinx | Node.js/MyST-MD |
| **Custom CSS** | `_config.yml` → `html.extra_css` | Requires theme customization |
| **Custom JS** | `_config.yml` → `html.extra_js` | Requires MyST plugins |
| **Config Files** | `_config.yml` + `_toc.yml` | `myst.yml` only |
| **Build Command** | `jupyter-book build .` | `jupyter-book start` |
| **Output** | Static HTML | Live dev server |

The migration would have required:
1. Rewriting all custom CSS for new theme system
2. Converting JavaScript to MyST plugins
3. Extensive testing and debugging
4. Time investment not justified for current project stage

## Files Preserved

**Backups created during migration (kept for reference):**
- `_config.yml.v1_backup` - Original Jupyter Book 1.x config
- `_toc.yml.v1_backup` - Original table of contents
- `myst.yml` - Jupyter Book 2.0 config (for future reference)
- `site/MIGRATION_TO_V2.md` - Migration documentation
- `JUPYTER_BOOK_2_FIXES.md` - Issues encountered during migration

## Future Considerations

If migrating to Jupyter Book 2.0 in the future:

1. **Plan for theme customization** - Custom CSS/JS requires different approach
2. **Budget significant time** - Not a simple config change
3. **Test thoroughly** - Completely different rendering engine
4. **Consider benefits vs. cost** - Is 2.0 worth the migration effort?

Jupyter Book 2.0 benefits:
- Modern Node.js-based tooling
- Live reload development server
- Better PDF generation (Typst)
- Client-side search
- Rich hover previews

Current assessment: **Stay on 1.x** until 2.0 matures and migration path is clearer.

## How to Build & Serve Going Forward

### Build Site
```bash
cd /Users/VJ/GitHub/TinyTorch/site
../.venv/bin/jupyter-book build . --all
```

### Serve Locally
```bash
cd /Users/VJ/GitHub/TinyTorch/site
python -m http.server 8000 --directory _build/html
# Open http://localhost:8000
```

### Important: Use .venv Jupyter Book
The system has multiple Python installations. Always use the venv version:
- ✅ **Correct:** `../.venv/bin/jupyter-book`
- ❌ **Wrong:** `jupyter-book` (might use system version)

---

**Status:** ✅ Rollback complete and verified working
