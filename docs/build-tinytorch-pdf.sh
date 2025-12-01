#!/bin/bash
# TinyTorch PDF Build Script
# Builds a high-quality PDF book from TinyTorch educational content
# Maintains single source of truth: src/*/ABOUT.md

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 TinyTorch PDF Book Builder"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check we're in the docs directory
if [ ! -f "_config_pdf.yml" ]; then
    echo "❌ Error: Must run from docs/ directory"
    echo "Usage: cd docs && ./build-tinytorch-pdf.sh"
    exit 1
fi

# Step 1: Ensure module symlinks exist
echo "📂 Step 1/5: Setting up module directory structure..."
mkdir -p modules
cd modules

for module in 01_tensor 02_activations 03_layers 04_losses 05_autograd \
              06_optimizers 07_training 08_dataloader 09_spatial \
              10_tokenization 11_embeddings 12_attention 13_transformers \
              14_profiling 15_quantization 16_compression 17_memoization \
              18_acceleration 19_benchmarking 20_capstone; do
    if [ ! -L "$module" ]; then
        ln -sf "../../modules/$module" "$module"
        echo "  ✓ Created symlink: $module"
    fi
done
cd ..
echo "  ✅ Module structure ready"
echo ""

# Step 2: Verify critical files exist
echo "📋 Step 2/5: Verifying required files..."
MISSING=0

# Check cover and front matter
for file in cover.md preface.md intro.md; do
    if [ ! -f "$file" ]; then
        echo "  ⚠️  Missing: $file"
        MISSING=1
    else
        echo "  ✓ $file"
    fi
done

# Check key chapters
if [ ! -f "chapters/00-introduction.md" ]; then
    echo "  ⚠️  Missing: chapters/00-introduction.md"
    MISSING=1
else
    echo "  ✓ chapters/00-introduction.md"
fi

# Check appendices
for file in chapters/milestones.md quickstart-guide.md tito-essentials.md resources.md; do
    if [ ! -f "$file" ]; then
        echo "  ⚠️  Missing: $file"
        MISSING=1
    else
        echo "  ✓ $file"
    fi
done

# Check module ABOUT files
MODULE_COUNT=$(find modules/*/ABOUT.md 2>/dev/null | wc -l | tr -d ' ')
echo "  📚 Found $MODULE_COUNT/20 module ABOUT.md files"

if [ "$MODULE_COUNT" -lt 20 ]; then
    echo "  ⚠️  Some module ABOUT.md files are missing"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  Warning: Some files are missing. PDF may be incomplete."
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "  ✅ All required files found"
fi
echo ""

# Step 3: Backup and swap to PDF configuration
echo "⚙️  Step 3/5: Activating PDF configuration..."
if [ -f "_config.yml" ]; then
    cp _config.yml _config_web_backup.yml
    echo "  ✓ Backed up _config.yml → _config_web_backup.yml"
fi

if [ -f "_toc.yml" ]; then
    cp _toc.yml _toc_web_backup.yml
    echo "  ✓ Backed up _toc.yml → _toc_web_backup.yml"
fi

cp _config_pdf.yml _config.yml
cp _toc_pdf.yml _toc.yml
echo "  ✅ Using PDF configuration"
echo ""

# Step 4: Clean previous builds
echo "🧹 Step 4/5: Cleaning previous builds..."
if command -v jupyter-book &> /dev/null; then
    jupyter-book clean . --all 2>/dev/null || true
    echo "  ✅ Build directory cleaned"
else
    echo "  ⚠️  jupyter-book not found, skipping clean"
fi
echo ""

# Step 5: Build PDF
echo "📚 Step 5/5: Building PDF (this may take 2-5 minutes)..."
echo ""

BUILD_SUCCESS=0

# Try Method 1: Direct PDF export
echo "Attempting Method 1: Direct PDF export..."
if jupyter-book build --pdf . 2>&1 | tee /tmp/jb-build.log; then
    if [ -f "_build/exports/tinytorch-book.pdf" ]; then
        BUILD_SUCCESS=1
        PDF_PATH="_build/exports/tinytorch-book.pdf"
    fi
fi

# Try Method 2: LaTeX build
if [ $BUILD_SUCCESS -eq 0 ]; then
    echo ""
    echo "Attempting Method 2: LaTeX build via MyST..."

    # First build to HTML/LaTeX
    if jupyter-book build . 2>&1 | tee -a /tmp/jb-build.log; then
        # Check if LaTeX was generated
        if [ -f "_build/latex/book.tex" ] || [ -f "_build/latex/python.tex" ]; then
            echo ""
            echo "LaTeX generated. Attempting PDF compilation..."

            cd _build/latex
            if command -v pdflatex &> /dev/null; then
                # Run pdflatex multiple times for cross-references
                pdflatex -interaction=nonstopmode python.tex || book.tex 2>&1 | tail -20
                pdflatex -interaction=nonstopmode python.tex || book.tex 2>&1 | tail -10

                if [ -f "python.pdf" ]; then
                    mv python.pdf tinytorch-book.pdf
                    BUILD_SUCCESS=1
                    PDF_PATH="../_build/latex/tinytorch-book.pdf"
                elif [ -f "book.pdf" ]; then
                    mv book.pdf tinytorch-book.pdf
                    BUILD_SUCCESS=1
                    PDF_PATH="../_build/latex/tinytorch-book.pdf"
                fi
            else
                echo "⚠️  pdflatex not found. Install LaTeX to compile PDF."
                echo "   macOS: brew install --cask mactex-no-gui"
                echo "   Ubuntu: sudo apt-get install texlive-latex-extra"
            fi
            cd ../..
        fi
    fi
fi

# Try Method 3: HTML to PDF (fallback)
if [ $BUILD_SUCCESS -eq 0 ]; then
    echo ""
    echo "Attempting Method 3: HTML build (PDF requires manual conversion)..."

    if jupyter-book build . 2>&1 | tee -a /tmp/jb-build.log; then
        if [ -f "_build/html/index.html" ]; then
            echo ""
            echo "✅ HTML build successful!"
            echo "📄 HTML output: _build/html/index.html"
            echo ""
            echo "To convert to PDF, you can:"
            echo "  1. Use print-to-PDF from browser (open _build/html/index.html)"
            echo "  2. Use wkhtmltopdf: wkhtmltopdf _build/html/index.html tinytorch-book.pdf"
            echo "  3. Use weasyprint: weasyprint _build/html/index.html tinytorch-book.pdf"
        fi
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Restore original configuration
echo ""
echo "🔄 Restoring original configuration..."
if [ -f "_config_web_backup.yml" ]; then
    mv _config_web_backup.yml _config.yml
    echo "  ✓ Restored _config.yml"
fi
if [ -f "_toc_web_backup.yml" ]; then
    mv _toc_web_backup.yml _toc.yml
    echo "  ✓ Restored _toc.yml"
fi

echo ""

# Report results
if [ $BUILD_SUCCESS -eq 1 ] && [ -n "$PDF_PATH" ] && [ -f "$PDF_PATH" ]; then
    PDF_SIZE=$(du -h "$PDF_PATH" | cut -f1)
    PDF_PAGES=$(pdfinfo "$PDF_PATH" 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SUCCESS! PDF book built successfully!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📄 Output: $PDF_PATH"
    echo "📊 Size: $PDF_SIZE"
    echo "📖 Pages: $PDF_PAGES"
    echo ""
    echo "To view the PDF:"
    echo "  macOS:  open $PDF_PATH"
    echo "  Linux:  xdg-open $PDF_PATH"
    echo "  Windows: start $PDF_PATH"
    echo ""

    # Copy to convenient location
    mkdir -p ../dist
    cp "$PDF_PATH" ../dist/tinytorch-book.pdf
    echo "📦 Also copied to: dist/tinytorch-book.pdf"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  PDF build incomplete"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Build log saved to: /tmp/jb-build.log"
    echo ""
    echo "Common issues:"
    echo "  1. Missing LaTeX: Install MacTeX (macOS) or texlive (Linux)"
    echo "  2. Missing files: Check that all ABOUT.md files exist in modules/"
    echo "  3. MyST version: Ensure jupyter-book is up to date (pip install -U jupyter-book)"
    echo ""
    echo "For detailed errors, check: /tmp/jb-build.log"
    echo ""

    # Check for HTML build as fallback
    if [ -f "_build/html/index.html" ]; then
        echo "💡 HTML version available at: _build/html/index.html"
        echo "   You can print this to PDF from your browser"
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
