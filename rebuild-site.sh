#!/bin/bash
# TinyTorch Site Rebuild Script
# Cleans and rebuilds the Jupyter Book site

echo "🧹 Cleaning old build..."
cd site
rm -rf _build/

echo "🔨 Building site..."

# Determine which jupyter-book to use
JUPYTER_BOOK=""

# Priority 1: Check for venv jupyter-book (most reliable)
if [ -f "../.venv/bin/jupyter-book" ]; then
    JUPYTER_BOOK="../.venv/bin/jupyter-book"
    echo "Using venv jupyter-book: $JUPYTER_BOOK"
# Priority 2: Check for system jupyter-book
elif command -v jupyter-book &> /dev/null; then
    JUPYTER_BOOK="jupyter-book"
    echo "Using system jupyter-book: $JUPYTER_BOOK"
# Priority 3: Check for bin/jupyter-book (if installed in project)
elif [ -f "../bin/jupyter-book" ]; then
    JUPYTER_BOOK="../bin/jupyter-book"
    echo "Using project bin jupyter-book: $JUPYTER_BOOK"
else
    echo "❌ Error: jupyter-book not found!"
    echo ""
    echo "Please install jupyter-book in your venv:"
    echo "   source .venv/bin/activate"
    echo "   pip install jupyter-book"
    echo ""
    echo "Or install system-wide:"
    echo "   pip install jupyter-book"
    exit 1
fi

# Verify _config.yml exists
if [ ! -f "_config.yml" ]; then
    echo "❌ Error: _config.yml not found in site/ directory"
    exit 1
fi

# Build the site
$JUPYTER_BOOK build . --all

BUILD_EXIT_CODE=$?

echo ""
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build complete!"
    echo ""
    echo "📂 To view locally, open: site/_build/html/index.html"
    echo "🌐 Or run: open site/_build/html/index.html"
else
    echo "❌ Build failed with exit code $BUILD_EXIT_CODE"
    exit $BUILD_EXIT_CODE
fi

echo ""
echo "💡 Tip: If navigation doesn't update, try hard refresh (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)"
