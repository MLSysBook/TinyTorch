#!/bin/bash
# TinyTorch Website Build Script
# Jupyter Book 1.x (Sphinx) Build System
# Quick and easy: ./site/build.sh (from root) or ./build.sh (from site/)

set -e  # Exit on error

echo "🏗️  Building TinyTorch documentation website (Jupyter Book 1.x)..."
echo ""

# Detect where we're running from and navigate to site directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR=""
PROJECT_ROOT=""

if [ -f "_config.yml" ]; then
    # Already in site directory
    SITE_DIR="$(pwd)"
    PROJECT_ROOT="$(dirname "$SITE_DIR")"
elif [ -f "site/_config.yml" ]; then
    # In root directory
    PROJECT_ROOT="$(pwd)"
    SITE_DIR="$(pwd)/site"
    cd "$SITE_DIR"
    echo "📂 Changed to site directory: $SITE_DIR"
else
    echo "❌ Error: Cannot find site directory with _config.yml"
    echo "   Run from project root or site/ directory"
    exit 1
fi

# Activate virtual environment if it exists and we're not already in it
if [ -z "$VIRTUAL_ENV" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Recommend running: source scripts/activate-tinytorch"
fi

# Verify jupyter-book is available
if ! command -v jupyter-book &> /dev/null; then
    echo "❌ Error: jupyter-book not found"
    echo "   Install with: pip install jupyter-book"
    exit 1
fi

echo "📦 Using: $(which jupyter-book)"
echo "   Version: $(jupyter-book --version | head -1)"
echo ""

# Clean previous build
if [ -d "_build" ]; then
    echo "🧹 Cleaning previous build..."
    jupyter-book clean .
    echo ""
fi

# Build the site
echo "🚀 Building Jupyter Book site..."
echo ""
jupyter-book build . --all

echo ""
echo "✅ Build complete!"
echo ""
echo "📖 To view the site locally:"
echo "   python -m http.server 8000 --directory _build/html"
echo "   Then open: http://localhost:8000"
echo ""
