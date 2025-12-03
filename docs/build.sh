#!/bin/bash
# TinyTorch Website Build Script
# Jupyter Book 1.x (Sphinx) Build System
# Quick and easy: ./docs/build.sh (from root) or ./build.sh (from docs/)

set -e  # Exit on error

echo "🏗️  Building TinyTorch documentation website (Jupyter Book 1.x)..."
echo ""

# Detect where we're running from and navigate to docs directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR=""
PROJECT_ROOT=""

if [ -f "_config.yml" ]; then
    # Already in docs directory
    DOCS_DIR="$(pwd)"
    PROJECT_ROOT="$(dirname "$DOCS_DIR")"
elif [ -f "docs/_config.yml" ]; then
    # In root directory
    PROJECT_ROOT="$(pwd)"
    DOCS_DIR="$(pwd)/docs"
    cd "$DOCS_DIR"
    echo "📂 Changed to docs directory: $DOCS_DIR"
else
    echo "❌ Error: Cannot find docs directory with _config.yml"
    echo "   Run from project root or docs/ directory"
    exit 1
fi

# Switch to Node.js v20 (required for Jupyter Book compatibility)
if command -v nvm &> /dev/null; then
    echo "🔧 Switching to Node.js v20..."
    source "$HOME/.nvm/nvm.sh"
    nvm use 20
    echo ""
elif [ -s "$HOME/.nvm/nvm.sh" ]; then
    echo "🔧 Switching to Node.js v20..."
    source "$HOME/.nvm/nvm.sh"
    nvm use 20
    echo ""
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
    rm -rf _build
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
echo "   jupyter-book start"
echo "   (This will start a MyST server and open your browser)"
echo ""
