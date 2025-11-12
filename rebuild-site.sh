#!/bin/bash
# TinyTorch Site Rebuild Script
# Cleans and rebuilds the Jupyter Book site

echo "🧹 Cleaning old build..."
cd site
rm -rf _build/

echo "🔨 Building site..."
# Try system jupyter-book first (more reliable), fallback to venv
if command -v jupyter-book &> /dev/null; then
    jupyter-book build . --all
else
    ../.venv/bin/jupyter-book build . --all
fi

echo ""
echo "✅ Build complete!"
echo ""
echo "📂 To view locally, open: site/_build/html/index.html"
echo "🌐 Or run: open site/_build/html/index.html"
echo ""
echo "💡 Tip: If navigation doesn't update, try hard refresh (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)"
