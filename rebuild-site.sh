#!/bin/bash
# TinyTorch Site Rebuild Script
# Cleans and rebuilds the Jupyter Book site

echo "🧹 Cleaning old build..."
cd site
rm -rf _build/

echo "🔨 Building site..."
jupyter-book build . --all

echo ""
echo "✅ Build complete!"
echo ""
echo "📂 To view locally, open: site/_build/html/index.html"
echo "🌐 Or run: open _build/html/index.html"
