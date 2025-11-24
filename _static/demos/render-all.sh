#!/bin/bash

# Render all Terminalizer GIFs
# Usage: ./render-all.sh

set -e  # Exit on error

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Use Node v16
nvm use 16

# Change to demos directory
cd "$(dirname "$0")"

echo "🎬 Rendering TinyTorch carousel GIFs..."
echo ""

echo "📹 Rendering 01-clone-setup.gif..."
terminalizer render 01-clone-setup -o 01-clone-setup.gif

echo "📹 Rendering 02-build-jupyter.gif..."
terminalizer render 02-build-jupyter -o 02-build-jupyter.gif

echo "📹 Rendering 03-export-tito.gif..."
terminalizer render 03-export-tito -o 03-export-tito.gif

echo "📹 Rendering 04-validate-history.gif..."
terminalizer render 04-validate-history -o 04-validate-history.gif

echo ""
echo "✅ All GIFs rendered successfully!"
echo ""
echo "Generated files:"
ls -lh *.gif
