#!/bin/bash
# Tiny🔥Torch Environment Activation & Setup

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "🆕 First time setup - creating environment..."
    python3 -m venv .venv || {
        echo "❌ Failed to create virtual environment"
        exit 1
    }
    echo "📦 Installing dependencies..."
    .venv/bin/pip install -r requirements.txt || {
        echo "❌ Failed to install dependencies"
        exit 1
    }
    echo "✅ Environment created!"
fi

echo "🔥 Activating Tiny🔥Torch environment..."
source .venv/bin/activate

# Create tito alias for convenience
alias tito="python3 bin/tito"

echo "✅ Ready to build ML systems!"
echo "💡 Quick commands:"
echo "   tito system info      - Check system status"
echo "   tito module test      - Run tests" 
echo "   tito system doctor    - Diagnose issues"
echo "   tito system jupyter   - Start Jupyter for interactive development"
