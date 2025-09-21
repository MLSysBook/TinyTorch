#!/bin/bash
# 🛡️ TinyTorch Core File Protection Script
# Industry-standard approach: Make generated files read-only

echo "🛡️ Setting up TinyTorch Core File Protection..."
echo "=" * 60

# Make all files in tinytorch/core/ read-only
if [ -d "tinytorch/core" ]; then
    echo "🔒 Making tinytorch/core/ files read-only..."
    chmod -R 444 tinytorch/core/*.py
    echo "✅ Core files are now read-only"
else
    echo "⚠️  tinytorch/core/ directory not found"
fi

# Create .gitattributes to mark files as generated (GitHub feature)
echo "📝 Setting up .gitattributes for generated file detection..."
cat > .gitattributes << 'EOF'
# Mark auto-generated files (GitHub will show "Generated" label)
tinytorch/core/*.py linguist-generated=true
tinytorch/**/*.py linguist-generated=true

# Exclude from diff by default (reduces noise)
tinytorch/core/*.py -diff
EOF

echo "✅ .gitattributes configured for generated file detection"

# Create EditorConfig to warn in common editors
echo "📝 Setting up .editorconfig for editor warnings..."
cat > .editorconfig << 'EOF'
# EditorConfig: Industry standard editor configuration
# Many editors will show warnings for files marked as generated

root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

# Mark generated files with special rules (some editors respect this)
[tinytorch/core/*.py]
# Some editors show warnings for files in generated directories
generated = true
EOF

echo "✅ .editorconfig configured for editor warnings"

# Create a pre-commit hook to warn about core file modifications
mkdir -p .git/hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# 🛡️ TinyTorch Pre-commit Hook: Prevent core file modifications

echo "🛡️ Checking for modifications to auto-generated files..."

# Check if any tinytorch/core files are staged
CORE_FILES_MODIFIED=$(git diff --cached --name-only | grep "^tinytorch/core/")

if [ ! -z "$CORE_FILES_MODIFIED" ]; then
    echo ""
    echo "🚨 ERROR: Attempting to commit auto-generated files!"
    echo "=========================================="
    echo ""
    echo "The following auto-generated files are staged:"
    echo "$CORE_FILES_MODIFIED"
    echo ""
    echo "🛡️ PROTECTION TRIGGERED: These files are auto-generated from modules/source/"
    echo ""
    echo "TO FIX:"
    echo "1. Unstage these files: git reset HEAD tinytorch/core/"
    echo "2. Make changes in modules/source/ instead"
    echo "3. Run: tito module complete <module_name>"
    echo "4. Commit the source changes, not the generated files"
    echo ""
    echo "⚠️  This protection prevents breaking CIFAR-10 training!"
    echo ""
    exit 1
fi

echo "✅ No auto-generated files being committed"
EOF

chmod +x .git/hooks/pre-commit
echo "✅ Git pre-commit hook installed"

echo ""
echo "🎉 TinyTorch Protection System Activated!"
echo "=" * 60
echo "🔒 Core files are read-only"
echo "📝 GitHub will label files as 'Generated'"
echo "⚙️  Editors will show generated file warnings"
echo "🚫 Git pre-commit hook prevents accidental commits"
echo ""
echo "🛡️ Students are now protected from accidentally breaking core functionality!"