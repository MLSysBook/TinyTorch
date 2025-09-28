#!/usr/bin/env python3
"""
Fix Unicode characters in module files that cause syntax errors.
"""

import os
import re

def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Track if changes were made
        original_content = content

        # Common Unicode fixes for Python compatibility
        replacements = {
            '│': '|',           # Box drawing vertical
            '├': '+',           # Box drawing vertical and right
            '┼': '+',           # Box drawing vertical and horizontal
            '┴': '+',           # Box drawing up and horizontal
            '┬': '+',           # Box drawing down and horizontal
            '╭': '+',           # Box drawing arc top left
            '╮': '+',           # Box drawing arc top right
            '╰': '+',           # Box drawing arc bottom left
            '╯': '+',           # Box drawing arc bottom right
            '─': '-',           # Box drawing horizontal
            '└': '+',           # Box drawing up and right
            '┘': '+',           # Box drawing up and left
            '┐': '+',           # Box drawing down and left
            '┌': '+',           # Box drawing down and right
            '→': '->',          # Right arrow
            '←': '<-',          # Left arrow
            '↓': 'v',           # Down arrow
            '↑': '^',           # Up arrow
            '▲': '^',           # Triangle up
            '►': '>',           # Triangle right
            '◄': '<',           # Triangle left
            '▼': 'v',           # Triangle down
            '╱': '/',           # Box drawing diagonal upper right to lower left
            '╲': '\\',          # Box drawing diagonal upper left to lower right
            '═': '=',           # Double horizontal line
            '║': '|',           # Double vertical line
            '╔': '+',           # Double line box drawing
            '╗': '+',
            '╚': '+',
            '╝': '+',
            '╠': '+',
            '╣': '+',
            '╦': '+',
            '╩': '+',
            '╬': '+',
            '≥': '>=',          # Greater than or equal
            '≤': '<=',          # Less than or equal
            '×': '*',           # Multiplication sign
            '÷': '/',           # Division sign
            '∂': 'd',           # Partial derivative
            '∇': 'grad',        # Nabla (gradient)
            'Σ': 'Sum',         # Sigma (summation)
            '∑': 'sum',         # Summation
            '√': 'sqrt',        # Square root
            '∞': 'inf',         # Infinity
            '≠': '!=',          # Not equal
            '≈': '~=',          # Approximately equal
            '∈': 'in',          # Element of
            '∉': 'not in',      # Not element of
            '⚠': 'WARNING',     # Warning sign
            '✓': 'OK',          # Check mark
            '✅': 'PASS',       # Check mark button
            '❌': 'FAIL',       # Cross mark
            '💡': 'TIP',        # Light bulb
            '💥': 'CRASH',      # Explosion
            '🔥': 'FIRE',       # Fire
            '🔗': 'LINK',       # Link
            '🚀': 'ROCKET',     # Rocket
            '🎯': 'TARGET',     # Target
            '🔍': 'MAGNIFY',    # Magnifying glass
            '🤔': 'THINK',      # Thinking face
            '🧪': 'TEST',       # Test tube
            '📈': 'PROGRESS',   # Chart increasing
            '📦': 'PACKAGE',    # Package
            '🎉': 'CELEBRATE',  # Party
            '⚡': 'SPEED',      # Lightning
        }

        # Apply replacements
        for unicode_char, replacement in replacements.items():
            content = content.replace(unicode_char, replacement)

        # Write back if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Unicode characters in: {filepath}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix Unicode in all module files."""
    modules_dir = '/Users/VJ/GitHub/TinyTorch/modules'
    fixed_count = 0

    # Find all Python files in modules
    for root, dirs, files in os.walk(modules_dir):
        for file in files:
            if file.endswith('_dev.py'):
                filepath = os.path.join(root, file)
                if fix_unicode_in_file(filepath):
                    fixed_count += 1

    print(f"\nFixed Unicode characters in {fixed_count} files.")

if __name__ == '__main__':
    main()