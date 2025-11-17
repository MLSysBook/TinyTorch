#!/usr/bin/env python3
"""
Fix box-drawing alignment in Terminalizer YAML files.

This script ensures all box lines have consistent width by:
1. Finding all box-drawing content lines
2. Calculating the correct width based on terminal columns (100)
3. Padding text to fit within the box
"""

import re
import yaml
from pathlib import Path

# Terminal width from configs
COLS = 100

def get_display_width(text):
    """
    Calculate display width accounting for emoji and special characters.
    Emojis typically take 2 display columns.
    """
    # Remove ANSI escape codes
    text = re.sub(r'\\e\[[0-9;]+m', '', text)

    width = 0
    for char in text:
        # Emoji and other wide characters
        if ord(char) > 0x1F300:  # Approximate emoji range
            width += 2
        else:
            width += 1
    return width

def fix_box_content_line(content):
    """Fix a box content line to be exactly 100 characters wide."""
    # This is a line like: │  ✅ Python 3.11.9                                         │

    if '│' not in content:
        return content

    # Extract the content between the │ characters
    # Pattern: \e[color]│\e[reset]  content  \e[color]│\e[reset]

    # Remove escape sequences to get clean content
    clean = content.replace('\\e[1;36m', '').replace('\\e[0m', '').replace('\\e[1;32m', '')
    clean = clean.replace('\\r\\n', '')

    if not clean.startswith('│') or not clean.endswith('│'):
        return content

    # Get the inner content
    inner = clean[1:-1]

    # Calculate how much padding we need
    # Total width should be 100: │ (1) + content (98) + │ (1)
    target_inner_width = 98
    current_width = get_display_width(inner)

    if current_width > target_inner_width:
        # Content is too wide, we need to truncate
        # This shouldn't happen with our content, but handle it
        print(f"Warning: Content too wide ({current_width} > {target_inner_width}): {inner[:50]}...")
        return content

    # Add padding spaces to the right
    padding_needed = target_inner_width - current_width
    padded_inner = inner + (' ' * padding_needed)

    # Reconstruct with ANSI codes
    result = f"\\e[1;36m│\\e[0m{padded_inner}\\e[1;36m│\\e[0m\\r\\n"

    return result

def process_yaml_file(filepath):
    """Process a single YAML file to fix box alignment."""
    print(f"Processing {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Find all content lines with box characters
    lines = content.split('\n')
    modified = False

    for i, line in enumerate(lines):
        if 'content:' in line and '│' in line:
            # Extract the content string
            match = re.search(r'content: "(.*)"', line)
            if match:
                original = match.group(1)

                # Skip top and bottom lines (they're already correct)
                if '╭' in original or '╰' in original:
                    continue

                fixed = fix_box_content_line(original)
                if fixed != original:
                    lines[i] = line.replace(original, fixed)
                    modified = True
                    print(f"  Fixed line {i}: {original[:50]}...")

    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  ✅ Fixed {filepath.name}")
    else:
        print(f"  ℹ️  No changes needed for {filepath.name}")

def main():
    """Process all Terminalizer YAML files."""
    demos_dir = Path(__file__).parent
    yaml_files = list(demos_dir.glob('[0-9][0-9]-*.yml'))

    print(f"Found {len(yaml_files)} YAML files to process\n")

    for yaml_file in sorted(yaml_files):
        process_yaml_file(yaml_file)

    print("\n✨ Done!")

if __name__ == '__main__':
    main()
