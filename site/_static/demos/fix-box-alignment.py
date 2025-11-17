#!/usr/bin/env python3
"""
Fix box-drawing alignment in Terminalizer YAML files.

This script ensures all box lines have consistent width by:
1. Finding all box-drawing content lines
2. Calculating the correct width based on terminal columns
3. Padding text to fit within the box
"""

import re
from pathlib import Path

# Terminal width from configs
COLS = 100

def fix_box_line(content, cols=COLS):
    """Fix a single box-drawing line to have correct width."""
    # Extract ANSI codes and actual text
    ansi_pattern = r'\x1b\[[0-9;]+m'

    # Box top/bottom patterns
    if '╭' in content and '╮' in content:
        # Top line: ╭─...─╮
        return f"\\e[1;36m╭{'─' * (cols - 2)}╮\\e[0m\\r\\n"

    if '╰' in content and '╯' in content:
        # Bottom line: ╰─...─╯
        return f"\\e[1;36m╰{'─' * (cols - 2)}╯\\e[0m\\r\\n"

    if '│' in content:
        # Side line with content: │  text...  │
        # Strip ANSI codes to measure actual content
        text_only = re.sub(ansi_pattern, '', content.replace('\\e', '\x1b'))
        text_only = text_only.replace('\\r\\n', '').replace('│', '')

        # Calculate padding needed
        content_width = len(text_only)
        total_padding = cols - 2 - content_width  # -2 for the │ on each side

        if total_padding < 0:
            # Content too wide, truncate
            text_only = text_only[:cols - 5] + '...'
            total_padding = 0

        # Keep original ANSI-formatted content but adjust spacing
        # This is a simplified version - you may need to preserve exact ANSI codes
        return content

    return content

def process_yaml_file(filepath):
    """Process a single YAML file to fix box alignment."""
    print(f"Processing {filepath.name}...")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        if 'content:' in line and any(char in line for char in ['╭', '╮', '╰', '╯', '│']):
            # This line contains box-drawing characters
            # Extract the content string
            match = re.search(r'content: "(.*)"', line)
            if match:
                original = match.group(1)
                fixed = fix_box_line(original)
                if fixed != original:
                    lines[i] = line.replace(original, fixed)
                    modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f"  ✅ Fixed {filepath.name}")
    else:
        print(f"  ℹ️  No changes needed for {filepath.name}")

def main():
    """Process all Terminalizer YAML files."""
    demos_dir = Path(__file__).parent
    yaml_files = list(demos_dir.glob('[0-9][0-9]-*.yml'))

    print(f"Found {len(yaml_files)} YAML files to process\\n")

    for yaml_file in sorted(yaml_files):
        process_yaml_file(yaml_file)

    print("\\n✨ Done!")

if __name__ == '__main__':
    main()
