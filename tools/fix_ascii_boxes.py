#!/usr/bin/env python3
"""
Fix ASCII Box Alignment

This script finds ASCII art boxes in Python files and ensures the right-side
vertical bars (│) are perfectly aligned.

Box characters detected:
  ┌ ┐ └ ┘ │ ─

Usage:
    python tools/fix_ascii_boxes.py              # Preview changes (dry run)
    python tools/fix_ascii_boxes.py --fix        # Apply fixes
    python tools/fix_ascii_boxes.py --verbose    # Show detailed info
"""

import re
import sys
from pathlib import Path


# Box drawing characters
BOX_CHARS = {
    'top_left': '┌',
    'top_right': '┐',
    'bottom_left': '└',
    'bottom_right': '┘',
    'vertical': '│',
    'horizontal': '─',
}


def find_boxes_in_content(content: str) -> list[tuple[int, int, list[str]]]:
    """
    Find all ASCII boxes in content.
    
    Returns list of (start_line, end_line, lines) tuples.
    """
    lines = content.split('\n')
    boxes = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for box start: line containing ┌ and ┐
        if '┌' in line and '┐' in line:
            box_start = i
            box_lines = [line]
            
            # Find the indentation level
            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ''
            
            # Collect box lines until we find └ and ┘
            j = i + 1
            while j < len(lines):
                current_line = lines[j]
                
                # Check if this line is part of the box (contains │ or box bottom)
                if '│' in current_line or ('└' in current_line and '┘' in current_line):
                    box_lines.append(current_line)
                    
                    # If this is the bottom of the box, we're done
                    if '└' in current_line and '┘' in current_line:
                        boxes.append((box_start, j, box_lines))
                        i = j
                        break
                    j += 1
                else:
                    # Not a box line, this box is malformed or ended
                    break
            
        i += 1
    
    return boxes


def fix_box_alignment(box_lines: list[str]) -> list[str]:
    """
    Fix alignment of a single ASCII box.
    
    Ensures all │ on the right side align vertically.
    """
    if len(box_lines) < 2:
        return box_lines
    
    # Determine the indentation from the first line
    indent_match = re.match(r'^(\s*)', box_lines[0])
    indent = indent_match.group(1) if indent_match else ''
    
    # Find the position of the left │ (or ┌/└)
    first_line = box_lines[0]
    left_pos = None
    for char in ['┌', '│', '└']:
        if char in first_line:
            left_pos = first_line.index(char)
            break
    
    if left_pos is None:
        return box_lines
    
    # Calculate max content width (content between │ markers)
    max_content_width = 0
    
    for line in box_lines:
        # For top/bottom lines (with ─), measure between corners
        if '┌' in line and '┐' in line:
            # Top line: ┌────────────────┐
            start = line.index('┌')
            end = line.index('┐')
            max_content_width = max(max_content_width, end - start - 1)
        elif '└' in line and '┘' in line:
            # Bottom line: └────────────────┘
            start = line.index('└')
            end = line.index('┘')
            max_content_width = max(max_content_width, end - start - 1)
        elif '│' in line:
            # Content line: │ content │
            # Find first and last │
            first_bar = line.index('│')
            last_bar = line.rindex('│')
            if first_bar != last_bar:
                content = line[first_bar + 1:last_bar]
                # Strip trailing spaces to get actual content length
                content_stripped = content.rstrip()
                max_content_width = max(max_content_width, len(content_stripped) + 1)  # +1 for trailing space
    
    # Now rebuild all lines with consistent width
    fixed_lines = []
    
    for line in box_lines:
        if '┌' in line and '┐' in line:
            # Top line
            start = line.index('┌')
            prefix = line[:start]
            fixed_line = prefix + '┌' + '─' * max_content_width + '┐'
            fixed_lines.append(fixed_line)
        elif '└' in line and '┘' in line:
            # Bottom line
            start = line.index('└')
            prefix = line[:start]
            fixed_line = prefix + '└' + '─' * max_content_width + '┘'
            fixed_lines.append(fixed_line)
        elif '│' in line:
            # Content line
            first_bar = line.index('│')
            last_bar = line.rindex('│')
            prefix = line[:first_bar]
            
            if first_bar != last_bar:
                content = line[first_bar + 1:last_bar].rstrip()
                # Pad content to max width (minus 1 for the space before │)
                padded_content = content.ljust(max_content_width - 1)
                fixed_line = prefix + '│' + padded_content + '│'
            else:
                # Only one │, might be malformed
                fixed_line = line
            
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    return fixed_lines


def process_file(filepath: Path, fix: bool = False, verbose: bool = False) -> tuple[bool, int]:
    """
    Process a single file, finding and optionally fixing ASCII boxes.
    
    Returns (has_changes, num_boxes_fixed).
    """
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Could not read {filepath}: {e}")
        return False, 0
    
    boxes = find_boxes_in_content(content)
    
    if not boxes:
        return False, 0
    
    lines = content.split('\n')
    changes_made = False
    boxes_fixed = 0
    
    # Process boxes in reverse order so line numbers stay valid
    for start_line, end_line, box_lines in reversed(boxes):
        fixed_lines = fix_box_alignment(box_lines)
        
        if fixed_lines != box_lines:
            changes_made = True
            boxes_fixed += 1
            
            if verbose:
                print(f"\n  📦 Box at lines {start_line + 1}-{end_line + 1}:")
                print("     Before:")
                for line in box_lines[:5]:  # Show first 5 lines
                    print(f"       {line}")
                if len(box_lines) > 5:
                    print(f"       ... ({len(box_lines) - 5} more lines)")
                print("     After:")
                for line in fixed_lines[:5]:
                    print(f"       {line}")
                if len(fixed_lines) > 5:
                    print(f"       ... ({len(fixed_lines) - 5} more lines)")
            
            # Replace lines in content
            lines[start_line:end_line + 1] = fixed_lines
    
    if changes_made and fix:
        new_content = '\n'.join(lines)
        filepath.write_text(new_content, encoding='utf-8')
    
    return changes_made, boxes_fixed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix ASCII box alignment in Python files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/fix_ascii_boxes.py              # Preview changes
    python tools/fix_ascii_boxes.py --fix        # Apply fixes
    python tools/fix_ascii_boxes.py src/         # Check specific directory
    python tools/fix_ascii_boxes.py file.py      # Check specific file
        """
    )
    parser.add_argument('paths', nargs='*', default=['.'], 
                        help='Files or directories to process (default: current directory)')
    parser.add_argument('--fix', action='store_true',
                        help='Apply fixes (default is dry-run)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed changes')
    
    args = parser.parse_args()
    
    # Collect all Python files to process
    py_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.py':
            py_files.append(path)
        elif path.is_dir():
            py_files.extend(path.rglob('*.py'))
    
    # Filter out virtual environment and cache directories
    py_files = [f for f in py_files if not any(
        part in f.parts for part in ['venv', '.venv', '__pycache__', 'lib', 'bin', '.git']
    )]
    
    if not py_files:
        print("No Python files found to process.")
        return
    
    print(f"🔍 Scanning {len(py_files)} Python files for ASCII boxes...\n")
    
    total_files_changed = 0
    total_boxes_fixed = 0
    files_with_issues = []
    
    for filepath in sorted(py_files):
        has_changes, num_boxes = process_file(filepath, fix=args.fix, verbose=args.verbose)
        
        if has_changes:
            total_files_changed += 1
            total_boxes_fixed += num_boxes
            files_with_issues.append((filepath, num_boxes))
            
            status = "✅ Fixed" if args.fix else "⚠️  Needs fixing"
            print(f"{status}: {filepath} ({num_boxes} box{'es' if num_boxes > 1 else ''})")
    
    print()
    if total_boxes_fixed == 0:
        print("✨ All ASCII boxes are properly aligned!")
    else:
        if args.fix:
            print(f"✅ Fixed {total_boxes_fixed} box{'es' if total_boxes_fixed > 1 else ''} in {total_files_changed} file{'s' if total_files_changed > 1 else ''}.")
        else:
            print(f"⚠️  Found {total_boxes_fixed} misaligned box{'es' if total_boxes_fixed > 1 else ''} in {total_files_changed} file{'s' if total_files_changed > 1 else ''}.")
            print("\nRun with --fix to apply corrections:")
            print("    python tools/fix_ascii_boxes.py --fix")


if __name__ == '__main__':
    main()

