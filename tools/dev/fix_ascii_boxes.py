#!/usr/bin/env python3
"""
Fix ASCII Box Alignment

This script finds simple ASCII art boxes in Python files and ensures the
right-side vertical bars (│) are perfectly aligned with the top border.

Handles:
- Simple boxes (content lines with exactly 2 │)
- Boxes with ├───┤ separator lines

Skips (requires manual review):
- Nested boxes (content lines with more than 2 │)
- Side-by-side boxes
- Dashed boxes

Usage:
    python tools/fix_ascii_boxes.py              # Preview changes (dry run)
    python tools/fix_ascii_boxes.py --fix        # Apply fixes
    python tools/fix_ascii_boxes.py --verbose    # Show detailed info
"""

import sys
from pathlib import Path


def find_simple_boxes(content: str) -> list[tuple[int, int, list[str]]]:
    """
    Find simple ASCII boxes (exactly 2 │ per content line).
    
    Returns list of (start_line, end_line, lines) tuples.
    """
    lines = content.split('\n')
    boxes = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for box start: ┌...┐
        if '┌' in line and '┐' in line:
            first_corner = line.index('┌')
            
            # Skip nested boxes (│ before ┌)
            if '│' in line[:first_corner]:
                i += 1
                continue
            
            # Skip side-by-side boxes
            if line.count('┌') > 1:
                i += 1
                continue
            
            # Skip dashed boxes
            if '─ ─' in line:
                i += 1
                continue
            
            box_start = i
            box_lines = [line]
            left_pos = first_corner
            is_simple = True
            
            # Collect box lines
            j = i + 1
            while j < len(lines) and j - i < 100:
                current_line = lines[j]
                
                if '─ ─' in current_line:
                    is_simple = False
                    break
                
                # Content line
                if '│' in current_line:
                    # Check if simple (exactly 2 │)
                    bar_count = current_line.count('│')
                    first_bar = current_line.index('│')
                    if bar_count != 2 or first_bar != left_pos:
                        is_simple = False
                    box_lines.append(current_line)
                    j += 1
                # Separator line
                elif '├' in current_line and '┤' in current_line:
                    first_sep = current_line.index('├')
                    if first_sep != left_pos:
                        is_simple = False
                    box_lines.append(current_line)
                    j += 1
                # Bottom line
                elif '└' in current_line and '┘' in current_line:
                    if current_line.count('└') > 1:
                        is_simple = False
                    box_lines.append(current_line)
                    if is_simple:
                        boxes.append((box_start, j, box_lines))
                    i = j
                    break
                else:
                    break
            
            i += 1
        else:
            i += 1
    
    return boxes


def needs_fixing(box_lines: list[str]) -> bool:
    """Check if box has misaligned right-side bars."""
    if len(box_lines) < 3:
        return False
    
    top_line = box_lines[0]
    target_right = top_line.index('┐')
    
    for line in box_lines[1:-1]:
        if '│' in line:
            last_bar = line.rindex('│')
            if last_bar != target_right:
                return True
        elif '├' in line and '┤' in line:
            last_corner = line.rindex('┤')
            if last_corner != target_right:
                return True
    
    bottom_line = box_lines[-1]
    if '┘' in bottom_line:
        if bottom_line.rindex('┘') != target_right:
            return True
    
    return False


def fix_box_alignment(box_lines: list[str]) -> list[str]:
    """Fix alignment of a simple box."""
    if len(box_lines) < 3:
        return box_lines
    
    top_line = box_lines[0]
    left_pos = top_line.index('┌')
    target_right = top_line.index('┐')
    inner_width = target_right - left_pos - 1
    
    fixed_lines = [top_line]
    
    for line in box_lines[1:-1]:
        if '├' in line and '┤' in line:
            prefix = line[:left_pos]
            last_pos = line.rindex('┤')
            after = line[last_pos + 1:] if last_pos + 1 < len(line) else ''
            fixed_line = prefix + '├' + '─' * inner_width + '┤' + after
            fixed_lines.append(fixed_line)
        elif '│' in line:
            first_bar = line.index('│')
            last_bar = line.rindex('│')
            
            if first_bar == last_bar:
                fixed_lines.append(line)
                continue
            
            prefix = line[:first_bar]
            content = line[first_bar + 1:last_bar]
            after = line[last_bar + 1:] if last_bar + 1 < len(line) else ''
            
            content_stripped = content.rstrip()
            padded_content = content_stripped.ljust(inner_width)
            
            fixed_line = prefix + '│' + padded_content + '│' + after
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    # Bottom line
    bottom_line = box_lines[-1]
    if '└' in bottom_line and '┘' in bottom_line:
        bottom_left = bottom_line.index('└')
        prefix = bottom_line[:bottom_left]
        bottom_right = bottom_line.rindex('┘')
        after = bottom_line[bottom_right + 1:] if bottom_right + 1 < len(bottom_line) else ''
        fixed_bottom = prefix + '└' + '─' * inner_width + '┘' + after
        fixed_lines.append(fixed_bottom)
    else:
        fixed_lines.append(bottom_line)
    
    return fixed_lines


def count_complex_boxes(content: str) -> int:
    """Count boxes that are too complex to auto-fix."""
    lines = content.split('\n')
    count = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        if '┌' in line and '┐' in line:
            first_corner = line.index('┌')
            # Nested box
            if '│' in line[:first_corner]:
                count += 1
            # Side-by-side
            elif line.count('┌') > 1:
                count += 1
            i += 1
        else:
            i += 1
    
    return count


def process_file(filepath: Path, fix: bool = False, verbose: bool = False) -> tuple[bool, int, int]:
    """Process a single file. Returns (has_changes, num_fixed, num_complex)."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return False, 0, 0
    
    boxes = find_simple_boxes(content)
    complex_count = count_complex_boxes(content)
    
    if not boxes and complex_count == 0:
        return False, 0, 0
    
    lines = content.split('\n')
    original_content = content
    boxes_fixed = 0
    
    for start_line, end_line, box_lines in reversed(boxes):
        if not needs_fixing(box_lines):
            continue
        
        fixed_lines = fix_box_alignment(box_lines)
        
        if fixed_lines != box_lines:
            boxes_fixed += 1
            
            if verbose:
                print(f"\n  📦 Box at lines {start_line + 1}-{end_line + 1}:")
                print("     Before:")
                for line in box_lines[:5]:
                    print(f"       {line}")
                if len(box_lines) > 5:
                    print(f"       ... ({len(box_lines) - 5} more lines)")
                print("     After:")
                for line in fixed_lines[:5]:
                    print(f"       {line}")
                if len(fixed_lines) > 5:
                    print(f"       ... ({len(fixed_lines) - 5} more lines)")
            
            lines[start_line:end_line + 1] = fixed_lines
    
    new_content = '\n'.join(lines)
    changes_made = new_content != original_content
    
    if changes_made and fix:
        filepath.write_text(new_content, encoding='utf-8')
    
    return changes_made, boxes_fixed, complex_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix ASCII box alignment in Python files')
    parser.add_argument('paths', nargs='*', default=['.'])
    parser.add_argument('--fix', action='store_true', help='Apply fixes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed changes')
    
    args = parser.parse_args()
    
    py_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.py':
            py_files.append(path)
        elif path.is_dir():
            py_files.extend(path.rglob('*.py'))
    
    py_files = [f for f in py_files if not any(
        part in f.parts for part in ['venv', '.venv', '__pycache__', 'lib', 'bin', '.git']
    )]
    
    if not py_files:
        print("No Python files found.")
        return
    
    print(f"🔍 Scanning {len(py_files)} Python files...\n")
    
    total_fixed = 0
    total_complex = 0
    files_changed = 0
    
    for filepath in sorted(py_files):
        has_changes, fixed, complex_count = process_file(filepath, fix=args.fix, verbose=args.verbose)
        total_fixed += fixed
        total_complex += complex_count
        
        if has_changes or complex_count > 0:
            if has_changes:
                files_changed += 1
            status = "✅ Fixed" if args.fix and fixed > 0 else "⚠️  Needs fixing" if fixed > 0 else ""
            parts = []
            if fixed > 0:
                parts.append(f"{fixed} simple")
            if complex_count > 0:
                parts.append(f"{complex_count} complex (manual)")
            if parts:
                print(f"{status}: {filepath} ({', '.join(parts)})" if status else f"📋 {filepath} ({', '.join(parts)})")
    
    print()
    if total_fixed == 0 and total_complex == 0:
        print("✨ All ASCII boxes are properly aligned.")
    else:
        if total_fixed > 0:
            if args.fix:
                print(f"✅ Fixed {total_fixed} simple box{'es' if total_fixed != 1 else ''} in {files_changed} file{'s' if files_changed != 1 else ''}.")
            else:
                print(f"⚠️  Found {total_fixed} misaligned simple box{'es' if total_fixed != 1 else ''}.")
                print("    Run with --fix to apply corrections.")
        if total_complex > 0:
            print(f"📋 Found {total_complex} complex/nested box{'es' if total_complex != 1 else ''} (require manual review).")


if __name__ == '__main__':
    main()
