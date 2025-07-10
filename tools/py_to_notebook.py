#!/usr/bin/env python3
"""
Convert Python files with cell markers to Jupyter notebooks.

Usage:
    python3 tools/py_to_notebook.py modules/tensor/tensor_dev.py
    python3 tools/py_to_notebook.py modules/tensor/tensor_dev.py --output custom_name.ipynb
"""

import argparse
import json
import re
import sys
from pathlib import Path

def convert_py_to_notebook(py_file: Path, output_file: Path = None):
    """Convert Python file with cell markers to notebook."""
    
    if not py_file.exists():
        print(f"❌ File not found: {py_file}")
        return False
    
    # Read the Python file
    with open(py_file, 'r') as f:
        content = f.read()
    
    # Split into cells based on # %% markers
    cells = re.split(r'^# %%.*$', content, flags=re.MULTILINE)
    cells = [cell.strip() for cell in cells if cell.strip()]
    
    # Create notebook structure
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    for i, cell_content in enumerate(cells):
        if not cell_content:
            continue
            
        # Check if this is a markdown cell
        if cell_content.startswith('# ') and '\n' in cell_content:
            lines = cell_content.split('\n')
            if lines[0].startswith('# ') and not any(line.strip() and not line.startswith('#') for line in lines[:5]):
                # This looks like a markdown cell
                cell = {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': []
                }
                
                for line in lines:
                    if line.startswith('# '):
                        cell['source'].append(line[2:] + '\n')
                    elif line.startswith('#'):
                        cell['source'].append(line[1:] + '\n')
                    elif line.strip() == '':
                        cell['source'].append('\n')
                
                notebook['cells'].append(cell)
                continue
        
        # Code cell
        cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': []
        }
        
        for line in cell_content.split('\n'):
            cell['source'].append(line + '\n')
        
        # Remove trailing newline from last line
        if cell['source'] and cell['source'][-1].endswith('\n'):
            cell['source'][-1] = cell['source'][-1][:-1]
        
        notebook['cells'].append(cell)
    
    # Determine output file
    if output_file is None:
        output_file = py_file.with_suffix('.ipynb')
    
    # Write notebook
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Converted {py_file} → {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert Python files to Jupyter notebooks")
    parser.add_argument('input_file', type=Path, help='Input Python file')
    parser.add_argument('--output', '-o', type=Path, help='Output notebook file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    success = convert_py_to_notebook(args.input_file, args.output)
    
    if not success:
        sys.exit(1)
    
    if args.verbose:
        print("🎉 Conversion complete!")

if __name__ == "__main__":
    main() 