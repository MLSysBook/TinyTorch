#!/usr/bin/env python3
"""
TinyTorch Student Notebook Generator

Transforms complete implementation notebooks into student exercise versions.
Uses special markers to identify what becomes student exercises.

Usage:
    python bin/generate_student_notebooks.py --module tensor
    python bin/generate_student_notebooks.py --all
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

class NotebookGenerator:
    """Transforms complete notebooks into student exercise versions."""
    
    def __init__(self):
        self.markers = {
            'exercise_start': '#| exercise_start',
            'exercise_end': '#| exercise_end', 
            'hint': '#| hint:',
            'solution_test': '#| solution_test:',
            'difficulty': '#| difficulty:',  # easy, medium, hard
            'keep_imports': '#| keep_imports',
            'remove_cell': '#| remove_cell'
        }
    
    def process_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """Transform a complete notebook into student version."""
        print(f"📝 Processing: {notebook_path}")
        
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        processed_cells = []
        
        for cell in notebook['cells']:
            processed_cell = self._process_cell(cell)
            if processed_cell:  # None means remove cell
                processed_cells.append(processed_cell)
        
        notebook['cells'] = processed_cells
        return notebook
    
    def _process_cell(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single notebook cell."""
        if cell['cell_type'] != 'code':
            return cell  # Keep markdown cells as-is
        
        source_lines = cell['source']
        if not source_lines:
            return cell
        
        # Check for remove_cell marker
        if any(self.markers['remove_cell'] in line for line in source_lines):
            return None  # Remove this cell
        
        # Check for exercise markers
        if any(self.markers['exercise_start'] in line for line in source_lines):
            return self._transform_exercise_cell(cell)
        
        # Check for keep_imports marker
        if any(self.markers['keep_imports'] in line for line in source_lines):
            return self._clean_markers(cell)
        
        return cell
    
    def _transform_exercise_cell(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a cell with exercise markers into student version."""
        source_lines = cell['source']
        new_lines = []
        
        in_exercise = False
        exercise_header_lines = []  # Store function def, docstring etc.
        hints = []
        solution_tests = []
        difficulty = "medium"
        
        for line in source_lines:
            if self.markers['exercise_start'] in line:
                in_exercise = True
                continue
            elif self.markers['exercise_end'] in line:
                in_exercise = False
                # Add the preserved header + exercise placeholder
                new_lines.extend(exercise_header_lines)
                new_lines.extend(self._create_exercise_placeholder(hints, solution_tests, difficulty))
                # Reset for next exercise
                exercise_header_lines = []
                hints = []
                solution_tests = []
                difficulty = "medium"
                continue
            elif self.markers['hint'] in line:
                hint = line.split(self.markers['hint'], 1)[1].strip()
                hints.append(hint)
                continue
            elif self.markers['solution_test'] in line:
                test = line.split(self.markers['solution_test'], 1)[1].strip()
                solution_tests.append(test)
                continue
            elif self.markers['difficulty'] in line:
                difficulty = line.split(self.markers['difficulty'], 1)[1].strip()
                continue
            elif in_exercise:
                # Preserve function signature and docstring, skip implementation
                if self._is_function_signature_or_docstring(line):
                    exercise_header_lines.append(line)
                # Skip implementation lines (but keep signature/docstring)
                continue
            else:
                # Keep non-exercise lines
                new_lines.append(line)
        
        cell['source'] = new_lines
        return cell
    
    def _is_function_signature_or_docstring(self, line: str) -> bool:
        """Check if line is part of function signature or docstring."""
        stripped = line.strip()
        
        # Empty lines
        if not stripped:
            return False
            
        # Function definition
        if (stripped.startswith('def ') or 
            stripped.startswith('class ') or
            stripped.startswith('@')):  # decorators
            return True
            
        # Function signature continuation (parameters on multiple lines) 
        if (stripped.endswith(',') or 
            stripped.endswith('\\') or
            stripped.startswith(')') or
            '->' in stripped):
            return True
            
        # Docstrings (triple quotes)
        if ('"""' in stripped or "'''" in stripped):
            return True
            
        # Docstring content (common patterns)
        if (stripped.startswith('Args:') or 
            stripped.startswith('Returns:') or
            stripped.startswith('Raises:') or
            stripped.startswith('Note:') or
            stripped.startswith('Example:')):
            return True
            
        # Implementation code (skip these)
        if (stripped.startswith('self.') or
            stripped.startswith('if ') or
            stripped.startswith('elif ') or
            stripped.startswith('else:') or
            stripped.startswith('for ') or
            stripped.startswith('while ') or
            stripped.startswith('return ') or
            stripped.startswith('raise ') or
            stripped.startswith('try:') or
            stripped.startswith('except ') or
            stripped.startswith('with ') or
            '=' in stripped and not stripped.startswith('"""') and not stripped.startswith("'''")):
            return False
            
        # Comments (keep them as they might be part of docstring)
        if stripped.startswith('#'):
            return True
            
        # If we're not sure and it's just text, assume it's docstring content
        # This catches parameter descriptions, etc.
        return True
    
    def _create_exercise_placeholder(self, hints: List[str], tests: List[str], difficulty: str) -> List[str]:
        """Create TODO placeholder for students."""
        lines = []
        
        # Add difficulty indicator and description
        difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
        lines.append(f"    # {difficulty_emoji.get(difficulty, '🟡')} TODO: Implement this method ({difficulty})\n")
        
        # Add hints
        for hint in hints:
            lines.append(f"    # HINT: {hint}\n")
        
        # Add test guidance
        for test in tests:
            lines.append(f"    # TEST: {test}\n")
        
        lines.append("    \n")
        lines.append("    # Your implementation here\n")
        lines.append("    pass\n")
        
        return lines
    
    def _clean_markers(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """Remove generator markers from cell."""
        source_lines = cell['source']
        cleaned_lines = []
        
        for line in source_lines:
            # Skip marker lines
            if any(marker in line for marker in self.markers.values()):
                continue
            cleaned_lines.append(line)
        
        cell['source'] = cleaned_lines
        return cell
    
    def save_student_notebook(self, notebook: Dict[str, Any], output_path: Path):
        """Save the student version notebook."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"✅ Student version saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate student exercise notebooks")
    parser.add_argument('--module', type=str, help='Generate for specific module')
    parser.add_argument('--all', action='store_true', help='Generate for all modules')
    parser.add_argument('--output-suffix', default='_student', help='Suffix for student notebooks')
    
    args = parser.parse_args()
    
    if not args.module and not args.all:
        parser.error("Must specify either --module or --all")
    
    generator = NotebookGenerator()
    modules_dir = Path("modules")
    
    if args.module:
        modules = [args.module]
    else:
        modules = [d.name for d in modules_dir.iterdir() if d.is_dir()]
    
    for module in modules:
        module_dir = modules_dir / module
        dev_notebook = module_dir / f"{module}_dev.ipynb"
        
        if not dev_notebook.exists():
            print(f"⚠️  No dev notebook found for {module}: {dev_notebook}")
            continue
        
        # Generate student version
        student_notebook = generator.process_notebook(dev_notebook)
        student_path = module_dir / f"{module}_dev{args.output_suffix}.ipynb"
        generator.save_student_notebook(student_notebook, student_path)
    
    print("🎉 Student notebook generation complete!")

if __name__ == "__main__":
    main() 