#!/usr/bin/env python3
"""Check NBGrader style guide compliance across all modules."""

import os
import re
from pathlib import Path

def analyze_module_compliance(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Count solution blocks
    solution_blocks = len(re.findall(r'### BEGIN SOLUTION', content))
    
    # Check for required sections
    has_todo = 'TODO:' in content
    has_step_by_step = 'STEP-BY-STEP IMPLEMENTATION:' in content
    has_example_usage = 'EXAMPLE USAGE:' in content or 'EXAMPLE:' in content
    has_hints = 'IMPLEMENTATION HINTS:' in content or 'HINTS:' in content
    has_connections = 'LEARNING CONNECTIONS:' in content or 'LEARNING CONNECTION:' in content
    
    # Check for alternative patterns (older style)
    has_approach = 'APPROACH:' in content
    has_your_code_here = 'YOUR CODE HERE' in content
    has_raise_notimpl = 'raise NotImplementedError' in content
    
    compliance_score = sum([has_todo, has_step_by_step, has_example_usage, has_hints, has_connections])
    
    return {
        'solution_blocks': solution_blocks,
        'compliance_score': compliance_score,
        'has_todo': has_todo,
        'has_step_by_step': has_step_by_step,
        'has_example_usage': has_example_usage,
        'has_hints': has_hints,
        'has_connections': has_connections,
        'has_old_patterns': has_approach or has_your_code_here or has_raise_notimpl
    }

# Analyze all modules
modules_dir = Path('modules/source')
results = {}

for module_dir in sorted(modules_dir.iterdir()):
    if module_dir.is_dir() and module_dir.name != 'utils':
        py_files = list(module_dir.glob('*_dev.py'))
        if py_files:
            module_file = py_files[0]
            results[module_dir.name] = analyze_module_compliance(module_file)

# Report results
print('=== NBGrader Style Guide Compliance Report ===\n')
print('Module            | Blocks | Score | TODO | STEP | EXAM | HINT | CONN | Old? |')
print('-' * 78)

for module_name in sorted(results.keys()):
    r = results[module_name]
    status_emoji = '✅' if r['compliance_score'] == 5 else '⚠️' if r['compliance_score'] >= 3 else '❌'
    
    print(f"{module_name:16} | {r['solution_blocks']:6} | {status_emoji} {r['compliance_score']}/5 | "
          f"{'✓' if r['has_todo'] else '✗':^4} | "
          f"{'✓' if r['has_step_by_step'] else '✗':^4} | "
          f"{'✓' if r['has_example_usage'] else '✗':^4} | "
          f"{'✓' if r['has_hints'] else '✗':^4} | "
          f"{'✓' if r['has_connections'] else '✗':^4} | "
          f"{'⚠️' if r['has_old_patterns'] else '✓':^4} |")

# Summary
fully_compliant = sum(1 for r in results.values() if r['compliance_score'] == 5)
needs_update = sum(1 for r in results.values() if r['compliance_score'] < 5)
has_old_patterns = sum(1 for r in results.values() if r['has_old_patterns'])

print('\n=== Summary ===')
print(f'Fully Compliant: {fully_compliant}/{len(results)}')
print(f'Needs Update: {needs_update}/{len(results)}')
print(f'Has Old Patterns: {has_old_patterns}/{len(results)}')

# List modules needing updates
print('\n=== Modules Needing Updates ===')
for module_name, r in sorted(results.items()):
    if r['compliance_score'] < 5:
        missing = []
        if not r['has_todo']: missing.append('TODO')
        if not r['has_step_by_step']: missing.append('STEP-BY-STEP')
        if not r['has_example_usage']: missing.append('EXAMPLE USAGE')
        if not r['has_hints']: missing.append('HINTS')
        if not r['has_connections']: missing.append('CONNECTIONS')
        print(f"{module_name}: Missing {', '.join(missing)}")