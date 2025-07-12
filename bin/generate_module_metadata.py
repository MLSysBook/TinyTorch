#!/usr/bin/env python3
"""
Generate module metadata template for TinyTorch modules.

Usage:
    python bin/generate_module_metadata.py <module_name>
    python bin/generate_module_metadata.py <module_name> --interactive
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

def generate_metadata_template(module_name: str, interactive: bool = False) -> str:
    """Generate a metadata template for a module."""
    
    # Default values
    defaults = {
        'name': module_name,
        'title': f"{module_name.title()} - Brief Description",
        'description': f"Implement {module_name} functionality for TinyTorch",
        'version': "1.0.0",
        'author': "TinyTorch Team",
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'status': "not_started",
        'implementation_status': "planned",
        'difficulty': "intermediate",
        'estimated_time': "4-6 hours",
        'pedagogical_pattern': "Build → Use → Understand",
        'exports_to': f"tinytorch.core.{module_name}",
        'export_directive': f"core.{module_name}",
        'test_coverage': "planned",
        'test_count': 0
    }
    
    # Interactive mode
    if interactive:
        print(f"Creating metadata for module: {module_name}")
        print("Press Enter to use default values shown in brackets\n")
        
        defaults['title'] = input(f"Title [{defaults['title']}]: ") or defaults['title']
        defaults['description'] = input(f"Description [{defaults['description']}]: ") or defaults['description']
        defaults['status'] = input(f"Status (complete/in_progress/not_started/deprecated) [{defaults['status']}]: ") or defaults['status']
        defaults['difficulty'] = input(f"Difficulty (beginner/intermediate/advanced) [{defaults['difficulty']}]: ") or defaults['difficulty']
        defaults['estimated_time'] = input(f"Estimated time [{defaults['estimated_time']}]: ") or defaults['estimated_time']
        
        print("\nGenerating metadata template...")
    
    # Generate template
    template = f"""# TinyTorch Module Metadata
# This file contains structured information about the module for CLI tools and documentation

# Basic Information
name: "{defaults['name']}"
title: "{defaults['title']}"
description: "{defaults['description']}"
version: "{defaults['version']}"
author: "{defaults['author']}"
last_updated: "{defaults['last_updated']}"

# Module Status
status: "{defaults['status']}"  # complete, in_progress, not_started, deprecated
implementation_status: "{defaults['implementation_status']}"  # stable, beta, alpha, experimental, planned

# Learning Information
learning_objectives:
  - "TODO: Add specific learning objectives"
  - "TODO: What will students understand after completing this module?"
  - "TODO: What skills will they develop?"

key_concepts:
  - "TODO: Key concept 1"
  - "TODO: Key concept 2"
  - "TODO: Key concept 3"

# Dependencies
dependencies:
  prerequisites: []  # TODO: List modules that must be completed first
  builds_on: []      # TODO: List direct dependencies
  enables: []        # TODO: List modules that depend on this one

# Educational Metadata
difficulty: "{defaults['difficulty']}"  # beginner, intermediate, advanced
estimated_time: "{defaults['estimated_time']}"
pedagogical_pattern: "{defaults['pedagogical_pattern']}"

# Implementation Details
components:
  - name: "TODO_ComponentName"
    type: "class"  # class, function, methods, system
    description: "TODO: What this component does"
    status: "not_started"  # complete, in_progress, not_started

# Package Export Information
exports_to: "{defaults['exports_to']}"
export_directive: "{defaults['export_directive']}"

# Testing Information
test_coverage: "{defaults['test_coverage']}"  # comprehensive, partial, minimal, none, planned
test_count: {defaults['test_count']}
test_categories:
  - "TODO: Test category 1"
  - "TODO: Test category 2"
  - "TODO: Test category 3"

# File Structure
required_files:
  - "{module_name}_dev.py"
  - "{module_name}_dev.ipynb"
  - "tests/test_{module_name}.py"
  - "README.md"

# Systems Focus
systems_concepts:
  - "TODO: Systems concept 1"
  - "TODO: Systems concept 2"
  - "TODO: Systems concept 3"

# Real-world Applications
applications:
  - "TODO: Application 1"
  - "TODO: Application 2"
  - "TODO: Application 3"

# Next Steps
next_modules: []  # TODO: List modules that should be completed after this one
completion_criteria:
  - "All tests pass"
  - "TODO: Specific completion criterion 1"
  - "TODO: Specific completion criterion 2"

# Implementation Notes (optional)
implementation_notes:
  - "TODO: Important implementation note 1"
  - "TODO: Important implementation note 2"
"""
    
    return template

def main():
    parser = argparse.ArgumentParser(description="Generate module metadata template")
    parser.add_argument("module_name", help="Name of the module")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--output", "-o", help="Output file path (default: modules/<module_name>/module.yaml)")
    
    args = parser.parse_args()
    
    # Validate module name
    if not args.module_name.isalnum() and '_' not in args.module_name:
        print("Error: Module name should contain only letters, numbers, and underscores")
        sys.exit(1)
    
    # Generate template
    template = generate_metadata_template(args.module_name, args.interactive)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"modules/{args.module_name}/module.yaml")
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Write template
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"✅ Generated metadata template: {output_path}")
    print(f"📝 Please edit the file to customize the metadata for your module")
    print(f"🧪 Test with: python bin/tito status --metadata")

if __name__ == "__main__":
    main() 