#!/usr/bin/env python3
"""
Generate module.yaml metadata template for TinyTorch modules.
"""

import argparse
import sys
from pathlib import Path

def generate_metadata_template(module_name: str) -> str:
    """Generate a simplified module metadata template."""
    
    template = f"""# TinyTorch Module Metadata
# Essential system information for CLI tools and build systems

name: "{module_name}"
title: "{module_name.title()}"
description: "Brief description of what this module does"

# Dependencies
dependencies:
  prerequisites: []  # e.g., ["setup", "tensor"]
  enables: []        # e.g., ["layers", "networks"]

# Package Export
exports_to: "tinytorch.core.{module_name}"

# File Structure
files:
  dev_file: "{module_name}_dev.py"
  test_file: "tests/test_{module_name}.py"
  readme: "README.md"

# Components
components:
  - "component1"
  - "component2"
  - "component3"
"""
    
    return template.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate module metadata template")
    parser.add_argument("module_name", help="Name of the module")
    parser.add_argument("--output", help="Output file path (default: modules/{module_name}/module.yaml)")
    
    args = parser.parse_args()
    
    # Generate template
    template = generate_metadata_template(args.module_name)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("modules") / args.module_name / "module.yaml"
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write template
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"✅ Generated metadata template: {output_path}")
    print(f"📝 Edit the file to customize the module information")
    print(f"💡 Module status will be determined automatically by test results")

if __name__ == "__main__":
    main() 