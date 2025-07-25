#!/usr/bin/env python3
"""
Convert module READMEs to Jupyter Book chapters.

This script takes README files from modules/source/*/README.md and converts them
to Jupyter Book chapters in book/chapters/ with proper frontmatter and web optimization.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional

def get_module_info(module_path: Path) -> Dict[str, str]:
    """Extract module information from module.yaml file."""
    yaml_path = module_path / "module.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            module_data = yaml.safe_load(f)
            return {
                'title': module_data.get('title', module_path.name.replace('_', ' ').title()),
                'description': module_data.get('description', ''),
                'difficulty': module_data.get('difficulty', 'Intermediate'),
                'time_estimate': module_data.get('time_estimate', '2-4 hours'),
                'prerequisites': module_data.get('prerequisites', []),
                'next_steps': module_data.get('next_steps', [])
            }
    return {}

def extract_learning_objectives(content: str) -> List[str]:
    """Extract learning objectives from README content."""
    objectives = []
    # Look for common patterns in READMEs
    patterns = [
        r'By the end of this module, you will:?\s*\n((?:- [^\n]+\n?)+)',
        r'Learning Goals?:?\s*\n((?:- [^\n]+\n?)+)',
        r'Learning Objectives?:?\s*\n((?:- [^\n]+\n?)+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            objectives_text = match.group(1)
            objectives = [line.strip('- ').strip() for line in objectives_text.split('\n') if line.strip().startswith('-')]
            break
    
    return objectives

def create_frontmatter(module_name: str, module_info: Dict[str, str], objectives: List[str]) -> str:
    """Create Jupyter Book frontmatter for the chapter."""
    # Clean up module name for title
    title = module_info.get('title', module_name.replace('_', ' ').title())
    
    frontmatter = f"""---
title: "{title}"
description: "{module_info.get('description', '')}"
difficulty: "{module_info.get('difficulty', 'Intermediate')}"
time_estimate: "{module_info.get('time_estimate', '2-4 hours')}"
prerequisites: {module_info.get('prerequisites', [])}
next_steps: {module_info.get('next_steps', [])}
learning_objectives: {objectives}
---

"""
    return frontmatter

def enhance_content_for_web(content: str, module_name: str, module_num: int) -> str:
    """Enhance README content for web presentation."""
    # Remove existing grid cards to prevent conflicts with new interactive elements
    # Pattern to match grid sections (from ```{grid} to closing ```)
    grid_pattern = r'```\{grid\}[^`]*?```'
    content = re.sub(grid_pattern, '', content, flags=re.DOTALL)
    
    # Also remove individual grid-item-card patterns that might be floating
    grid_item_pattern = r'\{grid-item-card\}[^`]*?```'
    content = re.sub(grid_item_pattern, '', content, flags=re.DOTALL)
    
    # Clean up any remaining grid-related patterns
    content = re.sub(r'\{grid-item-card\}[^\n]*\n', '', content)
    content = re.sub(r':link:[^\n]*\n', '', content)
    content = re.sub(r':class-[^:]*:[^\n]*\n', '', content)
    
    # Clean up multiple newlines that result from removals
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Add badges for difficulty and time
    difficulty = get_difficulty_stars(module_name)
    time_estimate = get_time_estimate(module_name)
    badges = f"\n```{{div}} badges\n{difficulty} | ⏱️ {time_estimate}\n```\n"
    
    # Get previous and next module names for navigation
    prev_module = f"{module_num-1:02d}_{get_prev_module_name(module_num)}" if module_num > 1 else None
    
    # Add interactive learning elements and navigation at the end
    interactive_elements = f"""

Choose your preferred way to engage with this module:

````{{grid}} 1 2 3 3

```{{grid-item-card}} 🚀 Launch Binder
:link: https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/{module_name}/{module_name.split('_', 1)[1]}_dev.ipynb
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{{grid-item-card}} ⚡ Open in Colab  
:link: https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/{module_name}/{module_name.split('_', 1)[1]}_dev.ipynb
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{{grid-item-card}} 📖 View Source
:link: https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/{module_name}/{module_name.split('_', 1)[1]}_dev.py
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{{admonition}} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

---

"""
    
    # Add navigation links
    nav_links = "<div class=\"prev-next-area\">\n"
    if prev_module:
        nav_links += f'<a class="left-prev" href="../chapters/{prev_module}.html" title="previous page">← Previous Module</a>\n'
    
    # Get total number of modules dynamically
    module_names = get_module_names()
    if module_num < len(module_names):
        next_module = f"{module_num+1:02d}_{get_next_module_name(module_num)}"
        nav_links += f'<a class="right-next" href="../chapters/{next_module}.html" title="next page">Next Module →</a>\n'
    
    nav_links += "</div>\n"
    
    # Combine interactive elements with navigation
    nav_links = interactive_elements + nav_links
    
    # Insert badges after the first heading
    lines = content.split('\n')
    enhanced_lines = []
    added_badges = False
    
    for i, line in enumerate(lines):
        # Keep the meaningful module headers but clean up the breadcrumb reference
        if line.startswith('# ') and not added_badges:
            # Keep "Module: CNN" format, just remove emoji for clean display
            if '🔥 Module:' in line:
                line = line.replace('🔥 ', '')  # Remove emoji, keep "Module: CNN"
        
        enhanced_lines.append(line)
        
        # Add badges after first heading
        if not added_badges and line.startswith('# '):
            enhanced_lines.append(badges)
            added_badges = True
    
    # Add navigation at the end
    enhanced_lines.append(nav_links)
    
    return '\n'.join(enhanced_lines)

def get_difficulty_stars(module_name: str) -> str:
    """Get difficulty stars from module.yaml file."""
    # Map module number to module folder name  
    module_path = Path(f'../modules/source/{module_name}')
    module_info = get_module_info(module_path)
    return module_info.get('difficulty', '⭐⭐')

def get_time_estimate(module_name: str) -> str:
    """Get time estimate from module.yaml file."""
    # Map module number to module folder name
    module_path = Path(f'../modules/source/{module_name}')
    module_info = get_module_info(module_path)
    return module_info.get('time_estimate', '3-4 hours')

def get_module_names() -> List[str]:
    """Get actual module names from module.yaml files."""
    modules_dir = Path("../modules/source")
    module_names = []
    
    # Get all module directories (sorted by number)
    module_dirs = []
    for item in modules_dir.iterdir():
        if item.is_dir() and item.name != 'utils':
            # Extract module number from directory name
            match = re.match(r'(\d+)_(.+)', item.name)
            if match:
                module_num = int(match.group(1))
                module_dirs.append((module_num, item))
    
    # Sort by module number
    module_dirs.sort(key=lambda x: x[0])
    
    # Read module names from module.yaml files
    for module_num, module_dir in module_dirs:
        module_yaml_path = module_dir / "module.yaml"
        if module_yaml_path.exists():
            module_info = get_module_info(module_dir)
            module_names.append(module_info.get('name', module_dir.name.split('_', 1)[1]))
        else:
            # Fallback to directory name
            module_names.append(module_dir.name.split('_', 1)[1])
    
    return module_names

def get_prev_module_name(module_num: int) -> str:
    """Get previous module name."""
    module_names = get_module_names()
    return module_names[module_num - 2] if module_num > 1 and module_num - 2 < len(module_names) else 'setup'

def get_next_module_name(module_num: int) -> str:
    """Get next module name."""
    module_names = get_module_names()
    return module_names[module_num] if module_num < len(module_names) else module_names[-1] if module_names else 'setup'

def convert_readme_to_chapter(readme_path: Path, chapter_path: Path, module_num: int):
    """Convert a single README to a Jupyter Book chapter."""
    print(f"Converting {readme_path} to {chapter_path}")
    
    # Read README content
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get module information
    module_path = readme_path.parent
    module_name = module_path.name
    module_info = get_module_info(module_path)
    
    # Extract learning objectives
    objectives = extract_learning_objectives(content)
    
    # Create frontmatter
    frontmatter = create_frontmatter(module_name, module_info, objectives)
    
    # Enhance content for web
    enhanced_content = enhance_content_for_web(content, module_name, module_num)
    
    # Write chapter file
    with open(chapter_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter)
        f.write(enhanced_content)
    
    print(f"✅ Created {chapter_path}")

def main():
    """Convert all module READMEs to Jupyter Book chapters."""
    # Setup paths
    modules_dir = Path("../modules/source")
    chapters_dir = Path("chapters")
    
    # Ensure chapters directory exists
    chapters_dir.mkdir(exist_ok=True)
    
    # Get all module directories (sorted by number)
    module_dirs = []
    for item in modules_dir.iterdir():
        if item.is_dir() and item.name != 'utils':
            # Extract module number from directory name
            match = re.match(r'(\d+)_(.+)', item.name)
            if match:
                module_num = int(match.group(1))
                module_dirs.append((module_num, item))
    
    # Sort by module number
    module_dirs.sort(key=lambda x: x[0])
    
    print(f"Found {len(module_dirs)} modules to convert")
    
    # Convert each README
    for module_num, module_dir in module_dirs:
        readme_path = module_dir / "README.md"
        if readme_path.exists():
            # Create chapter filename (just module number and name, no duplicate)
            chapter_filename = f"{module_num:02d}-{module_dir.name.split('_', 1)[1]}.md"
            chapter_path = chapters_dir / chapter_filename
            
            convert_readme_to_chapter(readme_path, chapter_path, module_num)
        else:
            print(f"⚠️  No README.md found in {module_dir}")
    
    print(f"\n🎉 Converted {len(module_dirs)} modules to chapters in {chapters_dir}")

if __name__ == "__main__":
    main() 