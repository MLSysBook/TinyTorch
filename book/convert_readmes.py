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
    """Enhance README content for web display."""
    # Add navigation breadcrumbs
    breadcrumb = f"""---
**Course Navigation:** [Home](../intro.html) → [Module {module_num}: {module_name.replace('_', ' ').title()}](#)

---
"""
    
    # Add difficulty and time badges
    badges = f"""
<div class="admonition note">
<p class="admonition-title">📊 Module Info</p>
<p><strong>Difficulty:</strong> ⭐ {get_difficulty_stars(module_name)} | <strong>Time:</strong> {get_time_estimate(module_name)}</p>
</div>

"""
    
    # Add interactive elements and navigation links at the end
    interactive_elements = f"""
---

## 🚀 Interactive Learning

<div class="admonition tip">
<p class="admonition-title">💡 Try It Yourself</p>
<p>Ready to start building? Choose your preferred environment:</p>
</div>

### 🔧 **Builder Environment**
<div class="admonition note">
<p class="admonition-title">🏗️ Quick Start</p>
<p>Jump directly into the implementation with our guided builder:</p>
</div>

<a href="https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/{module_name}/{module_name.split('_', 1)[1]}_dev.ipynb" target="_blank" class="btn btn-primary">
    🚀 Launch Builder
</a>

### 📓 **Jupyter Notebook**
<div class="admonition note">
<p class="admonition-title">📚 Full Development</p>
<p>Work with the complete development environment:</p>
</div>

<a href="https://mybinder.org/v2/gh/MLSysBook/TinyTorch/main?filepath=modules/source/{module_name}/{module_name.split('_', 1)[1]}_dev.ipynb" target="_blank" class="btn btn-success">
    📓 Open Jupyter
</a>

### 🎯 **Google Colab**
<div class="admonition note">
<p class="admonition-title">☁️ Cloud Environment</p>
<p>Use Google's cloud-based notebook environment:</p>
</div>

<a href="https://colab.research.google.com/github/MLSysBook/TinyTorch/blob/main/modules/source/{module_name}/{module_name.split('_', 1)[1]}_dev.ipynb" target="_blank" class="btn btn-info">
    ☁️ Open in Colab
</a>

---

<div class="prev-next-area">
"""
    
    # Build navigation links
    nav_links = ""
    if module_num > 1:
        prev_module = f"{module_num-1:02d}_{get_prev_module_name(module_num)}"
        nav_links += f'<a class="left-prev" href="../chapters/{prev_module}.html" title="previous page">← Previous Module</a>\n'
    
    if module_num < 14:  # Assuming 14 modules total
        next_module = f"{module_num+1:02d}_{get_next_module_name(module_num)}"
        nav_links += f'<a class="right-next" href="../chapters/{next_module}.html" title="next page">Next Module →</a>\n'
    
    nav_links += "</div>\n"
    
    # Combine interactive elements with navigation
    nav_links = interactive_elements + nav_links
    
    # Insert breadcrumb and badges after the first heading
    lines = content.split('\n')
    enhanced_lines = []
    added_breadcrumb = False
    added_badges = False
    
    for i, line in enumerate(lines):
        enhanced_lines.append(line)
        
        # Add breadcrumb after first heading
        if not added_breadcrumb and line.startswith('# '):
            enhanced_lines.append(breadcrumb)
            added_breadcrumb = True
        
        # Add badges after breadcrumb
        if added_breadcrumb and not added_badges:
            enhanced_lines.append(badges)
            added_badges = True
    
    # Add navigation at the end
    enhanced_lines.append(nav_links)
    
    return '\n'.join(enhanced_lines)

def get_difficulty_stars(module_name: str) -> str:
    """Get difficulty stars based on module name."""
    difficulty_map = {
        '01_setup': '⭐',
        '02_tensor': '⭐⭐',
        '03_activations': '⭐⭐',
        '04_layers': '⭐⭐⭐',
        '05_networks': '⭐⭐⭐',
        '06_cnn': '⭐⭐⭐⭐',
        '07_dataloader': '⭐⭐⭐',
        '08_autograd': '⭐⭐⭐⭐',
        '09_optimizers': '⭐⭐⭐⭐',
        '10_training': '⭐⭐⭐⭐',
        '11_compression': '⭐⭐⭐⭐⭐',
        '12_kernels': '⭐⭐⭐⭐⭐',
        '13_benchmarking': '⭐⭐⭐⭐⭐',
        '14_mlops': '⭐⭐⭐⭐⭐'
    }
    return difficulty_map.get(module_name, '⭐⭐')

def get_time_estimate(module_name: str) -> str:
    """Get time estimate based on module name."""
    time_map = {
        '01_setup': '1-2 hours',
        '02_tensor': '4-6 hours',
        '03_activations': '3-4 hours',
        '04_layers': '4-5 hours',
        '05_networks': '5-6 hours',
        '06_cnn': '6-8 hours',
        '07_dataloader': '4-5 hours',
        '08_autograd': '6-8 hours',
        '09_optimizers': '5-6 hours',
        '10_training': '6-8 hours',
        '11_compression': '4-5 hours',
        '12_kernels': '5-6 hours',
        '13_benchmarking': '4-5 hours',
        '14_mlops': '6-8 hours'
    }
    return time_map.get(module_name, '3-4 hours')

def get_prev_module_name(module_num: int) -> str:
    """Get previous module name."""
    module_names = [
        'setup', 'tensor', 'activations', 'layers', 'networks', 'cnn',
        'dataloader', 'autograd', 'optimizers', 'training', 'compression',
        'kernels', 'benchmarking', 'mlops'
    ]
    return module_names[module_num - 2] if module_num > 1 else 'setup'

def get_next_module_name(module_num: int) -> str:
    """Get next module name."""
    module_names = [
        'setup', 'tensor', 'activations', 'layers', 'networks', 'cnn',
        'dataloader', 'autograd', 'optimizers', 'training', 'compression',
        'kernels', 'benchmarking', 'mlops'
    ]
    return module_names[module_num] if module_num < len(module_names) else 'mlops'

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