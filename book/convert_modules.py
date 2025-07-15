#!/usr/bin/env python3
"""
Convert TinyTorch modules to Jupyter Book chapters.

This script processes modules/source/*_dev.py files and converts them to
student-ready notebooks for the Jupyter Book, stripping solutions manually.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ModuleConverter:
    """Convert TinyTorch modules to Jupyter Book chapters."""
    
    def __init__(self):
        # Use absolute paths relative to project root
        project_root = Path(__file__).parent.parent
        self.modules_dir = project_root / "modules/source"
        self.book_dir = project_root / "book"
        self.chapters_dir = self.book_dir / "chapters"
        
        # Module to chapter mapping
        self.module_mapping = {
            "01_setup": {"title": "Development Environment", "filename": "01-setup"},
            "02_tensor": {"title": "Tensors", "filename": "02-tensor"},
            "03_activations": {"title": "Activations", "filename": "03-activations"},
            "04_layers": {"title": "Layers", "filename": "04-layers"},
            "05_networks": {"title": "Networks", "filename": "05-networks"},
            "06_cnn": {"title": "CNNs", "filename": "06-cnn"},
            "07_dataloader": {"title": "DataLoader", "filename": "07-dataloader"},
            "08_autograd": {"title": "Autograd", "filename": "08-autograd"},
            "09_optimizers": {"title": "Optimizers", "filename": "09-optimizers"},
            "10_training": {"title": "Training", "filename": "10-training"},
            "11_compression": {"title": "Compression", "filename": "11-compression"},
            "12_kernels": {"title": "Kernels", "filename": "12-kernels"},
            "13_benchmarking": {"title": "Benchmarking", "filename": "13-benchmarking"},
            "14_mlops": {"title": "MLOps", "filename": "14-mlops"},
        }
        
        # Mapping from directory name to dev file name
        self.dev_file_mapping = {
            "01_setup": "setup_dev.py",
            "02_tensor": "tensor_dev.py", 
            "03_activations": "activations_dev.py",
            "04_layers": "layers_dev.py",
            "05_networks": "networks_dev.py",
            "06_cnn": "cnn_dev.py",
            "07_dataloader": "dataloader_dev.py",
            "08_autograd": "autograd_dev.py",
            "09_optimizers": "optimizers_dev.py",
            "10_training": "training_dev.py",
            "11_compression": "compression_dev.py",
            "12_kernels": "kernels_dev.py",
            "13_benchmarking": "benchmarking_dev.py",
            "14_mlops": "mlops_dev.py",
        }
    
    def convert_to_notebook(self, dev_file: Path) -> Optional[Path]:
        """Convert dev file to notebook using Jupytext."""
        print(f"📝 Converting {dev_file.name} to notebook")
        
        # Create temporary output file
        temp_notebook = dev_file.with_suffix('.temp.ipynb')
        
        # Use jupytext to convert .py to .ipynb
        cmd = ["jupytext", "--to", "ipynb", str(dev_file.absolute()), "--output", str(temp_notebook.absolute())]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to convert {dev_file} to notebook: {result.stderr}")
            return None
        
        return temp_notebook
    
    def remove_solutions(self, notebook_path: Path) -> Path:
        """Remove solutions from notebook."""
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Process each cell
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                new_source = []
                in_solution = False
                
                for line in source:
                    if '### BEGIN SOLUTION' in line:
                        in_solution = True
                        new_source.append(line)
                        new_source.append('    # YOUR CODE HERE\n')
                        new_source.append('    raise NotImplementedError()\n')
                        continue
                    elif '### END SOLUTION' in line:
                        in_solution = False
                        new_source.append(line)
                        continue
                    elif in_solution:
                        # Skip solution lines
                        continue
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
        
        # Save processed notebook
        output_path = notebook_path.with_suffix('.student.ipynb')
        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        return output_path
    
    def add_binder_config(self, notebook: Dict[str, Any], module_name: str) -> Dict[str, Any]:
        """Add Binder configuration to notebook metadata."""
        if 'metadata' not in notebook:
            notebook['metadata'] = {}
        
        notebook['metadata'].update({
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8+'
            },
            'mystnb': {
                'execution_mode': 'auto'
            }
        })
        
        return notebook
    
    def extract_learning_goals(self, dev_file: Path) -> str:
        """Extract learning goals from source file and format as admonition block."""
        with open(dev_file, 'r') as f:
            content = f.read()
        
        # Find the Learning Goals section
        goals_start = content.find('## Learning Goals\n')
        if goals_start == -1:
            return ""
        
        # Find the end of the goals section (next ## heading)
        goals_content_start = goals_start + len('## Learning Goals\n')
        next_section = content.find('\n## ', goals_content_start)
        
        if next_section == -1:
            # If no next section found, look for next markdown cell
            next_section = content.find('\n# %%', goals_content_start)
        
        if next_section == -1:
            goals_text = content[goals_content_start:].strip()
        else:
            goals_text = content[goals_content_start:next_section].strip()
        
        # Format as admonition block
        admonition = ['```{admonition} 🎯 Learning Goals\n']
        admonition.append(':class: tip\n')
        for line in goals_text.split('\n'):
            if line.strip():
                admonition.append(f'{line}\n')
        admonition.append('```\n\n')
        
        return ''.join(admonition)
    
    def extract_module_overview(self, dev_file: Path) -> str:
        """Extract first markdown cell content for book overview."""
        with open(dev_file, 'r') as f:
            content = f.read()
        
        # Find first markdown cell
        start = content.find('# %% [markdown]\n"""')
        if start == -1:
            return ""
            
        end = content.find('"""', start + 20)
        if end == -1:
            return ""
        
        # Extract and clean the content
        overview = content[start + len('# %% [markdown]\n"""'):end].strip()
        
        # Replace Learning Goals section with admonition block
        learning_goals = self.extract_learning_goals(dev_file)
        if learning_goals and '## Learning Goals' in overview:
            # Find and replace the Learning Goals section
            goals_start = overview.find('## Learning Goals')
            if goals_start != -1:
                # Find end of goals section
                next_section = overview.find('\n## ', goals_start + 1)
                if next_section == -1:
                    # Goals are at the end
                    overview = overview[:goals_start] + learning_goals
                else:
                    # Replace goals section with admonition
                    overview = (overview[:goals_start] + 
                              learning_goals + 
                              overview[next_section:])
        
        return overview
    
    def create_module_overview_page(self, module_name: str) -> bool:
        """Create a module overview page for the book (hybrid approach)."""
        if module_name not in self.module_mapping:
            return False
        
        module_dir = self.modules_dir / module_name
        dev_file_name = self.dev_file_mapping.get(module_name)
        if not dev_file_name:
            return False
        
        dev_file = module_dir / dev_file_name
        if not dev_file.exists():
            return False
        
        module_info = self.module_mapping[module_name]
        
        # Extract overview content
        overview = self.extract_module_overview(dev_file)
        
        # Create interactive launch buttons
        github_url = f"https://github.com/mlsysbook/TinyTorch/blob/main/modules/source/{module_name}/{dev_file_name}"
        binder_url = f"https://mybinder.org/v2/gh/mlsysbook/TinyTorch/main?filepath=modules/source/{module_name}/{dev_file_name.replace('.py', '.ipynb')}"
        colab_url = f"https://colab.research.google.com/github/mlsysbook/TinyTorch/blob/main/modules/source/{module_name}/{dev_file_name.replace('.py', '.ipynb')}"
        
        interactive_section = f"""
## 🚀 Interactive Learning

Choose your preferred way to engage with this module:

````{{grid}} 1 2 3 3

```{{grid-item-card}} 🚀 Launch Binder
:link: {binder_url}
:class-header: bg-light

Run this module interactively in your browser. No installation required!
```

```{{grid-item-card}} ⚡ Open in Colab  
:link: {colab_url}
:class-header: bg-light

Use Google Colab for GPU access and cloud compute power.
```

```{{grid-item-card}} 📖 View Source
:link: {github_url}
:class-header: bg-light

Browse the Python source code and understand the implementation.
```

````

```{{admonition}} 💾 Save Your Progress
:class: tip
**Binder sessions are temporary!** Download your completed notebook when done, or switch to local development for persistent work.

Ready for serious development? → [🏗️ Local Setup Guide](../usage-paths/serious-development.md)
```

"""
        
        # Combine everything
        page_content = overview + interactive_section
        
        # Save to chapters directory
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.chapters_dir / f"{module_info['filename']}.md"
        
        with open(output_file, 'w') as f:
            f.write(page_content)
        
        print(f"✅ Created overview page: {output_file}")
        return True
    
    def add_book_frontmatter(self, notebook: Dict[str, Any], module_name: str, title: str) -> Dict[str, Any]:
        """Add Jupyter Book frontmatter to the notebook."""
        
        # Create interactive learning admonition
        interactive_cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '```{admonition} Interactive Learning\n',
                ':class: tip\n',
                '🚀 **Launch Binder**: Click the rocket icon above to run this chapter interactively!\n',
                '\n', 
                '💾 **Save Your Work**: Download your completed notebook when done.\n',
                '\n',
                '🏗️ **Build Locally**: Ready for serious development? [Fork the repo](https://github.com/your-org/tinytorch) and work locally with the full `tito` workflow.\n',
                '```\n',
                '\n'
            ]
        }
        
        # Insert interactive cell after the first title cell
        cells = notebook.get('cells', [])
        
        # Find the first title cell and add interactive cell after it
        title_found = False
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if source.startswith('# '):
                    # Insert interactive cell after the title
                    cells.insert(i + 1, interactive_cell)
                    title_found = True
                    break
        
        if not title_found:
            cells.insert(0, interactive_cell)
        
        notebook['cells'] = cells
        return notebook
    
    def convert_module(self, module_name: str) -> bool:
        """Convert a single module to a chapter."""
        if module_name not in self.module_mapping:
            print(f"❌ Unknown module: {module_name}")
            return False
        
        module_dir = self.modules_dir / module_name
        if not module_dir.exists():
            print(f"❌ Module directory not found: {module_dir}")
            return False
        
        # Get the dev file name for this module
        dev_file_name = self.dev_file_mapping.get(module_name)
        if not dev_file_name:
            print(f"❌ No dev file mapping for {module_name}")
            return False
        
        dev_file = module_dir / dev_file_name
        if not dev_file.exists():
            print(f"❌ Dev file not found: {dev_file}")
            return False
        
        print(f"🔄 Converting {module_name}: {dev_file}")
        
        try:
            # Convert to notebook
            notebook_path = self.convert_to_notebook(dev_file)
            if not notebook_path:
                return False
            
            # Keep solutions (no NBGrader processing)
            # student_notebook_path = self.remove_solutions(notebook_path)  # Disabled - keep solutions
            
            # Load the full notebook with solutions
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Add book-specific enhancements
            module_info = self.module_mapping[module_name]
            notebook = self.add_binder_config(notebook, module_name)
            # notebook = self.add_book_frontmatter(notebook, module_name, module_info['title'])  # Disabled for raw export
            
            # Save to chapters directory
            self.chapters_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.chapters_dir / f"{module_info['filename']}.ipynb"
            
            with open(output_file, 'w') as f:
                json.dump(notebook, f, indent=2)
            
            print(f"✅ Created chapter: {output_file}")
            
            # Clean up temporary files
            notebook_path.unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Error converting {module_name}: {e}")
            return False
    
    def convert_all_modules(self) -> bool:
        """Convert all available modules."""
        print("🔄 Converting all TinyTorch modules to Jupyter Book chapters...")
        
        success_count = 0
        total_count = 0
        
        for module_name in self.module_mapping.keys():
            total_count += 1
            if self.convert_module(module_name):
                success_count += 1
        
        print(f"\n📊 Conversion Summary:")
        print(f"   ✅ Success: {success_count}/{total_count} modules")
        print(f"   📁 Output: {self.chapters_dir}")
        
        return success_count == total_count

def main():
    """Main conversion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TinyTorch modules to Jupyter Book")
    parser.add_argument('--module', help='Convert specific module (e.g., 00_setup)')
    parser.add_argument('--all', action='store_true', help='Convert all modules')
    parser.add_argument('--overview', action='store_true', help='Create overview pages instead of full notebooks')
    parser.add_argument('--overview-module', help='Create overview page for specific module')
    
    args = parser.parse_args()
    
    converter = ModuleConverter()
    
    if args.overview_module:
        success = converter.create_module_overview_page(args.overview_module)
        sys.exit(0 if success else 1)
    elif args.overview:
        # Create overview pages for all modules
        print("🔄 Creating module overview pages for Jupyter Book...")
        success_count = 0
        total_count = 0
        
        for module_name in converter.module_mapping.keys():
            total_count += 1
            if converter.create_module_overview_page(module_name):
                success_count += 1
        
        print(f"\n📊 Overview Creation Summary:")
        print(f"   ✅ Success: {success_count}/{total_count} modules")
        print(f"   📁 Output: {converter.chapters_dir}")
        
        success = success_count == total_count
        sys.exit(0 if success else 1)
    elif args.module:
        success = converter.convert_module(args.module)
        sys.exit(0 if success else 1)
    elif args.all:
        success = converter.convert_all_modules()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 