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
        self.book_dir = project_root / "book/tinytorch-course"
        self.chapters_dir = self.book_dir / "chapters"
        
        # Module to chapter mapping
        self.module_mapping = {
            "00_setup": {"title": "Development Environment", "filename": "00-setup"},
            "01_tensor": {"title": "Tensors", "filename": "01-tensor"},
            "02_activations": {"title": "Activations", "filename": "02-activations"},
            "03_layers": {"title": "Layers", "filename": "03-layers"},
            "04_networks": {"title": "Networks", "filename": "04-networks"},
            "05_cnn": {"title": "CNNs", "filename": "05-cnn"},
            "06_dataloader": {"title": "DataLoader", "filename": "06-dataloader"},
            "07_autograd": {"title": "Autograd", "filename": "07-autograd"},
            "08_optimizers": {"title": "Optimizers", "filename": "08-optimizers"},
            "09_training": {"title": "Training", "filename": "09-training"},
            "10_compression": {"title": "Compression", "filename": "10-compression"},
            "11_kernels": {"title": "Kernels", "filename": "11-kernels"},
            "12_benchmarking": {"title": "Benchmarking", "filename": "12-benchmarking"},
            "13_mlops": {"title": "MLOps", "filename": "13-mlops"},
        }
        
        # Mapping from directory name to dev file name
        self.dev_file_mapping = {
            "00_setup": "setup_dev.py",
            "01_tensor": "tensor_dev.py", 
            "02_activations": "activations_dev.py",
            "03_layers": "layers_dev.py",
            "04_networks": "networks_dev.py",
            "05_cnn": "cnn_dev.py",
            "06_dataloader": "dataloader_dev.py",
            "07_autograd": "autograd_dev.py",
            "08_optimizers": "optimizers_dev.py",
            "09_training": "training_dev.py",
            "10_compression": "compression_dev.py",
            "11_kernels": "kernels_dev.py",
            "12_benchmarking": "benchmarking_dev.py",
            "13_mlops": "mlops_dev.py",
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
    
    def add_book_frontmatter(self, notebook: Dict[str, Any], module_name: str, title: str) -> Dict[str, Any]:
        """Add Jupyter Book frontmatter to the notebook."""
        
        # Create title cell
        title_cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                f'# {title}\n',
                '\n',
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
        
        # Insert at the beginning (after any existing title)
        cells = notebook.get('cells', [])
        
        # Find if there's already a title cell
        title_inserted = False
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if source.startswith('# '):
                    # Replace existing title
                    cells[i] = title_cell
                    title_inserted = True
                    break
        
        if not title_inserted:
            cells.insert(0, title_cell)
        
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
            
            # Remove solutions
            student_notebook_path = self.remove_solutions(notebook_path)
            
            # Load the student notebook
            with open(student_notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Add book-specific enhancements
            module_info = self.module_mapping[module_name]
            notebook = self.add_binder_config(notebook, module_name)
            notebook = self.add_book_frontmatter(notebook, module_name, module_info['title'])
            
            # Save to chapters directory
            self.chapters_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.chapters_dir / f"{module_info['filename']}.ipynb"
            
            with open(output_file, 'w') as f:
                json.dump(notebook, f, indent=2)
            
            print(f"✅ Created chapter: {output_file}")
            
            # Clean up temporary files
            notebook_path.unlink(missing_ok=True)
            student_notebook_path.unlink(missing_ok=True)
            
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
    
    args = parser.parse_args()
    
    converter = ModuleConverter()
    
    if args.module:
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