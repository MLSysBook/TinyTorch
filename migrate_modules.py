#!/usr/bin/env python3
"""
TinyTorch Module Reorganization Script

This script handles the complex task of:
1. Removing 01_setup module (functionality moves to CLI)
2. Renumbering all modules (02_tensor becomes 01_tensor, etc.)
3. Updating all references, dependencies, and exports
4. Preserving all functionality while improving user experience

BEFORE RUNNING: Ensure you have a git backup!
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import json

class ModuleReorganizer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.modules_dir = project_root / "modules"
        self.tinytorch_dir = project_root / "tinytorch"
        
        # Current module mapping (what exists now)
        self.current_modules = {
            "01_setup": "setup",
            "02_tensor": "tensor", 
            "03_activations": "activations",
            "04_layers": "layers",
            "05_losses": "losses",
            "06_autograd": "autograd",
            "07_optimizers": "optimizers", 
            "08_training": "training",
            "09_spatial": "spatial",
            "10_dataloader": "dataloader",
            # Add more as needed
        }
        
        # New module mapping (what we want)
        self.new_modules = {
            # 01_setup is REMOVED - functionality goes to CLI
            "01_tensor": "tensor",      # was 02_tensor
            "02_activations": "activations",  # was 03_activations  
            "03_layers": "layers",      # was 04_layers
            "04_losses": "losses",      # was 05_losses
            "05_autograd": "autograd",  # was 06_autograd
            "06_optimizers": "optimizers", # was 07_optimizers
            "07_training": "training",  # was 08_training
            "08_spatial": "spatial",    # was 09_spatial
            "09_dataloader": "dataloader", # was 10_dataloader
            # Continue pattern for higher numbers
        }
        
        # Reverse mapping for lookups
        self.old_to_new = {
            "02_tensor": "01_tensor",
            "03_activations": "02_activations", 
            "04_layers": "03_layers",
            "05_losses": "04_losses",
            "06_autograd": "05_autograd",
            "07_optimizers": "06_optimizers",
            "08_training": "07_training", 
            "09_spatial": "08_spatial",
            "10_dataloader": "09_dataloader",
        }
    
    def discover_all_modules(self) -> Dict[str, str]:
        """Discover all existing modules dynamically."""
        modules = {}
        if not self.modules_dir.exists():
            return modules
            
        for item in self.modules_dir.iterdir():
            if item.is_dir() and re.match(r'\d+_\w+', item.name):
                # Extract module name (e.g., "tensor" from "02_tensor")
                module_name = item.name.split('_', 1)[1]
                modules[item.name] = module_name
        
        return modules
    
    def create_new_mapping(self, current_modules: Dict[str, str]) -> Dict[str, str]:
        """Create new module mapping, removing setup and renumbering."""
        new_mapping = {}
        counter = 1
        
        for old_dir, module_name in sorted(current_modules.items()):
            if module_name == "setup":
                # Skip setup - it gets removed
                print(f"🗑️  Removing {old_dir} (setup functionality moves to CLI)")
                continue
            
            new_dir = f"{counter:02d}_{module_name}"
            new_mapping[old_dir] = new_dir
            print(f"📦 {old_dir} → {new_dir}")
            counter += 1
        
        return new_mapping
    
    def backup_setup_module(self):
        """Archive the setup module before deletion."""
        setup_dir = self.modules_dir / "01_setup"
        if setup_dir.exists():
            archive_dir = self.project_root / "archive" / "setup_module"
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📁 Archiving setup module to {archive_dir}")
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            shutil.copytree(setup_dir, archive_dir)
    
    def rename_module_directories(self, mapping: Dict[str, str]):
        """Rename module directories according to mapping."""
        print("\n🔄 Renaming module directories...")
        
        # Create temporary names first to avoid conflicts
        temp_mapping = {}
        for old_name, new_name in mapping.items():
            old_path = self.modules_dir / old_name
            temp_name = f"temp_{new_name}"
            temp_path = self.modules_dir / temp_name
            
            if old_path.exists():
                print(f"  {old_name} → {temp_name} (temporary)")
                shutil.move(str(old_path), str(temp_path))
                temp_mapping[temp_name] = new_name
        
        # Now rename from temp to final names
        for temp_name, final_name in temp_mapping.items():
            temp_path = self.modules_dir / temp_name
            final_path = self.modules_dir / final_name
            print(f"  {temp_name} → {final_name} (final)")
            shutil.move(str(temp_path), str(final_path))
    
    def update_yaml_files(self, mapping: Dict[str, str]):
        """Update module.yaml files with new dependencies."""
        print("\n📝 Updating YAML files...")
        
        for new_dir in mapping.values():
            yaml_path = self.modules_dir / new_dir / "module.yaml"
            if yaml_path.exists():
                print(f"  Updating {yaml_path}")
                
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Update prerequisites
                if 'dependencies' in data and 'prerequisites' in data['dependencies']:
                    old_prereqs = data['dependencies']['prerequisites']
                    new_prereqs = []
                    
                    for prereq in old_prereqs:
                        if prereq == "setup":
                            # Remove setup dependency
                            continue
                        # Map old module names to new ones
                        for old_dir, new_dir_name in mapping.items():
                            if old_dir.endswith(f"_{prereq}"):
                                new_number = new_dir_name.split('_')[0]
                                new_prereqs.append(prereq)  # Keep just the name
                                break
                        else:
                            new_prereqs.append(prereq)
                    
                    data['dependencies']['prerequisites'] = new_prereqs
                
                with open(yaml_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
    
    def update_file_references(self, mapping: Dict[str, str]):
        """Update all file references to use new module numbers."""
        print("\n🔍 Updating file references...")
        
        # Files that commonly reference modules
        files_to_update = [
            "book/convert_modules.py",
            "test_educational_integration.py", 
            "_reviews/COMPREHENSIVE_READABILITY_ASSESSMENT.md",
            "MODULE_OVERVIEW.md"
        ]
        
        for file_path in files_to_update:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"  Updating {file_path}")
                self.update_file_content(full_path, mapping)
    
    def update_file_content(self, file_path: Path, mapping: Dict[str, str]):
        """Update content of a single file with new module references."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Update module directory references
            for old_dir, new_dir in mapping.items():
                # Update paths like "modules/02_tensor" → "modules/01_tensor"
                content = re.sub(
                    rf'\bmodules/{re.escape(old_dir)}\b',
                    f'modules/{new_dir}',
                    content
                )
                
                # Update references like "02_tensor" → "01_tensor"
                content = re.sub(
                    rf'\b{re.escape(old_dir)}\b',
                    new_dir,
                    content
                )
            
            # Remove setup references
            content = re.sub(r'\b01_setup\b', '', content)
            content = re.sub(r'modules/01_setup[^\s]*', '', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ Updated {file_path}")
            else:
                print(f"    ⏭️  No changes needed in {file_path}")
                
        except Exception as e:
            print(f"    ❌ Error updating {file_path}: {e}")
    
    def remove_setup_from_tinytorch_package(self):
        """Remove setup.py from tinytorch package exports."""
        print("\n🗑️  Removing setup from tinytorch package...")
        
        setup_py = self.tinytorch_dir / "core" / "setup.py"
        if setup_py.exists():
            print(f"  Removing {setup_py}")
            setup_py.unlink()
        
        # Update __init__.py files to remove setup imports
        core_init = self.tinytorch_dir / "core" / "__init__.py"
        if core_init.exists():
            with open(core_init, 'r') as f:
                content = f.read()
            
            # Remove setup imports
            content = re.sub(r'from \.setup import.*\n', '', content)
            content = re.sub(r'import \.setup.*\n', '', content)
            
            with open(core_init, 'w') as f:
                f.write(content)
            print("  ✅ Updated core/__init__.py")
    
    def run_migration(self):
        """Execute the complete migration process."""
        print("🚀 Starting TinyTorch Module Reorganization")
        print("=" * 50)
        
        # Discover current modules
        current_modules = self.discover_all_modules()
        print(f"📊 Found {len(current_modules)} modules")
        
        # Create new mapping
        mapping = self.create_new_mapping(current_modules)
        
        # Execute migration steps
        try:
            self.backup_setup_module()
            self.remove_setup_from_tinytorch_package()
            self.rename_module_directories(mapping)
            self.update_yaml_files(mapping)
            self.update_file_references(mapping)
            
            print("\n✅ Migration completed successfully!")
            print("\nNext steps:")
            print("1. Implement 'tito setup' CLI command")
            print("2. Add numeric shortcuts (tito 01, tito 02)")
            print("3. Test the reorganized system")
            print("4. Update documentation")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print("🔄 Consider restoring from git backup")
            raise

if __name__ == "__main__":
    project_root = Path(__file__).parent
    migrator = ModuleReorganizer(project_root)
    migrator.run_migration()
