"""
Export command for TinyTorch CLI: exports notebook code to Python package using nbdev.
"""

import subprocess
import sys
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand
from .checkpoint import CheckpointSystem

class ExportCommand(BaseCommand):
    # Module to checkpoint mapping - defines which checkpoint is triggered after each module
    MODULE_TO_CHECKPOINT = {
        "01_setup": "00",          # Setup → Environment checkpoint  
        "02_tensor": "01",         # Tensor → Foundation checkpoint
        "03_activations": "02",    # Activations → Intelligence checkpoint
        "04_layers": "03",         # Layers → Components checkpoint
        "05_dense": "04",          # Dense → Networks checkpoint
        "06_spatial": "05",        # Spatial → Learning checkpoint
        "07_attention": "06",      # Attention → Attention checkpoint
        "08_dataloader": "07",     # Dataloader → Stability checkpoint (data prep)
        "09_autograd": "08",       # Autograd → Differentiation checkpoint
        "10_optimizers": "09",     # Optimizers → Optimization checkpoint
        "11_training": "10",       # Training → Training checkpoint
        "12_compression": "11",    # Compression → Regularization checkpoint
        "13_kernels": "12",        # Kernels → Kernels checkpoint
        "14_benchmarking": "13",   # Benchmarking → Benchmarking checkpoint
        "15_mlops": "14",          # MLOps → Deployment checkpoint
        "16_tinygpt": "15",        # TinyGPT → Capstone checkpoint
    }

    @property
    def name(self) -> str:
        return "export"

    @property
    def description(self) -> str:
        return "Export notebook code to Python package"

    def add_arguments(self, parser: ArgumentParser) -> None:
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("module", nargs="?", help="Export specific module (e.g., setup, tensor)")
        group.add_argument("--all", action="store_true", help="Export all modules")
        parser.add_argument("--from-release", action="store_true", help="Export from release directory (student version) instead of source")
        parser.add_argument("--test-checkpoint", action="store_true", help="Run checkpoint test after successful export")

    def _get_export_target(self, module_path: Path) -> str:
        """
        Read the actual export target from the dev file's #| default_exp directive.
        This is the source of truth, not the YAML file.
        """
        # Extract the short name from the full module name
        module_name = module_path.name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name
        
        dev_file = module_path / f"{short_name}_dev.py"
        if not dev_file.exists():
            return "unknown"
        
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for #| default_exp directive with more flexible regex
                match = re.search(r'#\|\s*default_exp\s+([^\n\r]+)', content)
                if match:
                    return match.group(1).strip()
        except Exception as e:
            # Debug: print the error for troubleshooting
            print(f"Debug: Error reading {dev_file}: {e}")
        
        return "unknown"

    def _discover_modules(self) -> list:
        """Discover available modules from modules/source directory."""
        source_dir = Path("modules/source")
        modules = []
        
        if source_dir.exists():
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                    modules.append(module_dir.name)
        
        return sorted(modules)

    def _run_checkpoint_test(self, module_name: str) -> Dict:
        """Run checkpoint test for a module if mapping exists."""
        if module_name not in self.MODULE_TO_CHECKPOINT:
            return {"skipped": True, "reason": f"No checkpoint mapping for module {module_name}"}
        
        checkpoint_id = self.MODULE_TO_CHECKPOINT[module_name]
        checkpoint_system = CheckpointSystem(self.config)
        
        console = self.console
        console.print(f"\n[bold cyan]🧪 Running Checkpoint Test[/bold cyan]")
        
        checkpoint = checkpoint_system.CHECKPOINTS[checkpoint_id]
        console.print(f"[bold]Checkpoint {checkpoint_id}: {checkpoint['name']}[/bold]")
        console.print(f"[dim]Testing: {checkpoint['capability']}[/dim]")
        
        with console.status(f"[bold green]Running checkpoint {checkpoint_id} test...", spinner="dots"):
            result = checkpoint_system.run_checkpoint_test(checkpoint_id)
        
        return result

    def _show_checkpoint_results(self, result: Dict, module_name: str) -> None:
        """Display checkpoint test results with celebration or guidance."""
        console = self.console
        
        if result.get("skipped"):
            console.print(f"[dim]No checkpoint test for {module_name}[/dim]")
            return
        
        if result["success"]:
            # Celebration and progress feedback
            checkpoint_name = result.get("checkpoint_name", "Unknown")
            capability = result.get("capability", "")
            
            console.print(Panel(
                f"[bold green]🎉 Checkpoint Achieved![/bold green]\n\n"
                f"[green]✅ {checkpoint_name} checkpoint unlocked![/green]\n"
                f"[green]Capability: {capability}[/green]\n\n"
                f"[bold cyan]🚀 Progress Update[/bold cyan]\n"
                f"You've successfully built the {module_name} module and\n"
                f"proven your {checkpoint_name.lower()} capabilities!",
                title=f"Module {module_name} Complete",
                border_style="green"
            ))
            
            # Show next steps
            self._show_next_steps(module_name)
        else:
            console.print(Panel(
                f"[bold yellow]⚠️  Export Successful, Test Incomplete[/bold yellow]\n\n"
                f"[yellow]Module {module_name} exported successfully,[/yellow]\n"
                f"[yellow]but the checkpoint test failed.[/yellow]\n\n"
                f"[bold]This usually means:[/bold]\n"
                f"• Some functionality is still missing\n"
                f"• Implementation needs refinement\n"
                f"• Module requirements not fully met\n\n"
                f"[dim]Check the implementation and try again[/dim]",
                title="Integration Test Failed",
                border_style="yellow"
            ))
            
            # Show error details if available
            if "error" in result:
                console.print(f"\n[red]Error: {result['error']}[/red]")
            elif result.get("stderr"):
                console.print(f"\n[red]Test error output:[/red]")
                console.print(f"[dim]{result['stderr']}[/dim]")

    def _show_next_steps(self, completed_module: str) -> None:
        """Show next steps after successful module completion."""
        console = self.console
        
        # Get module number for next module suggestion
        if completed_module.startswith(tuple(f"{i:02d}_" for i in range(100))):
            try:
                module_num = int(completed_module[:2])
                next_num = module_num + 1
                
                # Suggest next module
                next_modules = {
                    1: ("02_tensor", "Tensor operations - the foundation of ML"),
                    2: ("03_activations", "Activation functions - adding intelligence"),
                    3: ("04_layers", "Neural layers - building blocks"),
                    4: ("05_dense", "Dense networks - complete architectures"),
                    5: ("06_spatial", "Spatial processing - convolutional operations"),
                    6: ("07_attention", "Attention mechanisms - sequence understanding"),
                    7: ("08_dataloader", "Data loading - efficient training"),
                    8: ("09_autograd", "Automatic differentiation - gradient computation"),
                    9: ("10_optimizers", "Optimization algorithms - sophisticated learning"),
                    10: ("11_training", "Training loops - end-to-end learning"),
                    11: ("12_compression", "Model compression - efficient deployment"),
                    12: ("13_kernels", "High-performance kernels - optimized computation"),
                    13: ("14_benchmarking", "Performance analysis - bottleneck identification"),
                    14: ("15_mlops", "MLOps - production deployment"),
                    15: ("16_capstone", "Capstone project - complete ML systems"),
                }
                
                if next_num in next_modules:
                    next_module, next_desc = next_modules[next_num]
                    console.print(f"\n[bold cyan]🎯 Continue Your Journey[/bold cyan]")
                    console.print(f"[bold]Next Module:[/bold] {next_module}")
                    console.print(f"[dim]{next_desc}[/dim]")
                    console.print(f"\n[green]Ready to continue? Run:[/green]")
                    console.print(f"[dim]  tito module view {next_module}[/dim]")
                elif next_num > 16:
                    console.print(f"\n[bold green]🏆 Congratulations![/bold green]")
                    console.print(f"[green]You've completed all TinyTorch modules![/green]")
                    console.print(f"[dim]Run 'tito checkpoint status' to see your full progress[/dim]")
            except (ValueError, IndexError):
                pass
        
        # General next steps
        console.print(f"\n[bold]Continue your ML systems journey:[/bold]")
        console.print(f"[dim]  tito checkpoint status    - View overall progress[/dim]")
        console.print(f"[dim]  tito checkpoint timeline  - Visual progress timeline[/dim]")

    def _add_autogenerated_warnings(self, console):
        """Add auto-generated warnings to all exported Python files."""
        console.print("[dim]🔧 Adding auto-generated warnings to exported files...[/dim]")
        
        tinytorch_path = Path("tinytorch")
        if not tinytorch_path.exists():
            return
        
        files_updated = 0
        for py_file in tinytorch_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue  # Skip __init__.py files
                
            try:
                # Read the current content
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if warning already exists
                if "AUTOGENERATED! DO NOT EDIT!" in content:
                    continue  # Already has warning
                
                # Find the source file for this export
                source_file = self._find_source_file_for_export(py_file)
                
                # Create auto-generated warning header
                warning_header = f"""# AUTOGENERATED! DO NOT EDIT! File to edit: {source_file}.

"""
                
                # Add warning at the top (after any existing shebang)
                lines = content.split('\n')
                insert_index = 0
                
                # Skip shebang line if present
                if lines and lines[0].startswith('#!'):
                    insert_index = 1
                
                # Insert warning
                lines.insert(insert_index, warning_header.rstrip())
                
                # Write back the modified content
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                files_updated += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add warning to {py_file}: {e}[/yellow]")
        
        if files_updated > 0:
            console.print(f"[green]✅ Added auto-generated warnings to {files_updated} files[/green]")

    def _find_source_file_for_export(self, exported_file: Path) -> str:
        """Find the source dev file that generated this export."""
        # Convert tinytorch/core/something.py back to source path
        rel_path = exported_file.relative_to(Path("tinytorch"))
        
        # Remove .py extension and convert to module path
        module_parts = rel_path.with_suffix('').parts
        
        # Common mappings
        source_mappings = {
            ('core', 'tensor'): 'modules/source/02_tensor/tensor_dev.py',
            ('core', 'activations'): 'modules/source/03_activations/activations_dev.py', 
            ('core', 'layers'): 'modules/source/04_layers/layers_dev.py',
            ('core', 'dense'): 'modules/source/05_dense/dense_dev.py',
            ('core', 'spatial'): 'modules/source/06_spatial/spatial_dev.py',
            ('core', 'attention'): 'modules/source/07_attention/attention_dev.py',
            ('core', 'dataloader'): 'modules/source/08_dataloader/dataloader_dev.py',
            ('core', 'autograd'): 'modules/source/09_autograd/autograd_dev.py',
            ('core', 'optimizers'): 'modules/source/10_optimizers/optimizers_dev.py',
            ('core', 'training'): 'modules/source/11_training/training_dev.py',
            ('core', 'compression'): 'modules/source/12_compression/compression_dev.py',
            ('core', 'kernels'): 'modules/source/13_kernels/kernels_dev.py',
            ('core', 'benchmarking'): 'modules/source/14_benchmarking/benchmarking_dev.py',
            ('core', 'networks'): 'modules/source/16_tinygpt/tinygpt_dev.ipynb',
        }
        
        if module_parts in source_mappings:
            return source_mappings[module_parts]
        
        # Fallback: try to guess based on the file name
        if len(module_parts) >= 2:
            module_name = module_parts[-1]  # e.g., 'tensor' from ('core', 'tensor')
            return f"modules/source/XX_{module_name}/{module_name}_dev.py"
        
        return "modules/source/[unknown]/[unknown]_dev.py"

    def _show_export_details(self, console, module_name: Optional[str] = None):
        """Show detailed export information including where each module exports to."""
        exports_text = Text()
        exports_text.append("📦 Export Details:\n", style="bold cyan")
        
        if module_name:
            # Single module export
            module_path = Path(f"modules/source/{module_name}")
            export_target = self._get_export_target(module_path)
            if export_target != "unknown":
                target_file = export_target.replace('.', '/') + '.py'
                exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
                
                # Extract the short name for display
                short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
                exports_text.append(f"     Source: modules/source/{module_name}/{short_name}_dev.py\n", style="dim")
                exports_text.append(f"     Target: tinytorch/{target_file}\n", style="dim")
            else:
                exports_text.append(f"  ❓ {module_name} → export target not found\n", style="yellow")
        else:
            # All modules export
            modules = self._discover_modules()
            for module_name in modules:
                module_path = Path(f"modules/source/{module_name}")
                export_target = self._get_export_target(module_path)
                if export_target != "unknown":
                    target_file = export_target.replace('.', '/') + '.py'
                    exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
        
        # Show what was actually created
        exports_text.append("\n📁 Generated Files:\n", style="bold cyan")
        tinytorch_path = Path("tinytorch")
        if tinytorch_path.exists():
            for py_file in tinytorch_path.rglob("*.py"):
                if py_file.name != "__init__.py" and py_file.stat().st_size > 100:  # Non-empty files
                    rel_path = py_file.relative_to(tinytorch_path)
                    exports_text.append(f"  ✅ tinytorch/{rel_path}\n", style="green")
        
        exports_text.append("\n💡 Next steps:\n", style="bold yellow")
        exports_text.append("  • Run: tito test --all\n", style="white")
        exports_text.append("  • Or: tito test <module_name>\n", style="white")
        exports_text.append("  • Or: tito export <module> --test-checkpoint\n", style="white")
        
        console.print(Panel(exports_text, title="Export Summary", border_style="bright_green"))

    def _validate_notebook_integrity(self, notebook_path: Path) -> Dict:
        """Validate notebook integrity and structure."""
        import json
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            
            # Basic structure checks
            issues = []
            warnings = []
            
            # Check required fields
            if 'cells' not in notebook_data:
                issues.append("Missing 'cells' field")
            elif not isinstance(notebook_data['cells'], list):
                issues.append("'cells' field is not a list")
            
            if 'metadata' not in notebook_data:
                warnings.append("Missing metadata field")
            
            if 'nbformat' not in notebook_data:
                warnings.append("Missing nbformat field")
            
            # Check cells for common issues
            cell_count = 0
            code_cells = 0
            markdown_cells = 0
            
            if 'cells' in notebook_data:
                for i, cell in enumerate(notebook_data['cells']):
                    cell_count += 1
                    
                    if 'cell_type' not in cell:
                        issues.append(f"Cell {i}: missing cell_type")
                        continue
                    
                    cell_type = cell['cell_type']
                    if cell_type == 'code':
                        code_cells += 1
                        if 'source' not in cell:
                            warnings.append(f"Code cell {i}: missing source")
                    elif cell_type == 'markdown':
                        markdown_cells += 1
                        if 'source' not in cell:
                            warnings.append(f"Markdown cell {i}: missing source")
                    else:
                        warnings.append(f"Cell {i}: unusual cell type '{cell_type}'")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "stats": {
                    "total_cells": cell_count,
                    "code_cells": code_cells,
                    "markdown_cells": markdown_cells
                }
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "issues": [f"Invalid JSON: {str(e)}"],
                "warnings": [],
                "stats": {}
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "stats": {}
            }

    def _convert_py_to_notebook(self, module_path: Path) -> bool:
        """Convert .py dev file to .ipynb using Jupytext with integrity checking."""
        module_name = module_path.name
        short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
        
        dev_file = module_path / f"{short_name}_dev.py"
        if not dev_file.exists():
            return False
        
        notebook_file = module_path / f"{short_name}_dev.ipynb"
        
        # Check if notebook is newer than .py file
        if notebook_file.exists():
            py_mtime = dev_file.stat().st_mtime
            nb_mtime = notebook_file.stat().st_mtime
            if nb_mtime > py_mtime:
                # Still validate existing notebook
                validation = self._validate_notebook_integrity(notebook_file)
                if not validation["valid"]:
                    self.console.print(f"[yellow]⚠️  Existing notebook has integrity issues: {', '.join(validation['issues'])}[/yellow]")
                    self.console.print("[yellow]Regenerating notebook...[/yellow]")
                else:
                    return True  # Notebook is up to date and valid
        
        try:
            # Try to find jupytext in virtual environment first
            jupytext_path = "jupytext"
            
            # Get the project root directory (where .venv should be)
            project_root = Path(__file__).parent.parent.parent
            venv_jupytext = project_root / ".venv" / "bin" / "jupytext"
            
            if venv_jupytext.exists():
                jupytext_path = str(venv_jupytext)
            
            result = subprocess.run([
                jupytext_path, "--to", "ipynb", str(dev_file)
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                # Validate the generated notebook
                validation = self._validate_notebook_integrity(notebook_file)
                if not validation["valid"]:
                    self.console.print(f"[red]❌ Generated notebook has integrity issues:[/red]")
                    for issue in validation["issues"]:
                        self.console.print(f"[red]  • {issue}[/red]")
                    return False
                
                if validation["warnings"]:
                    self.console.print("[yellow]⚠️  Notebook warnings:[/yellow]")
                    for warning in validation["warnings"]:
                        self.console.print(f"[yellow]  • {warning}[/yellow]")
                
                # Show notebook stats
                stats = validation["stats"]
                self.console.print(f"[dim]📊 Notebook: {stats.get('total_cells', 0)} cells "
                                 f"({stats.get('code_cells', 0)} code, {stats.get('markdown_cells', 0)} markdown)[/dim]")
                
                return True
            
            return False
        except FileNotFoundError:
            return False
    
    def _convert_all_modules(self) -> list:
        """Convert all modules' .py files to .ipynb files."""
        modules = self._discover_modules()
        converted = []
        
        for module_name in modules:
            module_path = Path(f"modules/source/{module_name}")
            if self._convert_py_to_notebook(module_path):
                converted.append(module_name)
        
        return converted

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Determine what to export
        if hasattr(args, 'module') and args.module:
            # Validate module exists
            module_path = Path(f"modules/source/{args.module}")
            if not module_path.exists():
                console.print(Panel(f"[red]❌ Module '{args.module}' not found in modules/source/[/red]", 
                                  title="Module Not Found", border_style="red"))
                
                # Show available modules
                available_modules = self._discover_modules()
                if available_modules:
                    help_text = Text()
                    help_text.append("Available modules:\n", style="bold yellow")
                    for module in available_modules:
                        help_text.append(f"  • {module}\n", style="white")
                    console.print(Panel(help_text, title="Available Modules", border_style="yellow"))
                
                return 1
            
            console.print(Panel(f"🔄 Exporting Module: {args.module}", 
                               title="Complete Export Workflow", border_style="bright_cyan"))
            
            # Step 1: Convert .py to .ipynb
            console.print(f"📝 Converting {args.module} Python file to notebook...")
            if not self._convert_py_to_notebook(module_path):
                console.print(Panel("[red]❌ Failed to convert .py file to notebook. Is jupytext installed?[/red]", 
                                  title="Conversion Error", border_style="red"))
                return 1
            
            console.print(f"🔄 Exporting {args.module} notebook to tinytorch package...")
            
            # Step 2: Use nbdev_export with --path for specific module
            cmd = ["nbdev_export", "--path", str(module_path)]
        elif hasattr(args, 'all') and args.all:
            console.print(Panel("🔄 Exporting All Modules to Package", 
                               title="Complete Export Workflow", border_style="bright_cyan"))
            
            # Step 1: Convert all .py files to .ipynb
            console.print("📝 Converting all Python files to notebooks...")
            converted = self._convert_all_modules()
            if not converted:
                console.print(Panel("[red]❌ No modules converted. Check if jupytext is installed and .py files exist.[/red]", 
                                  title="Conversion Error", border_style="red"))
                return 1
            
            console.print(f"✅ Converted {len(converted)} modules: {', '.join(converted)}")
            console.print("🔄 Exporting all notebook code to tinytorch package...")
            
            # Step 2: Use nbdev_export for all modules  
            cmd = ["nbdev_export"]
        else:
            console.print(Panel("[red]❌ Must specify either a module name or --all[/red]", 
                              title="Missing Arguments", border_style="red"))
            return 1
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                console.print(Panel("[green]✅ Successfully exported notebook code to tinytorch package![/green]", 
                                  title="Export Success", border_style="green"))
                
                # Add auto-generated warnings to exported files
                self._add_autogenerated_warnings(console)
                
                # Show detailed export information
                module_name = args.module if hasattr(args, 'module') and args.module else None
                self._show_export_details(console, module_name)
                
                # Run checkpoint test if requested and for single module exports
                if hasattr(args, 'test_checkpoint') and args.test_checkpoint and module_name:
                    checkpoint_result = self._run_checkpoint_test(module_name)
                    self._show_checkpoint_results(checkpoint_result, module_name)
                
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                console.print(Panel(f"[red]❌ Export failed:\n{error_msg}[/red]", 
                                  title="Export Error", border_style="red"))
                
                # Helpful error guidance
                help_text = Text()
                help_text.append("💡 Common issues:\n", style="bold yellow")
                help_text.append("  • Missing #| default_exp directive in notebook\n", style="white")
                help_text.append("  • Syntax errors in exported code\n", style="white")
                help_text.append("  • Missing settings.ini configuration\n", style="white")
                help_text.append("\n🔧 Run 'tito system doctor' for detailed diagnosis", style="cyan")
                
                console.print(Panel(help_text, title="Troubleshooting", border_style="yellow"))
                
            return result.returncode
            
        except FileNotFoundError:
            console.print(Panel("[red]❌ nbdev not found. Install with: pip install nbdev[/red]", 
                              title="Missing Dependency", border_style="red"))
            return 1 