#!/usr/bin/env python3
"""
Tito Demo Command - Show off your AI capabilities!
Runs progressive demos showing what TinyTorch can do at each stage.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .base import BaseCommand

console = Console()

class TinyTorchDemoMatrix:
    """Tracks and displays TinyTorch AI demo capabilities"""
    
    def __init__(self):
        self.demos = {
            'math': {
                'name': 'Mathematical Operations',
                'file': 'demo_tensor_math.py',
                'requires': ['02_tensor'],
                'description': 'Linear algebra, matrix operations, transformations'
            },
            'logic': {
                'name': 'Logical Reasoning', 
                'file': 'demo_activations.py',
                'requires': ['02_tensor', '03_activations'],
                'description': 'Boolean functions, XOR problem, decision boundaries'
            },
            'neuron': {
                'name': 'Single Neuron Learning',
                'file': 'demo_single_neuron.py', 
                'requires': ['02_tensor', '03_activations', '04_layers'],
                'description': 'Watch a neuron learn the AND gate'
            },
            'network': {
                'name': 'Multi-Layer Networks',
                'file': 'demo_xor_network.py',
                'requires': ['02_tensor', '03_activations', '04_layers', '05_dense'],
                'description': 'Solve the famous XOR problem'
            },
            'vision': {
                'name': 'Computer Vision',
                'file': 'demo_vision.py',
                'requires': ['02_tensor', '03_activations', '04_layers', '05_dense', '06_spatial'],
                'description': 'Image processing and pattern recognition'
            },
            'attention': {
                'name': 'Attention Mechanisms',
                'file': 'demo_attention.py',
                'requires': ['02_tensor', '03_activations', '04_layers', '05_dense', '07_attention'],
                'description': 'Sequence processing and attention'
            },
            'training': {
                'name': 'End-to-End Training',
                'file': 'demo_training.py',
                'requires': ['02_tensor', '03_activations', '04_layers', '05_dense', '11_training'],
                'description': 'Complete training pipelines'
            },
            'language': {
                'name': 'Language Generation',
                'file': 'demo_language.py',
                'requires': ['02_tensor', '03_activations', '04_layers', '05_dense', '07_attention', '16_tinygpt'],
                'description': 'AI text generation and language models'
            }
        }
    
    def check_module_exported(self, module_name):
        """Check if a module has been exported to the package"""
        try:
            if module_name == '02_tensor':
                import tinytorch.core.tensor
                return True
            elif module_name == '03_activations':
                import tinytorch.core.activations
                return True
            elif module_name == '04_layers':
                import tinytorch.core.layers
                return True
            elif module_name == '05_dense':
                import tinytorch.core.dense
                return True
            elif module_name == '06_spatial':
                import tinytorch.core.spatial
                return True
            elif module_name == '07_attention':
                import tinytorch.core.attention
                return True
            elif module_name == '11_training':
                import tinytorch.core.training
                return True
            elif module_name == '16_tinygpt':
                import tinytorch.tinygpt
                return True
            return False
        except ImportError:
            return False
    
    def get_demo_status(self, demo_name):
        """Get status of a demo: available, partial, or unavailable"""
        demo = self.demos[demo_name]
        required_modules = demo['requires']
        
        available_count = sum(1 for module in required_modules if self.check_module_exported(module))
        total_count = len(required_modules)
        
        if available_count == total_count:
            return '✅'  # Fully available
        elif available_count > 0:
            return '⚡'  # Partially available
        else:
            return '❌'  # Not available
    
    def show_matrix(self):
        """Display the demo capability matrix"""
        console.print("\n🤖 TinyTorch Demo Matrix", style="bold cyan")
        console.print("=" * 50)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Demo", style="cyan", width=20)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Description", style="dim")
        
        available_demos = []
        
        for demo_name, demo_info in self.demos.items():
            status = self.get_demo_status(demo_name)
            table.add_row(demo_info['name'], status, demo_info['description'])
            
            if status == '✅':
                available_demos.append(demo_name)
        
        console.print(table)
        console.print()
        
        if available_demos:
            console.print("🎯 Available Demos:", style="bold green")
            for demo in available_demos:
                console.print(f"  • tito demo {demo}")
            console.print()
        
        console.print("Legend: ✅ Ready  ⚡ Partial  ❌ Not Available")
        console.print()
    
    def run_demo(self, demo_name):
        """Run a specific demo"""
        if demo_name not in self.demos:
            console.print(f"❌ Unknown demo: {demo_name}", style="red")
            console.print("Available demos:", ', '.join(self.demos.keys()))
            return False
        
        demo = self.demos[demo_name]
        status = self.get_demo_status(demo_name)
        
        if status == '❌':
            console.print(f"❌ Demo '{demo_name}' not available", style="red")
            missing_modules = [m for m in demo['requires'] if not self.check_module_exported(m)]
            console.print(f"Missing modules: {', '.join(missing_modules)}")
            console.print(f"Run: tito export {' '.join(missing_modules)}")
            return False
        
        if status == '⚡':
            console.print(f"⚠️ Demo '{demo_name}' partially available", style="yellow")
            console.print("Some features may not work correctly.")
        
        # Find the demo file
        project_root = Path(__file__).parent.parent.parent
        demo_file = project_root / "demos" / demo['file']
        
        if not demo_file.exists():
            console.print(f"❌ Demo file not found: {demo_file}", style="red")
            return False
        
        console.print(f"🚀 Running {demo['name']} Demo...", style="bold green")
        console.print()
        
        # Run the demo
        try:
            result = subprocess.run([sys.executable, str(demo_file)], 
                                  capture_output=False, 
                                  text=True)
            return result.returncode == 0
        except Exception as e:
            console.print(f"❌ Demo failed: {e}", style="red")
            return False

class DemoCommand(BaseCommand):
    """Command for running TinyTorch AI capability demos"""
    
    def __init__(self, config):
        super().__init__(config)
        self.matrix = TinyTorchDemoMatrix()
    
    @property
    def name(self) -> str:
        return "demo"
    
    @property
    def description(self) -> str:
        return "Run AI capability demos"
    
    def add_arguments(self, parser):
        """Add demo command arguments"""
        parser.add_argument('demo_name', nargs='?', 
                           help='Name of demo to run (math, logic, neuron, network, etc.)')
        parser.add_argument('--all', action='store_true',
                           help='Run all available demos')
        parser.add_argument('--matrix', action='store_true',
                           help='Show capability matrix only')
    
    def run(self, args):
        """Execute the demo command"""
        # Just show matrix if no args or --matrix flag
        if not args.demo_name and not args.all or args.matrix:
            self.matrix.show_matrix()
            return
        
        # Run all available demos
        if args.all:
            self.matrix.show_matrix()
            available_demos = [name for name in self.matrix.demos.keys() 
                              if self.matrix.get_demo_status(name) == '✅']
            
            if not available_demos:
                console.print("❌ No demos available. Export some modules first!", style="red")
                return
            
            console.print(f"🚀 Running {len(available_demos)} available demos...", style="bold green")
            console.print()
            
            for demo_name in available_demos:
                console.print(f"\n{'='*60}")
                success = self.matrix.run_demo(demo_name)
                if not success:
                    console.print(f"❌ Demo {demo_name} failed", style="red")
            
            console.print(f"\n{'='*60}")
            console.print("🏆 All available demos completed!", style="bold green")
            return
        
        # Run specific demo
        if args.demo_name:
            self.matrix.run_demo(args.demo_name)

def main():
    """Standalone entry point for development"""
    import argparse
    parser = argparse.ArgumentParser()
    DemoCommand.add_parser(parser._subparsers_action.add_parser if hasattr(parser, '_subparsers_action') else parser.add_subparser)
    args = parser.parse_args()
    
    cmd = DemoCommand()
    cmd.execute(args)

if __name__ == "__main__":
    main()