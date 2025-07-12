"""
Info command for TinyTorch CLI: shows system information and module status.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree

from .base import BaseCommand

class InfoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "info"

    @property
    def description(self) -> str:
        return "Show system information and module status"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--hello", action="store_true", help="Show hello message")
        parser.add_argument("--show-architecture", action="store_true", help="Show system architecture")

    def run(self, args: Namespace) -> int:
        console = self.console
        self.print_banner()
        console.print()
        # System Information Panel
        info_text = Text()
        info_text.append(f"Python: {sys.version.split()[0]}\n", style="cyan")
        info_text.append(f"Platform: {sys.platform}\n", style="cyan")
        info_text.append(f"Working Directory: {os.getcwd()}\n", style="cyan")
        # Virtual environment check
        venv_path = Path(".venv")
        venv_exists = venv_path.exists()
        in_venv = (
            os.environ.get('VIRTUAL_ENV') is not None or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            hasattr(sys, 'real_prefix')
        )
        if venv_exists and in_venv:
            venv_style = "green"
            venv_icon = "✅"
            venv_status = "Ready & Active"
        elif venv_exists:
            venv_style = "yellow"
            venv_icon = "✅"
            venv_status = "Ready (Not Active)"
        else:
            venv_style = "red"
            venv_icon = "❌"
            venv_status = "Not Found"
        info_text.append(f"Virtual Environment: {venv_icon} ", style=venv_style)
        info_text.append(venv_status, style=f"bold {venv_style}")
        console.print(Panel(info_text, title="📋 System Information", border_style="bright_blue"))
        console.print()
        # Course Navigation Panel
        nav_text = Text()
        nav_text.append("📖 Course Overview: ", style="dim")
        nav_text.append("README.md\n", style="cyan underline")
        nav_text.append("🎯 Detailed Guide: ", style="dim")
        nav_text.append("COURSE_GUIDE.md\n", style="cyan underline")
        nav_text.append("🚀 Start Here: ", style="dim")
        nav_text.append("modules/setup/README.md", style="cyan underline")
        console.print(Panel(nav_text, title="📋 Course Navigation", border_style="bright_green"))
        console.print()
        # Implementation status
        modules = [
            ("Setup", "hello_tinytorch function", self.check_setup_status),
            ("Tensor", "basic tensor operations", self.check_tensor_status),
            ("Layers", "neural network building blocks", self.check_layers_status),
            ("Networks", "neural network architectures", self.check_networks_status),
            ("MLP", "multi-layer perceptron (manual)", self.check_mlp_status),
            ("CNN", "convolutional networks (basic)", self.check_cnn_status),
            ("DataLoader", "data loading pipeline", self.check_data_status),
            ("Training", "autograd engine & optimization", self.check_training_status),
            ("Profiling", "performance profiling", self.check_profiling_status),
            ("Compression", "model compression", self.check_compression_status),
            ("Kernels", "custom compute kernels", self.check_kernels_status),
            ("Benchmarking", "performance benchmarking", self.check_benchmarking_status),
            ("MLOps", "production monitoring", self.check_mlops_status),
        ]
        status_table = Table(title="🚀 Module Implementation Status", show_header=True, header_style="bold blue")
        status_table.add_column("ID", style="dim", width=3, justify="center")
        status_table.add_column("Project", style="bold cyan", width=12)
        status_table.add_column("Status", width=18, justify="center")
        status_table.add_column("Description", style="dim", width=40)
        for i, (name, desc, check_func) in enumerate(modules):
            status_text = check_func()
            if "✅" in status_text:
                status_style = "[green]✅ Implemented[/green]"
            elif "❌" in status_text:
                status_style = "[red]❌ Not Implemented[/red]"
            else:
                status_style = "[yellow]⏳ Not Started[/yellow]"
            status_table.add_row(str(i), name, status_style, desc)
        console.print(status_table)
        # Optionally show hello message or architecture
        if args.hello and self.check_setup_status() == "✅ Implemented":
            try:
                from tinytorch.core.utils import hello_tinytorch
                hello_text = Text(hello_tinytorch(), style="bold red")
                console.print()
                console.print(Panel(hello_text, style="bright_red", padding=(1, 2)))
            except ImportError:
                pass
        if args.show_architecture:
            console.print()
            arch_tree = Tree("🏗️ TinyTorch System Architecture", style="bold blue")
            cli_branch = arch_tree.add("CLI Interface", style="cyan")
            cli_branch.add("tito/ - Command line tools", style="dim")
            training_branch = arch_tree.add("Training Orchestration", style="cyan")
            training_branch.add("trainer.py - Training loop management", style="dim")
            core_branch = arch_tree.add("Core Components", style="cyan")
            model_sub = core_branch.add("Model Definition", style="yellow")
            model_sub.add("modules.py - Neural network layers", style="dim")
            data_sub = core_branch.add("Data Pipeline", style="yellow")
            data_sub.add("dataloader.py - Efficient data loading", style="dim")
            opt_sub = core_branch.add("Optimization", style="yellow")
            opt_sub.add("optimizer.py - SGD, Adam, etc.", style="dim")
            autograd_branch = arch_tree.add("Automatic Differentiation Engine", style="cyan")
            autograd_branch.add("autograd.py - Gradient computation", style="dim")
            tensor_branch = arch_tree.add("Tensor Operations & Storage", style="cyan")
            tensor_branch.add("tensor.py - Core tensor implementation", style="dim")
            system_branch = arch_tree.add("System Tools", style="cyan")
            system_branch.add("profiler.py - Performance measurement", style="dim")
            system_branch.add("mlops.py - Production monitoring", style="dim")
            console.print(Panel(arch_tree, title="🏗️ System Architecture", border_style="bright_blue"))
        return 0

    def print_banner(self):
        banner_text = Text("Tiny🔥Torch: Build ML Systems from Scratch", style="bold red")
        self.console.print(Panel(banner_text, style="bright_blue", padding=(1, 2)))

    # The following check_* methods are ported from bin/tito.py
    def check_setup_status(self):
        try:
            from tinytorch.core.utils import hello_tinytorch
            return "✅ Implemented"
        except ImportError:
            return "❌ Not Implemented"
    def check_tensor_status(self):
        try:
            from tinytorch.core.tensor import Tensor
            t1 = Tensor([1, 2, 3])
            t2 = Tensor([4, 5, 6])
            _ = t1 + t2
            return "✅ Implemented"
        except (ImportError, NotImplementedError):
            return "⏳ Not Started"
    
    def check_layers_status(self):
        try:
            from tinytorch.core.layers import Dense
            from tinytorch.core.activations import ReLU
            from tinytorch.core.tensor import Tensor
            layer = Dense(3, 4)
            activation = ReLU()
            x = Tensor([[1, 2, 3]])
            _ = activation(layer(x))
            return "✅ Implemented"
        except (ImportError, NotImplementedError):
            return "⏳ Not Started"
    
    def check_networks_status(self):
        try:
            from tinytorch.core.networks import Sequential
            from tinytorch.core.layers import Dense
            from tinytorch.core.activations import ReLU, Sigmoid
            from tinytorch.core.tensor import Tensor
            network = Sequential([Dense(3, 4), ReLU(), Dense(4, 2), Sigmoid()])
            x = Tensor([[1, 2, 3]])
            _ = network(x)
            return "✅ Implemented"
        except (ImportError, NotImplementedError):
            return "⏳ Not Started"
    def check_mlp_status(self):
        try:
            from tinytorch.core.modules import MLP
            mlp = MLP(input_size=10, hidden_size=5, output_size=2)
            from tinytorch.core.tensor import Tensor
            x = Tensor([[1,2,3,4,5,6,7,8,9,10]])
            _ = mlp(x)
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_cnn_status(self):
        try:
            from tinytorch.core.modules import Conv2d
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            from tinytorch.core.tensor import Tensor
            x = Tensor([[0]*32]*32)
            _ = conv(x)
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_data_status(self):
        try:
            from tinytorch.core.dataloader import DataLoader
            from tinytorch.core.tensor import Tensor
            import numpy as np
            data = [(Tensor(np.random.randn(3,32,32)), Tensor(np.array(i % 10))) for i in range(10)]
            loader = DataLoader(data, batch_size=2, shuffle=True)
            _ = next(iter(loader))
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError, StopIteration):
            return "⏳ Not Started"
    def check_training_status(self):
        try:
            from tinytorch.core.optimizer import SGD
            from tinytorch.core.tensor import Tensor
            t = Tensor([1.0,2.0,3.0], requires_grad=True)
            optimizer = SGD([t], lr=0.01)
            t.backward()
            optimizer.step()
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_profiling_status(self):
        try:
            from tinytorch.core.profiler import Profiler
            profiler = Profiler()
            profiler.start("test")
            profiler.end("test")
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_compression_status(self):
        try:
            from tinytorch.core.compression import Pruner
            pruner = Pruner(sparsity=0.5)
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_kernels_status(self):
        try:
            from tinytorch.core.kernels import optimized_matmul
            import numpy as np
            a = np.random.randn(3,3)
            b = np.random.randn(3,3)
            _ = optimized_matmul(a, b)
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_benchmarking_status(self):
        try:
            from tinytorch.core.benchmark import Benchmark
            benchmark = Benchmark()
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError):
            return "⏳ Not Started"
    def check_mlops_status(self):
        try:
            from tinytorch.core.mlops import ModelMonitor
            from tinytorch.core.tensor import Tensor
            monitor = ModelMonitor(model=None, baseline_metrics={})
            test_inputs = Tensor([1.0,2.0,3.0])
            test_predictions = Tensor([0.5,0.8,0.2])
            monitor.log_prediction(test_inputs, test_predictions)
            return "✅ Implemented"
        except (ImportError, NotImplementedError, AttributeError, TypeError):
            return "⏳ Not Started" 