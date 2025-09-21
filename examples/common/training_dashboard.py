#!/usr/bin/env python3
"""
TinyTorch Universal Training Dashboard

A beautiful, reusable Rich UI dashboard for any TinyTorch training script.
Features real-time ASCII plotting, progress tracking, and gorgeous formatting.

Usage:
    from examples.common.training_dashboard import TrainingDashboard
    
    dashboard = TrainingDashboard(title="Your Model Training")
    dashboard.start_training(num_epochs=10, target_accuracy=0.9)
    
    for epoch in range(num_epochs):
        # Training loop...
        dashboard.update_epoch(epoch+1, train_acc, test_acc, loss, extra_metrics={})
        
    dashboard.finish_training(final_accuracy=0.95)
"""

import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.columns import Columns

console = Console()

@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    train_accuracy: float
    test_accuracy: float
    loss: float
    extra_metrics: Dict[str, float]
    timestamp: float

class ASCIIPlotter:
    """Universal ASCII plotting for training metrics"""
    
    def __init__(self, width: int = 50, height: int = 10):
        self.width = width
        self.height = height
        self.metrics_history: List[TrainingMetrics] = []
        
    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics data point"""
        self.metrics_history.append(metrics)
        
        # Keep reasonable history for plotting
        max_points = self.width - 5
        if len(self.metrics_history) > max_points:
            self.metrics_history = self.metrics_history[-max_points:]
    
    def plot_accuracy(self) -> str:
        """Generate ASCII plot of accuracy over time"""
        if not self.metrics_history:
            return "No data yet..."
        
        train_accs = [m.train_accuracy for m in self.metrics_history]
        test_accs = [m.test_accuracy for m in self.metrics_history]
        
        # Normalize data to plot height
        all_accs = train_accs + test_accs
        min_acc = min(all_accs)
        max_acc = max(all_accs)
        range_acc = max_acc - min_acc if max_acc > min_acc else 0.1
        
        lines = []
        
        # Create plot grid
        for y in range(self.height):
            line = []
            threshold = max_acc - (y / (self.height - 1)) * range_acc
            
            for i in range(len(train_accs)):
                train_val = train_accs[i]
                test_val = test_accs[i]
                
                train_match = abs(train_val - threshold) < range_acc / (self.height * 2)
                test_match = abs(test_val - threshold) < range_acc / (self.height * 2)
                
                if train_match and test_match:
                    line.append('◉')  # Both
                elif train_match:
                    line.append('●')  # Train
                elif test_match:
                    line.append('○')  # Test
                else:
                    line.append(' ')
            
            # Pad and add y-axis label
            while len(line) < self.width - 8:
                line.append(' ')
                
            y_label = f"{threshold:.1%}"
            lines.append(f"{y_label:>6}│{''.join(line[:self.width-8])}")
        
        # Add x-axis and legend
        lines.append("      └" + "─" * (self.width - 8))
        lines.append("      ● Train  ○ Test  ◉ Both")
        
        return "\n".join(lines)
    
    def plot_loss(self) -> str:
        """Generate ASCII plot of loss over time"""
        if not self.metrics_history:
            return "No loss data yet..."
        
        losses = [m.loss for m in self.metrics_history]
        min_loss = min(losses)
        max_loss = max(losses)
        range_loss = max_loss - min_loss if max_loss > min_loss else 0.1
        
        lines = []
        height = 6  # Smaller height for loss plot
        
        for y in range(height):
            line = []
            threshold = max_loss - (y / (height - 1)) * range_loss
            
            for loss_val in losses:
                if abs(loss_val - threshold) < range_loss / (height * 2):
                    line.append('▓')
                else:
                    line.append(' ')
            
            while len(line) < self.width - 8:
                line.append(' ')
                
            y_label = f"{threshold:.2f}"
            lines.append(f"{y_label:>6}│{''.join(line[:self.width-8])}")
        
        lines.append("      └" + "─" * (self.width - 8))
        lines.append("      Loss Trend")
        
        return "\n".join(lines)
    
    def plot_custom_metric(self, metric_name: str) -> str:
        """Plot any custom metric from extra_metrics"""
        if not self.metrics_history:
            return f"No {metric_name} data yet..."
        
        values = []
        for m in self.metrics_history:
            if metric_name in m.extra_metrics:
                values.append(m.extra_metrics[metric_name])
        
        if not values:
            return f"No {metric_name} data found..."
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 0.1
        
        lines = []
        height = 6
        
        for y in range(height):
            line = []
            threshold = max_val - (y / (height - 1)) * range_val
            
            for val in values:
                if abs(val - threshold) < range_val / (height * 2):
                    line.append('■')
                else:
                    line.append(' ')
            
            while len(line) < self.width - 8:
                line.append(' ')
                
            y_label = f"{threshold:.3f}"
            lines.append(f"{y_label:>6}│{''.join(line[:self.width-8])}")
        
        lines.append("      └" + "─" * (self.width - 8))
        lines.append(f"      {metric_name}")
        
        return "\n".join(lines)

class TrainingDashboard:
    """Universal training dashboard for TinyTorch examples"""
    
    def __init__(self, title: str = "TinyTorch Training", subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.plotter = ASCIIPlotter()
        self.start_time = None
        self.best_accuracy = 0.0
        self.target_accuracy = None
        self.current_epoch = 0
        self.total_epochs = 0
        
        # Initialize console
        self.console = console
        
    def show_welcome(self, model_info: Dict[str, str] = None, config: Dict[str, str] = None):
        """Show welcome screen with model and config info"""
        
        # Title banner
        if self.subtitle:
            welcome_text = Text()
            welcome_text.append(self.title, style="bold green")
            welcome_text.append("\n")
            welcome_text.append(self.subtitle, style="italic cyan")
        else:
            welcome_text = Text(self.title, style="bold green")
            
        title_panel = Panel(
            welcome_text, 
            title="🚀 TinyTorch Training Dashboard", 
            border_style="blue",
            padding=(1, 2)
        )
        
        panels = [title_panel]
        
        # Model info
        if model_info:
            model_table = Table(show_header=False, box=box.SIMPLE)
            model_table.add_column("", style="cyan")
            model_table.add_column("", style="white")
            
            for key, value in model_info.items():
                model_table.add_row(f"{key}:", str(value))
            
            model_panel = Panel(
                model_table,
                title="🏗️ Model Architecture",
                border_style="green"
            )
            panels.append(model_panel)
        
        # Config info
        if config:
            config_table = Table(show_header=False, box=box.SIMPLE)
            config_table.add_column("", style="yellow")
            config_table.add_column("", style="white")
            
            for key, value in config.items():
                config_table.add_row(f"{key}:", str(value))
            
            config_panel = Panel(
                config_table,
                title="⚙️ Training Configuration",
                border_style="yellow"
            )
            panels.append(config_panel)
        
        # Display panels
        for panel in panels:
            self.console.print(panel)
        
        self.console.print()
    
    def start_training(self, num_epochs: int, target_accuracy: Optional[float] = None):
        """Initialize training session"""
        self.total_epochs = num_epochs
        self.target_accuracy = target_accuracy
        self.start_time = time.time()
        
        target_text = f" (Target: {target_accuracy:.1%})" if target_accuracy else ""
        self.console.print(f"[bold red]🎯 Starting Training for {num_epochs} epochs{target_text}[/bold red]\n")
    
    def update_epoch(self, epoch: int, train_acc: float, test_acc: float, loss: float, 
                    extra_metrics: Dict[str, float] = None):
        """Update dashboard with new epoch results"""
        
        self.current_epoch = epoch
        
        # Track best accuracy
        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
        
        # Add to plotter
        metrics = TrainingMetrics(
            epoch=epoch,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            loss=loss,
            extra_metrics=extra_metrics or {},
            timestamp=time.time()
        )
        self.plotter.add_metrics(metrics)
        
        # Create display
        time_elapsed = time.time() - self.start_time
        self._display_current_state(train_acc, test_acc, loss, time_elapsed, extra_metrics)
        
        # Check target achievement
        if self.target_accuracy and test_acc >= self.target_accuracy:
            self.console.print(f"\n🎊 [bold green]TARGET ACHIEVED![/bold green] {self.target_accuracy:.1%}+ accuracy reached!")
    
    def _display_current_state(self, train_acc: float, test_acc: float, loss: float, 
                             time_elapsed: float, extra_metrics: Dict[str, float] = None):
        """Display current training state"""
        
        # Main stats table
        stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan", no_wrap=True)
        stats_table.add_column("Current", style="green")
        stats_table.add_column("Best", style="yellow")
        
        stats_table.add_row("Epoch", f"{self.current_epoch}/{self.total_epochs}", "—")
        stats_table.add_row("Train Accuracy", f"{train_acc:.1%}", "—")
        stats_table.add_row("Test Accuracy", f"{test_acc:.1%}", f"{self.best_accuracy:.1%}")
        stats_table.add_row("Loss", f"{loss:.3f}", "—")
        stats_table.add_row("Time Elapsed", f"{time_elapsed:.1f}s", "—")
        
        # Add extra metrics
        if extra_metrics:
            for name, value in extra_metrics.items():
                stats_table.add_row(name, f"{value:.3f}", "—")
        
        # Create panels
        stats_panel = Panel(stats_table, title="📊 Training Statistics", border_style="blue")
        acc_panel = Panel(self.plotter.plot_accuracy(), title="📈 Accuracy Progress", border_style="green")
        loss_panel = Panel(self.plotter.plot_loss(), title="📉 Loss Progress", border_style="red")
        
        # Display
        self.console.print(stats_panel)
        
        # Show plots side by side if possible
        columns = Columns([acc_panel, loss_panel], equal=True)
        self.console.print(columns)
        
        # Show custom metric plots if any
        if extra_metrics:
            for metric_name in extra_metrics.keys():
                if len(self.plotter.metrics_history) > 2:  # Need some history
                    custom_plot = self.plotter.plot_custom_metric(metric_name)
                    custom_panel = Panel(custom_plot, title=f"📊 {metric_name}", border_style="cyan")
                    self.console.print(custom_panel)
        
        self.console.print(Rule(style="dim"))
    
    def show_batch_progress(self, epoch: int, description: str = "", total_batches: int = None):
        """Show progress bar for batches within an epoch"""
        return Progress(
            TextColumn(f"[progress.description]Epoch {epoch}/{self.total_epochs}: {description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True
        )
    
    def finish_training(self, final_accuracy: float):
        """Show final training results"""
        total_time = time.time() - self.start_time
        
        self.console.print("\n" + "=" * 70, style="bold blue")
        self.console.print("🎯 TRAINING COMPLETE", style="bold green", justify="center")
        self.console.print("=" * 70, style="bold blue")
        
        # Final results table
        results_table = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_column("Status", style="yellow")
        
        results_table.add_row("Final Accuracy", f"{final_accuracy:.1%}", "")
        results_table.add_row("Best Accuracy", f"{self.best_accuracy:.1%}", "")
        results_table.add_row("Total Time", f"{total_time:.1f}s", "")
        results_table.add_row("Epochs Completed", f"{self.current_epoch}", "")
        
        # Add benchmark comparisons
        results_table.add_row("", "", "")  # Spacer
        results_table.add_row("Random Chance", "10.0%", "❌")
        results_table.add_row("Typical Baseline", "40-50%", "✅" if self.best_accuracy >= 0.40 else "📈")
        
        if self.target_accuracy:
            target_status = "🎊" if self.best_accuracy >= self.target_accuracy else "📈"
            results_table.add_row(f"Target ({self.target_accuracy:.0%})", f"{self.target_accuracy:.1%}", target_status)
        
        self.console.print(Panel(results_table, title="📊 Final Results", border_style="green"))
        
        # Success message
        if self.target_accuracy and self.best_accuracy >= self.target_accuracy:
            self.console.print("\n🏆 [bold green]TARGET ACHIEVED![/bold green]")
        elif self.best_accuracy >= 0.50:
            self.console.print("\n✅ [bold yellow]EXCELLENT PERFORMANCE![/bold yellow]")
        elif self.best_accuracy >= 0.40:
            self.console.print("\n📈 [bold blue]GOOD PERFORMANCE![/bold blue]")
        else:
            self.console.print("\n⚡ [bold cyan]TRAINING SUCCESSFUL![/bold cyan]")
        
        # Final visualization
        final_plot = Panel(
            self.plotter.plot_accuracy(), 
            title="📈 Complete Training Journey", 
            border_style="blue"
        )
        self.console.print(final_plot)
        
        return {
            'final_accuracy': final_accuracy,
            'best_accuracy': self.best_accuracy,
            'total_time': total_time,
            'epochs': self.current_epoch
        }

# Convenience functions for common use cases
def create_cifar10_dashboard():
    """Pre-configured dashboard for CIFAR-10 training"""
    return TrainingDashboard(
        title="CIFAR-10 Image Classification",
        subtitle="Real-time training with beautiful visualization"
    )

def create_xor_dashboard():
    """Pre-configured dashboard for XOR training"""
    return TrainingDashboard(
        title="XOR Neural Network",
        subtitle="Classic non-linear function learning"
    )

def create_custom_dashboard(title: str, subtitle: str = ""):
    """Create custom dashboard for any task"""
    return TrainingDashboard(title=title, subtitle=subtitle)