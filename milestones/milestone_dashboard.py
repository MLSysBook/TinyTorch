#!/usr/bin/env python3
"""
TinyTorch Milestone Dashboard System
====================================

Provides a standardized, engaging dashboard for all milestone demonstrations.
Keeps milestone files clean while providing consistent student experience.

Features:
- Live training metrics
- System resource monitoring
- Achievement tracking
- Progress visualization
- Historical comparison
"""

import os
import sys
import time
import json
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box

console = Console()


# ============================================================================
# Achievement System
# ============================================================================

class AchievementSystem:
    """Track and display achievements across milestones."""
    
    ACHIEVEMENTS = {
        # Core milestones
        "first_blood": {
            "icon": "🥇",
            "title": "First Blood",
            "description": "Completed your first successful training run"
        },
        "perceptron_master": {
            "icon": "🧠",
            "title": "Perceptron Master",
            "description": "Trained the 1957 Rosenblatt Perceptron"
        },
        "xor_slayer": {
            "icon": "⚔️",
            "title": "XOR Slayer",
            "description": "Defeated the problem that stumped AI for 17 years"
        },
        "mlp_pioneer": {
            "icon": "🏗️",
            "title": "MLP Pioneer",
            "description": "Built multi-layer networks on real data"
        },
        "conv_master": {
            "icon": "🖼️",
            "title": "Convolution Master",
            "description": "Mastered spatial feature extraction with CNNs"
        },
        "transformer_wizard": {
            "icon": "🧙",
            "title": "Transformer Wizard",
            "description": "Built GPT from scratch with YOUR code"
        },
        
        # Performance achievements
        "speed_demon": {
            "icon": "⚡",
            "title": "Speed Demon",
            "description": "Trained a model in under 30 seconds"
        },
        "perfectionist": {
            "icon": "💯",
            "title": "Perfectionist",
            "description": "Achieved 99%+ accuracy on a task"
        },
        "marathon_runner": {
            "icon": "🏃",
            "title": "Marathon Runner",
            "description": "Trained for over 1000 epochs"
        },
        
        # Meta achievements
        "historian": {
            "icon": "📚",
            "title": "Historian",
            "description": "Completed all 5 historical milestones"
        },
        "completionist": {
            "icon": "🎯",
            "title": "Completionist",
            "description": "Completed all TinyTorch milestones"
        },
    }
    
    def __init__(self, progress_file=None):
        if progress_file is None:
            self.progress_file = Path(__file__).parent / ".milestone_progress.json"
        else:
            self.progress_file = Path(progress_file)
        
        self.progress = self._load_progress()
    
    def _load_progress(self):
        """Load progress from disk."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "milestones": {},
            "achievements": [],
            "stats": {
                "total_training_time": 0,
                "total_epochs": 0,
                "best_accuracy": 0,
                "fastest_training": float('inf')
            }
        }
    
    def _save_progress(self):
        """Save progress to disk."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def unlock_achievement(self, achievement_id):
        """Unlock an achievement and display it."""
        if achievement_id not in self.progress["achievements"]:
            self.progress["achievements"].append(achievement_id)
            self._save_progress()
            
            achievement = self.ACHIEVEMENTS[achievement_id]
            console.print("\n")
            console.print(Panel.fit(
                f"[bold yellow]🏆 ACHIEVEMENT UNLOCKED![/bold yellow]\n\n"
                f"[bold]{achievement['icon']} {achievement['title']}[/bold]\n"
                f"[dim]{achievement['description']}[/dim]\n\n"
                f"[dim]Progress: {len(self.progress['achievements'])}/{len(self.ACHIEVEMENTS)} achievements[/dim]",
                title="🎉 New Achievement",
                border_style="yellow",
                box=box.DOUBLE
            ))
            console.print("\n")
    
    def record_milestone(self, milestone_name, metrics):
        """Record milestone completion with metrics."""
        self.progress["milestones"][milestone_name] = {
            "completed_at": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Update global stats
        stats = self.progress["stats"]
        if "training_time" in metrics:
            stats["total_training_time"] += metrics["training_time"]
            if metrics["training_time"] < stats["fastest_training"]:
                stats["fastest_training"] = metrics["training_time"]
        
        if "epochs" in metrics:
            stats["total_epochs"] += metrics["epochs"]
        
        if "accuracy" in metrics:
            if metrics["accuracy"] > stats["best_accuracy"]:
                stats["best_accuracy"] = metrics["accuracy"]
        
        self._save_progress()
        
        # Check for achievements
        self._check_achievements(milestone_name, metrics)
    
    def _check_achievements(self, milestone_name, metrics):
        """Check if any achievements should be unlocked."""
        # First milestone
        if len(self.progress["milestones"]) == 1:
            self.unlock_achievement("first_blood")
        
        # Milestone-specific achievements
        milestone_map = {
            "1957_perceptron": "perceptron_master",
            "1969_xor": "xor_slayer",
            "1986_mlp": "mlp_pioneer",
            "1998_cnn": "conv_master",
            "2017_transformer": "transformer_wizard",
        }
        
        for key, achievement_id in milestone_map.items():
            if key in milestone_name:
                self.unlock_achievement(achievement_id)
        
        # Performance-based achievements
        if metrics.get("training_time", float('inf')) < 30:
            self.unlock_achievement("speed_demon")
        
        if metrics.get("accuracy", 0) >= 99.0:
            self.unlock_achievement("perfectionist")
        
        if metrics.get("epochs", 0) >= 1000:
            self.unlock_achievement("marathon_runner")
        
        # Meta achievements
        milestone_count = len(self.progress["milestones"])
        if milestone_count >= 5:
            self.unlock_achievement("historian")
    
    def show_progress(self):
        """Display overall progress."""
        console.print("\n")
        console.print(Panel.fit(
            self._format_progress(),
            title="📊 Your TinyTorch Journey",
            border_style="cyan",
            box=box.ROUNDED
        ))
        console.print("\n")
    
    def _format_progress(self):
        """Format progress display."""
        milestones = self.progress["milestones"]
        achievements = self.progress["achievements"]
        stats = self.progress["stats"]
        
        lines = []
        lines.append("[bold cyan]Milestones Completed:[/bold cyan]")
        
        if milestones:
            for name, data in milestones.items():
                metrics = data["metrics"]
                acc = metrics.get("accuracy", 0)
                lines.append(f"  ✅ {name} ({acc:.1f}% accuracy)")
        else:
            lines.append("  [dim]No milestones completed yet[/dim]")
        
        lines.append("\n[bold yellow]Achievements Unlocked:[/bold yellow]")
        if achievements:
            for ach_id in achievements[:5]:  # Show first 5
                ach = self.ACHIEVEMENTS[ach_id]
                lines.append(f"  {ach['icon']} {ach['title']}")
            if len(achievements) > 5:
                lines.append(f"  [dim]... and {len(achievements) - 5} more[/dim]")
        else:
            lines.append("  [dim]Complete your first milestone to unlock achievements![/dim]")
        
        lines.append("\n[bold green]Statistics:[/bold green]")
        lines.append(f"  Total training time: {stats['total_training_time']:.1f}s")
        lines.append(f"  Total epochs: {stats['total_epochs']:,}")
        lines.append(f"  Best accuracy: {stats['best_accuracy']:.1f}%")
        
        return "\n".join(lines)


# ============================================================================
# Live Training Dashboard
# ============================================================================

class TrainingDashboard:
    """Live dashboard for training visualization."""
    
    def __init__(self, milestone_name, model_info, dataset_info):
        """
        Initialize training dashboard.
        
        Args:
            milestone_name: Name of milestone (e.g., "1957 Perceptron")
            model_info: Dict with model details (architecture, params, etc.)
            dataset_info: Dict with dataset details (samples, classes, etc.)
        """
        self.milestone_name = milestone_name
        self.model_info = model_info
        self.dataset_info = dataset_info
        
        # Training state
        self.start_time = None
        self.epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0
        self.current_accuracy = 0.0
        self.best_accuracy = 0.0
        
        # History
        self.loss_history = []
        self.accuracy_history = []
        
        # System monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Milestone events (for drama!)
        self.events = []
    
    def start_training(self, total_epochs):
        """Begin training session."""
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.epoch = 0
    
    def update(self, epoch, loss, accuracy=None):
        """Update dashboard with training metrics."""
        self.epoch = epoch
        self.current_loss = loss
        self.loss_history.append(loss)
        
        if accuracy is not None:
            self.current_accuracy = accuracy
            self.accuracy_history.append(accuracy)
            
            # Track best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                if epoch > 10:  # After some training
                    self.add_event(f"🎯 New best: {accuracy:.1f}% accuracy!")
        
        # Check for interesting events
        self._check_events(epoch, loss, accuracy)
    
    def add_event(self, message):
        """Add a milestone event."""
        timestamp = time.time() - self.start_time if self.start_time else 0
        self.events.append((timestamp, message))
    
    def _check_events(self, epoch, loss, accuracy):
        """Check for milestone events during training."""
        # First major accuracy jump
        if len(self.accuracy_history) > 10:
            recent_improvement = accuracy - self.accuracy_history[-10]
            if recent_improvement > 20:
                self.add_event(f"🔥 Breakthrough! +{recent_improvement:.1f}% in 10 epochs")
        
        # Loss milestones
        if loss < 0.1 and not any("Loss < 0.1" in e[1] for e in self.events):
            self.add_event("📉 Loss < 0.1 achieved!")
        
        # Convergence detection
        if len(self.loss_history) > 20:
            recent_losses = self.loss_history[-20:]
            variance = np.var(recent_losses)
            if variance < 0.0001 and not any("Converged" in e[1] for e in self.events):
                self.add_event("✨ Model converged!")
    
    def get_live_layout(self):
        """Generate live dashboard layout."""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split main into metrics and system
        layout["main"].split_row(
            Layout(name="metrics", ratio=2),
            Layout(name="system", ratio=1)
        )
        
        # Header: Milestone info
        layout["header"].update(Panel(
            f"[bold cyan]{self.milestone_name}[/bold cyan]",
            style="cyan"
        ))
        
        # Metrics: Training progress
        layout["metrics"].update(self._build_metrics_panel())
        
        # System: Resource monitoring
        layout["system"].update(self._build_system_panel())
        
        # Footer: Recent events
        layout["footer"].update(self._build_events_panel())
        
        return layout
    
    def _build_metrics_panel(self):
        """Build training metrics display."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress_pct = (self.epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0
        
        # Progress bar
        bar_length = 30
        filled = int(bar_length * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Build content
        lines = [
            f"[bold]Progress:[/bold] {self.epoch}/{self.total_epochs} epochs",
            f"[{bar}] {progress_pct:.1f}%",
            "",
            f"[bold]Loss:[/bold]      {self.current_loss:.4f}",
        ]
        
        if self.accuracy_history:
            lines.append(f"[bold]Accuracy:[/bold]  {self.current_accuracy:.1f}%")
            lines.append(f"[bold]Best:[/bold]      {self.best_accuracy:.1f}%")
        
        lines.append("")
        lines.append(f"[bold]Time:[/bold]      {elapsed:.1f}s")
        
        if elapsed > 0 and self.epoch > 0:
            speed = self.epoch / elapsed
            eta = (self.total_epochs - self.epoch) / speed if speed > 0 else 0
            lines.append(f"[bold]Speed:[/bold]     {speed:.1f} epochs/s")
            lines.append(f"[bold]ETA:[/bold]       {eta:.0f}s")
        
        return Panel("\n".join(lines), title="📊 Training Metrics", border_style="green")
    
    def _build_system_panel(self):
        """Build system resource monitoring."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.initial_memory
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        lines = [
            f"[bold]Memory:[/bold]",
            f"  Current: {current_memory:.1f} MB",
            f"  Delta: +{memory_delta:.1f} MB",
            "",
            f"[bold]CPU:[/bold] {cpu_percent:.1f}%",
            "",
            f"[bold]Model:[/bold]",
            f"  Params: {self.model_info.get('params', 'N/A')}",
        ]
        
        return Panel("\n".join(lines), title="💻 System", border_style="blue")
    
    def _build_events_panel(self):
        """Build recent events display."""
        if not self.events:
            return Panel("[dim]No events yet...[/dim]", title="📰 Events", border_style="yellow")
        
        # Show last 3 events
        recent_events = self.events[-3:]
        event_lines = []
        for timestamp, message in recent_events:
            event_lines.append(f"[{timestamp:.0f}s] {message}")
        
        return Panel("\n".join(event_lines), title="📰 Recent Events", border_style="yellow")
    
    def show_final_summary(self):
        """Display final training summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Build comparison table
        table = Table(title="Training Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Initial", style="yellow", width=15)
        table.add_column("Final", style="green", width=15)
        table.add_column("Change", style="magenta", width=15)
        
        # Loss comparison
        initial_loss = self.loss_history[0] if self.loss_history else 0
        final_loss = self.loss_history[-1] if self.loss_history else 0
        loss_change = initial_loss - final_loss
        
        table.add_row(
            "Loss",
            f"{initial_loss:.4f}",
            f"{final_loss:.4f}",
            f"-{loss_change:.4f}" if loss_change > 0 else f"+{abs(loss_change):.4f}"
        )
        
        # Accuracy comparison
        if self.accuracy_history:
            initial_acc = self.accuracy_history[0]
            final_acc = self.accuracy_history[-1]
            acc_change = final_acc - initial_acc
            
            table.add_row(
                "Accuracy",
                f"{initial_acc:.1f}%",
                f"{final_acc:.1f}%",
                f"+{acc_change:.1f}%" if acc_change > 0 else f"{acc_change:.1f}%"
            )
        
        # Time stats
        table.add_row(
            "Training Time",
            "0s",
            f"{elapsed:.1f}s",
            f"{elapsed:.1f}s"
        )
        
        if elapsed > 0 and self.epoch > 0:
            speed = self.epoch / elapsed
            table.add_row(
                "Speed",
                "-",
                f"{speed:.1f} epochs/s",
                "-"
            )
        
        console.print("\n")
        console.print(table)
        console.print("\n")
        
        # Show key events
        if self.events:
            console.print(Panel.fit(
                "\n".join([f"  {msg}" for _, msg in self.events]),
                title="🎯 Key Milestones During Training",
                border_style="yellow"
            ))
            console.print("\n")


# ============================================================================
# Milestone Context Manager
# ============================================================================

class MilestoneRunner:
    """
    Context manager for running milestones with standardized dashboard.
    
    Usage:
        with MilestoneRunner("1957 Perceptron", model_info, dataset_info) as runner:
            runner.start_training(epochs=100)
            
            for epoch in range(100):
                loss, acc = train_one_epoch()
                runner.update(epoch, loss, acc)
            
            runner.record_completion({"accuracy": final_acc, "epochs": 100})
    """
    
    def __init__(self, milestone_name, model_info, dataset_info, use_live=True):
        """
        Initialize milestone runner.
        
        Args:
            milestone_name: Human-readable milestone name
            model_info: Dict with model details
            dataset_info: Dict with dataset details
            use_live: Whether to use live updating dashboard
        """
        self.milestone_name = milestone_name
        self.model_info = model_info
        self.dataset_info = dataset_info
        self.use_live = use_live
        
        self.dashboard = TrainingDashboard(milestone_name, model_info, dataset_info)
        self.achievement_system = AchievementSystem()
        self.live = None
    
    def __enter__(self):
        """Start milestone context."""
        # Show initial info
        self._show_welcome()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End milestone context."""
        if self.live:
            self.live.stop()
        
        # Show final summary
        self.dashboard.show_final_summary()
        
        # Show progress
        self.achievement_system.show_progress()
        
        return False
    
    def _show_welcome(self):
        """Show welcome panel."""
        console.print("\n")
        console.print(Panel.fit(
            f"[bold cyan]{self.milestone_name}[/bold cyan]\n\n"
            f"[bold]Model:[/bold] {self.model_info.get('architecture', 'N/A')}\n"
            f"[bold]Parameters:[/bold] {self.model_info.get('params', 'N/A')}\n\n"
            f"[bold]Dataset:[/bold] {self.dataset_info.get('name', 'N/A')}\n"
            f"[bold]Samples:[/bold] {self.dataset_info.get('samples', 'N/A')}\n\n"
            f"[dim]Using YOUR TinyTorch implementation![/dim]",
            title="🎯 Starting Milestone",
            border_style="cyan",
            box=box.DOUBLE
        ))
        console.print("\n")
    
    def start_training(self, total_epochs, use_live=None):
        """Start training session."""
        if use_live is None:
            use_live = self.use_live
        
        self.dashboard.start_training(total_epochs)
        
        if use_live:
            self.live = Live(self.dashboard.get_live_layout(), refresh_per_second=2, console=console)
            self.live.start()
    
    def update(self, epoch, loss, accuracy=None):
        """Update training metrics."""
        self.dashboard.update(epoch, loss, accuracy)
        
        if self.live:
            self.live.update(self.dashboard.get_live_layout())
    
    def add_event(self, message):
        """Add a training event."""
        self.dashboard.add_event(message)
    
    def record_completion(self, metrics):
        """Record milestone completion."""
        # Generate milestone ID from name
        milestone_id = self.milestone_name.lower().replace(" ", "_")
        
        # Add training time
        if self.dashboard.start_time:
            elapsed = time.time() - self.dashboard.start_time
            metrics["training_time"] = elapsed
        
        # Record in achievement system
        self.achievement_system.record_milestone(milestone_id, metrics)


# ============================================================================
# Utility Functions
# ============================================================================

# Note: PyTorch comparison removed - we don't want that dependency!


# ============================================================================
# Main (for testing)
# ============================================================================

def main():
    """Test the dashboard system."""
    console.print("[bold cyan]Testing TinyTorch Milestone Dashboard[/bold cyan]\n")
    
    # Test achievement system
    achievements = AchievementSystem()
    achievements.unlock_achievement("first_blood")
    achievements.show_progress()
    
    # Test training dashboard
    model_info = {
        "architecture": "Perceptron (Linear + Sigmoid)",
        "params": "2,418"
    }
    
    dataset_info = {
        "name": "Synthetic Linearly Separable",
        "samples": "1,000"
    }
    
    with MilestoneRunner("1957 Perceptron", model_info, dataset_info) as runner:
        runner.start_training(total_epochs=50, use_live=False)
        
        # Simulate training
        for epoch in range(50):
            loss = 0.7 * np.exp(-epoch / 20) + 0.1 + np.random.normal(0, 0.01)
            accuracy = 50 + 45 * (1 - np.exp(-epoch / 15)) + np.random.normal(0, 1)
            
            runner.update(epoch, loss, accuracy)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                console.print(f"Epoch {epoch+1}/50  Loss: {loss:.4f}  Accuracy: {accuracy:.1f}%")
            
            time.sleep(0.1)  # Simulate training time
        
        # Record completion
        runner.record_completion({
            "accuracy": 95.2,
            "epochs": 50,
            "loss": loss
        })


if __name__ == "__main__":
    main()



