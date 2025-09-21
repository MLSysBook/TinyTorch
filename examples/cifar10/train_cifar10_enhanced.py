#!/usr/bin/env python3
"""
TinyTorch CIFAR-10 Enhanced Training with Rich UI and Real-time Plotting

This script demonstrates TinyTorch's capability with beautiful Rich UI,
real-time ASCII plotting, and extended training for higher accuracy.

Features:
- Rich console with progress bars and live tables
- Real-time ASCII plots of training progress  
- Extended training for 55%+ accuracy
- Beautiful formatted output

Performance Target: 55%+ accuracy with engaging visual feedback
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import Variable
from tinytorch.core.layers import Dense
from tinytorch.core.activations import ReLU
from tinytorch.core.training import CrossEntropyLoss
from tinytorch.core.optimizers import Adam
from tinytorch.core.dataloader import DataLoader, CIFAR10Dataset

# Rich imports for beautiful UI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich import box
import threading
import queue

console = Console()

class ASCIIPlotter:
    """Real-time ASCII plotting for training metrics"""
    
    def __init__(self, width=60, height=12):
        self.width = width
        self.height = height
        self.train_acc_history = []
        self.test_acc_history = []
        self.loss_history = []
        
    def add_data(self, train_acc, test_acc, loss):
        """Add new data point"""
        self.train_acc_history.append(train_acc)
        self.test_acc_history.append(test_acc)
        self.loss_history.append(loss)
        
        # Keep only recent history for plotting
        max_points = self.width - 10
        if len(self.train_acc_history) > max_points:
            self.train_acc_history = self.train_acc_history[-max_points:]
            self.test_acc_history = self.test_acc_history[-max_points:]
            self.loss_history = self.loss_history[-max_points:]
    
    def plot_accuracy(self):
        """Generate ASCII plot of accuracy over time"""
        if not self.train_acc_history:
            return "No data yet..."
        
        # Normalize data to plot height
        all_acc = self.train_acc_history + self.test_acc_history
        min_acc = min(all_acc)
        max_acc = max(all_acc)
        range_acc = max_acc - min_acc if max_acc > min_acc else 1
        
        lines = []
        
        # Create plot grid
        for y in range(self.height):
            line = []
            threshold = max_acc - (y / (self.height - 1)) * range_acc
            
            for x in range(len(self.train_acc_history)):
                train_val = self.train_acc_history[x]
                test_val = self.test_acc_history[x] if x < len(self.test_acc_history) else 0
                
                if abs(train_val - threshold) < range_acc / (self.height * 2):
                    line.append('●')  # Train accuracy
                elif abs(test_val - threshold) < range_acc / (self.height * 2):
                    line.append('○')  # Test accuracy  
                else:
                    line.append(' ')
            
            # Pad line to full width
            while len(line) < self.width - 10:
                line.append(' ')
            
            # Add y-axis label
            y_label = f"{threshold:.1%}"
            lines.append(f"{y_label:>6}│{''.join(line[:self.width-10])}")
        
        # Add x-axis
        x_axis = "      └" + "─" * (self.width - 10)
        lines.append(x_axis)
        
        # Add legend
        legend = "      ● Train  ○ Test"
        lines.append(legend)
        
        return "\n".join(lines)
    
    def plot_loss(self):
        """Generate ASCII plot of loss over time"""
        if not self.loss_history:
            return "No loss data yet..."
        
        # Normalize loss data
        min_loss = min(self.loss_history)
        max_loss = max(self.loss_history)
        range_loss = max_loss - min_loss if max_loss > min_loss else 1
        
        lines = []
        
        for y in range(8):  # Smaller height for loss
            line = []
            threshold = max_loss - (y / 7) * range_loss
            
            for x in range(len(self.loss_history)):
                loss_val = self.loss_history[x]
                
                if abs(loss_val - threshold) < range_loss / 16:
                    line.append('▓')
                else:
                    line.append(' ')
            
            # Pad and add label
            while len(line) < self.width - 10:
                line.append(' ')
                
            y_label = f"{threshold:.2f}"
            lines.append(f"{y_label:>6}│{''.join(line[:self.width-10])}")
        
        # Add x-axis  
        lines.append("      └" + "─" * (self.width - 10))
        lines.append("      Loss over time")
        
        return "\n".join(lines)

class EnhancedCIFAR10_MLP:
    """Enhanced MLP with better architecture for higher accuracy"""
    
    def __init__(self):
        # Larger architecture for better accuracy
        self.fc1 = Dense(3072, 1024)  # Bigger first layer
        self.fc2 = Dense(1024, 512)
        self.fc3 = Dense(512, 256)
        self.fc4 = Dense(256, 10)
        
        self.relu = ReLU()
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        self._initialize_weights()
        
        total_params = sum(np.prod(layer.weights.shape) + np.prod(layer.bias.shape) 
                          for layer in self.layers)
        
        console.print(f"[bold green]✅ Model Architecture:[/bold green] 3072 → 1024 → 512 → 256 → 10")
        console.print(f"[bold blue]📊 Parameters:[/bold blue] {total_params:,}")
    
    def _initialize_weights(self):
        """Improved initialization"""
        for i, layer in enumerate(self.layers):
            fan_in = layer.weights.shape[0]
            
            if i == len(self.layers) - 1:  # Output layer
                std = 0.01
            else:  # Hidden layers
                std = np.sqrt(2.0 / fan_in) * 0.6  # Slightly more aggressive
            
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        """Forward pass"""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        logits = self.fc4(h3)
        return logits
    
    def parameters(self):
        """Get all parameters"""
        params = []
        for layer in self.layers:
            params.extend([layer.weights, layer.bias])
        return params

def preprocess_images_enhanced(images, training=True):
    """Enhanced preprocessing with better augmentation"""
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    
    if training:
        # Enhanced augmentation
        augmented = np.copy(images_np)
        for i in range(batch_size):
            # Horizontal flip
            if np.random.random() > 0.5:
                augmented[i] = np.flip(augmented[i], axis=2)
            
            # Brightness
            brightness = np.random.uniform(0.85, 1.15)
            augmented[i] = np.clip(augmented[i] * brightness, 0, 1)
            
            # Small rotation (approximate with shifts)
            if np.random.random() > 0.7:
                shift_x = np.random.randint(-2, 3)
                shift_y = np.random.randint(-2, 3)
                augmented[i] = np.roll(augmented[i], shift_x, axis=2)
                augmented[i] = np.roll(augmented[i], shift_y, axis=1)
        
        images_np = augmented
    
    # Improved normalization
    flat = images_np.reshape(batch_size, -1)
    normalized = (flat - 0.485) / 0.229  # Better normalization
    
    return Tensor(normalized.astype(np.float32))

def evaluate_model_enhanced(model, dataloader, max_batches=100):
    """Enhanced evaluation with more thorough testing"""
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        x = Variable(preprocess_images_enhanced(images, training=False), requires_grad=False)
        logits = model.forward(x)
        
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        predictions = np.argmax(logits_np, axis=1)
        
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
        
        # Per-class accuracy
        for i in range(len(labels_np)):
            label = labels_np[i]
            class_total[label] += 1
            if predictions[i] == label:
                class_correct[label] += 1
    
    accuracy = correct / total if total > 0 else 0
    class_accuracies = class_correct / np.maximum(class_total, 1)
    
    return accuracy, class_accuracies

def create_training_display(plotter, epoch, total_epochs, train_acc, test_acc, best_acc, current_loss, time_elapsed):
    """Create rich display layout"""
    
    # Main stats table
    stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Current", style="green")
    stats_table.add_column("Best", style="yellow")
    
    stats_table.add_row("Epoch", f"{epoch}/{total_epochs}", f"—")
    stats_table.add_row("Train Accuracy", f"{train_acc:.1%}", f"—")
    stats_table.add_row("Test Accuracy", f"{test_acc:.1%}", f"{best_acc:.1%}")
    stats_table.add_row("Loss", f"{current_loss:.3f}", f"—")
    stats_table.add_row("Time Elapsed", f"{time_elapsed:.1f}s", f"—")
    
    # Accuracy plot
    acc_plot = plotter.plot_accuracy()
    
    # Loss plot
    loss_plot = plotter.plot_loss()
    
    # Create panels
    stats_panel = Panel(stats_table, title="📊 Training Statistics", border_style="blue")
    acc_panel = Panel(acc_plot, title="📈 Accuracy Progress", border_style="green")
    loss_panel = Panel(loss_plot, title="📉 Loss Progress", border_style="red")
    
    return stats_panel, acc_panel, loss_panel

def main():
    """Enhanced main training loop with Rich UI"""
    
    # Rich welcome
    console.print("\n" + "=" * 70, style="bold blue")
    console.print("🚀 TinyTorch CIFAR-10 Enhanced Training", style="bold green", justify="center")
    console.print("Real-time plots • Rich UI • Higher accuracy target", style="italic", justify="center")
    console.print("=" * 70 + "\n", style="bold blue")
    
    # Initialize plotter
    plotter = ASCIIPlotter()
    
    # Load dataset with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading CIFAR-10 dataset...", total=None)
        
        train_dataset = CIFAR10Dataset(train=True, root='data')
        test_dataset = CIFAR10Dataset(train=False, root='data')
        
        progress.update(task, description="Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Larger batch
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        progress.update(task, description="✅ Dataset loaded!")
    
    console.print(f"[bold green]✅ Dataset:[/bold green] {len(train_dataset):,} train + {len(test_dataset):,} test samples")
    
    # Create model
    console.print("\n[bold yellow]🏗️ Building Enhanced Model...[/bold yellow]")
    model = EnhancedCIFAR10_MLP()
    
    # Setup training
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=0.002)  # Higher learning rate
    
    console.print(f"\n[bold cyan]⚙️ Training Configuration:[/bold cyan]")
    console.print(f"• Optimizer: Adam (LR: {optimizer.learning_rate})")
    console.print(f"• Batch size: 64")
    console.print(f"• Batches per epoch: 300")
    console.print(f"• Target accuracy: 55%+")
    
    # Training parameters
    num_epochs = 20  # More epochs for higher accuracy
    best_test_accuracy = 0
    batches_per_epoch = 300
    
    console.print(f"\n[bold red]🎯 Starting Training (Target: 55%+ accuracy)[/bold red]\n")
    
    # Training loop with live display
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase with progress bar
        train_losses = []
        train_correct = 0
        train_total = 0
        
        with Progress(
            TextColumn("[progress.description]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            
            train_task = progress.add_task(f"Epoch {epoch+1}/{num_epochs}", total=batches_per_epoch)
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= batches_per_epoch:
                    break
                
                # Training step
                x = Variable(preprocess_images_enhanced(images, training=True), requires_grad=False)
                y_true = Variable(labels, requires_grad=False)
                
                logits = model.forward(x)
                loss = loss_fn(logits, y_true)
                
                # Track metrics
                loss_val = float(loss.data.data) if hasattr(loss.data, 'data') else float(loss.data._data)
                train_losses.append(loss_val)
                
                logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
                preds = np.argmax(logits_np, axis=1)
                labels_np = y_true.data._data if hasattr(y_true.data, '_data') else y_true.data
                train_correct += np.sum(preds == labels_np)
                train_total += len(labels_np)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                progress.update(train_task, advance=1, description=f"Epoch {epoch+1}/{num_epochs} (Loss: {loss_val:.3f})")
        
        # Evaluation
        train_accuracy = train_correct / train_total
        test_accuracy, class_accuracies = evaluate_model_enhanced(model, test_loader, max_batches=80)
        
        # Update best accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        
        # Add to plotter
        avg_loss = np.mean(train_losses)
        plotter.add_data(train_accuracy, test_accuracy, avg_loss)
        
        # Create display
        time_elapsed = time.time() - start_time
        stats_panel, acc_panel, loss_panel = create_training_display(
            plotter, epoch+1, num_epochs, train_accuracy, test_accuracy, 
            best_test_accuracy, avg_loss, time_elapsed
        )
        
        # Print results
        console.print(stats_panel)
        console.print(acc_panel)
        console.print(loss_panel)
        
        # Success check
        if test_accuracy > 0.55:
            console.print("\n🎊 [bold green]TARGET ACHIEVED![/bold green] 55%+ accuracy reached!")
        
        # Learning rate schedule
        if epoch == 10:
            optimizer.learning_rate *= 0.5
            console.print(f"[yellow]📉 Learning rate reduced to {optimizer.learning_rate:.4f}[/yellow]")
        
        console.print(Rule(style="dim"))
    
    # Final results
    total_time = time.time() - start_time
    
    console.print("\n" + "=" * 70, style="bold blue")
    console.print("🎯 FINAL RESULTS", style="bold green", justify="center")
    console.print("=" * 70, style="bold blue")
    
    # Final evaluation
    final_accuracy, final_class_acc = evaluate_model_enhanced(model, test_loader, max_batches=None)
    
    # Results table
    results_table = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_column("Comparison", style="yellow")
    
    results_table.add_row("Final Accuracy", f"{final_accuracy:.1%}", "")
    results_table.add_row("Best Accuracy", f"{best_test_accuracy:.1%}", "")
    results_table.add_row("Training Time", f"{total_time:.1f} seconds", "")
    results_table.add_row("Random Chance", "10.0%", "❌")
    results_table.add_row("CS231n Baseline", "50-55%", "✅" if best_test_accuracy >= 0.50 else "📈")
    results_table.add_row("Target (55%)", "55.0%", "🎊" if best_test_accuracy >= 0.55 else "📈")
    
    console.print(Panel(results_table, title="📊 Performance Summary", border_style="green"))
    
    # Success assessment
    if best_test_accuracy >= 0.55:
        console.print("\n🏆 [bold green]OUTSTANDING SUCCESS![/bold green]")
        console.print("🎉 TinyTorch achieves excellent performance on real dataset!")
    elif best_test_accuracy >= 0.50:
        console.print("\n✅ [bold yellow]STRONG PERFORMANCE![/bold yellow]")
        console.print("🎯 TinyTorch matches professional ML course benchmarks!")
    else:
        console.print("\n📈 [bold blue]GOOD PROGRESS![/bold blue]")
        console.print("⚡ TinyTorch demonstrates working ML system!")
    
    # Final plot
    console.print(Panel(plotter.plot_accuracy(), title="📈 Final Training Progress", border_style="blue"))
    
    console.print(f"\n💡 [bold cyan]Key Achievements:[/bold cyan]")
    console.print(f"   • Built complete neural network from scratch")
    console.print(f"   • Achieved {best_test_accuracy:.1%} on real image classification")
    console.print(f"   • Trained in {total_time:.1f} seconds with beautiful UI")
    console.print(f"   • Proved TinyTorch enables real ML development")

if __name__ == "__main__":
    main()