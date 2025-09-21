#!/usr/bin/env python3
"""
CIFAR-10 Training with TinyTorch

Train a neural network on CIFAR-10 images with Rich progress display.
Achieves ~55% accuracy, demonstrating real learning on complex data.

Expected performance:
- Random baseline: ~10% (untrained network)
- Trained network: 53-55% (5.5× improvement)
- Training time: ~2 minutes
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

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()

class CIFAR10_MLP:
    """Multi-layer perceptron for CIFAR-10 classification"""
    
    def __init__(self):
        # Architecture: 4 hidden layers tapering down
        self.fc1 = Dense(3072, 1024)  # Input: 32x32x3 = 3072
        self.fc2 = Dense(1024, 512)
        self.fc3 = Dense(512, 256)
        self.fc4 = Dense(256, 10)     # Output: 10 classes
        
        self.relu = ReLU()
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate total parameters
        self.total_params = sum(np.prod(layer.weights.shape) + np.prod(layer.bias.shape) 
                               for layer in self.layers)
    
    def _initialize_weights(self):
        """Initialize weights with scaled random values"""
        for i, layer in enumerate(self.layers):
            fan_in = layer.weights.shape[0]
            
            if i == len(self.layers) - 1:  # Output layer
                std = 0.01
            else:  # Hidden layers
                std = np.sqrt(2.0 / fan_in) * 0.6
            
            layer.weights._data = np.random.randn(*layer.weights.shape).astype(np.float32) * std
            layer.bias._data = np.zeros(layer.bias.shape, dtype=np.float32)
            
            layer.weights = Variable(layer.weights.data, requires_grad=True)
            layer.bias = Variable(layer.bias.data, requires_grad=True)
    
    def forward(self, x):
        """Forward pass through the network"""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return self.fc4(h3)
    
    def parameters(self):
        """Get all trainable parameters"""
        params = []
        for layer in self.layers:
            params.extend([layer.weights, layer.bias])
        return params

def preprocess_images(images, training=True):
    """Preprocess images with optional augmentation"""
    batch_size = images.shape[0]
    images_np = images.data if hasattr(images, 'data') else images._data
    
    if training:
        # Simple augmentation: horizontal flip
        augmented = np.copy(images_np)
        for i in range(batch_size):
            if np.random.random() > 0.5:
                augmented[i] = np.flip(augmented[i], axis=2)
        images_np = augmented
    
    # Flatten and normalize
    flat = images_np.reshape(batch_size, -1)
    normalized = (flat - 0.485) / 0.229
    
    return Tensor(normalized.astype(np.float32))

def evaluate_model(model, dataloader, max_batches=60):
    """Evaluate model accuracy on test set"""
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        x = Variable(preprocess_images(images, training=False), requires_grad=False)
        logits = model.forward(x)
        
        logits_np = logits.data._data if hasattr(logits.data, '_data') else logits.data
        predictions = np.argmax(logits_np, axis=1)
        labels_np = labels.data if hasattr(labels, 'data') else labels._data
        
        correct += np.sum(predictions == labels_np)
        total += len(labels_np)
    
    return correct / total if total > 0 else 0

def create_stats_table(epoch, total_epochs, train_acc, test_acc, loss, lr, best_acc):
    """Create a rich table showing current training stats"""
    table = Table(title="Training Statistics", box=box.ROUNDED)
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Epoch", f"{epoch}/{total_epochs}")
    table.add_row("Train Accuracy", f"{train_acc:.1%}")
    table.add_row("Test Accuracy", f"{test_acc:.1%}")
    table.add_row("Best Test Accuracy", f"{best_acc:.1%}")
    table.add_row("Loss", f"{loss:.3f}")
    table.add_row("Learning Rate", f"{lr:.4f}")
    
    return table

def main():
    """Main training loop"""
    
    # Show header
    console.print(Panel.fit(
        "[bold green]CIFAR-10 Image Classification[/bold green]\n"
        "Training neural network on 32×32 color images",
        title="🔥 TinyTorch",
        border_style="blue"
    ))
    
    # Load dataset
    console.print("\n[yellow]Loading CIFAR-10 dataset...[/yellow]")
    train_dataset = CIFAR10Dataset(train=True, root='data')
    test_dataset = CIFAR10Dataset(train=False, root='data')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = CIFAR10_MLP()
    
    # Show model info
    info_table = Table(title="Model Configuration", box=box.SIMPLE)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Architecture", "3072 → 1024 → 512 → 256 → 10")
    info_table.add_row("Parameters", f"{model.total_params:,}")
    info_table.add_row("Optimizer", "Adam")
    info_table.add_row("Learning Rate", "0.002 (with decay)")
    info_table.add_row("Batch Size", "64")
    info_table.add_row("Target", "55%+ accuracy")
    
    console.print(info_table)
    console.print()
    
    # Setup training
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=0.002)
    
    # Training parameters
    num_epochs = 15
    batches_per_epoch = 250
    best_accuracy = 0.0
    
    console.print(f"[bold cyan]Training for {num_epochs} epochs...[/bold cyan]\n")
    
    # Training loop with live display
    with Live(refresh_per_second=4) as live:
        
        for epoch in range(num_epochs):
            # Learning rate decay
            if epoch == 8:
                optimizer.learning_rate *= 0.5
            elif epoch == 12:
                optimizer.learning_rate *= 0.5
            
            # Training phase
            train_losses = []
            train_correct = 0
            train_total = 0
            
            # Progress bar for batches
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                
                train_task = progress.add_task(
                    f"[cyan]Epoch {epoch+1}/{num_epochs}[/cyan]", 
                    total=batches_per_epoch
                )
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    if batch_idx >= batches_per_epoch:
                        break
                    
                    # Training step
                    x = Variable(preprocess_images(images, training=True), requires_grad=False)
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
                    progress.update(train_task, advance=1)
            
            # Evaluation
            train_accuracy = train_correct / train_total
            test_accuracy = evaluate_model(model, test_loader, max_batches=60)
            avg_loss = np.mean(train_losses)
            
            # Track best
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
            
            # Update live display
            stats_table = create_stats_table(
                epoch + 1, num_epochs,
                train_accuracy, test_accuracy,
                avg_loss, optimizer.learning_rate,
                best_accuracy
            )
            live.update(stats_table)
            
            # Check if target reached
            if test_accuracy >= 0.55:
                console.print(f"\n[bold green]🎉 Target accuracy reached: {test_accuracy:.1%}![/bold green]")
                break
    
    # Final evaluation
    console.print("\n[yellow]Running final evaluation on full test set...[/yellow]")
    final_accuracy = evaluate_model(model, test_loader, max_batches=None)
    
    # Show results
    console.print("\n" + "="*50)
    
    results_table = Table(title="🏆 Final Results", box=box.DOUBLE)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_column("Notes", style="yellow")
    
    results_table.add_row("Final Accuracy", f"{final_accuracy:.1%}", "")
    results_table.add_row("Best Accuracy", f"{best_accuracy:.1%}", "")
    results_table.add_row("Random Baseline", "10.0%", "Untrained network")
    results_table.add_row("Improvement", f"{final_accuracy/0.10:.1f}×", "vs random baseline")
    
    console.print(results_table)
    
    # Success message
    if final_accuracy >= 0.55:
        console.print("\n[bold green]✅ SUCCESS: Achieved 55%+ accuracy![/bold green]")
        console.print("[green]The network successfully learned complex patterns from images.[/green]")
    elif final_accuracy >= 0.50:
        console.print("\n[bold yellow]📈 Good Performance: >50% accuracy[/bold yellow]")
        console.print("[yellow]Solid learning demonstrated, 5× better than random.[/yellow]")
    else:
        console.print("\n[bold cyan]🔧 Training Complete[/bold cyan]")
        console.print(f"[cyan]Achieved {final_accuracy:.1%}, which is {final_accuracy/0.10:.1f}× better than random.[/cyan]")
    
    return final_accuracy

if __name__ == "__main__":
    main()