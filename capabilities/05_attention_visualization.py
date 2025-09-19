#!/usr/bin/env python3
"""
🚀 CAPABILITY SHOWCASE: Attention Visualization
After Module 07 (Attention)

"Look what you built!" - Your attention mechanism focuses on important parts!
"""

import sys
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.align import Align

# Import from YOUR TinyTorch implementation
try:
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.attention import MultiHeadAttention, ScaledDotProductAttention
except ImportError:
    print("❌ TinyTorch attention layers not found. Make sure you've completed Module 07 (Attention)!")
    sys.exit(1)

console = Console()

def create_sample_sentence():
    """Create a sample sentence with clear attention patterns."""
    # Simple sentence: "The cat sat on the mat"
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Create simple embeddings (6 tokens × 4 dimensions)
    # In reality, these would come from word embeddings
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],  # The
        [0.8, 0.1, 0.9, 0.2],  # cat (subject)
        [0.3, 0.7, 0.1, 0.8],  # sat (verb)
        [0.2, 0.3, 0.4, 0.1],  # on (preposition)
        [0.1, 0.2, 0.3, 0.4],  # the (same as first "the")
        [0.6, 0.4, 0.7, 0.3],  # mat (object)
    ]
    
    return tokens, embeddings

def visualize_attention_heatmap(attention_weights, tokens, title):
    """Create ASCII heatmap of attention weights."""
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    
    # Create attention table
    table = Table(title="Attention Heatmap (Each row shows what that token attends to)")
    table.add_column("Token", style="white", width=8)
    
    # Add columns for each token
    for token in tokens:
        table.add_column(token, style="yellow", width=6)
    
    # Add rows with attention weights
    for i, (token, weights) in enumerate(zip(tokens, attention_weights)):
        row = [f"[bold]{token}[/bold]"]
        
        for weight in weights:
            # Convert weight to visual representation
            intensity = int(weight * 5)  # Scale to 0-5
            chars = " ░▒▓█"
            visual = chars[min(intensity, 4)]
            row.append(f"{weight:.2f}{visual}")
        
        table.add_row(*row)
    
    console.print(table)

def demonstrate_self_attention():
    """Show self-attention mechanism."""
    console.print(Panel.fit("🎯 SELF-ATTENTION MECHANISM", style="bold green"))
    
    tokens, embeddings = create_sample_sentence()
    
    console.print("📝 Sample sentence: \"The cat sat on the mat\"")
    console.print("🎯 Let's see which words pay attention to which other words!")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating attention layer...", total=None)
        time.sleep(1)
        
        # Create attention layer with YOUR implementation
        d_model = 4  # Embedding dimension
        attention = ScaledDotProductAttention(d_model)
        
        progress.update(task, description="✅ Attention layer ready!")
        time.sleep(0.5)
    
    # Convert to tensor
    input_tensor = Tensor([embeddings])  # Shape: (1, seq_len, d_model)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing attention weights...", total=None)
        time.sleep(1)
        
        # Compute attention with YOUR implementation
        output, attention_weights = attention.forward(input_tensor, input_tensor, input_tensor)
        
        # Extract attention weights (shape: seq_len × seq_len)
        attn_matrix = np.array(attention_weights.data[0])
        
        progress.update(task, description="✅ Attention computed!")
        time.sleep(0.5)
    
    visualize_attention_heatmap(attn_matrix, tokens, "Self-Attention Weights")
    
    console.print("\n💡 [bold]Key Observations:[/bold]")
    console.print("   🎯 'cat' and 'sat' might attend to each other (subject-verb)")
    console.print("   🎯 'sat' and 'mat' might connect (verb-object relationship)")  
    console.print("   🎯 'the' tokens might have similar attention patterns")
    console.print("   🎯 Each word considers ALL other words when deciding meaning!")

def demonstrate_multi_head_attention():
    """Show multi-head attention mechanism."""
    console.print(Panel.fit("🧠 MULTI-HEAD ATTENTION", style="bold blue"))
    
    console.print("🔍 Why multiple attention heads?")
    console.print("   💡 Different heads can focus on different relationships:")
    console.print("      • Head 1: Syntactic relationships (noun-verb)")
    console.print("      • Head 2: Semantic relationships (related concepts)")
    console.print("      • Head 3: Positional relationships (nearby words)")
    console.print("      • Head 4: Long-range dependencies")
    console.print()
    
    tokens, embeddings = create_sample_sentence()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating multi-head attention...", total=None)
        time.sleep(1)
        
        # Create multi-head attention with YOUR implementation
        d_model = 4
        num_heads = 2  # Keep it simple for visualization
        mha = MultiHeadAttention(d_model, num_heads)
        
        progress.update(task, description="✅ Multi-head attention ready!")
        time.sleep(0.5)
    
    input_tensor = Tensor([embeddings])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing multi-head attention...", total=None)
        time.sleep(1)
        
        # Compute multi-head attention
        output = mha.forward(input_tensor, input_tensor, input_tensor)
        
        progress.update(task, description="✅ Multi-head computation complete!")
        time.sleep(0.5)
    
    console.print("🎯 [bold]Multi-Head Output:[/bold]")
    console.print(f"   Input shape:  {input_tensor.shape}")
    console.print(f"   Output shape: {output.shape}")
    console.print(f"   Number of heads: {num_heads}")
    console.print()
    console.print("🔄 What happened internally:")
    console.print("   1️⃣ Split into multiple attention heads")
    console.print("   2️⃣ Each head computed its own attention pattern")
    console.print("   3️⃣ Heads were concatenated and projected")
    console.print("   4️⃣ Result captures multiple types of relationships!")

def demonstrate_sequence_modeling():
    """Show how attention enables sequence modeling."""
    console.print(Panel.fit("📚 SEQUENCE MODELING POWER", style="bold yellow"))
    
    console.print("🔍 Translation example: \"Hello world\" → \"Hola mundo\"")
    console.print()
    
    # Simulate translation attention pattern
    english_tokens = ["Hello", "world"]
    spanish_tokens = ["Hola", "mundo"]
    
    # Simulated cross-attention weights (Spanish attending to English)
    # In real translation, Spanish words attend to relevant English words
    cross_attention = [
        [0.9, 0.1],  # "Hola" attends mostly to "Hello"
        [0.2, 0.8],  # "mundo" attends mostly to "world"
    ]
    
    table = Table(title="Cross-Attention in Translation")
    table.add_column("Spanish", style="cyan")
    table.add_column("→ Hello", style="yellow")
    table.add_column("→ world", style="yellow")
    table.add_column("Meaning", style="green")
    
    for i, (spanish, weights) in enumerate(zip(spanish_tokens, cross_attention)):
        visual_weights = []
        for w in weights:
            intensity = int(w * 5)
            chars = " ░▒▓█"
            visual_weights.append(f"{w:.1f}{chars[min(intensity, 4)]}")
        
        meaning = "Direct match!" if weights[i] > 0.5 else "Cross-reference"
        table.add_row(spanish, visual_weights[0], visual_weights[1], meaning)
    
    console.print(table)
    
    console.print("\n💡 [bold]Attention enables:[/bold]")
    console.print("   🌍 Machine Translation (Google Translate)")
    console.print("   📝 Text Summarization (GPT, BERT)")
    console.print("   🗣️ Speech Recognition (Whisper)")
    console.print("   💬 Conversational AI (ChatGPT)")

def show_transformer_architecture():
    """Show how attention fits into the transformer."""
    console.print(Panel.fit("🏗️ TRANSFORMER ARCHITECTURE", style="bold magenta"))
    
    console.print("🧠 Your attention is the heart of the Transformer:")
    console.print()
    console.print("   📥 Input Embeddings")
    console.print("   ↓")
    console.print("   📊 Positional Encoding")
    console.print("   ↓")
    console.print("   🎯 [bold cyan]Multi-Head Attention[/bold cyan]  ← YOUR CODE!")
    console.print("   ↓")
    console.print("   🔄 Add & Norm")
    console.print("   ↓")
    console.print("   🧮 Feed Forward Network")
    console.print("   ↓")
    console.print("   🔄 Add & Norm")
    console.print("   ↓")
    console.print("   📤 Output")
    console.print()
    console.print("🎯 [bold]Transformer Applications:[/bold]")
    console.print("   • GPT family (text generation)")
    console.print("   • BERT (text understanding)")
    console.print("   • T5 (text-to-text)")
    console.print("   • Vision Transformer (images)")
    console.print("   • DALL-E (text-to-image)")

def show_computational_complexity():
    """Show the computational trade-offs of attention."""
    console.print(Panel.fit("⚡ COMPUTATIONAL COMPLEXITY", style="bold red"))
    
    console.print("🧮 Attention Complexity Analysis:")
    console.print()
    
    # Create complexity comparison table
    table = Table(title="Sequence Modeling Approaches")
    table.add_column("Method", style="cyan")
    table.add_column("Time Complexity", style="yellow")
    table.add_column("Parallelizable?", style="green")
    table.add_column("Long Dependencies?", style="magenta")
    
    table.add_row("RNN/LSTM", "O(n)", "❌ Sequential", "❌ Vanishing gradient")
    table.add_row("CNN", "O(n log n)", "✅ Parallel", "❌ Limited receptive field")
    table.add_row("[bold]Attention[/bold]", "[bold]O(n²)[/bold]", "✅ Parallel", "✅ Direct connections")
    
    console.print(table)
    
    console.print("\n💡 [bold]Trade-offs:[/bold]")
    console.print("   ✅ Perfect parallelization → faster training")
    console.print("   ✅ Direct long-range connections → better understanding")
    console.print("   ⚠️ Quadratic memory → challenging for very long sequences")
    console.print("   🚀 Solutions: Sparse attention, linear attention, hierarchical methods")
    
    console.print("\n🎯 [bold]Production Optimizations:[/bold]")
    console.print("   • Flash Attention: Memory-efficient computation")
    console.print("   • Gradient checkpointing: Trade compute for memory")
    console.print("   • Mixed precision: FP16/BF16 for speed")
    console.print("   • Model parallelism: Split across multiple GPUs")

def main():
    """Main showcase function."""
    console.clear()
    
    # Header
    header = Panel.fit(
        "[bold cyan]🚀 CAPABILITY SHOWCASE: ATTENTION VISUALIZATION[/bold cyan]\n"
        "[yellow]After Module 07 (Attention)[/yellow]\n\n"
        "[green]\"Look what you built!\" - Your attention mechanism focuses on important parts![/green]",
        border_style="bright_blue"
    )
    console.print(Align.center(header))
    console.print()
    
    try:
        demonstrate_self_attention()
        console.print("\n" + "="*60)
        
        demonstrate_multi_head_attention()
        console.print("\n" + "="*60)
        
        demonstrate_sequence_modeling()
        console.print("\n" + "="*60)
        
        show_transformer_architecture()
        console.print("\n" + "="*60)
        
        show_computational_complexity()
        
        # Celebration
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]🎉 ATTENTION MECHANISM MASTERY! 🎉[/bold green]\n\n"
            "[cyan]You've implemented the CORE innovation that revolutionized AI![/cyan]\n\n"
            "[white]Your attention mechanism powers:[/white]\n"
            "[white]• GPT and ChatGPT (language generation)[/white]\n"
            "[white]• Google Translate (language translation)[/white]\n"
            "[white]• DALL-E (image generation)[/white]\n"
            "[white]• GitHub Copilot (code generation)[/white]\n\n"
            "[yellow]Next up: Normalization (Module 08) - Stabilizing deep networks![/yellow]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error running showcase: {e}")
        console.print("💡 Make sure you've completed Module 07 and your attention layers work!")
        import traceback
        console.print(f"Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()