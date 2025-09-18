#!/usr/bin/env python3
"""
TinyTorch Demo 07: Attention Mechanisms - The AI Revolution
Shows how attention transforms sequence processing and enables modern AI!
"""

import sys
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.columns import Columns

def demo_attention():
    """Demo attention mechanisms for sequence understanding and modern AI"""
    
    console = Console()
    
    try:
        # Import TinyTorch modules
        import tinytorch.core.tensor as tt
        import tinytorch.core.activations as act
        import tinytorch.core.layers as layers
        import tinytorch.core.dense as dense
        import tinytorch.core.attention as attention
        
        # Main header
        console.print(Panel.fit(
            "🎯 TinyTorch Attention Mechanisms Demo\nThe breakthrough that enabled ChatGPT and modern AI!",
            style="bold cyan",
            border_style="bright_blue"
        ))
        console.print()
        
        # Demo 1: The Attention Problem
        console.print(Panel(
            "From fixed-size bottlenecks to dynamic focus...",
            title="🧠 Demo 1: Why Attention Revolutionized AI",
            style="green"
        ))
        
        # Simulate a sequence processing problem
        sequence = ["The", "cat", "sat", "on", "the", "mat"]
        console.print(f"[bold cyan]Input sequence:[/bold cyan] {' '.join(sequence)}")
        console.print()
        
        # Create comparison table
        comparison_table = Table(show_header=True, header_style="bold magenta")
        comparison_table.add_column("Traditional RNN", style="red")
        comparison_table.add_column("Attention Mechanism", style="green")
        
        rnn_steps = [
            "[The] → h1",
            "[cat] + h1 → h2", 
            "[sat] + h2 → h3",
            "[on] + h3 → h4",
            "[the] + h4 → h5",
            "[mat] + h5 → h6 (final)"
        ]
        
        attention_steps = [
            "Process ALL positions simultaneously:",
            "[The, cat, sat, on, the, mat]",
            "",
            "For each output:",
            "Look at ALL inputs with learned weights",
            "Direct access to any information!"
        ]
        
        for rnn, attn in zip(rnn_steps, attention_steps):
            comparison_table.add_row(rnn, attn)
        
        console.print(comparison_table)
        console.print()
        
        # Problems and solutions
        problems_panel = Panel(
            "❌ Problem: h6 must encode ALL previous information!\n❌ Result: Information loss, especially for long sequences",
            title="Traditional RNN Issues",
            style="red"
        )
        
        solutions_panel = Panel(
            "✅ Solution: Direct access to any previous information!\n✅ Result: No information bottleneck!",
            title="Attention Solution",
            style="green"
        )
        
        console.print(Columns([problems_panel, solutions_panel]))
        console.print()
        
        # Demo 2: Basic Attention Mechanism
        print("🔍 Demo 2: Basic Attention Computation")
        print("Computing attention weights step by step...")
        print()
        
        # Create simple sequence embeddings (3 words, 4 dimensions each)
        sequence_length = 3
        embed_dim = 4
        
        # Word embeddings for "cat sat mat"
        embeddings = tt.Tensor([
            [1.0, 0.5, 0.2, 0.8],  # "cat"
            [0.3, 1.0, 0.7, 0.1],  # "sat"
            [0.6, 0.2, 1.0, 0.4]   # "mat"
        ])
        
        print("Word embeddings (3 words × 4 dimensions):")
        for i, word in enumerate(["cat", "sat", "mat"]):
            emb = embeddings.data[i]
            print(f"  {word}: [{emb[0]:.1f}, {emb[1]:.1f}, {emb[2]:.1f}, {emb[3]:.1f}]")
        print()
        
        # Simple attention: query attends to all keys
        query = embeddings.data[1]  # "sat" is attending
        keys = embeddings.data      # to all words
        
        print(f"Query (word 'sat'): {query}")
        print()
        
        # Compute attention scores (dot product)
        scores = np.dot(keys, query)
        print("Attention scores (how much 'sat' attends to each word):")
        for i, (word, score) in enumerate(zip(["cat", "sat", "mat"], scores)):
            print(f"  'sat' → '{word}': {score:.3f}")
        print()
        
        # Softmax to get attention weights
        exp_scores = np.exp(scores)
        attention_weights = exp_scores / np.sum(exp_scores)
        
        print("Attention weights (after softmax):")
        for i, (word, weight) in enumerate(zip(["cat", "sat", "mat"], attention_weights)):
            print(f"  'sat' → '{word}': {weight:.3f} ({weight*100:.1f}%)")
        print(f"Total: {np.sum(attention_weights):.3f}")
        print()
        
        # Compute attended output
        attended_output = np.sum(keys * attention_weights.reshape(-1, 1), axis=0)
        print(f"Attended output for 'sat': {attended_output}")
        print("(Weighted combination of all word embeddings)")
        print()
        
        # Demo 3: Multi-Head Attention
        print("🧩 Demo 3: Multi-Head Attention - Multiple Perspectives")
        print("Like having multiple experts focus on different aspects...")
        print()
        
        # Create multi-head attention layer
        num_heads = 2
        head_dim = embed_dim // num_heads
        
        print(f"Multi-head setup: {num_heads} heads, {head_dim} dimensions each")
        print()
        
        # Simulate different attention heads
        print("Head 1 (Syntax Expert) - Focuses on grammatical relationships:")
        syntax_scores = np.array([0.2, 0.7, 0.1])  # Focuses on current word
        syntax_weights = np.exp(syntax_scores) / np.sum(np.exp(syntax_scores))
        for word, weight in zip(["cat", "sat", "mat"], syntax_weights):
            print(f"  '{word}': {weight:.3f}")
        
        print()
        print("Head 2 (Semantic Expert) - Focuses on meaning relationships:")
        semantic_scores = np.array([0.4, 0.2, 0.4])  # Focuses on related objects
        semantic_weights = np.exp(semantic_scores) / np.sum(np.exp(semantic_scores))
        for word, weight in zip(["cat", "sat", "mat"], semantic_weights):
            print(f"  '{word}': {weight:.3f}")
        
        print()
        print("💡 Key insight: Different heads learn different types of relationships!")
        print()
        
        # Demo 4: Self-Attention in Practice
        print("🎭 Demo 4: Self-Attention - Words Talking to Each Other")
        print("Every word attends to every other word...")
        print()
        
        # Create attention layer
        attn_layer = attention.SelfAttention(d_model=4)
        
        print("Self-attention matrix (who attends to whom):")
        print("         cat   sat   mat")
        
        # Simulate attention weights for visualization
        attention_matrix = np.array([
            [0.4, 0.3, 0.3],  # cat attends to...
            [0.2, 0.6, 0.2],  # sat attends to...
            [0.3, 0.2, 0.5]   # mat attends to...
        ])
        
        for i, word in enumerate(["cat", "sat", "mat"]):
            weights = attention_matrix[i]
            print(f"  {word}:  {weights[0]:.1f}   {weights[1]:.1f}   {weights[2]:.1f}")
        
        print()
        print("Interpretation:")
        print("  • 'cat' focuses on itself (0.4) and context words")
        print("  • 'sat' focuses mainly on itself (0.6) - the action")
        print("  • 'mat' balances between all words")
        print()
        
        # Demo 5: Scaled Dot-Product Attention
        console.print(Panel(
            "The mathematical foundation of modern AI",
            title="⚖️ Demo 5: Scaled Dot-Product Attention - The Core Formula",
            style="blue"
        ))
        
        # Display the attention formula with syntax highlighting
        formula_code = """
# The Attention Formula that Changed Everything
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Where:
  Q = Queries (what we're looking for)
  K = Keys    (what's available to match against)  
  V = Values  (what we actually retrieve)
  d_k = key dimension (for scaling)
"""
        
        console.print(Syntax(formula_code, "python", theme="monokai", line_numbers=False))
        console.print()
        
        # Create Q, K, V matrices
        d_k = 4  # key dimension
        scale_factor = 1.0 / np.sqrt(d_k)
        
        Q = embeddings  # Queries
        K = embeddings  # Keys  
        V = embeddings  # Values
        
        print(f"Q (Queries): {Q.data.shape}")
        print(f"K (Keys): {K.data.shape}")
        print(f"V (Values): {V.data.shape}")
        print(f"Scale factor: 1/√{d_k} = {scale_factor:.3f}")
        print()
        
        # Compute attention
        QK = np.dot(Q.data, K.data.T)  # Query-Key similarity
        scaled_QK = QK * scale_factor   # Scale to prevent large values
        attn_weights = np.exp(scaled_QK) / np.sum(np.exp(scaled_QK), axis=1, keepdims=True)
        output = np.dot(attn_weights, V.data)
        
        print("Attention weights matrix:")
        for i in range(3):
            print(f"  [{attn_weights[i,0]:.3f}, {attn_weights[i,1]:.3f}, {attn_weights[i,2]:.3f}]")
        
        print()
        print("Output (attended representations):")
        for i, word in enumerate(["cat", "sat", "mat"]):
            out = output[i]
            print(f"  {word}: [{out[0]:.3f}, {out[1]:.3f}, {out[2]:.3f}, {out[3]:.3f}]")
        
        print()
        
        # Demo 6: Transformer Architecture Preview
        console.print(Panel(
            "How attention enables modern language models...",
            title="🏗️ Demo 6: Transformer Architecture - The Full Picture",
            style="magenta"
        ))
        
        # Transformer architecture diagram
        transformer_arch = """
┌─────────────────────┐
│   Input Embeddings  │
└─────────────────────┘
           ↓
┌─────────────────────┐
│ Multi-Head Self-    │
│    Attention        │
└─────────────────────┘
           ↓ + (residual)
┌─────────────────────┐
│ Layer Normalization │
└─────────────────────┘
           ↓
┌─────────────────────┐
│ Feed-Forward        │
│    Network          │
└─────────────────────┘
           ↓ + (residual)
┌─────────────────────┐
│ Layer Normalization │
└─────────────────────┘
           ↓
┌─────────────────────┐
│      Output         │
└─────────────────────┘
"""
        
        console.print(Panel(transformer_arch, title="Transformer Block", style="cyan"))
        
        # Why it works table
        why_table = Table(show_header=True, header_style="bold magenta")
        why_table.add_column("Component", style="cyan")
        why_table.add_column("Purpose", style="yellow")
        
        why_table.add_row("Self-attention", "Captures long-range dependencies")
        why_table.add_row("Multi-head", "Multiple types of relationships")
        why_table.add_row("Residual connections", "Stable training")
        why_table.add_row("Layer normalization", "Normalized activations")
        why_table.add_row("Feed-forward", "Non-linear transformations")
        
        console.print(why_table)
        console.print()
        
        # Demo 7: Real-World Applications
        print("🌍 Demo 7: Real-World Impact")
        print("Where attention mechanisms changed everything...")
        print()
        
        applications = [
            ("Language Translation", "Attention shows which source words align with target words"),
            ("ChatGPT/GPT-4", "Self-attention enables understanding of entire conversation context"),
            ("Image Captioning", "Visual attention focuses on relevant image regions"),
            ("Document Analysis", "Attention connects information across long documents"),
            ("Code Generation", "Attention relates variable names and function calls"),
            ("Scientific Discovery", "Attention finds patterns in massive datasets")
        ]
        
        print("Revolutionary applications:")
        for app, description in applications:
            print(f"  • {app}: {description}")
        
        print()
        
        # Demo 8: Scaling Analysis
        print("📈 Demo 8: Why Attention Scales")
        print("Understanding computational complexity...")
        print()
        
        print("Attention complexity analysis:")
        print("  Sequence length: n")
        print("  Embedding dimension: d")
        print("  ")
        print("  Self-attention: O(n² × d)")
        print("  Feed-forward: O(n × d²)")
        print("  ")
        print("  For long sequences: attention dominates")
        print("  For wide embeddings: feed-forward dominates")
        print()
        
        print("Example scaling:")
        for n in [100, 1000, 10000]:
            attn_ops = n * n * 512
            ff_ops = n * 512 * 2048
            print(f"  n={n}: Attention={attn_ops:,} ops, Feed-forward={ff_ops:,} ops")
        
        print()
        
        # Success summary
        console.print(Panel.fit(
            "🎯 Achievements:\n"
            "• Understood the attention revolution and why it matters\n"
            "• Computed attention weights and attended outputs\n"
            "• Explored multi-head attention for different perspectives\n"
            "• Analyzed self-attention matrices\n"
            "• Implemented scaled dot-product attention formula\n"
            "• Previewed complete Transformer architecture\n"
            "• Connected to real-world AI applications\n"
            "• Analyzed computational scaling properties\n\n"
            "🔥 Next: End-to-end training pipelines!",
            title="🏆 TinyTorch Attention Demo Complete!",
            style="bold green",
            border_style="bright_green"
        ))
        
        return True
        
    except ImportError as e:
        console.print(Panel(
            f"Could not import TinyTorch modules: {e}\n\n💡 Make sure to run: tito export 07_attention",
            title="❌ Import Error",
            style="bold red"
        ))
        return False
    except Exception as e:
        console.print(Panel(
            f"Demo failed: {e}",
            title="❌ Error",
            style="bold red"
        ))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_attention()
    sys.exit(0 if success else 1)