#!/usr/bin/env python3
"""
Verify that the Jupyter Book build is complete and all pages are present.
"""

import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def verify_book_build():
    """Verify the book build is complete."""
    build_dir = Path("book/_build/html")
    
    if not build_dir.exists():
        console.print("❌ Build directory not found! Run 'tito book build' first.")
        return False
    
    # Pages that must exist
    required_pages = {
        "Main Pages": [
            "index.html",
            "intro.html",
            "setup.html",
            "instructor-guide.html",
            "system-architecture.html"
        ],
        "Module Chapters": [
            f"chapters/{i:02d}-{name}.html" for i, name in enumerate([
                "introduction", "setup", "tensor", "activations", "layers",
                "dense", "spatial", "attention", "dataloader", "autograd",
                "optimizers", "training", "compression", "kernels", 
                "benchmarking", "mlops", "tinygpt"
            ], 0)
        ],
        "New Documentation": [
            "testing-framework.html",
            "kiss-principle.html"
        ],
        "Usage Paths": [
            "usage-paths/quick-start.html",
            "usage-paths/browse-online.html",
            "usage-paths/serious-development.html"
        ]
    }
    
    # Check each category
    results = {}
    for category, pages in required_pages.items():
        results[category] = []
        for page in pages:
            full_path = build_dir / page
            exists = full_path.exists()
            size = full_path.stat().st_size if exists else 0
            results[category].append({
                'page': page,
                'exists': exists,
                'size': size
            })
    
    # Display results
    console.print(Panel.fit(
        "📚 [bold blue]TinyTorch Jupyter Book Verification[/bold blue]",
        border_style="blue"
    ))
    
    all_good = True
    for category, checks in results.items():
        console.print(f"\n[bold]{category}[/bold]")
        
        for check in checks:
            if check['exists']:
                if check['size'] > 100:  # More than just a redirect
                    console.print(f"  ✅ {check['page']} ({check['size']:,} bytes)")
                else:
                    console.print(f"  ⚠️  {check['page']} (small: {check['size']} bytes)")
            else:
                console.print(f"  ❌ {check['page']} (missing)")
                all_good = False
    
    # Summary
    if all_good:
        console.print(Panel.fit(
            "✨ [bold green]All documentation pages built successfully![/bold green]\n"
            f"🌐 View at: file://{build_dir.absolute()}/index.html",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "⚠️  [bold yellow]Some pages are missing![/bold yellow]\n"
            "Run 'tito book build' to rebuild the documentation.",
            border_style="yellow"
        ))
    
    return all_good

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)  # Go to project root
    success = verify_book_build()
    exit(0 if success else 1)