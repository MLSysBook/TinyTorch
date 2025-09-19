#!/usr/bin/env python3
"""
🚀 TinyTorch Capability Showcase Launcher

Easy way to run capability showcases and see what you've built!
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

console = Console()

def get_available_showcases():
    """Get list of available capability showcases."""
    capabilities_dir = Path(__file__).parent
    showcases = []
    
    showcase_files = sorted(capabilities_dir.glob("*_*.py"))
    
    for file_path in showcase_files:
        if file_path.name.startswith(("test_", "run_")):
            continue
            
        # Extract info from filename and docstring
        module_num = file_path.stem.split("_")[0]
        name = " ".join(file_path.stem.split("_")[1:]).title()
        
        # Try to get description from file
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                description = ""
                for line in lines:
                    if '"Look what you built!"' in line:
                        description = line.strip().replace('"""', '').replace('"', '')
                        break
                
                if not description:
                    description = f"Capability showcase for {name}"
                    
        except:
            description = f"Capability showcase for {name}"
        
        showcases.append({
            'number': module_num,
            'name': name,
            'description': description,
            'file': str(file_path),
            'filename': file_path.name
        })
    
    return showcases

def display_showcase_menu(showcases):
    """Display the showcase selection menu."""
    console.print(Panel.fit(
        "[bold cyan]🚀 TinyTorch Capability Showcases[/bold cyan]\n\n"
        "[green]\"Look what you built!\" - Celebrate your achievements![/green]",
        border_style="bright_blue"
    ))
    
    table = Table(title="Available Showcases")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Showcase", style="yellow", width=25)
    table.add_column("Description", style="green")
    
    for showcase in showcases:
        table.add_row(
            showcase['number'],
            showcase['name'],
            showcase['description']
        )
    
    console.print(table)
    console.print()

def run_showcase(showcase_file):
    """Run a specific showcase."""
    console.print(f"🚀 Running showcase: {Path(showcase_file).stem}")
    console.print("="*60)
    
    try:
        result = subprocess.run([sys.executable, showcase_file], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            console.print("\n✅ Showcase completed successfully!")
        else:
            console.print("\n⚠️ Showcase had some issues, but that's okay!")
            console.print("💡 Make sure you've completed the prerequisite modules.")
            
    except Exception as e:
        console.print(f"\n❌ Error running showcase: {e}")

def main():
    """Main launcher function."""
    showcases = get_available_showcases()
    
    if not showcases:
        console.print("❌ No capability showcases found!")
        return
    
    while True:
        console.clear()
        display_showcase_menu(showcases)
        
        console.print("[bold]Options:[/bold]")
        console.print("   • Enter showcase ID (e.g., '01', '02', '11')")
        console.print("   • Type 'all' to run all showcases")
        console.print("   • Type 'list' to see this menu again")
        console.print("   • Type 'quit' or 'exit' to exit")
        console.print()
        
        choice = Prompt.ask("Your choice").strip().lower()
        
        if choice in ['quit', 'exit', 'q']:
            console.print("👋 Thanks for using TinyTorch showcases!")
            break
            
        elif choice == 'all':
            console.print("🚀 Running all available showcases...")
            for showcase in showcases:
                console.print(f"\n🎯 Starting {showcase['name']}...")
                run_showcase(showcase['file'])
                
                if showcase != showcases[-1]:  # Not the last one
                    console.print("\n" + "="*60)
                    input("Press Enter to continue to next showcase...")
            
            console.print("\n🎉 All showcases completed!")
            input("Press Enter to return to menu...")
            
        elif choice == 'list':
            continue
            
        elif choice.isdigit() or choice.zfill(2).isdigit():
            # Handle numeric choice
            choice_id = choice.zfill(2)
            
            matching_showcases = [s for s in showcases if s['number'] == choice_id]
            
            if matching_showcases:
                showcase = matching_showcases[0]
                console.clear()
                run_showcase(showcase['file'])
                console.print("\n" + "="*60)
                input("Press Enter to return to menu...")
            else:
                console.print(f"❌ No showcase found with ID '{choice_id}'")
                input("Press Enter to continue...")
                
        else:
            console.print(f"❌ Invalid choice: '{choice}'")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()