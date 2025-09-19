"""
Logo command for TinyTorch CLI: displays beautiful ASCII art logo.
"""

from argparse import ArgumentParser, Namespace
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
import time

from .base import BaseCommand
from ..core.preferences import UserPreferences

class LogoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "logo"

    @property
    def description(self) -> str:
        return "Display the TinyTorch ASCII art logo with theme support"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--animate", action="store_true", help="Show animated flame effect")
        parser.add_argument("--simple", action="store_true", help="Show simple logo without extra elements")
        parser.add_argument("--bright", action="store_true", help="Use bright/vivid theme with yellow center")
        parser.add_argument("--theme", choices=["standard", "bright"], help="Choose logo theme")
        parser.add_argument("--save-theme", action="store_true", help="Save the selected theme as default")

    def run(self, args: Namespace) -> int:
        # Determine theme
        theme = self._get_theme(args)
        
        # Save theme preference if requested
        if args.save_theme:
            self._save_theme_preference(theme)
            self.console.print(f"[green]✅ Saved '{theme}' as default theme[/green]")
        
        if args.animate:
            self.show_animated_logo(theme=theme)
        elif args.simple:
            self.show_simple_logo(theme=theme)
        else:
            self.show_full_logo(theme=theme)
        return 0

    def show_full_logo(self, theme: str = "standard"):
        """Show the complete logo with version and tagline."""
        console = self.console
        
        # ASCII Art Logo (now returns a Text object)
        logo_art = self.get_logo_art(theme=theme)
        
        # Add additional tagline
        tagline = Text()
        tagline.append("\nBuild Complete Neural Networks from First Principles", style="dim cyan")
        
        # Combine logo and tagline
        full_content = Text()
        full_content.append(logo_art)
        full_content.append(tagline)
        
        # Display with rich styling
        border_style = "bright_blue" if theme == "standard" else "bright_yellow"
        console.print()
        console.print(Panel(
            Align.center(full_content),
            border_style=border_style,
            padding=(1, 2)
        ))
        console.print()
        
        # Show quick stats
        self.show_progress_stats(theme=theme)

    def show_simple_logo(self, theme: str = "standard"):
        """Show just the ASCII art logo."""
        console = self.console
        logo_art = self.get_logo_art(theme=theme)  # Now returns a Text object
        
        console.print()
        console.print(Align.center(logo_art))
        console.print()

    def show_animated_logo(self, theme: str = "standard"):
        """Show logo with animated flame effect."""
        console = self.console
        
        # Animation frames - different for each theme
        if theme == "bright":
            flame_frames = ["✨", "⚡", "🔥", "💫", "✨"]  # More energetic
        else:
            flame_frames = ["🔥", "🔶", "🔸", "✨", "🔥"]  # Professional
        
        border_style = "bright_blue" if theme == "standard" else "bright_yellow"
        
        for i in range(10):  # Show 10 frames
            console.clear()
            
            # Get current flame
            flame = flame_frames[i % len(flame_frames)]
            
            # Create animated logo (now returns a Text object)
            logo_art = self.get_logo_art(theme=theme, flame_char=flame)
            
            tagline = Text()
            tagline.append("\nBuild Complete Neural Networks from First Principles", style="dim cyan")
            
            full_content = Text()
            full_content.append(logo_art)
            full_content.append(tagline)
            
            console.print()
            console.print(Panel(
                Align.center(full_content),
                border_style=border_style,
                padding=(1, 2)
            ))
            
            time.sleep(0.2)
        
        # Show final static version
        self.show_full_logo(theme=theme)

    def get_logo_art(self, theme: str = "standard", flame_char: str = None):
        """Generate the ASCII art logo matching the actual TinyTorch design."""
        # Set default flame character based on theme
        if flame_char is None:
            flame_char = "✨" if theme == "bright" else "🔥"
        
        # Create a Text object to properly handle Rich markup
        logo_text = Text()
        
        # Theme-specific styling
        if theme == "bright":
            # Bright theme: vivid yellow center, orange "tiny" text
            top_flames = "✨🔥✨"
        else:
            # Standard theme: professional orange/red tones
            top_flames = f"{flame_char}{flame_char}"
        
        # ASCII art representing the real TinyTorch logo:
        # Flame on left with neural network inside, "tiny TORCH" on right
        logo_lines = [
            f"         {top_flames}      ",
            "        ╱──╲      ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗",
            "       ╱ ⚡ ╲     ╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║",
            "      ╱ ╱ ╲ ╲       ██║   ██║   ██║██████╔╝██║     ███████║",
            "     │ ●━━━● │      ██║   ██║   ██║██╔══██╗██║     ██╔══██║",
            "     │ │ ⚡ │ │      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║",
            "     │ ●━┬━● │      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝",
            "      ╲ ╲●╱ ╱                                                       ",
            "       ╲━━━╱      ",
            "         ╲╱       "
        ]
        
        # Add the first line with styled "tiny"
        if theme == "bright":
            logo_text.append(logo_lines[0])
            logo_text.append("tiny", style="bold orange1")
            logo_text.append("                                              \n")
        else:
            logo_text.append(logo_lines[0])
            logo_text.append("tiny", style="dim")
            logo_text.append("                                              \n")
        
        # Add the main ASCII art lines
        for line in logo_lines[1:8]:
            logo_text.append(line + "\n")
        
        # Add the tagline with proper styling
        logo_text.append(logo_lines[8])
        if theme == "bright":
            logo_text.append(flame_char, style="bright_yellow")
            logo_text.append(" Learn ML Systems by Building Them", style="bold orange1")
        else:
            logo_text.append(f"{flame_char} Learn ML Systems by Building Them", style="orange1")
        logo_text.append("\n")
        
        # Add the final line
        logo_text.append(logo_lines[9])
        
        return logo_text

    def show_progress_stats(self, theme: str = "standard"):
        """Show current progress through the course."""
        console = self.console
        
        # Check how many modules are completed
        # This is a simplified version - could be enhanced to check actual module status
        
        # Theme-specific emoji and styling
        if theme == "bright":
            journey_emoji = "🌟"
            fire_emoji = "⚡"
            start_emoji = "💫"
            border_style = "bright_yellow"
        else:
            journey_emoji = "📊"
            fire_emoji = "🔥"
            start_emoji = "🚀"
            border_style = "bright_green"
        
        stats_table = f"""
[bold cyan]{journey_emoji} Your TinyTorch Journey[/bold cyan]

[bold green]Module Progress:[/bold green]
  🎯 17 Total Modules Available
  📚 Build from Tensors → Training → TinyGPT
  {fire_emoji} Learn by Implementation, Not Theory

[bold yellow]Quick Commands:[/bold yellow]
  [dim]tito module status[/dim]        - Check module progress
  [dim]tito checkpoint timeline[/dim]   - Visual progress tracker
  [dim]tito module view[/dim]          - Start coding in Jupyter Lab
  [dim]tito system doctor[/dim]        - Diagnose any issues

[bold magenta]Ready to Build ML Systems? Start with:[/bold magenta]
  [dim]tito module view 01_setup[/dim]  - Configure your environment
"""
        
        console.print(Panel(
            Text.from_markup(stats_table),
            title=f"{start_emoji} Getting Started",
            border_style=border_style,
            padding=(1, 2)
        ))
    
    def _get_theme(self, args: Namespace) -> str:
        """Determine the theme to use based on arguments and preferences."""
        # Check command line arguments first
        if args.bright:
            return "bright"
        if args.theme:
            return args.theme
        
        # Fall back to saved preferences
        prefs = UserPreferences.load_from_file()
        return prefs.logo_theme
    
    def _save_theme_preference(self, theme: str) -> None:
        """Save the theme preference to config file."""
        prefs = UserPreferences.load_from_file()
        prefs.logo_theme = theme
        prefs.save_to_file()