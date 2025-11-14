"""
Book command for TinyTorch CLI: builds and manages the Jupyter Book.
"""

import os
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel

from .base import BaseCommand

NOTEBOOKS_DIR = "modules"

class BookCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "book"

    @property
    def description(self) -> str:
        return "Build and manage the TinyTorch Jupyter Book"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='book_command',
            help='Book management commands',
            metavar='COMMAND'
        )
        
        # Build command
        build_parser = subparsers.add_parser(
            'build',
            help='Build the Jupyter Book locally'
        )
        
        # Publish command
        publish_parser = subparsers.add_parser(
            'publish',
            help='Generate content, commit, and publish to GitHub'
        )
        publish_parser.add_argument(
            '--message',
            type=str,
            default='📚 Update book content',
            help='Commit message (default: "📚 Update book content")'
        )
        publish_parser.add_argument(
            '--branch',
            type=str,
            default='main',
            help='Branch to push to (default: main)'
        )
        
        # Clean command
        clean_parser = subparsers.add_parser(
            'clean',
            help='Clean built book files'
        )
        
        # Serve command
        serve_parser = subparsers.add_parser(
            'serve',
            help='Build and serve the Jupyter Book locally'
        )
        serve_parser.add_argument(
            '--port',
            type=int,
            default=8001,
            help='Port to serve on (default: 8001)'
        )
        serve_parser.add_argument(
            '--no-build',
            action='store_true',
            help='Skip building and serve existing files'
        )

    def run(self, args: Namespace) -> int:
        console = self.console
        
        # Check if we're in the right directory
        if not Path("site").exists():
            console.print(Panel(
                "[red]❌ site/ directory not found. Run this command from the TinyTorch root directory.[/red]",
                title="Error",
                border_style="red"
            ))
            return 1
        
        # Handle subcommands
        if not hasattr(args, 'book_command') or not args.book_command:
            console.print(Panel(
                "[bold cyan]📚 TinyTorch Book Management[/bold cyan]\n\n"
                "[bold]Available Commands:[/bold]\n"
                "  [bold green]build[/bold green]      - Build the complete Jupyter Book\n"
                "  [bold green]serve[/bold green]      - Build and serve the Jupyter Book locally\n"
                "  [bold green]publish[/bold green]   - Generate content, commit, and publish to GitHub\n"
                "  [bold green]clean[/bold green]     - Clean built book files\n\n"
                "[bold]Quick Start:[/bold]\n"
                "  [dim]tito book publish[/dim]       - Generate, commit, and publish to GitHub\n"
                "  [dim]tito book clean[/dim]         - Clean built book files",
                title="Book Commands",
                border_style="bright_blue"
            ))
            return 0
        
        if args.book_command == 'build':
            return self._build_book(args)
        elif args.book_command == 'serve':
            return self._serve_book(args)
        elif args.book_command == 'publish':
            return self._publish_book(args)
        elif args.book_command == 'clean':
            return self._clean_book()
        else:
            console.print(f"[red]Unknown book command: {args.book_command}[/red]")
            return 1

    def _generate_overview(self) -> int:
        """Generate overview pages from modules."""
        console = self.console
        console.print("🔄 Generating overview pages from modules...")
        
        try:
            os.chdir("site")
            result = subprocess.run(
                ["python3", "convert_readmes.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("✅ Overview pages generated successfully")
                # Show summary from the output
                for line in result.stdout.split('\n'):
                    if "✅ Created" in line or "🎉 Converted" in line:
                        console.print(f"   {line.strip()}")
                return 0
            else:
                console.print(f"[red]❌ Failed to generate overview pages: {result.stderr}[/red]")
                return 1
                
        except FileNotFoundError:
            console.print("[red]❌ Python3 not found or convert_readmes.py missing[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]❌ Error generating overview pages: {e}[/red]")
            return 1
        finally:
            os.chdir("..")

    def _generate_all(self) -> int:
        """Verify that all book chapters exist."""
        console = self.console
        console.print("📝 Verifying book chapters...")
        
        # Check that the chapters directory exists
        chapters_dir = Path("site/chapters")
        if not chapters_dir.exists():
            console.print("[red]❌ site/chapters directory not found[/red]")
            return 1
        
        # Count markdown files in chapters directory
        chapter_files = list(chapters_dir.glob("*.md"))
        if chapter_files:
            console.print(f"✅ Found {len(chapter_files)} chapter files")
        else:
            console.print("[yellow]⚠️  No chapter files found in site/chapters/[/yellow]")
        
        return 0

    def _build_book(self, args: Namespace) -> int:
        """Build the Jupyter Book locally."""
        console = self.console
        
        # First generate all content (notebooks + overview pages)
        console.print("📄 Step 1: Generating all content...")
        if self._generate_all() != 0:
            return 1
        
        # Then build the book
        console.print("📚 Step 2: Building Jupyter Book...")
        
        try:
            os.chdir("site")
            result = subprocess.run(
                ["jupyter-book", "build", "."],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("✅ Book built successfully!")
                
                # Extract and show the file path
                if "file://" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "file://" in line:
                            console.print(f"🌐 View at: {line.strip()}")
                            break
                
                console.print("📁 HTML files available in: site/_build/html/")
                return 0
            else:
                console.print(f"[red]❌ Failed to build book[/red]")
                if result.stderr:
                    console.print(f"Error details: {result.stderr}")
                return 1
                
        except FileNotFoundError:
            console.print("[red]❌ jupyter-book not found. Install with: pip install jupyter-book[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]❌ Error building book: {e}[/red]")
            return 1
        finally:
            os.chdir("..")

    def _serve_book(self, args: Namespace) -> int:
        """Build and serve the Jupyter Book locally."""
        console = self.console
        
        # Build the book first unless --no-build is specified
        if not args.no_build:
            console.print("📚 Step 1: Building the book...")
            if self._build_book(args) != 0:
                return 1
            console.print()
        
        # Start the HTTP server
        console.print("🌐 Step 2: Starting development server...")
        console.print(f"📖 Open your browser to: [bold blue]http://localhost:{args.port}[/bold blue]")
        console.print("🛑 Press [bold]Ctrl+C[/bold] to stop the server")
        console.print()
        
        book_dir = Path("site/_build/html")
        if not book_dir.exists():
            console.print("[red]❌ Built book not found. Run with --no-build=False to build first.[/red]")
            return 1
        
        try:
            # Use Python's built-in HTTP server
            subprocess.run([
                "python3", "-m", "http.server", str(args.port),
                "--directory", str(book_dir)
            ])
        except KeyboardInterrupt:
            console.print("\n🛑 Development server stopped")
        except FileNotFoundError:
            console.print("[red]❌ Python3 not found in PATH[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]❌ Error starting server: {e}[/red]")
            return 1
        
        return 0

    def _clean_book(self) -> int:
        """Clean built book files."""
        console = self.console
        console.print("🧹 Cleaning book build files...")
        
        try:
            os.chdir("site")
            result = subprocess.run(
                ["jupyter-book", "clean", "."],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("✅ Book files cleaned successfully")
                return 0
            else:
                console.print(f"[red]❌ Failed to clean book files: {result.stderr}[/red]")
                return 1
                
        except FileNotFoundError:
            console.print("[red]❌ jupyter-book not found[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]❌ Error cleaning book: {e}[/red]")
            return 1
        finally:
            os.chdir("..")

    def _publish_book(self, args: Namespace) -> int:
        """Generate content, commit, and publish to GitHub."""
        console = self.console
        
        console.print("🚀 Starting book publishing workflow...")
        
        # Step 1: Generate all content
        console.print("📝 Step 1: Generating all content...")
        if self._generate_all() != 0:
            console.print("[red]❌ Failed to generate content. Aborting publish.[/red]")
            return 1
        
        # Step 2: Check git status
        console.print("🔍 Step 2: Checking git status...")
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.returncode != 0:
                console.print("[red]❌ Git not available or not a git repository[/red]")
                return 1
            
            changes = result.stdout.strip()
            if not changes:
                console.print("✅ No changes to publish")
                return 0
                
        except Exception as e:
            console.print(f"[red]❌ Error checking git status: {e}[/red]")
            return 1
        
        # Step 3: Add and commit changes
        console.print("📦 Step 3: Committing changes...")
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], check=True, cwd=".")
            
            # Commit with message
            subprocess.run([
                "git", "commit", "-m", args.message
            ], check=True, cwd=".")
            
            console.print(f"✅ Committed with message: {args.message}")
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Failed to commit changes: {e}[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]❌ Error during commit: {e}[/red]")
            return 1
        
        # Step 4: Push to GitHub
        console.print(f"⬆️  Step 4: Pushing to {args.branch} branch...")
        try:
            result = subprocess.run([
                "git", "push", "origin", args.branch
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                console.print(f"✅ Successfully pushed to {args.branch}")
            else:
                console.print(f"[red]❌ Failed to push: {result.stderr}[/red]")
                return 1
                
        except Exception as e:
            console.print(f"[red]❌ Error during push: {e}[/red]")
            return 1
        
        # Step 5: Show deployment info
        console.print("🌐 Step 5: Deployment initiated...")
        console.print("✅ GitHub Actions will now:")
        console.print("   📚 Build the Jupyter Book")
        console.print("   🚀 Deploy to GitHub Pages")
        console.print("   🔗 Update live website")
        
        # Try to get repository info for deployment URL
        try:
            result = subprocess.run([
                "git", "remote", "get-url", "origin"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                remote_url = result.stdout.strip()
                if "github.com" in remote_url:
                    # Extract owner/repo from git URL
                    if remote_url.endswith(".git"):
                        remote_url = remote_url[:-4]
                    if remote_url.startswith("git@github.com:"):
                        repo_path = remote_url.replace("git@github.com:", "")
                    elif remote_url.startswith("https://github.com/"):
                        repo_path = remote_url.replace("https://github.com/", "")
                    else:
                        repo_path = None
                    
                    if repo_path:
                        console.print(f"\n🔗 Monitor deployment: https://github.com/{repo_path}/actions")
                        console.print(f"📖 Live website: https://{repo_path.split('/')[0]}.github.io/{repo_path.split('/')[1]}/")
                        
        except Exception:
            # Don't fail the whole command if we can't get repo info
            pass
        
        console.print("\n🎉 Publishing workflow complete!")
        console.print("💡 Check GitHub Actions for deployment status")
        
        return 0 