"""
TinyTorch Competition Submission Command

Validates and prepares TinyMLPerf competition submissions from Module 20.
"""

import json
import platform
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any, Optional

from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.console import Group
from rich.align import Align

from .base import BaseCommand
from ..core.exceptions import TinyTorchCLIError


class SubmitCommand(BaseCommand):
    """Validate and submit TinyMLPerf competition entries"""
    
    @property
    def name(self) -> str:
        return "submit"
    
    @property
    def description(self) -> str:
        return "Validate and prepare TinyMLPerf competition submission"
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add submit command arguments."""
        parser.add_argument(
            'submission_file',
            type=str,
            default='submission.json',
            nargs='?',
            help='Path to submission.json file (default: submission.json)'
        )
        parser.add_argument(
            '--github-repo',
            type=str,
            help='GitHub repository URL (overrides submission file)'
        )
        parser.add_argument(
            '--skip-validation',
            action='store_true',
            help='Skip validation checks (not recommended)'
        )
    
    def run(self, args: Namespace) -> int:
        """Execute submit command."""
        submission_file = Path(args.submission_file)
        
        # Check if submission file exists
        if not submission_file.exists():
            self.console.print(Panel(
                f"[red]❌ Submission file not found:[/red] {submission_file}\n\n"
                f"[yellow]Expected file:[/yellow] submission.json\n"
                f"[cyan]How to generate:[/cyan]\n"
                f"  1. Open Module 20 (TinyMLPerf Competition)\n"
                f"  2. Run the generate_submission() function\n"
                f"  3. This will create submission.json\n\n"
                f"[dim]Then run: tito submit submission.json[/dim]",
                title="❌ File Not Found",
                border_style="red"
            ))
            return 1
        
        # Load submission
        try:
            with open(submission_file, 'r') as f:
                submission = json.load(f)
        except json.JSONDecodeError as e:
            self.console.print(Panel(
                f"[red]❌ Invalid JSON file:[/red] {submission_file}\n\n"
                f"[yellow]Error:[/yellow] {str(e)}\n\n"
                f"[cyan]Please regenerate submission.json from Module 20[/cyan]",
                title="❌ Invalid Submission",
                border_style="red"
            ))
            return 1
        
        # Welcome
        self._show_welcome()
        
        # Validate submission
        if not args.skip_validation:
            validation_result = self._validate_submission(submission)
            if not validation_result["valid"]:
                self.console.print(Panel(
                    "[red]❌ Submission validation failed[/red]\n\n"
                    "[yellow]Please fix the errors and regenerate submission.json[/yellow]",
                    title="Validation Failed",
                    border_style="red"
                ))
                return 1
        
        # Display scorecard
        self._display_scorecard(submission)
        
        # Collect additional info
        submission = self._collect_additional_info(submission, args)
        
        # Honor code
        if not self._confirm_honor_code():
            self.console.print(Panel(
                "[yellow]Submission cancelled[/yellow]\n\n"
                "Honor code agreement is required for competition submission.",
                title="Cancelled",
                border_style="yellow"
            ))
            return 1
        
        submission["honor_code"] = True
        
        # Save final submission
        final_file = self._save_final_submission(submission)
        
        # Show next steps
        self._show_next_steps(final_file, submission)
        
        return 0
    
    def _show_welcome(self) -> None:
        """Show welcome message."""
        self.console.print(Panel(
            Group(
                Align.center("[bold bright_green]🏅 TinyMLPerf Competition Submission 🏅[/bold bright_green]"),
                "",
                "Welcome to the TinyMLPerf competition submission system!",
                "",
                "This tool will:",
                "• [green]✓ Validate your submission[/green]",
                "• [blue]✓ Display your normalized scorecard[/blue]",
                "• [yellow]✓ Collect GitHub repository for verification[/yellow]",
                "• [magenta]✓ Prepare final submission package[/magenta]",
                "",
                "[dim]Following MLPerf principles: Reproducible, fair, and transparent[/dim]",
            ),
            title="🔥 Competition Submission",
            border_style="bright_green",
            padding=(1, 2)
        ))
    
    def _validate_submission(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        """Validate submission with sanity checks."""
        self.console.print("\n🔍 [bold cyan]Validating Submission...[/bold cyan]\n")
        
        checks = []
        warnings = []
        errors = []
        
        # Extract metrics
        normalized = submission.get("normalized_scores", {})
        speedup = normalized.get("speedup", 1.0)
        compression = normalized.get("compression_ratio", 1.0)
        accuracy_delta = normalized.get("accuracy_delta", 0.0)
        
        # Check 1: Speedup is reasonable
        if speedup > 50:
            errors.append(f"❌ Speedup {speedup:.1f}x seems unrealistic (>50x)")
        elif speedup > 20:
            warnings.append(f"⚠️  Speedup {speedup:.1f}x is very high - please verify measurements")
        else:
            checks.append(f"✅ Speedup {speedup:.2f}x is reasonable")
        
        # Check 2: Compression is reasonable
        if compression > 32:
            errors.append(f"❌ Compression {compression:.1f}x seems unrealistic (>32x)")
        elif compression > 16:
            warnings.append(f"⚠️  Compression {compression:.1f}x is very high - please verify")
        else:
            checks.append(f"✅ Compression {compression:.2f}x is reasonable")
        
        # Check 3: Accuracy preservation (Closed Division)
        division = submission.get("division", "closed")
        if division == "closed" and accuracy_delta > 1.0:
            errors.append(f"❌ Accuracy improved {accuracy_delta:.1f}pp - did you train the model?")
        elif accuracy_delta > 0.5:
            warnings.append(f"⚠️  Accuracy improved {accuracy_delta:.1f}pp - verify no training")
        else:
            checks.append(f"✅ Accuracy Δ {accuracy_delta:+.2f}pp is reasonable")
        
        # Check 4: Required fields
        required = ["division", "event", "athlete_name", "baseline", "optimized", "normalized_scores"]
        missing = [f for f in required if f not in submission]
        if missing:
            errors.append(f"❌ Missing fields: {', '.join(missing)}")
        else:
            checks.append("✅ All required fields present")
        
        # Check 5: Techniques documented
        techniques = submission.get("techniques_applied", [])
        if not techniques or "TODO" in str(techniques):
            warnings.append("⚠️  No optimization techniques listed")
        else:
            checks.append(f"✅ Techniques: {', '.join(techniques[:3])}{'...' if len(techniques) > 3 else ''}")
        
        # Display results
        for check in checks:
            self.console.print(f"  {check}")
        for warning in warnings:
            self.console.print(f"  {warning}")
        for error in errors:
            self.console.print(f"  {error}")
        
        if errors:
            self.console.print("\n[red]❌ Validation failed - please fix errors above[/red]\n")
        elif warnings:
            self.console.print("\n[yellow]⚠️  Warnings detected - review before submitting[/yellow]\n")
        else:
            self.console.print("\n[green]✅ All validation checks passed![/green]\n")
        
        return {
            "valid": len(errors) == 0,
            "checks": checks,
            "warnings": warnings,
            "errors": errors
        }
    
    def _display_scorecard(self, submission: Dict[str, Any]) -> None:
        """Display MLPerf-style scorecard with normalized metrics."""
        normalized = submission.get("normalized_scores", {})
        division = submission.get("division", "closed").upper()
        event = submission.get("event", "all_around").replace("_", " ").title()
        athlete = submission.get("athlete_name", "Unknown")
        
        # Create scorecard table
        table = Table(
            title=f"🏆 TinyMLPerf Scorecard - {athlete}",
            title_style="bold bright_green",
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Metric", style="bold", width=25)
        table.add_column("Value", style="green", justify="right", width=20)
        table.add_column("Rating", style="yellow", width=15)
        
        # Division & Event
        table.add_row("Division", division, self._get_division_badge(division.lower()))
        table.add_row("Event", event, self._get_event_badge(submission.get("event", "")))
        table.add_section()
        
        # Normalized scores
        speedup = normalized.get("speedup", 1.0)
        compression = normalized.get("compression_ratio", 1.0)
        accuracy_delta = normalized.get("accuracy_delta", 0.0)
        efficiency = normalized.get("efficiency_score", 1.0)
        
        table.add_row(
            "Speedup",
            f"{speedup:.2f}x faster",
            self._get_performance_rating(speedup, "speedup")
        )
        table.add_row(
            "Compression",
            f"{compression:.2f}x smaller",
            self._get_performance_rating(compression, "compression")
        )
        table.add_row(
            "Accuracy Δ",
            f"{accuracy_delta:+.2f}pp",
            self._get_accuracy_rating(accuracy_delta)
        )
        table.add_row(
            "Efficiency Score",
            f"{efficiency:.2f}",
            self._get_efficiency_rating(efficiency)
        )
        
        self.console.print(table)
    
    def _collect_additional_info(self, submission: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
        """Collect additional information for submission."""
        self.console.print("\n📝 [bold cyan]Additional Information[/bold cyan]\n")
        
        # GitHub repo (required for verification)
        current_repo = submission.get("github_repo", "")
        if args.github_repo:
            github_repo = args.github_repo
        elif current_repo:
            github_repo = Prompt.ask(
                "[bold]GitHub repository URL[/bold] (for code verification)",
                default=current_repo
            )
        else:
            github_repo = Prompt.ask(
                "[bold]GitHub repository URL[/bold] (required for verification)",
                default="https://github.com/YourUsername/TinyTorch"
            )
        
        submission["github_repo"] = github_repo
        
        # Environment info (for reproducibility)
        submission["environment"] = {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "Unknown"
        }
        
        return submission
    
    def _confirm_honor_code(self) -> bool:
        """Confirm honor code agreement."""
        self.console.print("\n📜 [bold cyan]Honor Code Agreement[/bold cyan]\n")
        
        self.console.print(Panel(
            "[bold]TinyMLPerf Honor Code[/bold]\n\n"
            "I affirm that:\n"
            "• This submission represents my own work\n"
            "• All techniques and optimizations are documented\n"
            "• Results are reproducible from the provided repository\n"
            "• I followed the competition rules for my division\n"
            "• Measurements were conducted fairly and honestly\n\n"
            "[yellow]Closed Division:[/yellow] No model training, architecture changes, or data modifications\n"
            "[blue]Open Division:[/blue] Innovations are documented and reproducible",
            title="Honor Code",
            border_style="bright_blue"
        ))
        
        return Confirm.ask("\n[bold]Do you agree to the honor code?[/bold]", default=True)
    
    def _save_final_submission(self, submission: Dict[str, Any]) -> Path:
        """Save final submission with all information."""
        final_file = Path("submission_final.json")
        
        with open(final_file, 'w') as f:
            json.dump(submission, f, indent=2)
        
        self.console.print(f"\n✅ [green]Final submission saved:[/green] {final_file}")
        
        return final_file
    
    def _show_next_steps(self, final_file: Path, submission: Dict[str, Any]) -> None:
        """Show next steps for submission."""
        github_repo = submission.get("github_repo", "")
        
        self.console.print(Panel(
            Group(
                "[bold bright_green]🎉 Submission Ready![/bold bright_green]",
                "",
                "[bold]Your submission package:[/bold]",
                f"• [green]Submission file:[/green] {final_file}",
                f"• [blue]GitHub repo:[/blue] {github_repo}",
                f"• [yellow]Honor code:[/yellow] Confirmed ✓",
                "",
                "[bold bright_blue]📤 Next Steps:[/bold bright_blue]",
                "",
                "[bold]Option 1: Email Submission[/bold]",
                "  1. Email [cyan]submission_final.json[/cyan] to your instructor",
                "  2. Include your GitHub repo link in the email",
                "  3. Subject: [dim]TinyMLPerf Submission - [YourName][/dim]",
                "",
                "[bold]Option 2: Platform Upload[/bold]",
                "  1. Go to competition platform (if available)",
                "  2. Upload [cyan]submission_final.json[/cyan]",
                "  3. Verify GitHub repo link is correct",
                "",
                "[bold green]🔍 Verification:[/bold green]",
                "• Instructor will clone your GitHub repo",
                "• Reproduce your results using your code",
                "• Verify optimizations match documentation",
                "",
                "[dim]💡 Tip: Make sure your README.md explains how to reproduce your results![/dim]",
            ),
            title="✅ Submission Complete",
            border_style="bright_green",
            padding=(1, 2)
        ))
    
    # Helper methods for ratings
    def _get_division_badge(self, division: str) -> str:
        if division == "closed":
            return "🔒 Optimization"
        elif division == "open":
            return "🔓 Innovation"
        return "❓ Unknown"
    
    def _get_event_badge(self, event: str) -> str:
        badges = {
            "latency_sprint": "🏃 Speed",
            "memory_challenge": "🏋️ Size",
            "accuracy_contest": "🎯 Accuracy",
            "all_around": "🏋️‍♂️ Balanced",
            "extreme_push": "🚀 Extreme"
        }
        return badges.get(event, "📊 Custom")
    
    def _get_performance_rating(self, value: float, metric: str) -> str:
        if value >= 10:
            return "⭐⭐⭐ Elite"
        elif value >= 5:
            return "⭐⭐ Strong"
        elif value >= 2:
            return "⭐ Good"
        else:
            return "📊 Baseline"
    
    def _get_accuracy_rating(self, delta: float) -> str:
        if abs(delta) <= 0.5:
            return "⭐⭐⭐ Excellent"
        elif abs(delta) <= 2.0:
            return "⭐⭐ Good"
        elif abs(delta) <= 5.0:
            return "⭐ Acceptable"
        else:
            return "⚠️ High Loss"
    
    def _get_efficiency_rating(self, score: float) -> str:
        if score >= 20:
            return "⭐⭐⭐ Elite"
        elif score >= 10:
            return "⭐⭐ Strong"
        elif score >= 5:
            return "⭐ Good"
        else:
            return "📊 Baseline"

