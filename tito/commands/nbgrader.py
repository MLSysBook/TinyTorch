"""
NBGrader integration commands for TinyTorch.

This module provides commands for managing nbgrader assignments,
auto-grading, and feedback generation.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

from .base import BaseCommand
from ..core.console import console
from ..core.exceptions import TitoError


class NBGraderCommand(BaseCommand):
    """NBGrader integration commands."""
    
    def __init__(self):
        super().__init__()
        self.assignments_dir = Path("assignments")
        self.source_dir = self.assignments_dir / "source"
        self.release_dir = self.assignments_dir / "release"
        self.submitted_dir = self.assignments_dir / "submitted"
        self.autograded_dir = self.assignments_dir / "autograded"
        self.feedback_dir = self.assignments_dir / "feedback"
        
    def init(self):
        """Initialize nbgrader environment."""
        console.print("🔧 Initializing NBGrader environment...")
        
        # Check if nbgrader is installed
        try:
            result = subprocess.run(
                ["nbgrader", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"✅ NBGrader version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("❌ NBGrader not found. Please install with: pip install nbgrader")
            return False
        
        # Create directory structure
        directories = [
            self.assignments_dir,
            self.source_dir,
            self.release_dir,
            self.submitted_dir,
            self.autograded_dir,
            self.feedback_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"📁 Created directory: {directory}")
        
        # Check if nbgrader_config.py exists
        config_file = Path("nbgrader_config.py")
        if config_file.exists():
            console.print(f"✅ Found nbgrader config: {config_file}")
        else:
            console.print("⚠️  NBGrader config not found. Please create nbgrader_config.py")
            return False
        
        # Initialize nbgrader database
        try:
            subprocess.run(
                ["nbgrader", "db", "upgrade"],
                check=True,
                capture_output=True
            )
            console.print("✅ NBGrader database initialized")
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to initialize database: {e}")
            return False
        
        console.print("🎉 NBGrader environment initialized successfully!")
        return True
    
    def generate(self, module: Optional[str] = None, all_modules: bool = False):
        """Generate nbgrader assignment from TinyTorch module."""
        if not module and not all_modules:
            console.print("❌ Must specify either --module or --all")
            return False
        
        modules_to_process = []
        
        if all_modules:
            # Find all modules
            modules_dir = Path("modules")
            if not modules_dir.exists():
                console.print("❌ Modules directory not found")
                return False
            
            modules_to_process = [
                d.name for d in modules_dir.iterdir() 
                if d.is_dir() and not d.name.startswith(".")
            ]
        else:
            modules_to_process = [module]
        
        console.print(f"🔄 Generating assignments for modules: {modules_to_process}")
        
        for module_name in modules_to_process:
            success = self._generate_single_module(module_name)
            if not success:
                console.print(f"❌ Failed to generate assignment for {module_name}")
                return False
        
        console.print("✅ All assignments generated successfully!")
        return True
    
    def _generate_single_module(self, module_name: str) -> bool:
        """Generate assignment from a single module."""
        console.print(f"📝 Generating assignment for module: {module_name}")
        
        # Find the module development file
        module_dir = Path("modules") / module_name
        dev_file = module_dir / f"{module_name}_dev_enhanced.py"
        
        if not dev_file.exists():
            # Try regular dev file
            dev_file = module_dir / f"{module_name}_dev.py"
            
        if not dev_file.exists():
            console.print(f"❌ Module file not found: {dev_file}")
            return False
        
        # Convert to notebook using enhanced generator
        try:
            from ...bin.generate_student_notebooks import NotebookGenerator
            
            # Generate nbgrader version
            generator = NotebookGenerator(use_nbgrader=True)
            
            # First convert .py to .ipynb
            console.print(f"🔄 Converting {dev_file} to notebook...")
            notebook_file = module_dir / f"{module_name}_dev.ipynb"
            
            # Use jupytext to convert
            subprocess.run([
                "jupytext", "--to", "ipynb", str(dev_file)
            ], check=True)
            
            # Process with nbgrader generator
            notebook = generator.process_notebook(notebook_file)
            
            # Save to assignments/source
            assignment_dir = self.source_dir / module_name
            assignment_dir.mkdir(parents=True, exist_ok=True)
            
            assignment_file = assignment_dir / f"{module_name}.ipynb"
            generator.save_student_notebook(notebook, assignment_file)
            
            console.print(f"✅ Assignment created: {assignment_file}")
            return True
            
        except Exception as e:
            console.print(f"❌ Error generating assignment: {e}")
            return False
    
    def validate(self, assignment: str):
        """Validate an assignment."""
        console.print(f"🔍 Validating assignment: {assignment}")
        
        try:
            subprocess.run([
                "nbgrader", "validate", assignment
            ], check=True)
            console.print(f"✅ Assignment {assignment} is valid")
            return True
        except subprocess.CalledProcessError:
            console.print(f"❌ Assignment {assignment} validation failed")
            return False
    
    def release(self, assignment: str):
        """Release assignment to students."""
        console.print(f"🚀 Releasing assignment: {assignment}")
        
        try:
            subprocess.run([
                "nbgrader", "generate_assignment", assignment
            ], check=True)
            console.print(f"✅ Assignment {assignment} released")
            return True
        except subprocess.CalledProcessError:
            console.print(f"❌ Failed to release assignment {assignment}")
            return False
    
    def collect(self, assignment: str):
        """Collect student submissions."""
        console.print(f"📥 Collecting submissions for: {assignment}")
        
        try:
            subprocess.run([
                "nbgrader", "collect", assignment
            ], check=True)
            console.print(f"✅ Submissions collected for {assignment}")
            return True
        except subprocess.CalledProcessError:
            console.print(f"❌ Failed to collect submissions for {assignment}")
            return False
    
    def autograde(self, assignment: str):
        """Auto-grade submissions."""
        console.print(f"🎯 Auto-grading assignment: {assignment}")
        
        try:
            subprocess.run([
                "nbgrader", "autograde", assignment
            ], check=True)
            console.print(f"✅ Assignment {assignment} auto-graded")
            return True
        except subprocess.CalledProcessError:
            console.print(f"❌ Failed to auto-grade {assignment}")
            return False
    
    def feedback(self, assignment: str):
        """Generate feedback for students."""
        console.print(f"📋 Generating feedback for: {assignment}")
        
        try:
            subprocess.run([
                "nbgrader", "generate_feedback", assignment
            ], check=True)
            console.print(f"✅ Feedback generated for {assignment}")
            return True
        except subprocess.CalledProcessError:
            console.print(f"❌ Failed to generate feedback for {assignment}")
            return False
    
    def status(self):
        """Show status of all assignments."""
        console.print("📊 Assignment Status:")
        
        # List source assignments
        if self.source_dir.exists():
            source_assignments = list(self.source_dir.iterdir())
            console.print(f"📚 Source assignments: {len(source_assignments)}")
            for assignment in source_assignments:
                if assignment.is_dir():
                    console.print(f"  - {assignment.name}")
        
        # List released assignments
        if self.release_dir.exists():
            released_assignments = list(self.release_dir.iterdir())
            console.print(f"🚀 Released assignments: {len(released_assignments)}")
            for assignment in released_assignments:
                if assignment.is_dir():
                    console.print(f"  - {assignment.name}")
        
        # List submitted assignments
        if self.submitted_dir.exists():
            submitted_assignments = list(self.submitted_dir.iterdir())
            console.print(f"📥 Submitted assignments: {len(submitted_assignments)}")
            for assignment in submitted_assignments:
                if assignment.is_dir():
                    console.print(f"  - {assignment.name}")
        
        # List graded assignments
        if self.autograded_dir.exists():
            graded_assignments = list(self.autograded_dir.iterdir())
            console.print(f"🎯 Graded assignments: {len(graded_assignments)}")
            for assignment in graded_assignments:
                if assignment.is_dir():
                    console.print(f"  - {assignment.name}")
    
    def batch_release(self):
        """Release all pending assignments."""
        console.print("🚀 Batch releasing all assignments...")
        
        if not self.source_dir.exists():
            console.print("❌ No source assignments found")
            return False
        
        assignments = [d.name for d in self.source_dir.iterdir() if d.is_dir()]
        
        for assignment in assignments:
            console.print(f"🔄 Releasing {assignment}...")
            if not self.release(assignment):
                console.print(f"❌ Failed to release {assignment}")
                return False
        
        console.print("✅ All assignments released successfully!")
        return True
    
    def batch_collect(self):
        """Collect all submitted assignments."""
        console.print("📥 Batch collecting all submissions...")
        
        if not self.release_dir.exists():
            console.print("❌ No released assignments found")
            return False
        
        assignments = [d.name for d in self.release_dir.iterdir() if d.is_dir()]
        
        for assignment in assignments:
            console.print(f"🔄 Collecting {assignment}...")
            if not self.collect(assignment):
                console.print(f"❌ Failed to collect {assignment}")
                return False
        
        console.print("✅ All submissions collected successfully!")
        return True
    
    def batch_autograde(self):
        """Auto-grade all submitted assignments."""
        console.print("🎯 Batch auto-grading all submissions...")
        
        if not self.submitted_dir.exists():
            console.print("❌ No submitted assignments found")
            return False
        
        assignments = [d.name for d in self.submitted_dir.iterdir() if d.is_dir()]
        
        for assignment in assignments:
            console.print(f"🔄 Auto-grading {assignment}...")
            if not self.autograde(assignment):
                console.print(f"❌ Failed to auto-grade {assignment}")
                return False
        
        console.print("✅ All assignments auto-graded successfully!")
        return True
    
    def batch_feedback(self):
        """Generate feedback for all graded assignments."""
        console.print("📋 Batch generating all feedback...")
        
        if not self.autograded_dir.exists():
            console.print("❌ No graded assignments found")
            return False
        
        assignments = [d.name for d in self.autograded_dir.iterdir() if d.is_dir()]
        
        for assignment in assignments:
            console.print(f"🔄 Generating feedback for {assignment}...")
            if not self.feedback(assignment):
                console.print(f"❌ Failed to generate feedback for {assignment}")
                return False
        
        console.print("✅ All feedback generated successfully!")
        return True
    
    def analytics(self, assignment: str):
        """Show analytics for an assignment."""
        console.print(f"📈 Analytics for assignment: {assignment}")
        
        # This would integrate with nbgrader's gradebook
        # For now, show basic file counts
        
        assignment_dir = self.submitted_dir / assignment
        if not assignment_dir.exists():
            console.print(f"❌ No submissions found for {assignment}")
            return False
        
        submissions = list(assignment_dir.iterdir())
        console.print(f"📊 Total submissions: {len(submissions)}")
        
        # Show grading status
        graded_dir = self.autograded_dir / assignment
        if graded_dir.exists():
            graded_submissions = list(graded_dir.iterdir())
            console.print(f"✅ Graded submissions: {len(graded_submissions)}")
            console.print(f"⏳ Pending submissions: {len(submissions) - len(graded_submissions)}")
        
        return True
    
    def report(self, format: str = "csv"):
        """Export grades report."""
        console.print(f"📊 Generating grades report in {format} format...")
        
        try:
            if format == "csv":
                subprocess.run([
                    "nbgrader", "export"
                ], check=True)
                console.print("✅ Grades report exported to grades.csv")
            else:
                console.print(f"❌ Unsupported format: {format}")
                return False
            
            return True
        except subprocess.CalledProcessError:
            console.print("❌ Failed to generate grades report")
            return False 