# NBGrader Configuration for TinyTorch ML Systems Course

c = get_config()

# Course Information
c.CourseDirectory.course_id = "tinytorch-ml-systems"

# Directory Structure
c.CourseDirectory.source_directory = "assignments/source"
c.CourseDirectory.release_directory = "assignments/release"
c.CourseDirectory.submitted_directory = "assignments/submitted"
c.CourseDirectory.autograded_directory = "assignments/autograded"
c.CourseDirectory.feedback_directory = "assignments/feedback"

# Solution Removal Configuration
c.ClearSolutions.code_stub = {
    "python": "# YOUR CODE HERE\nraise NotImplementedError()"
}

# Text Stub for written responses
c.ClearSolutions.text_stub = "YOUR ANSWER HERE"

c.ClearSolutions.begin_solution_delimeter = "### BEGIN SOLUTION"
c.ClearSolutions.end_solution_delimeter = "### END SOLUTION"
c.ClearSolutions.begin_text_delimeter = "### BEGIN TEXT SOLUTION"
c.ClearSolutions.end_text_delimeter = "### END TEXT SOLUTION"

# Enforce Metadata (require proper cell metadata for grading)
c.ClearSolutions.enforce_metadata = True

# Validation Configuration
c.Validate.ignore_checksums = False

# Logging Configuration
c.NbGrader.log_level = "INFO"

# Assignment and Student Configuration
c.AssignApp.generate_unique_ids = True  # Ensure uniqueness of cell IDs
c.AssignApp.create_assignment = True

# Autograder Configuration
c.AutogradeApp.update = True  # Update existing autograded notebooks
c.AutogradeApp.create = True  # Create new entries if they don't exist

# Feedback Configuration
c.FeedbackApp.generate_feedback = True
c.FeedbackApp.output_dir = "modules/feedback"

# Student IDs (optional: helps when testing locally or in custom workflows)
c.Gradebook.db_url = "sqlite:///gradebook.db"

# Hide hidden test cells in release notebooks
c.ClearSolutions.remove_hidden_tests = True

# Set default language for code cells (if mixed notebooks used)
c.ClearSolutions.language = "python"
