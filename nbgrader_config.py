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

# Solution delimiters (corrected spelling)
c.ClearSolutions.begin_solution_delimeter = "### BEGIN SOLUTION"
c.ClearSolutions.end_solution_delimeter = "### END SOLUTION"

# Enforce Metadata (require proper cell metadata for grading)
c.ClearSolutions.enforce_metadata = True

# Validation Configuration
c.Validate.ignore_checksums = False

# Logging Configuration
c.Application.log_level = "INFO"

# Database Configuration
c.CourseDirectory.db_url = "sqlite:///gradebook.db"

# Preprocessor Configuration
c.ClearSolutions.begin_solution_delimeter = "### BEGIN SOLUTION"
c.ClearSolutions.end_solution_delimeter = "### END SOLUTION"

# Cell timeout for execution (30 seconds per cell)
c.ExecutePreprocessor.timeout = 30

# Don't allow infinite loops to hang the system
c.ExecutePreprocessor.interrupt_on_timeout = True