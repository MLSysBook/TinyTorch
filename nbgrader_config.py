# NBGrader Configuration for TinyTorch ML Systems Course

c = get_config()

# Course Information
c.CourseDirectory.course_id = "tinytorch-ml-systems"
c.CourseDirectory.assignment_id = ""  # Will be set per assignment

# Directory Structure
c.CourseDirectory.root = "."
c.CourseDirectory.source_directory = "assignments/source"
c.CourseDirectory.release_directory = "assignments/release"
c.CourseDirectory.submitted_directory = "assignments/submitted"
c.CourseDirectory.autograded_directory = "assignments/autograded"
c.CourseDirectory.feedback_directory = "assignments/feedback"

# Student Configuration
c.CourseDirectory.student_id = "*"  # All students
c.CourseDirectory.student_id_exclude = ""

# Database Configuration
c.CourseDirectory.db_assignments = []
c.CourseDirectory.db_students = []

# Auto-grading Configuration
c.Execute.timeout = 300  # 5 minutes per cell
c.Execute.allow_errors = True
c.Execute.error_on_timeout = True
c.Execute.interrupt_on_timeout = True

# Solution Removal Configuration
c.ClearSolutions.code_stub = {
    "python": "# YOUR CODE HERE\nraise NotImplementedError()",
    "javascript": "// YOUR CODE HERE\nthrow new Error();",
    "R": "# YOUR CODE HERE\nstop('No Answer Given!')",
    "matlab": "% YOUR CODE HERE\nerror('No Answer Given!')",
    "octave": "% YOUR CODE HERE\nerror('No Answer Given!')",
    "sage": "# YOUR CODE HERE\nraise NotImplementedError()",
    "scala": "// YOUR CODE HERE\n???"
}

# Text Stub for written responses
c.ClearSolutions.text_stub = "YOUR ANSWER HERE"

# Preprocessor Configuration
c.ClearSolutions.begin_solution_delimeter = "BEGIN SOLUTION"
c.ClearSolutions.end_solution_delimeter = "END SOLUTION"
c.ClearSolutions.begin_hidden_tests_delimeter = "BEGIN HIDDEN TESTS"
c.ClearSolutions.end_hidden_tests_delimeter = "END HIDDEN TESTS"

# Enforce Metadata (require proper cell metadata)
c.ClearSolutions.enforce_metadata = True

# Grade Calculation
c.TotalPoints.total_points = 100  # Each module is worth 100 points

# Validation Configuration
c.Validate.ignore_checksums = False

# Feedback Configuration
c.GenerateFeedback.force = False
c.GenerateFeedback.max_dir_size = 1000000  # 1MB max feedback size

# Exchange Configuration (for distributing assignments)
c.Exchange.course_id = "tinytorch-ml-systems"
c.Exchange.timezone = "UTC"

# Notebook Configuration
c.NbGraderConfig.logfile = "nbgrader.log"
c.NbGraderConfig.log_level = "INFO"

# Custom TinyTorch Configuration (stored as comments for reference)
# Each module is worth 100 points:
# - setup: 100 points (easy, 1-2 hours)
# - tensor: 100 points (medium, 2-3 hours) 
# - activations: 100 points (medium, 2-3 hours)
# - layers: 100 points (hard, 3-4 hours)
#
# Grading policy:
# - Partial credit enabled
# - Late penalty: 10% per day
# - Max attempts: 3 