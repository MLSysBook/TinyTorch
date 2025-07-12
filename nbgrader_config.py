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

# Enforce Metadata (require proper cell metadata)
c.ClearSolutions.enforce_metadata = True

# Validation Configuration
c.Validate.ignore_checksums = False

# Logging Configuration
c.NbGrader.log_level = "INFO" 