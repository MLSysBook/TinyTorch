# VHS Tape: ⚡ Zero to Ready in 30 Seconds
# Purpose: Show how easy TinyTorch setup is
# Duration: 15-20 seconds

Output "gifs/01-zero-to-ready.gif"

# Window bar for realistic terminal look (must be at top)
Set WindowBar Colorful

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme "Catppuccin Mocha"
Set Padding 60
Set Framerate 30
Set TypingSpeed 100ms
Set LoopOffset 0%

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "

# Start in /tmp for clean setup
Type "cd /tmp"
Sleep 300ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 500ms

# Clone the TinyTorch repository
Type "git clone https://github.com/mlsysbook/TinyTorch.git"
Sleep 400ms
Enter
Wait+Line@120s /profvjreddi/  # Wait for clone to complete (up to 120s)
Sleep 500ms

# Navigate into the project
Type "cd TinyTorch"
Sleep 300ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 500ms

# Setup automatically detects your environment (conda/venv)
Type "./setup-environment.sh"
Sleep 400ms
Enter
Wait+Line@120s /profvjreddi/  # Setup creates virtual environment (up to 120s)
Sleep 1s

# Activate the environment to get tito command
Type "source activate.sh"
Sleep 300ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 1s

# Check module status - shows all 20 modules ready to complete
Type "tito module status"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 2s

# Learn the story and motivation
Type "tito logo"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 3s

# Ready to build!
Type "# Ready to build your ML framework from scratch 🔥"
Sleep 2s
