# VHS Tape: ⚡ Zero to Ready in 30 Seconds
# Purpose: Show how easy TinyTorch setup is
# Duration: 15-20 seconds

Output "01-zero-to-ready.gif"

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set Theme "Catppuccin Mocha"
Set Padding 30
Set Framerate 30

# Window bar for realistic terminal look
Set WindowBar Colorful
Set TypingSpeed 100ms

# Start in /tmp for demo purposes
Type "cd /tmp"
Sleep 300ms
Enter
Sleep 500ms

# Clone repository
Type "git clone https://github.com/mlsysbook/TinyTorch.git"
Sleep 400ms
Enter
Sleep 3s

Type "cd TinyTorch"
Sleep 300ms
Enter
Sleep 500ms

# Setup environment (measured: ~9s)
Type "./setup-environment.sh"
Sleep 400ms
Enter
Sleep 9s

# Activate (instant)
Type "source activate.sh"
Sleep 300ms
Enter
Sleep 1s

# Check module status (measured: ~0.2s, show for 3s)
Type "tito module status"
Sleep 400ms
Enter
Sleep 3s

# Show the TinyTorch brand and philosophy (measured: ~0.1s, show for 6s)
Type "tito logo"
Sleep 400ms
Enter
Sleep 6s

# Ready to start!
Type "# Start tiny. Go deep. Build big. 🔥"
Sleep 2s
