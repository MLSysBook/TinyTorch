# VHS Tape: 🔨 Build → Test → Ship (THE CONVERSION GIF)
# Purpose: Show complete module workflow with detailed test output
# Duration: 30-35 seconds

Output "gifs/02-build-test-ship.gif"

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme "Catppuccin Mocha"
Set Padding 30
Set Framerate 30

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "

# Window bar for realistic terminal look
Set WindowBar Colorful
Set TypingSpeed 100ms

# Activate TinyTorch environment
Hide
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 500ms
Show
Type "source activate.sh"
Sleep 400ms
Enter
Sleep 2s

# Start module 02
Type "tito module start 02"
Sleep 400ms
Enter
Sleep 3s

# Show working indicator
Type "# ... working in Jupyter Lab ..."
Sleep 1500ms
Enter
Sleep 800ms

# Complete module - THE MONEY SHOT
Type "tito module complete 02"
Sleep 400ms
Enter
Sleep 8s  # Long pause to read all test output

# Verify it works in Python
Type "python3"
Sleep 400ms
Enter
Sleep 1s

Type "from tinytorch import Sigmoid"
Sleep 400ms
Enter
Sleep 500ms

Type "from tinytorch import Tensor"
Sleep 400ms
Enter
Sleep 500ms

Type "sig = Sigmoid()"
Sleep 400ms
Enter
Sleep 500ms

Type "sig(Tensor([0.0]))"
Sleep 400ms
Enter
Sleep 2s

Type "exit()"
Sleep 300ms
Enter
Sleep 1s

# Final message
Type "# You didn't import it. You BUILT it! 🎉"
Sleep 3s

