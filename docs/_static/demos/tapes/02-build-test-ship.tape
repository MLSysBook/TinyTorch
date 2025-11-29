# VHS Tape: 🔨 Build → Test → Ship (THE CONVERSION GIF)
# Purpose: Show complete module workflow with detailed test output
# Duration: 30-35 seconds

Output "gifs/02-build-test-ship.gif"

# Window bar for realistic terminal look
Set WindowBar Colorful

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme { "name": "TinyTorch", "black": "#1E1E2E", "red": "#F38BA8", "green": "#A6E3A1", "yellow": "#F9E2AF", "blue": "#89B4FA", "magenta": "#CBA6F7", "cyan": "#94E2D5", "white": "#CDD6F4", "brightBlack": "#585B70", "brightRed": "#F38BA8", "brightGreen": "#A6E3A1", "brightYellow": "#F9E2AF", "brightBlue": "#89B4FA", "brightMagenta": "#CBA6F7", "brightCyan": "#94E2D5", "brightWhite": "#CDD6F4", "background": "#1E1E2E", "foreground": "#CDD6F4", "selection": "#585B70", "cursor": "#F5E0DC" }
Set Padding 30
Set Framerate 30
Set Margin 20
Set MarginFill "#1E1E2E"
Set BorderRadius 10
Set LoopOffset 10%

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "
Set TypingSpeed 100ms

# Opening: Show what this demo is about
Type "# Build → Test → Ship 🔨"
Sleep 2s
Enter
Sleep 500ms

# Show everything - users see the full setup
Type "cd /Users/VJ/GitHub/TinyTorch"
Sleep 400ms
Enter
Sleep 1s

Type "source activate.sh"
Sleep 400ms
Enter
Sleep 3s

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

Type "sig(Tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))"
Sleep 400ms
Enter
Sleep 2s

Type "exit()"
Sleep 300ms
Enter
Sleep 1s

# Final message
Type "# You didn't just import it. You BUILT it! 🎉"
Sleep 3s

