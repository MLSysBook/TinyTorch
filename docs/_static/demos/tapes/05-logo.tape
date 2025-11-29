# VHS Tape: 🔥 The TinyTorch Story - Philosophy & Vision
# Purpose: Show the beautiful story behind TinyTorch with tito logo
# Duration: 45-50 seconds

Output "gifs/05-logo.gif"

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

# ==============================================================================
# SETUP PHASE (HIDDEN): Like a unit test, set up the environment cleanly first
# ==============================================================================
Hide

# Navigate to project and activate environment
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 1s
Type "source activate.sh"
Enter
Sleep 3s

Show
# ==============================================================================
# DEMO BEGINS (VISIBLE): Now show the philosophy and story
# ==============================================================================

# Opening comment explaining what we'll demonstrate
Type "# The Story Behind TinyTorch 🔥"
Sleep 2s
Enter
Sleep 500ms

# Show the beautiful logo and story
Type "tito logo"
Sleep 400ms
Enter
Sleep 5s  # Let the logo appear

# Scroll down to show the flame symbolism
Type ""
Enter
Sleep 3s

# Continue scrolling through the philosophy
Type ""
Enter
Sleep 3s

# Scroll to show "Why Tiny?"
Type ""
Enter
Sleep 3s

# Scroll to show "Why Torch?"
Type ""
Enter
Sleep 3s

# Scroll to show the hidden network
Type ""
Enter
Sleep 3s

# Scroll to show the philosophy section
Type ""
Enter
Sleep 3s

# Scroll to show Professor Reddi's message
Type ""
Enter
Sleep 4s

# Scroll to the final message
Type ""
Enter
Sleep 3s

# Final comment
Type "# Don't just import it. Build it. 🔥"
Sleep 3s
