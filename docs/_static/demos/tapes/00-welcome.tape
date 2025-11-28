# VHS Tape: 🔥 Welcome to TinyTorch - The Big Logo
# Purpose: Show the beautiful branded welcome screen with full command reference
# Duration: 10-12 seconds

Output "gifs/00-welcome.gif"

# Window bar for realistic terminal look (must be at top)
Set WindowBar Colorful

# Taller dimensions to show full tito output
Set Width 1280
Set Height 1000
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme "Catppuccin Mocha"
Set Padding 30
Set Framerate 30
Set TypingSpeed 100ms

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "

# Activate TinyTorch environment (hidden from recording)
Hide
Type "source activate.sh"
Enter
Sleep 2s
Show

# Show the beautiful welcome screen
Type "tito"
Sleep 400ms
Enter
Sleep 8s

# Final message
Type "# Welcome to TinyTorch! 🔥"
Sleep 3s
