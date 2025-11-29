# VHS Tape: 🌍 Share Your Journey - Community & Progress
# Purpose: Show community engagement and progress tracking
# Duration: 20-25 seconds

Output "gifs/04-share-journey.gif"

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

# Check progress
Type "tito module status"
Sleep 400ms
Enter
Sleep 3s
Sleep 3s

# Join community
Type "tito community join"
Sleep 400ms
Enter
Sleep 2s  # Wait for Question 1/3 (GitHub username)

# Question 1: GitHub username (required)
Type "profvjreddi"
Sleep 400ms
Enter
Sleep 2s  # Wait for Question 2/3 (Country)

# Question 2: Country (optional)
Type "United States"
Sleep 400ms
Enter
Sleep 2s  # Wait for Question 3/3 (Institution)

# Question 3: Institution (optional)
Type "Harvard University"
Sleep 400ms
Enter
Sleep 3s  # Wait for confirmation

# Share achievement after more progress
Type "tito community update"
Sleep 400ms
Enter
Sleep 3s
Sleep 3s

# Final message
Type "# Join 2,847 builders worldwide! 🌍"
Sleep 3s

