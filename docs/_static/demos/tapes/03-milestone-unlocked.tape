# VHS Tape: 🏆 Milestone Unlocked - Recreate ML History
# Purpose: Show milestone unlock and training with historical context
# Duration: 30-35 seconds

Output "gifs/03-milestone-unlocked.gif"

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
Type "# Milestone: Recreate ML History 🏆"
Sleep 2s
Enter
Sleep 500ms

# Show cd and activate, then fast-forward module completions (hidden)
Type "cd /Users/VJ/GitHub/TinyTorch"
Sleep 400ms
Enter
Sleep 1s

Type "source activate.sh"
Sleep 400ms
Enter
Sleep 3s

# Fast-forward: Complete modules 01-06 to unlock module 07 (hidden for speed)
Hide
Type "tito module complete 01"
Enter
Sleep 5s
Type "tito module complete 02"
Enter
Sleep 5s
Type "tito module complete 03"
Enter
Sleep 5s
Type "tito module complete 04"
Enter
Sleep 5s
Type "tito module complete 05"
Enter
Sleep 5s
Type "tito module complete 06"
Enter
Sleep 5s
Show

# Show where we are in the journey (modules 01-06 completed)
Type "tito module status"
Sleep 400ms
Enter
Sleep 4s

# Complete module 07 which unlocks milestone 03
Type "tito module complete 07"
Sleep 400ms
Enter
Sleep 10s

# Check available milestones
Type "tito milestones list"
Sleep 400ms
Enter
Sleep 3s

# Run milestone 03 (MLP Revival 1986)
Type "tito milestones run 03"
Sleep 400ms
Enter
Sleep 2s
Type ""  # Press Enter to proceed past "Press Enter to begin"
Enter
Sleep 60s  # Wait for milestone to complete
Sleep 3s

# Celebrate the achievement
Type "# You recreated Rumelhart's 1986 breakthrough! 🏆"
Sleep 3s

