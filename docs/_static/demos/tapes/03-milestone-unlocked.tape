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
Set Theme "Catppuccin Mocha"
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

# Setup: cd, activate, and complete modules 01-06 (all hidden)
Hide
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 500ms
Type "source activate.sh"
Enter
Sleep 2s
# Complete modules 01-06 to unlock module 07 and milestone 03
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

