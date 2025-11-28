# VHS Tape: 🏆 Milestone Unlocked - Recreate ML History
# Purpose: Show milestone unlock and training with historical context
# Duration: 30-35 seconds

Output "03-milestone-unlocked.gif"

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
Env PS1 "🔥 "

# Window bar for realistic terminal look
Set WindowBar Colorful
Set TypingSpeed 100ms

# Complete module that unlocks milestone
Type "tito module complete 07"
Sleep 400ms
Enter
Wait Line "🔥"  # Wait for module completion
Sleep 3s

# Check available milestones
Type "tito milestone list"
Sleep 400ms
Enter
Wait Line "🔥"  # Wait for list output
Sleep 3s

# Run milestone 03 (MLP Revival 1986) - SKIP CHECKS FOR DEMO
Type "tito milestone run 03 --skip-checks"
Sleep 400ms
Enter
Sleep 500ms
Type ""  # Press Enter to proceed past "Press Enter to begin"
Enter
Wait 60s Line "▶"  # Wait for milestone to complete (max 60s)
Sleep 3s

# Celebrate the achievement
Type "# You recreated Rumelhart's 1986 breakthrough! 🏆"
Sleep 3s

