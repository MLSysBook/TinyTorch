# VHS Tape: 🏆 Milestone Unlocked - Recreate ML History
# Purpose: Show milestone unlock and training with historical context
# Duration: 30-35 seconds

Output "gifs/03-milestone-unlocked.gif"

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

# Window bar for realistic terminal look
Set WindowBar Colorful
Set TypingSpeed 100ms

# Setup: cd, activate, and prepare modules 01-07 (all hidden)
Hide
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 500ms
Type "source activate.sh"
Enter
Sleep 2s
# Reset and complete modules 01-06 to set up for module 07
Type "tito src export 01_tensor 02_activations 03_layers 04_losses 05_autograd 06_optimizers"
Enter
Sleep 10s
Show

# Show where we are in the journey
Type "tito module status"
Sleep 400ms
Enter
Sleep 3s

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

