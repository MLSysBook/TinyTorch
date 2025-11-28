# VHS Tape: 🌍 Share Your Journey - Community & Progress
# Purpose: Show community engagement and progress tracking
# Duration: 20-25 seconds

Output "04-share-journey.gif"

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
Sleep 2s  # Wait for first prompt

# Simulate interactive prompts
Type "United States"
Sleep 400ms
Enter
Sleep 2s  # Wait for next prompt

Type "Harvard University"
Sleep 400ms
Enter
Sleep 2s  # Wait for next prompt

Type "intermediate"
Sleep 400ms
Enter
Sleep 3s
Sleep 3s

# Share achievement after more progress
Type "tito community update"
Sleep 400ms
Enter
Sleep 3s
Sleep 3s

# Final message
Type "# Join 2,847 builders worldwide! 🌍"
Sleep 3s

