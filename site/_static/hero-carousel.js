/**
 * Hero Carousel for TinyTorch User Journey GIFs
 * Handles sliding between different workflow demonstrations
 */

let currentSlideIndex = 0;

function moveCarousel(direction) {
    const slides = document.querySelectorAll('.carousel-item');
    const dots = document.querySelectorAll('.indicator');

    if (slides.length === 0) return;

    // Hide current slide
    if (slides[currentSlideIndex]) {
        slides[currentSlideIndex].classList.remove('active');
    }
    if (dots.length > 0 && dots[currentSlideIndex]) {
        dots[currentSlideIndex].classList.remove('active');
    }

    // Calculate new slide index
    currentSlideIndex = currentSlideIndex + direction;

    // Wrap around
    if (currentSlideIndex >= slides.length) {
        currentSlideIndex = 0;
    } else if (currentSlideIndex < 0) {
        currentSlideIndex = slides.length - 1;
    }

    // Show new slide
    if (slides[currentSlideIndex]) {
        slides[currentSlideIndex].classList.add('active');
    }
    if (dots.length > 0 && dots[currentSlideIndex]) {
        dots[currentSlideIndex].classList.add('active');
    }
}

function currentSlide(index) {
    const slides = document.querySelectorAll('.carousel-item');
    const dots = document.querySelectorAll('.indicator');

    if (slides.length === 0) return;

    // Hide current slide
    if (slides[currentSlideIndex]) {
        slides[currentSlideIndex].classList.remove('active');
    }
    if (dots.length > 0 && dots[currentSlideIndex]) {
        dots[currentSlideIndex].classList.remove('active');
    }

    // Update index
    currentSlideIndex = index;

    // Show new slide
    if (slides[currentSlideIndex]) {
        slides[currentSlideIndex].classList.add('active');
    }
    if (dots.length > 0 && dots[currentSlideIndex]) {
        dots[currentSlideIndex].classList.add('active');
    }
}

// Optional: Auto-advance carousel every 8 seconds
let autoAdvanceInterval;

function startAutoAdvance() {
    autoAdvanceInterval = setInterval(() => {
        moveCarousel(1);
    }, 8000); // 8 seconds per slide
}

function stopAutoAdvance() {
    if (autoAdvanceInterval) {
        clearInterval(autoAdvanceInterval);
    }
}

// Start auto-advance on page load
document.addEventListener('DOMContentLoaded', function() {
    // Only start auto-advance if carousel exists
    const slides = document.querySelectorAll('.carousel-item');
    if (slides.length > 0) {
        // Start auto-advancing
        startAutoAdvance();

        // Pause auto-advance when user hovers over carousel
        const carousel = document.querySelector('.hero-carousel-compact');
        if (carousel) {
            carousel.addEventListener('mouseenter', stopAutoAdvance);
            carousel.addEventListener('mouseleave', startAutoAdvance);
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') {
                stopAutoAdvance();
                moveCarousel(-1);
                startAutoAdvance();
            } else if (e.key === 'ArrowRight') {
                stopAutoAdvance();
                moveCarousel(1);
                startAutoAdvance();
            }
        });
    }
});
