/**
 * Work-in-Progress Banner JavaScript
 * Handles banner toggle, collapse, and dismiss functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    const banner = document.getElementById('wip-banner');
    const toggleBtn = document.getElementById('wip-banner-toggle');
    const closeBtn = document.getElementById('wip-banner-close');

    if (!banner) return;

    // Check if banner was previously dismissed
    const dismissed = localStorage.getItem('wip-banner-dismissed');
    if (dismissed === 'true') {
        banner.style.display = 'none';
        return;
    }

    // Check if banner was previously collapsed
    const collapsed = localStorage.getItem('wip-banner-collapsed');
    if (collapsed === 'true') {
        banner.classList.add('collapsed');
        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
            toggleBtn.title = 'Expand banner';
        }
    }

    // Toggle collapse/expand
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            const isCollapsed = banner.classList.contains('collapsed');

            if (isCollapsed) {
                banner.classList.remove('collapsed');
                toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
                toggleBtn.title = 'Collapse banner';
                localStorage.setItem('wip-banner-collapsed', 'false');
            } else {
                banner.classList.add('collapsed');
                toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
                toggleBtn.title = 'Expand banner';
                localStorage.setItem('wip-banner-collapsed', 'true');
            }
        });
    }

    // Dismiss banner completely
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            banner.style.display = 'none';
            localStorage.setItem('wip-banner-dismissed', 'true');
        });
    }

    // Add smooth transitions
    banner.style.transition = 'all 0.3s ease';
});