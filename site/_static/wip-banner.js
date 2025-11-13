/**
 * Work-in-Progress Banner JavaScript
 * Handles banner injection and toggle/collapse functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Banner injection: Create banner dynamically if not present in HTML
    // This allows the banner to work even if not included in the page template
    let banner = document.getElementById('wip-banner');
    if (!banner) {
        const bannerHTML = `
            <div class="wip-banner" id="wip-banner">
                <div class="wip-banner-content">
                    <div class="wip-banner-title">
                        <span class="icon">🚧</span>
                        <span>Under Construction</span>
                        <span class="icon">🚧</span>
                    </div>
                    <div class="wip-banner-description">
                        TinyTorch is currently under active development. Public release planned for December 2025. Expect changes and improvements!
                    </div>
                </div>
                <button class="wip-banner-toggle" id="wip-banner-toggle" title="Collapse banner" aria-label="Toggle banner">▲</button>
            </div>
        `;
        document.body.insertAdjacentHTML('afterbegin', bannerHTML);
        banner = document.getElementById('wip-banner');
    }

    const toggleBtn = document.getElementById('wip-banner-toggle');

    if (!banner) return;

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

    // Add smooth transitions
    banner.style.transition = 'all 0.3s ease';
});