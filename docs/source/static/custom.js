// Custom JavaScript for NCZarr Viewer Documentation

// Add custom functionality after the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add custom classes to external links
    const externalLinks = document.querySelectorAll('a[href^="http"]');
    externalLinks.forEach(link => {
        link.classList.add('external');
    });
    
    // Add custom styling to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        block.style.fontSize = '0.9em';
    });
});
