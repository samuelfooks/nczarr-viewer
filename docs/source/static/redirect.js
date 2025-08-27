// Redirect script for GitHub Pages compatibility
(function() {
    // Function to handle internal navigation
    function handleInternalNavigation() {
        // Get all internal links in the document
        const links = document.querySelectorAll('a[href^="./"]');
        
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                
                // If it's a relative link that might need redirection
                if (href && href.startsWith('./')) {
                    // Check if we're on GitHub Pages (the current path structure)
                    const currentPath = window.location.pathname;
                    
                    // If we're already in the build/html directory, no need to redirect
                    if (currentPath.includes('/build/html/')) {
                        return;
                    }
                    
                    // If we're at the docs root, redirect to build/html
                    if (currentPath.endsWith('/docs/') || currentPath.endsWith('/docs')) {
                        e.preventDefault();
                        const newHref = href.replace('./', './build/html/');
                        window.location.href = newHref;
                    }
                }
            });
        });
    }
    
    // Function to handle form submissions (like search)
    function handleForms() {
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', function(e) {
                const action = this.getAttribute('action');
                if (action && action.startsWith('./')) {
                    const currentPath = window.location.pathname;
                    if (currentPath.endsWith('/docs/') || currentPath.endsWith('/docs')) {
                        const newAction = action.replace('./', './build/html/');
                        this.setAttribute('action', newAction);
                    }
                }
            });
        });
    }
    
    // Function to handle GitHub Pages specific navigation
    function handleGitHubPagesNavigation() {
        const currentPath = window.location.pathname;
        
        // If we're at the root of the repository on GitHub Pages
        if (currentPath === '/nczarr-viewer/' || currentPath === '/nczarr-viewer') {
            // Redirect to the docs
            window.location.href = '/nczarr-viewer/docs/';
            return;
        }
        
        // If we're at the docs root, redirect to the built Sphinx docs
        if (currentPath === '/nczarr-viewer/docs/' || currentPath === '/nczarr-viewer/docs') {
            window.location.href = '/nczarr-viewer/docs/build/html/';
            return;
        }
    }
    
    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            handleInternalNavigation();
            handleForms();
            handleGitHubPagesNavigation();
        });
    } else {
        handleInternalNavigation();
        handleForms();
        handleGitHubPagesNavigation();
    }
})();
