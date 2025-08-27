# Makefile for generating presentations from Marp markdown files

# Default target
all: pdf html sphinx sphinx-deploy

# Generate PDF from Marp markdown
pdf: dash_nczarr_viewer_presentation.pdf

# Generate HTML from Marp markdown
html: dash_nczarr_viewer_presentation.html

# Generate Sphinx HTML documentation
sphinx: sphinx-html

# Sphinx HTML generation
sphinx-html:
	@echo "Building Sphinx HTML documentation..."
	../.venv/bin/sphinx-build -b html -D html_baseurl=https://samuelfooks.github.io/nczarr-viewer/build/html/ source build/html

# Copy Sphinx index to root for GitHub Pages
sphinx-deploy: sphinx-html
	@echo "Copying Sphinx index to docs root for GitHub Pages..."
	cp build/html/index.html ./sphinx-index.html
	@echo "GitHub Pages index.html is ready"

# PDF generation
dash_nczarr_viewer_presentation.pdf: dash_nczarr_viewer_presentation.md
	@echo "Generating PDF from $<..."
	marp --theme-set ../assets/marp.css --pdf $< --output $@

# HTML generation
dash_nczarr_viewer_presentation.html: dash_nczarr_viewer_presentation.md
	@echo "Generating HTML from $<..."
	marp --theme-set ../assets/marp.css --html $< --output $@

# Generate both formats
presentations: pdf html
	@echo "Generated both PDF and HTML presentations"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f *.pdf *.html
	rm -rf build/

# Install Marp CLI (if not already installed)
install-marp:
	@echo "Installing Marp CLI..."
	npm install -g @marp-team/marp-cli

# Check if Marp is installed
check-marp:
	@which marp || (echo "Marp CLI not found. Run 'make install-marp' to install it." && exit 1)

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Generate both PDF and HTML presentations and Sphinx docs (default)"
	@echo "  pdf              - Generate PDF presentation"
	@echo "  html             - Generate HTML presentation"
	@echo "  sphinx           - Generate Sphinx HTML documentation"
	@echo "  sphinx-html      - Generate Sphinx HTML documentation"
	@echo "  presentations    - Generate both PDF and HTML presentations"
	@echo "  clean            - Remove generated files"
	@echo "  install-marp     - Install Marp CLI"
	@echo "  check-marp       - Check if Marp CLI is installed"
	@echo "  help             - Show this help message"

# Ensure Marp is installed before generating files
pdf html: check-marp

.PHONY: all pdf html sphinx sphinx-html presentations clean install-marp check-marp help