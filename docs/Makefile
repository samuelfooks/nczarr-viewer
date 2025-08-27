# Makefile for generating presentations from Marp markdown files

# Default target
doall: pdf html sphinx

# Generate PDF from Marp markdown
pdf: dash_nczarr_viewer_presentation.pdf

# Generate HTML from Marp markdown
html: dash_nczarr_viewer_presentation.html

# Generate Sphinx HTML documentation
sphinx: sphinx-html

# Sphinx HTML generation
sphinx-html:
	@echo "Building Sphinx HTML documentation..."
	../.venv/bin/sphinx-build -b html source build/html

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
	rm -f *.pdf
	rm -f dash_nczarr_viewer_presentation.html
	rm -rf build/
	rm -rf _build/
	rm -rf _static/
	rm -rf _sources/
	rm -f *.js
	rm -f .buildinfo
	rm -f objects.inv
	rm -f searchindex.js

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
	@echo "  doall            - Generate both PDF and HTML presentations and Sphinx docs (default)"
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

.PHONY: doall pdf html sphinx sphinx-html presentations clean install-marp check-marp help