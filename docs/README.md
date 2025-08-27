# NCZarr Viewer Documentation

This directory contains the documentation for the NCZarr Viewer project.

## Building Documentation

To build all documentation, run from the `docs` directory:

```bash
make doall
```

This will:
- Generate Sphinx HTML documentation in `build/html/`
- Generate PDF presentation from Marp markdown
- Generate HTML presentation from Marp markdown

## Individual Build Targets

- `make sphinx` - Build only Sphinx HTML documentation
- `make pdf` - Generate only PDF presentation
- `make html` - Generate only HTML presentation
- `make clean` - Remove all generated files

## Viewing Documentation

After building, you can view the documentation:

- **Sphinx Docs**: Open `build/html/index.html` in your browser
- **HTML Presentation**: Open `dash_nczarr_viewer_presentation.html` in your browser
- **PDF Presentation**: Open `dash_nczarr_viewer_presentation.pdf`

## GitHub Pages

The documentation is configured to work with GitHub Pages. The main entry point (`index.html`) will automatically redirect to the built Sphinx documentation.

## Requirements

- Python 3.10+ with virtual environment
- Marp CLI for presentation generation
- Sphinx with Read the Docs theme

## Installation

```bash
# Install Marp CLI (if not already installed)
make install-marp

# Check if Marp is installed
make check-marp
```
