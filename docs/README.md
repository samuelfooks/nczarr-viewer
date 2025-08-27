# NCZarr Viewer Documentation

This directory contains the documentation for the NCZarr Viewer project.

## Building Documentation

### Prerequisites
- Python 3.10+
- Sphinx
- Pandoc (for presentation generation)

### Build Commands

```bash
# Build Sphinx HTML documentation
make html

# Build PDF presentation
make pdf

# Build HTML presentation
make html-presentation

# Clean build artifacts
make clean
```

### Generated Files

After building, you'll find:

- **Sphinx HTML**: Open `build/html/index.html` in your browser
- **HTML Presentation**: Open `nczarr_viewer_presentation.html` in your browser
- **PDF Presentation**: Open `nczarr_viewer_presentation.pdf`

## Documentation Structure

- `source/` - Sphinx source files
- `build/` - Generated documentation (after building)
- `nczarr_viewer_presentation.md` - Main presentation file
- `Makefile` - Build configuration

## Notes

- The Sphinx documentation is configured to work with GitHub Pages
- Presentations are generated using Pandoc from Markdown
- The `_modules/` directory contains auto-generated module documentation
