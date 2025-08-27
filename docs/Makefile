# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build Sphinx HTML documentation
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build PDF presentation
presentationpdf: nczarr_viewer_presentation.pdf

# Build HTML presentation
presentationhtml: nczarr_viewer_presentation.html

# Rule to build PDF from markdown using pandoc
nczarr_viewer_presentation.pdf: nczarr_viewer_presentation.md
	pandoc $< -o $@ --pdf-engine=xelatex -V geometry:margin=1in

# Rule to build HTML from markdown using pandoc
nczarr_viewer_presentation.html: nczarr_viewer_presentation.md
	pandoc $< -o $@ --standalone --css=../assets/marp.css
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	touch $(BUILDDIR)/html/.nojekyll
clean:
	rm -rf $(BUILDDIR)/*
	rm -f nczarr_viewer_presentation.html