# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= LANG=C sphinx-build

# Internal variables
SPHINXPROJ    = NumPyEnhancementProposals
SOURCEDIR     = .
BUILDDIR      = _build
ALLSPHINXOPTS = -WT --keep-going -n -d $(SPHINXOPTS)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)

.PHONY: help Makefile index

index:
	python3 tools/build_index.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile index
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
