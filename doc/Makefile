# Makefile for Sphinx documentation
#

PYVER = 2.7
PYTHON = python$(PYVER)

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = LANG=C sphinx-build
PAPER         =

FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html web pickle htmlhelp latex changes linkcheck \
        dist dist-build gitwash-update

#------------------------------------------------------------------------------

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  html-scipyorg  to make standalone HTML files with scipy.org theming"
	@echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  dist PYVER=... to make a distribution-ready tree"
	@echo "  gitwash-update GITWASH=path/to/gitwash  update gitwash developer docs"
	@echo "  upload USERNAME=... RELEASE=... to upload built docs to docs.scipy.org"

clean:
	-rm -rf build/* source/reference/generated

gitwash-update:
	rm -rf source/dev/gitwash
	install -d source/dev/gitwash
	python $(GITWASH)/gitwash_dumper.py source/dev NumPy \
	    --repo-name=numpy \
	    --github-user=numpy
	cat source/dev/gitwash_links.txt >> source/dev/gitwash/git_links.inc

#------------------------------------------------------------------------------
# Automated generation of all documents
#------------------------------------------------------------------------------

# Build the current numpy version, and extract docs from it.
# We have to be careful of some issues:
# 
# - Everything must be done using the same Python version
# - We must use eggs (otherwise they might override PYTHONPATH on import).
# - Different versions of easy_install install to different directories (!)
#


INSTALL_DIR = $(CURDIR)/build/inst-dist/
INSTALL_PPH = $(INSTALL_DIR)/lib/python$(PYVER)/site-packages:$(INSTALL_DIR)/local/lib/python$(PYVER)/site-packages:$(INSTALL_DIR)/lib/python$(PYVER)/dist-packages:$(INSTALL_DIR)/local/lib/python$(PYVER)/dist-packages
UPLOAD_DIR=/srv/docs_scipy_org/doc/numpy-$(RELEASE)

DIST_VARS=SPHINXBUILD="LANG=C PYTHONPATH=$(INSTALL_PPH) python$(PYVER) `which sphinx-build`" PYTHON="PYTHONPATH=$(INSTALL_PPH) python$(PYVER)" SPHINXOPTS="$(SPHINXOPTS)"

dist:
	make $(DIST_VARS) real-dist

real-dist: dist-build html html-scipyorg
	test -d build/latex || make latex
	make -C build/latex all-pdf
	-test -d build/htmlhelp || make htmlhelp-build
	-rm -rf build/dist
	cp -r build/html-scipyorg build/dist
	cd build/html && zip -9r ../dist/numpy-html.zip .
	cp build/latex/numpy-ref.pdf build/dist
	cp build/latex/numpy-user.pdf build/dist
	cd build/dist && tar czf ../dist.tar.gz *
	chmod ug=rwX,o=rX -R build/dist
	find build/dist -type d -print0 | xargs -0r chmod g+s

dist-build:
	rm -f ../dist/*.egg
	cd .. && $(PYTHON) setup.py bdist_egg
	install -d $(subst :, ,$(INSTALL_PPH))
	$(PYTHON) `which easy_install` --prefix=$(INSTALL_DIR) ../dist/*.egg

upload:
	# SSH must be correctly configured for this to work.
	# Assumes that ``make dist`` was already run
	# Example usage: ``make upload USERNAME=rgommers RELEASE=1.10.1``
	ssh $(USERNAME)@docs.scipy.org mkdir $(UPLOAD_DIR)
	scp build/dist.tar.gz $(USERNAME)@docs.scipy.org:$(UPLOAD_DIR)
	ssh $(USERNAME)@docs.scipy.org tar xvC $(UPLOAD_DIR) \
	    -zf $(UPLOAD_DIR)/dist.tar.gz
	ssh $(USERNAME)@docs.scipy.org mv $(UPLOAD_DIR)/numpy-ref.pdf \
	    $(UPLOAD_DIR)/numpy-ref-$(RELEASE).pdf
	ssh $(USERNAME)@docs.scipy.org mv $(UPLOAD_DIR)/numpy-user.pdf \
	    $(UPLOAD_DIR)/numpy-user-$(RELEASE).pdf
	ssh $(USERNAME)@docs.scipy.org mv $(UPLOAD_DIR)/numpy-html.zip \
	    $(UPLOAD_DIR)/numpy-html-$(RELEASE).zip
	ssh $(USERNAME)@docs.scipy.org rm $(UPLOAD_DIR)/dist.tar.gz
	ssh $(USERNAME)@docs.scipy.org ln -snf numpy-$(RELEASE) /srv/docs_scipy_org/doc/numpy

#------------------------------------------------------------------------------
# Basic Sphinx generation rules for different formats
#------------------------------------------------------------------------------

generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst)
	mkdir -p build
	touch build/generate-stamp

html: generate
	mkdir -p build/html build/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html $(FILES)
	$(PYTHON) postprocess.py html build/html/*.html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

html-scipyorg:
	mkdir -p build/html build/doctrees
	$(SPHINXBUILD) -t scipyorg -b html $(ALLSPHINXOPTS) build/html-scipyorg $(FILES)
	@echo
	@echo "Build finished. The HTML pages are in build/html."

pickle: generate
	mkdir -p build/pickle build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) build/pickle $(FILES)
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web build/pickle"
	@echo "to start the sphinx-web server."

web: pickle

htmlhelp: generate
	mkdir -p build/htmlhelp build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) build/htmlhelp $(FILES)
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

htmlhelp-build: htmlhelp build/htmlhelp/numpy.chm
%.chm: %.hhp
	-hhc.exe $^

qthelp: generate
	mkdir -p build/qthelp build/doctrees
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) build/qthelp $(FILES)

latex: generate
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex $(FILES)
	$(PYTHON) postprocess.py tex build/latex/*.tex
	perl -pi -e 's/\t(latex.*|pdflatex) (.*)/\t-$$1 -interaction batchmode $$2/' build/latex/Makefile
	@echo
	@echo "Build finished; the LaTeX files are in build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

coverage: build
	mkdir -p build/coverage build/doctrees
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) build/coverage $(FILES)
	@echo "Coverage finished; see c.txt and python.txt in build/coverage"

changes: generate
	mkdir -p build/changes build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) build/changes $(FILES)
	@echo
	@echo "The overview file is in build/changes."

linkcheck: generate
	mkdir -p build/linkcheck build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/linkcheck $(FILES)
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) build/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in build/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) build/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C build/texinfo info
	@echo "makeinfo finished; the Info files are in build/texinfo."
