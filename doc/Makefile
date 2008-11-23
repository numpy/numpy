# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = LANG=C sphinx-build
PAPER         =

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html web pickle htmlhelp latex changes linkcheck

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  dist      to make a distribution-ready tree"
	@echo "  html      to make standalone HTML files"
	@echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"

clean:
	-rm -rf build/* source/reference/generated

dist: html
	test -d build/latex || make latex
	make -C build/latex all-pdf
	-rm -rf build/dist
	cp -r build/html build/dist
	perl -pi -e 's#^\s*(<li><a href=".*?">NumPy.*?Manual.*?&raquo;</li>)#<li><a href="/">Numpy and Scipy Documentation</a> &raquo;</li>#;' build/dist/*.html build/dist/*/*.html build/dist/*/*/*.html
	cd build/html && zip -9r ../dist/numpy-html.zip .
	cp build/latex/*.pdf build/dist
	cd build/dist && tar czf ../dist.tar.gz *

generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst) ext
	mkdir -p build
	./ext/autosummary_generate.py source/reference/*.rst \
		-p dump.xml -o source/reference/generated 
	touch build/generate-stamp

ext:
	svn co http://sphinx.googlecode.com/svn/contrib/trunk/numpyext ext

html: generate
	mkdir -p build/html build/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html
	python postprocess.py html build/html/*.html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

pickle: generate
	mkdir -p build/pickle build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web build/pickle"
	@echo "to start the sphinx-web server."

web: pickle

htmlhelp: generate
	mkdir -p build/htmlhelp build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

latex: generate
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex
	python postprocess.py tex build/latex/*.tex
	@echo
	@echo "Build finished; the LaTeX files are in build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

coverage: build
	mkdir -p build/coverage build/doctrees
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) build/coverage
	@echo "Coverage finished; see c.txt and python.txt in build/coverage"

changes: generate
	mkdir -p build/changes build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) build/changes
	@echo
	@echo "The overview file is in build/changes."

linkcheck: generate
	mkdir -p build/linkcheck build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."
