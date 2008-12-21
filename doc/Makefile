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
	@echo "  upload USER=...  to upload results to docs.scipy.org"

clean:
	-rm -rf build/* source/reference/generated

upload:
	@test -e build/dist || { echo "make dist is required first"; exit 1; }
	@test output-is-fine -nt build/dist || { \
		echo "Review the output in build/dist, and do 'touch output-is-fine' before uploading."; exit 1; }
	rsync -r -z --delete-after -p build/dist/ $(USER)@docs.scipy.org:/home/docserver/www-root/doc/numpy/

dist: html
	test -d build/latex || make latex
	make -C build/latex all-pdf
	-test -d build/htmlhelp || make htmlhelp-build
	-rm -rf build/dist
	cp -r build/html build/dist
	perl -pi -e 's#^\s*(<li><a href=".*?">NumPy.*?Manual.*?&raquo;</li>)#<li><a href="/">Numpy and Scipy Documentation</a> &raquo;</li>#;' build/dist/*.html build/dist/*/*.html build/dist/*/*/*.html
	cd build/html && zip -9r ../dist/numpy-html.zip .
	cp build/latex/numpy-*.pdf build/dist
	-zip build/dist/numpy-chm.zip build/htmlhelp/numpy.chm
	cd build/dist && tar czf ../dist.tar.gz *
	chmod ug=rwX,o=rX -R build/dist
	find build/dist -type d -print0 | xargs -0r chmod g+s

generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst)
	mkdir -p build
	./sphinxext/autosummary_generate.py source/reference/*.rst \
		-p dump.xml -o source/reference/generated 
	touch build/generate-stamp

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

htmlhelp-build: htmlhelp build/htmlhelp/numpy.chm
%.chm: %.hhp
	-hhc.exe $^

latex: generate
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex
	python postprocess.py tex build/latex/*.tex
	perl -pi -e 's/\t(latex.*|pdflatex) (.*)/\t-$$1 -interaction batchmode $$2/' build/latex/Makefile
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
