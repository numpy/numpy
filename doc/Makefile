# Makefile for Sphinx documentation
#

# PYVER needs to be major.minor, just "3" doesn't work - it will result in
# issues with the amendments to PYTHONPATH and install paths (see DIST_VARS).

# Use explicit "version_info" indexing since make cannot handle colon characters, and
# evaluate it now to allow easier debugging when printing the variable

PYVER:=$(shell python3 -c 'from sys import version_info as v; print("{0}.{1}".format(v[0], v[1]))')
PYTHON = python$(PYVER)

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= LANG=C sphinx-build
PAPER         ?=
# For merging a documentation archive into a git checkout of numpy/doc
# Turn a tag like v1.18.0 into 1.18
# Use sed -n -e 's/patttern/match/p' to return a blank value if no match
TAG ?= $(shell git describe --tag | sed -n -e's,v\([1-9]\.[0-9]*\)\.[0-9].*,\1,p')

FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -WT --keep-going -d build/doctrees $(PAPEROPT_$(PAPER)) \
  $(SPHINXOPTS) source

.PHONY: help clean html web pickle htmlhelp latex changes linkcheck \
		dist dist-build gitwash-update version-check html-build latex-build \
		merge-doc

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
	@echo "  merge-doc TAG=... to clone numpy/doc and archive documentation into it"

clean:
	-rm -rf build/* 
	find . -name generated -type d -prune -exec rm -rf "{}" ";"

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


INSTALL_DIR = $(CURDIR)/build/inst-dist
INSTALL_PPH = $(INSTALL_DIR)/lib/python$(PYVER)/site-packages:$(INSTALL_DIR)/local/lib/python$(PYVER)/site-packages:$(INSTALL_DIR)/lib/python$(PYVER)/dist-packages:$(INSTALL_DIR)/local/lib/python$(PYVER)/dist-packages
UPLOAD_DIR=/srv/docs_scipy_org/doc/numpy-$(RELEASE)

DIST_VARS=SPHINXBUILD="LANG=C PYTHONPATH=$(INSTALL_PPH) python$(PYVER) `which sphinx-build`" PYTHON="PYTHONPATH=$(INSTALL_PPH) python$(PYVER)" 

NUMPYVER:=$(shell $(PYTHON) -c "import numpy; print(numpy.version.git_revision[:10])" 2>/dev/null)
GITVER ?= $(shell cd ..; $(PYTHON) -c "from setup import git_version; \
				print(git_version()[:10])")

version-check:
ifeq "$(GITVER)" "Unknown"
	# @echo sdist build with unlabeled sources
else ifeq ("", "$(NUMPYVER)")
	@echo numpy not found, cannot build documentation without successful \"import numpy\"
	@exit 1
else ifneq ($(NUMPYVER),$(GITVER))
	@echo installed numpy $(NUMPYVER) != current repo git version \'$(GITVER)\'
	@echo use '"make dist"' or '"GITVER=$(NUMPYVER) make $(MAKECMDGOALS) ..."'
	@exit 1
else
	# for testing
	# @echo installed numpy $(NUMPYVER) matches git version $(GITVER); exit 1
endif


dist: build/dist.tar.gz

build/dist.tar.gz:
	make $(DIST_VARS) real-dist

real-dist: dist-build html-build html-scipyorg
	test -d build/latex || make latex-build
	make -C build/latex all-pdf
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

upload: build/dist.tar.gz
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


merge-doc: build/dist.tar.gz
ifeq "$(TAG)" ""
	echo tag "$(TAG)" not of the form 1.18;
	exit 1;
endif
	@# Only clone if the directory does not exist
	@if ! test -d build/merge; then \
		git clone https://github.com/numpy/doc build/merge; \
	fi;
	@# Remove any old content and copy in the new, add it to git
	-rm -rf build/merge/$(TAG)/*
	-mkdir -p build/merge/$(TAG)
	@# -C changes working directory
	tar -C build/merge/$(TAG) -xf build/dist.tar.gz
	git -C build/merge add $(TAG)
	@# For now, the user must do this. If it is onerous, automate it and change
	@# the instructions in doc/HOWTO_RELEASE.rst.txt
	@echo " "
	@echo New documentation archive added to ./build/merge.
	@echo Now add/modify the appropiate section after
	@echo "    <!-- insert here -->"
	@echo in build/merge/index.html,
	@echo then \"git commit\", \"git push\"
	

#------------------------------------------------------------------------------
# Basic Sphinx generation rules for different formats
#------------------------------------------------------------------------------

generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst)
	mkdir -p build
	touch build/generate-stamp

html: version-check html-build
html-build: generate
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

pickle: generate version-check
	mkdir -p build/pickle build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) build/pickle $(FILES)
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web build/pickle"
	@echo "to start the sphinx-web server."

web: pickle

htmlhelp: generate version-check
	mkdir -p build/htmlhelp build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) build/htmlhelp $(FILES)
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

htmlhelp-build: htmlhelp build/htmlhelp/numpy.chm
%.chm: %.hhp
	-hhc.exe $^

qthelp: generate version-check
	mkdir -p build/qthelp build/doctrees
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) build/qthelp $(FILES)

latex: version-check latex-build
latex-build: generate
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex $(FILES)
	$(PYTHON) postprocess.py tex build/latex/*.tex
	perl -pi -e 's/\t(latex.*|pdflatex) (.*)/\t-$$1 -interaction batchmode $$2/' build/latex/Makefile
	@echo
	@echo "Build finished; the LaTeX files are in build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

coverage: build version-check
	mkdir -p build/coverage build/doctrees
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) build/coverage $(FILES)
	@echo "Coverage finished; see c.txt and python.txt in build/coverage"

changes: generate version-check
	mkdir -p build/changes build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) build/changes $(FILES)
	@echo
	@echo "The overview file is in build/changes."

linkcheck: generate version-check
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
