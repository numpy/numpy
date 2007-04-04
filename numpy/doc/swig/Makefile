# SWIG
INTERFACES = Vector.i Matrix.i Tensor.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = $(INTERFACES:.i=.py      )

# ReStructured Text
RST2HTML = rst2html.py
RST2LATEX = rst2latex.py
RFLAGS = --generator --time
HTML_FLAGS = --no-xml-declaration
LATEX_FLAGS =
LATEX = pdflatex

# Web pages that need to be made
WEB_PAGES = numpy_swig.html

# LaTeX files that need to be made
LATEX_FILES = numpy_swig.tex

# PDF files that need to be made
PDF_FILES = numpy_swig.pdf

# List all of the subdirectories here for recursive make
SUBDIRS = 

all: $(WRAPPERS) Vector.cxx Vector.h Matrix.cxx Matrix.h Tensor.cxx Tensor.h
	./setup.py build

test: all
	test1D.py
	test2D.py
	test3D.py

doc: html pdf

%_wrap.cxx: %.i %.h numpy.i
	swig -c++ -python $<

html: $(WEB_PAGES)

%.html: %.txt
	$(RST2HTML) $(RFLAGS) $(HTML_FLAGS) $< $@

tex: $(LATEX_FILES)

%.tex: %.txt
	$(RST2LATEX) $(RFLAGS) $(LATEX_FLAGS) $< $@

pdf: $(PDF_FILES)

%.pdf: %.tex
	$(LATEX) $<
	$(LATEX) $<

clean:
	$(RM) -r build
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)
	$(RM) $(LATEX_FILES)
	$(RM) *.pyc *.aux *.dvi *.log *.out *~

.PHONY : all test doc html tex pdf clean
