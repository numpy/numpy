# ReStructured Text
RST2HTML = rst2html.py
RST2LATEX = rst2latex.py
RFLAGS = --generator --time
HTML_FLAGS = --no-xml-declaration
LATEX_FLAGS =
LATEX = pdflatex

# Web pages that need to be made
WEB_PAGES = numpy_swig.html testing.html

# LaTeX files that need to be made
LATEX_FILES = numpy_swig.tex testing.tex

# PDF files that need to be made
PDF_FILES = numpy_swig.pdf testing.pdf

# Default target: documentation
.PHONY : doc
doc: html pdf

# HTML target
.PHONY : html
html: $(WEB_PAGES)

# Rule: %.txt -> %.html
%.html: %.txt
	$(RST2HTML) $(RFLAGS) $(HTML_FLAGS) $< $@

# LaTeX target
.PHONY : tex
tex: $(LATEX_FILES)

# Rule: %.txt -> %.tex
%.tex: %.txt
	$(RST2LATEX) $(RFLAGS) $(LATEX_FLAGS) $< $@

# PDF target
.PHONY : pdf
pdf: $(PDF_FILES)

# Rule: %.tex -> %.pdf
%.pdf: %.tex
	$(LATEX) $<
	$(LATEX) $<

# Clean target
.PHONY : clean
clean:
	$(RM) $(LATEX_FILES)
	$(RM) *.pyc *.aux *.dvi *.log *.out *~
