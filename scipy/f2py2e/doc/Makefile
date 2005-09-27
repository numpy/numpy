# Makefile for compiling f2py2e documentation (dvi, ps, html)
# Pearu Peterson <pearu@ioc.ee>

REL=4
TOP = usersguide
LATEXSRC = bugs.tex  commands.tex  f2py2e.tex  intro.tex  notes.tex  signaturefile.tex
MAINLATEX = f2py2e

LATEX = latex
PDFLATEX = pdflatex

COLLECTINPUT = ./collectinput.py
INSTALLDATA = install -m 644 -c

TTH = tth
TTHFILTER = sed -e "s/{{}\\\verb@/\\\texttt{/g" | sed -e "s/@{}}/}/g" | $(TTH) -L$(MAINLATEX) -i
TTHFILTER2 = sed -e "s/{{}\\\verb@/\\\texttt{/g" | sed -e "s/@{}}/}/g" | $(TTH) -Lpython9 -i
TTHFILTER3 = sed -e "s/{{}\\\verb@/\\\texttt{/g" | sed -e "s/@{}}/}/g" | $(TTH) -Lfortranobject -i
TTHMISSING = "\
***************************************************************\n\
Warning:        Could not find tth (a TeX to HTML translator)  \n\
                or an error arised was by tth\n\
You can download tth from http://hutchinson.belmont.ma.us/tth/  \n\
or\n\
use your favorite LaTeX to HTML translator on file tmp_main.tex\n\
***************************************************************\
"

all: dvi ps html clean
$(MAINLATEX).dvi: $(LATEXSRC)
	$(LATEX) $(MAINLATEX).tex
	$(LATEX) $(MAINLATEX).tex
	$(LATEX) $(MAINLATEX).tex
	$(PDFLATEX) $(MAINLATEX).tex
$(TOP).dvi: $(MAINLATEX).dvi
	cp -f $(MAINLATEX).dvi $(TOP).dvi
	mv -f $(MAINLATEX).pdf $(TOP).pdf
$(TOP).ps: $(TOP).dvi
	dvips $(TOP).dvi -o
$(TOP).html: $(LATEXSRC)
	$(COLLECTINPUT) < $(MAINLATEX).tex > tmp_$(MAINLATEX).tex
	@test `which $(TTH)` && cat tmp_$(MAINLATEX).tex | $(TTHFILTER) > $(TOP).html\
	 || echo -e $(TTHMISSING)
dvi: $(TOP).dvi
ps: $(TOP).ps
	gzip -f $(TOP).ps
html: $(TOP).html

python9:
	cp -f python9.tex f2python9-final/src/
	cd f2python9-final && mk_html.sh
	cd f2python9-final && mk_ps.sh
	cd f2python9-final && mk_pdf.sh
pyfobj:
	$(LATEX) fortranobject.tex
	$(LATEX) fortranobject.tex
	$(LATEX) fortranobject.tex
	@test `which $(TTH)` && cat fortranobject.tex | $(TTHFILTER3) > pyfobj.html\
	 || echo -e $(TTHMISSING)
	dvips fortranobject.dvi -o pyfobj.ps
	gzip -f pyfobj.ps
	pdflatex fortranobject.tex
	mv fortranobject.pdf pyfobj.pdf

WWWDIR=/net/cens/home/www/unsecure/projects/f2py2e/
wwwpage: all
	$(INSTALLDATA) index.html $(TOP).html $(TOP).ps.gz $(TOP).dvi $(TOP).pdf \
	Release-$(REL).x.txt ../NEWS.txt win32_notes.txt $(WWWDIR)
	$(INSTALLDATA) pyfobj.{ps.gz,pdf,html} $(WWWDIR)
	$(INSTALLDATA) f2python9-final/f2python9.{ps.gz,pdf,html} f2python9-final/{flow,structure,aerostructure}.jpg $(WWWDIR)
clean:
	rm -f tmp_$(MAINLATEX).* $(MAINLATEX).{aux,dvi,log,toc}
distclean:
	rm -f tmp_$(MAINLATEX).* $(MAINLATEX).{aux,dvi,log,toc}
	rm -f $(TOP).{ps,dvi,html,pdf,ps.gz}
	rm -f *~
