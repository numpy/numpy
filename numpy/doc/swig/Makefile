# SWIG
INTERFACES = Series.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = $(INTERFACES:.i=.py      )

# ReStructured Text
RST2HTML = rst2html.py
RFLAGS = --generator --time --no-xml-declaration

# Web pages that need to be made
WEB_PAGES = numpy_swig.html

# List all of the subdirectories here for recursive make
SUBDIRS = 

all: $(WRAPPERS) series.cxx series.h html
	./setup.py build

%_wrap.cxx: %.i numpy.i series.h
	swig -c++ -python $<

html: $(WEB_PAGES)

%.html: %.txt
	$(RST2HTML) $(RFLAGS) $< $@

clean:
	$(RM) -r build
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)

.PHONY : html clean
