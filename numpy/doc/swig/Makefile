INTERFACES = Series.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = $(INTERFACES:.i=.py      )

all: $(WRAPPERS) series.cxx series.h
	./setup.py build

%_wrap.cxx: %.i numpy.i series.h
	swig -c++ -python $<

clean:
	$(RM) -r build
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)
