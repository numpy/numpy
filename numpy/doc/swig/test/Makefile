# SWIG
INTERFACES = Array.i Vector.i Matrix.i Tensor.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = $(INTERFACES:.i=.py      )

# Default target: build the tests
.PHONY : all
all: $(WRAPPERS) Array1.cxx Array1.h Vector.cxx Vector.h Matrix.cxx Matrix.h \
	Tensor.cxx Tensor.h
	./setup.py build

# Test target: run the tests
.PHONY : test
test: all
	python testVector.py
	python testMatrix.py
	python testTensor.py
	python testArray.py

# Rule: %.i -> %_wrap.cxx
%_wrap.cxx: %.i %.h ../numpy.i
	swig -c++ -python $<
%_wrap.cxx: %.i %1.h %2.h ../numpy.i
	swig -c++ -python $<

# Clean target
.PHONY : clean
clean:
	$(RM) -r build
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)
