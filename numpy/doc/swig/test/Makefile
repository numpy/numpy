# SWIG
INTERFACES = Vector.i Matrix.i Tensor.i
WRAPPERS   = $(INTERFACES:.i=_wrap.cxx)
PROXIES    = $(INTERFACES:.i=.py      )

# Default target: build the tests
.PHONY : all
all: $(WRAPPERS) Vector.cxx Vector.h Matrix.cxx Matrix.h Tensor.cxx Tensor.h
	./setup.py build

# Test target: run the tests
.PHONY : test
test: all
	testVector.py
	testMatrix.py
	testTensor.py

# Rule: %.i -> %_wrap.cxx
%_wrap.cxx: %.i %.h ../numpy.i
	swig -c++ -python $<

# Clean target
.PHONY : clean
clean:
	$(RM) -r build
	$(RM) $(WRAPPERS)
	$(RM) $(PROXIES)
