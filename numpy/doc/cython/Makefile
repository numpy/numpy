# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

help:
	@echo "Numpy/Cython tasks.  Available tasks:"
	@echo "ext  -> build the Cython extension module."
	@echo "html -> create annotated HTML from the .pyx sources"
	@echo "test -> run a simple test demo."
	@echo "all  -> Call ext, html and finally test."

all: ext html test

ext: numpyx.so

test:   ext
	python run_test.py

html: numpyx.pyx.html

numpyx.so: numpyx.pyx numpyx.c
	python setup.py build_ext --inplace

numpyx.pyx.html: numpyx.pyx
	cython -a numpyx.pyx
	@echo "Annotated HTML of the C code generated in numpyx.html"

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build

# Suffix rules
%.c : %.pyx
	cython $<
