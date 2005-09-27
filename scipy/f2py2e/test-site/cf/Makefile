
all:
compile:
	f2py test.f -m test2
	ccc -c -O -I`f2py -I` test2module.c  -o test2module.o
	fort -c test.f
	fort -Wl,-s -shared test2module.o test.o -o test2module.so
test:
	python -c "import test2;print dir(test2);test2.foo(5);print 'ok'"
clean:
	rm -f *.o *.so core
