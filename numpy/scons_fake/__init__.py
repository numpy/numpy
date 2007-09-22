from numpy.ctypeslib import load_library

_FOO = load_library("libfoo.so", __file__)
def foo():
    _FOO.foo()
    
