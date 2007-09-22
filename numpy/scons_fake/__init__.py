from numpy.ctypeslib import load_library

_FOO = load_library("foo.dll", __file__)
def foo():
    _FOO.foo()
    
