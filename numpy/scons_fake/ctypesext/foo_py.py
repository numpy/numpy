try:
	from numpy.ctypeslib import load_library
	_FOO = load_library("foo", __file__)
	def foo():
	    _FOO.foo()
except ImportError:
	print "ctypes not available"
	def foo():
	    pass

