try:
	from foo_py import foo
	__all__ = ['foo']
except ImportError, e:
	print "Warning: Error importing pyext, error was %s" % e
	__all__ = []

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
