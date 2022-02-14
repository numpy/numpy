"""Fortran to Python Interface Generator."""

__all__ = ['run_main', 'compile', 'get_include']

from numpy.f2py import frontend
from numpy.f2py import __version__
# Helpers
from numpy.f2py.utils.pathhelper import get_include
# Build tool
from numpy.f2py.utils.npdist import compile
from numpy.f2py.frontend.f2py2e import main

run_main = numpy.f2py.frontend.f2py2e.run_main

if __name__ == "__main__":
    sys.exit(main())

def __getattr__(attr):

    # Avoid importing things that aren't needed for building
    # which might import the main numpy module
    if attr == "test":
        from numpy.f2py import f2py_testing
        from numpy._pytesttester import PytestTester
        test = PytestTester(__name__)
        return test

    else:
        raise AttributeError("module {!r} has no attribute "
                              "{!r}".format(__name__, attr))


def __dir__():
    return list(globals().keys() | {"test"})
