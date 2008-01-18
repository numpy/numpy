"""
NumPy
==========

You can support the development of NumPy and SciPy by purchasing
the book "Guide to NumPy" at

  http://www.trelgol.com

It is being distributed for a fee until Oct. 2010 to
cover some of the costs of development.  After the restriction period
it will also be freely available.

Documentation is available in the docstrings and at

http://www.scipy.org.
"""

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False


if __NUMPY_SETUP__:
    import sys as _sys
    print >> _sys.stderr, 'Running from numpy source directory.'
    del _sys
else:
    from numpy.__config__ import show as show_config
    from version import version as __version__

    from _import_tools import PackageLoader

    def pkgload(*packages, **options):
        loader = PackageLoader(infunc=True)
        return loader(*packages, **options)

    pkgload.__doc__ = PackageLoader.__call__.__doc__
    import testing
    from testing import ScipyTest, NumpyTest
    import core
    from core import *
    import lib
    from lib import *
    import linalg
    import fft
    import random
    import ctypeslib
    import ma

    # Make these accessible from numpy name-space
    #  but not imported in from numpy import *
    from __builtin__ import bool, int, long, float, complex, \
         object, unicode, str
    from core import round, abs, max, min

    __all__ = ['__version__', 'pkgload', 'PackageLoader',
               'ScipyTest', 'NumpyTest', 'show_config']
    __all__ += core.__all__
    __all__ += lib.__all__
    __all__ += ['linalg', 'fft', 'random', 'ctypeslib']

    if __doc__ is not None:
        __doc__ += """

Available subpackages
---------------------
core      --- Defines a multi-dimensional array and useful procedures
              for Numerical computation.
lib       --- Basic functions used by several sub-packages and useful
              to have in the main name-space.
random    --- Core Random Tools
linalg    --- Core Linear Algebra Tools
fft       --- Core FFT routines
testing   --- Numpy testing tools

  These packages require explicit import
f2py      --- Fortran to Python Interface Generator.
distutils --- Enhancements to distutils with support for
              Fortran compilers support and more.


Global symbols from subpackages
-------------------------------
core    --> * (use numpy.* not numpy.core.*)
lib     --> * (use numpy.* not numpy.lib.*)
testing --> NumpyTest
"""

    def test(*args, **kw):
        import os, sys
        print 'Numpy is installed in %s' % (os.path.split(__file__)[0],)
        print 'Numpy version %s' % (__version__,)
        print 'Python version %s' % (sys.version.replace('\n', '',),)
        return NumpyTest().test(*args, **kw)
    test.__doc__ = NumpyTest.test.__doc__

    import add_newdocs

    __all__.extend(['add_newdocs'])

    if __doc__ is not None:
        __doc__ += """

Utility tools
-------------

  test        --- Run numpy unittests
  pkgload     --- Load numpy packages
  show_config --- Show numpy build configuration
  dual        --- Overwrite certain functions with high-performance Scipy tools
  matlib      --- Make everything matrices.
  __version__ --- Numpy version string
"""
