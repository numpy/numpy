"""\
NumPy
==========

You can support the development of NumPy and SciPy by purchasing
documentation at

  http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise
money for development.

Documentation is also available in the docstrings.

Available subpackages
---------------------

core      - Defines a multi-dimensional array and useful procedures
            for Numerical computation.
lib       - Basic functions used by several sub-packages and useful
            to have in the main name-space
dft       - Core FFT routines
linalg    - Core Linear Algebra Tools
random    - Core Random Tools
testing   - Scipy testing tools
distutils - Enhanced distutils with Fortran compiler support and more
f2py      - Fortran to Python interface generator
dual      - Overwrite certain functions with high-performance Scipy tools

Available tools
---------------

core, lib namespaces
fft, ifft   - Functions for FFT and inverse FFT
rand, randn - Functions for uniform and normal random numbers.
test        - Method to run numpy unittests.
__version__ - Numpy version string
show_config - Show numpy build configuration
"""

try:
    from __config__ import show as show_config
except ImportError:
    show_config = None

try:
    import pkg_resources as _pk # activate namespace packages (manipulates __path__)
    del _pk
except ImportError:
    pass

if show_config is None:
    import sys as _sys
    print >> _sys.stderr, 'Running from numpy source directory.'
    del _sys
else:
    from version import version as __version__
    from testing import ScipyTest
    from core import *
    from lib import *
    from dft import fft, ifft
    from random import rand, randn

    __all__ = ['ScipyTest','fft','ifft','rand','randn']
    __all__ += core.__all__
    __all__ += lib.__all__

    test = ScipyTest('numpy').test

    import add_newdocs
