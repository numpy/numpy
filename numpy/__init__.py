"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <http://www.scipy.org>`_.

We recommend exploring the docstrings using
`IPython <http://ipython.scipy.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as `np`::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)

For some objects, ``np.info(obj)`` may provide additional help.

To search for objects of which the documentation contains keywords, do::

  >>> np.lookfor('keyword')

Available subpackages
---------------------
core
    Defines a multi-dimensional array and useful procedures
    for Numerical computation.
lib
    Basic functions used by several sub-packages and useful
    to have in the main name-space.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
testing
    Numpy testing tools

The following sub-packages must be explicitly imported:

f2py
    Fortran to Python Interface Generator.
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.

Global symbols from subpackages
-------------------------------
Do not import directly from `core` and `lib`: those functions
have been imported into the `numpy` namespace.

Utility tools
-------------
test
    Run numpy unittests
pkgload
    Load numpy packages
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance Scipy tools
matlib
    Make everything matrices.
__version__
    Numpy version string

Viewing documentation using IPython
-----------------------------------
Start IPython with the NumPy profile (``ipython -p numpy``), which will
import `numpy` under the alias `np`.  Then, use the ``cpaste`` command to
paste examples into the shell.  To see which functions are available in
`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the methods in `numpy` return a copy of the array argument (e.g.,
`sort`).  In-place versions of these methods are often available as
array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.  Exceptions to
this rule are documented.

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
    try:
        from numpy.__config__ import show as show_config
    except ImportError, e:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python intepreter from there."""
        raise ImportError(msg)
    from version import version as __version__

    from _import_tools import PackageLoader

    def pkgload(*packages, **options):
        loader = PackageLoader(infunc=True)
        return loader(*packages, **options)

    import add_newdocs
    __all__ = ['add_newdocs']

    pkgload.__doc__ = PackageLoader.__call__.__doc__

    from testing import Tester
    test = Tester().test
    bench = Tester().bench

    import core
    from core import *
    import lib
    from lib import *
    import linalg
    import fft
    import random
    import ctypeslib
    import ma
    import doc

    # Make these accessible from numpy name-space
    #  but not imported in from numpy import *
    from __builtin__ import bool, int, long, float, complex, \
         object, unicode, str
    from core import round, abs, max, min

    __all__.extend(['__version__', 'pkgload', 'PackageLoader',
               'show_config'])
    __all__.extend(core.__all__)
    __all__.extend(lib.__all__)
    __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma', 'doc'])

