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
"""

import os, sys
NUMPY_IMPORT_VERBOSE = int(os.environ.get('NUMPY_IMPORT_VERBOSE','0'))

try:
    from __core_config__ import show as show_core_config
except ImportError:
    show_core_config = None

try:
    import pkg_resources # activate namespace packages (manipulates __path__)
except ImportError:
    pass

import _import_tools
pkgload = _import_tools.PackageLoader()

if show_core_config is None:
    print >> sys.stderr, 'Running from numpy core source directory.'
else:
    from version import version as __version__

    pkgload('testing','core','lib','dft','linalg','random',
            verbose=NUMPY_IMPORT_VERBOSE)

    test = ScipyTest('numpy').test
    __all__.append('test')

__numpy_doc__ = """

NumPy: A scientific computing package for Python
================================================

Available subpackages
---------------------
"""

if show_core_config is None:
    show_numpy_config = None
else:
    try:
        from __numpy_config__ import show as show_numpy_config
    except ImportError:
        show_numpy_config = None


if show_numpy_config is not None:
    from numpy_version import numpy_version as __numpy_version__
    __doc__ += __numpy_doc__
    pkgload(verbose=NUMPY_IMPORT_VERBOSE,postpone=True)
