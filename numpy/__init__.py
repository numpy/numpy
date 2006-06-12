"""\
NumPy
==========

You can support the development of NumPy and SciPy by purchasing
extended documentation at

  http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise
money for development.

Documentation is also available in the docstrings.

"""

try:
    from __config__ import show as show_config
except ImportError:
    show_config = None

if show_config is None:
    import sys as _sys
    print >> _sys.stderr, 'Running from numpy source directory.'
    del _sys
else:
    from version import version as __version__

    import os as _os
    NUMPY_IMPORT_VERBOSE = int(_os.environ.get('NUMPY_IMPORT_VERBOSE','0'))
    del _os
    from _import_tools import PackageLoader
    pkgload = PackageLoader()
    pkgload('testing','core','lib','linalg','dft','random','f2py',
            verbose=NUMPY_IMPORT_VERBOSE,postpone=False)
        
    if __doc__ is not None:
        __doc__ += """

Available subpackages
---------------------
"""
    if __doc__ is not None:
        __doc__ += pkgload.get_pkgdocs()

    def test(level=1, verbosity=1):
        return NumpyTest().test(level, verbosity)

    import add_newdocs

    if __doc__ is not None:
        __doc__ += """

Utility tools
-------------

  test        --- Run numpy unittests
  pkgload     --- Load numpy packages
  show_config --- Show numpy build configuration
  dual        --- Overwrite certain functions with high-performance Scipy tools
  __version__ --- Numpy version string

Environment variables
---------------------

  NUMPY_IMPORT_VERBOSE --- pkgload verbose flag, default is 0.
"""
