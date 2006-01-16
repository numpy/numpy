"""\
NumPy
==========

You can support the development of NumPy and SciPy by purchasing
documentation at

  http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise
money for development.

Documentation is also available in the docstrings.

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

    import os as _os
    NUMPY_IMPORT_VERBOSE = int(_os.environ.get('NUMPY_IMPORT_VERBOSE','0'))
    del _os
    from _import_tools import PackageLoader
    pkgload = PackageLoader()
    pkgload('testing','core','linalg','lib','dft','random','f2py','distutils',
            verbose=NUMPY_IMPORT_VERBOSE,postpone=False)

    __doc__ += """

Available subpackages
---------------------
"""
    __doc__ += pkgload.get_pkgdocs()

    test = ScipyTest('numpy').test
    import add_newdocs

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
