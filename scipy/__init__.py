"""\
SciPy Core
==========

You can support the development of SciPy by purchasing documentation
at

  http://www.trelgol.com

It is being distributed for a fee for a limited time to try and raise
money for development.

Documentation is also available in the docstrings.

Available subpackages
---------------------
"""

import os, sys
NO_SCIPY_IMPORT = os.environ.get('NO_SCIPY_IMPORT',None)

try:
    from __core_config__ import show as show_core_config
except ImportError:
    show_core_config = None

if show_core_config is None:
    print >> sys.stderr, 'Running from scipy core source directory.'
else:
    from _import_tools import PackageImport
    from core_version import version as __core_version__
    __doc__ += PackageImport().import_packages(\
        ['test','basic','base'])
    test = ScipyTest('scipy').test

__scipy_doc__ = """

SciPy: A scientific computing package for Python
================================================

Available subpackages
---------------------
"""

if NO_SCIPY_IMPORT is not None:
    print >> sys.stderr, 'Skip importing scipy packages (NO_SCIPY_IMPORT=%s)' % (NO_SCIPY_IMPORT)
    show_scipy_config = None
elif show_core_config is None:
    show_scipy_config = None
else:
    try:
        from __scipy_config__ import show as show_scipy_config
    except ImportError:
        show_scipy_config = None


if show_scipy_config is not None:
    from scipy_version import scipy_version as __scipy_version__
    __doc__ += __scipy_doc__
    __doc__ += PackageImport().import_packages()

