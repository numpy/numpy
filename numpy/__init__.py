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
    from __config__ import show as show_config
except ImportError:
    show_config = None

try:
    import pkg_resources # activate namespace packages (manipulates __path__)
except ImportError:
    pass

if show_config is None:
    print >> sys.stderr, 'Running from numpy source directory.'
else:
    from version import version as __version__
    from testing import ScipyTest
    from core import *
    from lib import *
    from dft import fft, ifft
    from random import rand, randn
    __all__ = filter(lambda s:not s.startswith('_'),dir())

    test = ScipyTest('numpy').test

    # TODO: Fix __doc__
