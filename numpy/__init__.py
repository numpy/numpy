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

import os as _os
import sys as _sys
NUMPY_IMPORT_VERBOSE = int(_os.environ.get('NUMPY_IMPORT_VERBOSE','0'))

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
    print >> _sys.stderr, 'Running from numpy source directory.'
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

    __all__ = filter(lambda s:not s.startswith('_'),dir())

    test = ScipyTest('numpy').test

    import add_newdocs

    # TODO: Fix __doc__

del _os, _sys
