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

import os
NO_SCIPY_IMPORT = os.environ.get('NO_SCIPY_IMPORT',None)

try:
    from __core_config__ import show as show_core_config
except ImportError:
    show_core_config = None

if show_core_config is None:
    print 'Running from scipy core source directory.'
else:
    from scipy.base import *
    import scipy.basic as basic
    from scipy.basic.fft import fft, ifft
    from scipy.basic.random import rand, randn
    import scipy.basic.fft as fftpack
    import scipy.basic.linalg as linalg
    import scipy.basic.random as random
    from core_version import version as __core_version__
    from scipy.test.testing import ScipyTest
    test = ScipyTest('scipy.base').test

__scipy_doc__ = """\
SciPy: A scientific computing package for Python
================================================

Available subpackages
---------------------
"""

if NO_SCIPY_IMPORT is not None:
    print 'Skip importing scipy packages (NO_SCIPY_IMPORT=%s)' % (NO_SCIPY_IMPORT)
    show_scipy_config = None
elif show_core_config is None:
    show_scipy_config = None
else:
    try:
        from __scipy_config__ import show as show_scipy_config
    except ImportError:
        show_scipy_config = None


if show_scipy_config is not None:
    __doc__ += __scipy_doc__
    from scipy_version import scipy_version as __scipy_version__
    from _import_tools import import_packages
    import_packages()
