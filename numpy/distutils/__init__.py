
from __version__ import version as __version__

# Must import local ccompiler ASAP in order to get
# customized CCompiler.spawn effective.
import ccompiler
import unixccompiler

from info import __doc__

try:
    import __config__
    _INSTALLED = True
except ImportError:
    _INSTALLED = False

if _INSTALLED:
    def test(level=1, verbosity=1):
        from numpy.testing import NumpyTest
        return NumpyTest().test(level, verbosity)
