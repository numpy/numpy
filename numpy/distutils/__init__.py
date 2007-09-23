
from __version__ import version as __version__

# Check that distutils has not been imported before.
import sys
if 'distutils' in sys.modules:
    sys.stderr.write('''\
********************************************************
WARNING!WARNING!WARNING!WARNING!WARNING!WARNING!WARNING!

distutils has been imported before numpy.distutils
and now numpy.distutils cannot apply all of its
customizations to distutils effectively.

To avoid this warning, make sure that numpy.distutils
is imported *before* distutils.
********************************************************
''')

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
