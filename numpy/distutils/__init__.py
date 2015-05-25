from __future__ import division, absolute_import, print_function

import sys

from .__version__ import version as __version__
# Must import local ccompiler ASAP in order to get
# customized CCompiler.spawn effective.
from . import ccompiler
from . import unixccompiler

from .info import __doc__
from .npy_pkg_config import *

try:
    from . import __config__
    _INSTALLED = True
except ImportError:
    _INSTALLED = False

if _INSTALLED:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
