from __future__ import division as _, absolute_import as _, print_function as _

import sys

from .__version__ import version as __version__
# Must import local ccompiler ASAP in order to get
# customized CCompiler.spawn effective.
from . import ccompiler
from . import unixccompiler

from .info import __doc__
from .npy_pkg_config import *

# If numpy is installed, add distutils.test()
try:
    from . import __config__
    # Normally numpy is installed if the above import works, but an interrupted
    # in-place build could also have left a __config__.py.  In that case the
    # next import may still fail, so keep it inside the try block.
    from numpy.testing._private.pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester
except ImportError:
    pass


def customized_fcompiler(plat=None, compiler=None):
    from numpy.distutils.fcompiler import new_fcompiler
    c = new_fcompiler(plat=plat, compiler=compiler)
    c.customize()
    return c

def customized_ccompiler(plat=None, compiler=None):
    c = ccompiler.new_compiler(plat=plat, compiler=compiler)
    c.customize('')
    return c
