# Need to do something here to get distutils subsumed...
from info_scipy_distutils import __doc__
from scipy_distutils_version import scipy_distutils_version as __version__

import sys

# Must import local ccompiler ASAP in order to get
# customized CCompiler.spawn effective.
import ccompiler
import unixccompiler
