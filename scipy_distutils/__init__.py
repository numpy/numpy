"""scipy_distutils

   Modified version of distutils to handle fortran source code, f2py,
   and other issues in the scipy build process.
"""

# Need to do something here to get distutils subsumed...

from scipy_distutils_version import scipy_distutils_version as __version__

import sys

# Must import local ccompiler ASAP in order to get
# customized CCompiler.spawn effective.
import ccompiler
import unixccompiler
