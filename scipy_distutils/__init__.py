"""scipy_distutils

   Modified version of distutils to handle fortran source code, f2py,
   and other issues in the scipy build process.
"""

# Need to do something here to get distutils subsumed...

from scipy_distutils_version import scipy_distutils_version as __version__

import sys

# Replace distutils.ccompiler with scipy_distutils.ccompiler
assert not sys.modules.has_key('distutils.ccompiler'),\
       'distutils has been imported before scipy_distutils'
import ccompiler
sys.modules['distutils.ccompiler'] = ccompiler

assert not sys.modules.has_key('distutils.unixccompiler'),\
       'cannot override distutils.unixccompiler'
import unixccompiler
sys.modules['distutils.unixccompiler'] = unixccompiler
