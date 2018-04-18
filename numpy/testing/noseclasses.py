"""
Back compatibility noseclasses module. It will import the appropriate
set of tools
"""
from __future__ import division as _, absolute_import as _, print_function as _

import warnings

# 2018-04-04, numpy 1.15.0
warnings.warn("Importing from numpy.testing.noseclasses is deprecated, "
              "import from numpy.testing instead",
              DeprecationWarning, stacklevel=2)

from ._private.noseclasses import *
