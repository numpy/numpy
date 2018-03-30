"""
Back compatibility noseclasses module. It will import the appropriate
set of tools
"""
import warnings

warnings.warn("Import from numpy.testing, not numpy.testing.noseclasses",
              ImportWarning)

from ._private.noseclasses import *

