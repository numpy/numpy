"""
Back compatibility noseclasses module. It will import the appropriate
set of tools
"""
import warnings

warnings.warn(ImportWarning,
    "Import from numpy.testing, not numpy.testing.noseclasses")

from ._private.noseclasses import *

