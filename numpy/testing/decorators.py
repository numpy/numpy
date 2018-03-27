"""
Back compatibility decorators module. It will import the appropriate
set of tools

"""
import warnings

warnings.warn(ImportWarning,
    "Import from numpy.testing, not numpy.testing.decorators")

from ._private.decorators import *
